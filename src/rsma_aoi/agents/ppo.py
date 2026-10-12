"""Proximal Policy Optimization for the hybrid RSMA/AoI action space.

Implements the clipped surrogate objective of Schulman et al. (2017)::

    ratio = exp(log pi_new(z) - log pi_old(z))
    L_clip = E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)]
    loss   = -L_clip + c_v * L_value - c_H * H[pi]

``z`` is the stored *latent* action (see
:mod:`rsma_aoi.agents.distributions`), so old and new log-probabilities always
refer to the same random variables and the same scheduling masks.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from rsma_aoi.agents.distributions import HybridPolicyDistribution, LatentAction
from rsma_aoi.agents.networks import HybridActorCritic, RecurrentHybridActorCritic
from rsma_aoi.agents.rollout_buffer import RecurrentRolloutBuffer, RolloutBuffer
from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.utils.config import EnvConfig, PPOConfig, dump_config_snapshot
from rsma_aoi.utils.logging import RunLogger, get_logger, git_commit_hash
from rsma_aoi.utils.running_stats import RunningMeanStd
from rsma_aoi.utils.seeding import (
    SeedBundle, make_torch_generator, seed_everything, spawn_seeds,
)
from rsma_aoi.utils.vec_env import SyncVecEnv


def resolve_schedule_spec(env_cfg: EnvConfig, ppo_cfg: PPOConfig) -> tuple[int, tuple[int, ...]]:
    """Default the curriculum schedule sizes to the spatial multiplexing limit."""
    limit = max(1, min(env_cfg.num_users, env_cfg.num_antennas))
    size = ppo_cfg.schedule_size or limit
    sizes = tuple(ppo_cfg.schedule_sizes or range(1, limit + 1))
    return size, sizes


def max_aoi_masks(states: list[dict[str, Any]], num_scheduled: int) -> np.ndarray:
    """Deterministic max-weighted-AoI-first schedule for the fixed_top_m stage."""
    masks = []
    for state in states:
        priority = np.asarray(state["weights"]) * np.asarray(state["aoi"])
        order = np.argsort(-priority, kind="stable")
        mask = np.zeros(priority.shape[0], dtype=bool)
        mask[order[: max(1, num_scheduled)]] = True
        masks.append(mask)
    return np.stack(masks)


def resolve_device(name: str) -> torch.device:
    """Validate and return a torch device, failing early with a clear message."""
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={name!r} was requested but CUDA is not available in this "
            "environment; use device: cpu (the CPU path is the tested default)"
        )
    return device


class RewardNormalizer:
    """Scale rewards by the running std of the discounted return."""

    def __init__(self, num_envs: int, gamma: float) -> None:
        self.rms = RunningMeanStd(shape=())
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.gamma = gamma

    def __call__(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        self.returns = self.returns * self.gamma + rewards
        self.rms.update(self.returns)
        scaled = rewards / np.sqrt(self.rms.var + 1e-8)
        self.returns = np.where(dones, 0.0, self.returns)
        return np.clip(scaled, -10.0, 10.0).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"rms": self.rms.state_dict(), "returns": self.returns.copy()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rms.load_state_dict(state["rms"])
        self.returns = np.asarray(state["returns"], dtype=np.float64)


def latent_to_env_actions(
    dist: HybridPolicyDistribution, latent: LatentAction
) -> list[dict[str, np.ndarray]]:
    """Transform a batched latent action into per-environment action dicts.

    Args:
        dist: the distribution the latent was drawn from (its masked softmax
            transforms are what make the action feasible).
        latent: batched latent action of size ``B``.

    Returns:
        ``B`` action dicts matching :class:`~rsma_aoi.envs.aoi_env.AoIRSMAEnv`'s
        ``Dict`` action space.
    """
    with torch.no_grad():
        batched = dist.to_env_action(latent)
    actions: list[dict[str, np.ndarray]] = []
    n = latent.power_latent.shape[0]
    for i in range(n):
        action = {"schedule": batched["schedule"][i].cpu().numpy().astype(np.int8)}
        if dist.direct_fraction:
            action.update({
                "common_power_fraction": batched["common_power_fraction"][i].cpu().numpy().astype(np.float64),
                "private_power_fractions": batched["private_power_fractions"][i].cpu().numpy().astype(np.float64),
                "common_payload_fractions": batched["common_payload_fractions"][i].cpu().numpy().astype(np.float64),
            })
        else:
            action.update({
                "power_fractions": batched["power_fractions"][i].cpu().numpy().astype(np.float64),
                "rate_fractions": batched["rate_fractions"][i].cpu().numpy().astype(np.float64),
                "common_payload_scale": batched["common_payload_scale"][i].cpu().numpy().astype(np.float64),
            })
        if "power_scale" in batched:
            action["power_scale"] = batched["power_scale"][i].cpu().numpy().astype(np.float64)
        actions.append(action)
    return actions


class PPOPolicy:
    """Inference-only wrapper (shared by evaluation and checkpoint loading)."""

    name = "ppo"

    def __init__(
        self,
        model: HybridActorCritic | RecurrentHybridActorCritic,
        obs_rms: RunningMeanStd | None,
        device: torch.device | str = "cpu",
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        self.model = model
        self.obs_rms = obs_rms
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.generator = generator
        self.schedule_size = getattr(model, "schedule_size", None) or 1
        self.actor_hidden: torch.Tensor | None = None
        self.critic_hidden: torch.Tensor | None = None

    def reset(self, seed: int | None = None) -> None:
        """Reset recurrent memory at an episode boundary."""
        del seed
        self.actor_hidden = None
        self.critic_hidden = None

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms is None:
            return obs.astype(np.float32)
        return self.obs_rms.normalize(obs)

    def act(
        self, obs: np.ndarray, states: list[dict[str, Any]] | None = None
    ) -> list[dict[str, np.ndarray]]:
        """Map a batch of observations to environment actions."""
        obs_t = torch.as_tensor(self.normalize(obs), dtype=torch.float32, device=self.device)
        forced = None
        if self.model.scheduling_head == "fixed_top_m":
            if states is None:
                raise ValueError(
                    "the fixed_top_m policy needs the structured environment state "
                    "to compute its deterministic schedule"
                )
            forced = torch.as_tensor(
                max_aoi_masks(states, self.schedule_size), dtype=torch.bool, device=self.device
            )
        with torch.no_grad():
            if isinstance(self.model, RecurrentHybridActorCritic):
                batch = obs_t.shape[0]
                if self.actor_hidden is None or self.actor_hidden.shape[0] != batch:
                    self.actor_hidden, self.critic_hidden = self.model.initial_state(
                        batch, self.device
                    )
                assert self.critic_hidden is not None
                dist, _value, self.actor_hidden, self.critic_hidden = self.model.recurrent_step(
                    obs_t, self.actor_hidden, self.critic_hidden, forced_mask=forced
                )
            else:
                dist = self.model.distribution(obs_t, forced_mask=forced)
            latent = dist.mode() if self.deterministic else dist.sample(self.generator)
        return latent_to_env_actions(dist, latent)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str = "cpu", deterministic: bool = True
    ) -> tuple["PPOPolicy", EnvConfig, PPOConfig]:
        """Rebuild a policy (and its configs) from a training checkpoint."""
        device = str(resolve_device(device))
        ckpt = torch.load(Path(path), map_location=device, weights_only=False)
        env_cfg = EnvConfig(**ckpt["env_config"])
        ppo_cfg = PPOConfig(**ckpt["ppo_config"])
        probe = AoIRSMAEnv(env_cfg)
        _size, _sizes = resolve_schedule_spec(env_cfg, ppo_cfg)
        model_cls = RecurrentHybridActorCritic if ppo_cfg.recurrent_enabled else HybridActorCritic
        model_kwargs = dict(
            obs_dim=int(np.prod(probe.observation_space.shape)),
            num_users=env_cfg.num_users,
            hidden_sizes=ppo_cfg.hidden_sizes,
            activation=ppo_cfg.activation,
            log_std_init=ppo_cfg.log_std_init,
            scalar_dist=ppo_cfg.scalar_dist,
            scheduling_enabled=env_cfg.scheduling_enabled,
            power_scale_action=env_cfg.power_scale_action,
            scheduling_head=ppo_cfg.scheduling_head,
            conditional_allocation=ppo_cfg.conditional_allocation,
            schedule_size=_size,
            schedule_sizes=_sizes,
            count_prior=ppo_cfg.count_prior,
            allocation_control=ppo_cfg.allocation_control,
            fixed_common_power_fraction=ppo_cfg.fixed_common_power_fraction,
            fixed_common_payload_scale=ppo_cfg.fixed_common_payload_scale,
            initial_common_power_fraction=ppo_cfg.initial_common_power_fraction,
            initial_common_payload_scale=ppo_cfg.effective_initial_common_fraction,
            initial_scalar_concentration=ppo_cfg.initial_scalar_concentration,
            common_gate_enabled=ppo_cfg.common_gate_enabled,
            rate_split_mode=env_cfg.rate_split_mode,
            initial_common_on_probability=ppo_cfg.initial_common_on_probability,
        )
        if ppo_cfg.recurrent_enabled:
            model_kwargs["recurrent_hidden_size"] = ppo_cfg.recurrent_hidden_size
        model = model_cls(**model_kwargs).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        obs_rms = None
        if ckpt.get("obs_rms") is not None:
            obs_rms = RunningMeanStd(shape=probe.observation_space.shape)
            obs_rms.load_state_dict(ckpt["obs_rms"])
        probe.close()
        return cls(model, obs_rms, device=device, deterministic=deterministic), env_cfg, ppo_cfg


class PPOTrainer:
    """End-to-end PPO training loop with vectorised environments."""

    def __init__(
        self,
        env_cfg: EnvConfig,
        ppo_cfg: PPOConfig,
        run_dir: str | Path,
        seed: int = 0,
        env_factory: Callable[[EnvConfig], AoIRSMAEnv] | None = None,
    ) -> None:
        self.env_cfg = env_cfg
        if ppo_cfg.allocation_control != "joint":
            valid_rate_mode = (
                env_cfg.rate_split_mode == "simplex_scaled"
                or (
                    ppo_cfg.allocation_control == "sdma_power"
                    and env_cfg.rate_split_mode == "per_user_fraction"
                )
            )
            if (
                env_cfg.allocator_mode != "none"
                or env_cfg.power_scale_action
                or env_cfg.multiple_access != "rsma"
                or not valid_rate_mode
            ):
                raise ValueError(
                    "allocation ablations require direct RSMA with fixed total power; "
                    "per_user_fraction is supported only by sdma_power"
                )
        if ppo_cfg.common_gate_enabled and (
            ppo_cfg.allocation_control != "joint" or env_cfg.allocator_mode != "none"
            or env_cfg.power_scale_action or env_cfg.multiple_access != "rsma"
            or env_cfg.rate_split_mode != "simplex_scaled" or ppo_cfg.conditional_allocation
        ):
            raise ValueError("common gate requires joint direct simplex_scaled RSMA with fixed total power")
        if ppo_cfg.recurrent_enabled and ppo_cfg.reference_kl_coef > 0.0:
            raise ValueError("reference_kl_coef is not yet supported by recurrent PPO")
        self.cfg = ppo_cfg
        self.seeds = SeedBundle(seed)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(ppo_cfg.device)
        self.logger = get_logger()

        seed_everything(self.seeds.agent)
        self.torch_gen = make_torch_generator(self.seeds.agent, device="cpu")
        self.np_rng = np.random.default_rng(self.seeds.agent)

        factory = env_factory or (lambda cfg: AoIRSMAEnv(cfg))
        self.envs = SyncVecEnv([lambda: factory(env_cfg) for _ in range(ppo_cfg.num_envs)])
        self.obs_dim = self.envs.obs_dim
        self.schedule_size, self.schedule_sizes = resolve_schedule_spec(env_cfg, ppo_cfg)
        self.uses_forced_mask = ppo_cfg.scheduling_head == "fixed_top_m"

        model_cls = RecurrentHybridActorCritic if ppo_cfg.recurrent_enabled else HybridActorCritic
        model_kwargs = dict(
            obs_dim=self.obs_dim,
            num_users=env_cfg.num_users,
            hidden_sizes=ppo_cfg.hidden_sizes,
            activation=ppo_cfg.activation,
            log_std_init=ppo_cfg.log_std_init,
            scalar_dist=ppo_cfg.scalar_dist,
            scheduling_enabled=env_cfg.scheduling_enabled,
            power_scale_action=env_cfg.power_scale_action,
            scheduling_head=ppo_cfg.scheduling_head,
            conditional_allocation=ppo_cfg.conditional_allocation,
            schedule_size=self.schedule_size,
            schedule_sizes=self.schedule_sizes,
            count_prior=ppo_cfg.count_prior,
            allocation_control=ppo_cfg.allocation_control,
            fixed_common_power_fraction=ppo_cfg.fixed_common_power_fraction,
            fixed_common_payload_scale=ppo_cfg.fixed_common_payload_scale,
            initial_common_power_fraction=ppo_cfg.initial_common_power_fraction,
            initial_common_payload_scale=ppo_cfg.effective_initial_common_fraction,
            initial_scalar_concentration=ppo_cfg.initial_scalar_concentration,
            common_gate_enabled=ppo_cfg.common_gate_enabled,
            rate_split_mode=env_cfg.rate_split_mode,
            initial_common_on_probability=ppo_cfg.initial_common_on_probability,
        )
        if ppo_cfg.recurrent_enabled:
            model_kwargs["recurrent_hidden_size"] = ppo_cfg.recurrent_hidden_size
        self.model = model_cls(**model_kwargs).to(self.device)
        self.reference_model: HybridActorCritic | None = None
        self.warm_start_meta: dict[str, Any] = {}

        # load_state_dict mutates parameters in place, so building the optimizer
        # before the warm start still binds the right tensors
        critic_names = (
            "critic_trunk", "value_head", "critic_gru", "critic_recurrent_proj"
        )
        critic_params = [
            p for n, p in self.model.named_parameters() if n.startswith(critic_names)
        ]
        actor_params = [
            p for n, p in self.model.named_parameters() if not n.startswith(critic_names)
        ]
        self.optimizer = torch.optim.Adam(
            [
                {"params": actor_params, "lr": ppo_cfg.actor_lr or ppo_cfg.learning_rate,
                 "name": "actor"},
                {"params": critic_params, "lr": ppo_cfg.critic_lr or ppo_cfg.learning_rate,
                 "name": "critic"},
            ],
            eps=1e-5,
        )
        self._base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        buffer_cls = RecurrentRolloutBuffer if ppo_cfg.recurrent_enabled else RolloutBuffer
        buffer_kwargs = dict(
            num_steps=ppo_cfg.rollout_steps,
            num_envs=ppo_cfg.num_envs,
            obs_dim=self.obs_dim,
            num_users=env_cfg.num_users,
            power_scale_action=env_cfg.power_scale_action,
            device=self.device,
            schedule_dim=self.model.schedule_action_dim,
            common_gate_enabled=ppo_cfg.common_gate_enabled,
        )
        if ppo_cfg.recurrent_enabled:
            buffer_kwargs.update(
                hidden_size=ppo_cfg.recurrent_hidden_size,
                sequence_length=ppo_cfg.recurrent_seq_len,
            )
        self.buffer = buffer_cls(**buffer_kwargs)
        self.obs_rms = (
            RunningMeanStd(shape=(self.obs_dim,)) if ppo_cfg.normalize_observation else None
        )
        self.reward_norm = (
            RewardNormalizer(ppo_cfg.num_envs, ppo_cfg.gamma)
            if ppo_cfg.normalize_reward
            else None
        )
        # the warm start may restore the observation normaliser, so it has to run
        # after obs_rms exists
        if ppo_cfg.pretrained_actor:
            self._load_pretrained(ppo_cfg.pretrained_actor)

        self.run_logger = RunLogger(self.run_dir)
        self.global_step = 0
        self.num_updates = max(1, ppo_cfg.total_timesteps // ppo_cfg.batch_size)
        self._episode_stats: list[dict[str, float]] = []

        self.probe = None
        if ppo_cfg.probe_dataset and ppo_cfg.probe_interval:
            from rsma_aoi.agents.imitation import OracleDataset, OracleProbe

            probe_split = Path(ppo_cfg.probe_dataset) / "test.npz"
            if probe_split.is_file():
                self.probe = OracleProbe(
                    OracleDataset.load(probe_split), env_cfg, size=ppo_cfg.probe_size
                )
                self.logger.info("oracle probe set: %d states", self.probe.size)
            else:
                self.logger.warning("probe dataset %s not found; probes disabled", probe_split)

        env_seeds = spawn_seeds(self.seeds.env, ppo_cfg.num_envs)
        self._obs, _ = self.envs.reset(env_seeds)
        self._episode_starts = np.ones(ppo_cfg.num_envs, dtype=bool)
        if ppo_cfg.recurrent_enabled:
            assert isinstance(self.model, RecurrentHybridActorCritic)
            self._actor_hidden, self._critic_hidden = self.model.initial_state(
                ppo_cfg.num_envs, self.device
            )
        dump_config_snapshot(self.run_dir / "config_snapshot.yaml", env=env_cfg, ppo=ppo_cfg)
        self.run_logger.log_meta(
            {
                "seed": seed,
                "git_commit": git_commit_hash(Path(__file__).resolve().parents[3]),
                "env_config": env_cfg.to_dict(),
                "ppo_config": ppo_cfg.to_dict(),
                "num_updates": self.num_updates,
            }
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _normalize_obs(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if self.obs_rms is None:
            return obs.astype(np.float32)
        if update and not self.cfg.freeze_observation_stats:
            self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs)

    def _tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def _value_of(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.model.value(self._tensor(obs)).cpu().numpy()

    # ------------------------------------------------------------------
    # rollout
    # ------------------------------------------------------------------
    def collect_rollout(self) -> dict[str, float]:
        """Fill the buffer with ``rollout_steps`` vectorised transitions."""
        self.buffer.reset()
        self.model.eval()
        step_metrics: dict[str, list[float]] = {
            "weighted_aoi": [], "mean_aoi": [], "max_aoi": [], "tail_excess": [],
            "total_power": [], "common_power": [], "mean_bler": [],
            "num_scheduled": [], "constraint_violation": [], "common_payload": [],
        }
        violations = 0
        transitions = 0

        for t in range(self.cfg.rollout_steps):
            norm_obs = self._normalize_obs(self._obs, update=True)
            obs_t = self._tensor(norm_obs)
            forced = None
            if self.uses_forced_mask:
                forced = torch.as_tensor(
                    max_aoi_masks(self.envs.get_states(), self.schedule_size),
                    dtype=torch.bool, device=self.device,
                )
            with torch.no_grad():
                if self.cfg.recurrent_enabled:
                    assert isinstance(self.model, RecurrentHybridActorCritic)
                    actor_hidden_in = self._actor_hidden.detach().clone()
                    critic_hidden_in = self._critic_hidden.detach().clone()
                    starts_t = torch.as_tensor(
                        self._episode_starts, dtype=torch.bool, device=self.device
                    )
                    dist, value, next_actor_hidden, next_critic_hidden = (
                        self.model.recurrent_step(
                            obs_t,
                            self._actor_hidden,
                            self._critic_hidden,
                            episode_start=starts_t,
                            forced_mask=forced,
                        )
                    )
                else:
                    dist = self._distribution(obs_t, forced)
                    value = self.model.value(obs_t)
                latent = dist.sample(self.torch_gen)
                log_prob = dist.log_prob(latent)
                head_ent = dist.head_entropies(latent.active_mask)
                counts = latent.active_mask.sum(-1).cpu().numpy()
                raw_empty = (
                    float((latent.schedule_raw.sum(-1) == 0).float().mean())
                    if self.model.scheduling_head in ("bernoulli", "fixed_top_m")
                    else 0.0
                )
            for name, value_t in head_ent.items():
                step_metrics.setdefault(f"entropy_{name}", []).append(float(value_t.mean()))
            step_metrics.setdefault("raw_all_zero_rate", []).append(raw_empty)
            for c in range(1, self.env_cfg.num_users + 1):
                step_metrics.setdefault(f"p_count_{c}", []).append(float((counts == c).mean()))
            actions = latent_to_env_actions(dist, latent)
            next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)

            done = np.logical_or(terminated, truncated)
            if self.reward_norm is not None:
                rewards = self.reward_norm(rewards.astype(np.float64), done)

            add_args = (
                t, norm_obs, latent, log_prob.cpu().numpy(), value.cpu().numpy(),
                rewards, terminated, truncated,
            )
            if self.cfg.recurrent_enabled:
                assert isinstance(self.buffer, RecurrentRolloutBuffer)
                self.buffer.add(
                    *add_args,
                    actor_hidden=actor_hidden_in.cpu().numpy(),
                    critic_hidden=critic_hidden_in.cpu().numpy(),
                    episode_starts=self._episode_starts,
                )
            else:
                self.buffer.add(*add_args)

            # V(s_{t+1}) with terminal observations restored for bootstrapping
            value_obs = np.stack(
                [
                    infos[i]["final_observation"] if done[i] else next_obs[i]
                    for i in range(len(infos))
                ]
            )
            value_norm = self._normalize_obs(value_obs, update=False)
            if self.cfg.recurrent_enabled:
                with torch.no_grad():
                    next_value_t, _ = self.model.recurrent_value_step(
                        self._tensor(value_norm), next_critic_hidden
                    )
                next_values = next_value_t.cpu().numpy()
            else:
                next_values = self._value_of(value_norm)
            self.buffer.set_next_values(t, next_values)

            for info in infos:
                source = info.get("final_info", info)
                for key in step_metrics:
                    if key in source:
                        step_metrics[key].append(float(np.mean(source[key])))
                violations += int(source.get("constraint_violation", 0.0) > self.env_cfg.feasibility_tol)
                transitions += 1
                if "episode" in info:
                    self._episode_stats.append(dict(info["episode"]))

            self._obs = next_obs
            self._episode_starts = done
            if self.cfg.recurrent_enabled:
                keep = torch.as_tensor(
                    ~done, dtype=torch.float32, device=self.device
                ).unsqueeze(-1)
                self._actor_hidden = next_actor_hidden * keep
                self._critic_hidden = next_critic_hidden * keep
            self.global_step += self.cfg.num_envs

        metrics = {f"env/{k}": float(np.mean(v)) for k, v in step_metrics.items() if v}
        metrics["env/violation_rate"] = violations / max(transitions, 1)
        return metrics

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------
    def update(self) -> dict[str, float]:
        """Run the PPO epochs over the collected rollout."""
        cfg = self.cfg
        self.model.train()
        advantages, returns = self.buffer.compute_returns(cfg.gamma, cfg.gae_lambda)

        stats = {
            "policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [],
            "clip_fraction": [], "grad_norm": [], "reference_kl": [],
        }
        early_stop = False
        epochs_done = 0
        for _ in range(cfg.update_epochs):
            epochs_done += 1
            epoch_kl: list[float] = []
            batches = (
                self.buffer.iter_sequence_minibatches(cfg.minibatch_size, self.np_rng)
                if cfg.recurrent_enabled
                else self.buffer.iter_minibatches(cfg.minibatch_size, self.np_rng)
            )
            for batch in batches:
                if cfg.recurrent_enabled:
                    assert isinstance(self.model, RecurrentHybridActorCritic)
                    actor_hidden = batch.initial_actor_hidden
                    critic_hidden = batch.initial_critic_hidden
                    log_probs, entropies, values = [], [], []
                    for t in range(batch.obs.shape[0]):
                        forced = batch.latent.active_mask[t] if self.uses_forced_mask else None
                        dist, value_t, actor_hidden, critic_hidden = self.model.recurrent_step(
                            batch.obs[t], actor_hidden, critic_hidden,
                            episode_start=batch.episode_starts[t], forced_mask=forced,
                        )
                        latent_t = LatentAction(
                            schedule_raw=batch.latent.schedule_raw[t],
                            active_mask=batch.latent.active_mask[t],
                            power_latent=batch.latent.power_latent[t],
                            rate_latent=batch.latent.rate_latent[t],
                            rho_r_latent=batch.latent.rho_r_latent[t],
                            rho_p_latent=(
                                None if batch.latent.rho_p_latent is None
                                else batch.latent.rho_p_latent[t]
                            ),
                            common_gate_raw=(
                                None if batch.latent.common_gate_raw is None
                                else batch.latent.common_gate_raw[t]
                            ),
                        )
                        log_probs.append(dist.log_prob(latent_t))
                        entropies.append(dist.entropy(latent_t.active_mask))
                        values.append(value_t)
                    new_log_prob = torch.stack(log_probs).reshape(-1)
                    entropy = torch.stack(entropies).mean()
                    value = torch.stack(values).reshape(-1)
                    old_log_probs = batch.log_probs.reshape(-1)
                    old_values = batch.values.reshape(-1)
                    batch_returns = batch.returns.reshape(-1)
                    adv = batch.advantages.reshape(-1)
                else:
                    dist = self._distribution(batch.obs, batch.latent.active_mask)
                    new_log_prob = dist.log_prob(batch.latent)
                    entropy = dist.entropy(batch.latent.active_mask).mean()
                    value = self.model.value(batch.obs)
                    old_log_probs = batch.log_probs
                    old_values = batch.values
                    batch_returns = batch.returns
                    adv = batch.advantages

                log_ratio = new_log_prob - old_log_probs
                ratio = torch.exp(log_ratio)

                if cfg.normalize_advantage:
                    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

                pg_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * adv,
                ).mean()

                if cfg.clip_value_loss:
                    v_clipped = old_values + torch.clamp(
                        value - old_values, -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss = 0.5 * torch.max(
                        (value - batch_returns) ** 2, (v_clipped - batch_returns) ** 2
                    ).mean()
                else:
                    v_loss = 0.5 * ((value - batch_returns) ** 2).mean()

                loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy

                kl_coef = self._reference_kl_coef()
                if kl_coef > 0.0 and self.reference_model is not None:
                    with torch.no_grad():
                        ref_dist = (
                            self.reference_model.distribution(
                                batch.obs, forced_mask=batch.latent.active_mask
                            )
                            if self.uses_forced_mask
                            else self.reference_model.distribution(batch.obs)
                        )
                        ref_log_prob = ref_dist.log_prob(batch.latent)
                    ref_kl = (new_log_prob - ref_log_prob).mean()
                    loss = loss + kl_coef * ref_kl
                    ref_kl_value = float(ref_kl.detach())
                else:
                    ref_kl_value = 0.0

                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO loss became non-finite")

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for param in self._frozen_parameters():
                    if param.grad is not None:
                        param.grad.zero_()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()
                stats["policy_loss"].append(pg_loss.detach().item())
                stats["value_loss"].append(v_loss.detach().item())
                stats["entropy"].append(entropy.detach().item())
                stats["approx_kl"].append(approx_kl.item())
                stats["clip_fraction"].append(clip_frac.item())
                stats["grad_norm"].append(float(grad_norm.detach()))
                stats["reference_kl"].append(ref_kl_value)
                epoch_kl.append(approx_kl.item())

            # early stop on the mean KL of the epoch, not of a single minibatch
            if cfg.target_kl is not None and epoch_kl and float(np.mean(epoch_kl)) > cfg.target_kl:
                early_stop = True
                break

        flat_values = self.buffer.values.reshape(-1)
        flat_returns = returns.reshape(-1)
        var_y = float(np.var(flat_returns))
        explained_var = float("nan") if var_y == 0 else 1.0 - float(np.var(flat_returns - flat_values)) / var_y

        out = {f"train/{k}": float(np.mean(v)) for k, v in stats.items() if v}
        out["train/explained_variance"] = explained_var
        out["train/epochs_completed"] = float(epochs_done)
        out["train/early_stop"] = float(early_stop)
        out["train/learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        out["train/actor_lr"] = float(self.optimizer.param_groups[0]["lr"])
        out["train/critic_lr"] = float(self.optimizer.param_groups[1]["lr"])
        out["train/reference_kl_coef"] = self._reference_kl_coef()
        out["train/frozen_parameters"] = float(len(self._frozen_parameters()))
        return out

    # ------------------------------------------------------------------
    def _anneal_lr(self, update_index: int) -> None:
        if not self.cfg.anneal_lr:
            return
        frac = 1.0 - update_index / float(self.num_updates)
        for group, base in zip(self.optimizer.param_groups, self._base_lrs):
            group["lr"] = max(base * frac, 1e-8)

    # ------------------------------------------------------------------
    def _load_pretrained(self, path: str) -> None:
        """Initialise the actor from a behaviour-cloning checkpoint."""
        from rsma_aoi.agents.imitation import load_pretrained_actor

        info = load_pretrained_actor(self.model, path, strict=False)
        self.warm_start_meta = {k: v for k, v in info.items() if k != "obs_rms"}
        if self.cfg.warm_start_log_std is not None:
            with torch.no_grad():
                self.model.power_log_std.fill_(self.cfg.warm_start_log_std)
                self.model.rate_log_std.fill_(self.cfg.warm_start_log_std)
            self.logger.info(
                "warm-start latent log_std overridden to %.3f",
                self.cfg.warm_start_log_std,
            )
        if info.get("obs_rms") is not None and self.obs_rms is not None:
            # the actor was cloned under this observation scale; PPO must continue it
            self.obs_rms.load_state_dict(info["obs_rms"])
        if self.cfg.reset_critic:
            for module in (self.model.critic_trunk, self.model.value_head):
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        layer.reset_parameters()
        if self.cfg.reference_kl_coef > 0.0:
            import copy

            self.reference_model = copy.deepcopy(self.model).eval()
            for param in self.reference_model.parameters():
                param.requires_grad_(False)
        self.logger.info(
            "warm start from %s (missing=%s, unexpected=%s, critic_reset=%s)",
            path, info["missing"], info["unexpected"], self.cfg.reset_critic,
        )

    def _reference_kl_coef(self) -> float:
        """Current reference-KL coefficient (linearly annealed when configured)."""
        coef = self.cfg.reference_kl_coef
        if coef <= 0.0 or self.reference_model is None:
            return 0.0
        if self.cfg.reference_kl_anneal_steps <= 0:
            return coef
        frac = 1.0 - self.global_step / float(self.cfg.reference_kl_anneal_steps)
        return float(max(coef * frac, 0.0))

    def _frozen_parameters(self) -> list[torch.nn.Parameter]:
        """Parameters that must not receive gradients at the current step."""
        if self.global_step < self.cfg.freeze_actor_steps:
            return [
                p for name, p in self.model.named_parameters()
                if not name.startswith(
                    ("critic_trunk", "value_head", "critic_gru", "critic_recurrent_proj")
                )
            ]
        frozen: list[torch.nn.Parameter] = []
        if self.global_step < self.cfg.freeze_scheduling_steps:
            frozen += list(self.model.schedule_head.parameters())
        if self.global_step < self.cfg.freeze_allocation_steps:
            for head in (self.model.power_head, self.model.rate_head, self.model.rho_r_head):
                if head is not None:
                    frozen += list(head.parameters())
            frozen += [self.model.power_log_std, self.model.rate_log_std]
            if self.model.rho_p_head is not None:
                frozen += list(self.model.rho_p_head.parameters())
            if self.model.common_gate_head is not None:
                frozen += list(self.model.common_gate_head.parameters())
        return frozen

    def _distribution(self, obs: torch.Tensor, active_mask: torch.Tensor | None = None):
        """Build the policy distribution, supplying the forced mask when needed."""
        if self.uses_forced_mask:
            if active_mask is None:
                raise ValueError("the fixed_top_m head requires an active mask")
            return self.model.distribution(obs, forced_mask=active_mask)
        return self.model.distribution(obs)

    # ------------------------------------------------------------------
    def train(self, progress_callback: Callable[[int, dict[str, float]], None] | None = None) -> dict[str, float]:
        """Run the full training budget; returns the final metric dictionary."""
        start = time.time()
        metrics: dict[str, float] = {}
        for update in range(self.num_updates):
            self._anneal_lr(update)
            rollout_metrics = self.collect_rollout()
            update_metrics = self.update()
            elapsed = max(time.time() - start, 1e-9)
            metrics = {**rollout_metrics, **update_metrics}
            metrics["perf/sps"] = self.global_step / elapsed
            metrics["perf/global_step"] = float(self.global_step)
            if self._episode_stats:
                recent = self._episode_stats[-50:]
                for key in ("r", "weighted_aoi", "success_rate", "mean_power", "violation_rate"):
                    metrics[f"episode/{key}"] = float(np.mean([e[key] for e in recent]))

            # probe BEFORE logging so the diagnostics reach scalars.csv
            if self.probe is not None and self.cfg.probe_interval \
                    and update % self.cfg.probe_interval == 0:
                metrics.update(self.probe.evaluate(self.model, self.obs_rms))

            if self.cfg.log_interval and update % self.cfg.log_interval == 0:
                self.run_logger.log_scalars(self.global_step, metrics)
                self.logger.info(
                    "update %d/%d step %d | AoI %.3f | ret %.3f | pl %.4f vl %.4f ent %.3f kl %.4f",
                    update + 1, self.num_updates, self.global_step,
                    metrics.get("env/weighted_aoi", float("nan")),
                    metrics.get("episode/r", float("nan")),
                    metrics.get("train/policy_loss", float("nan")),
                    metrics.get("train/value_loss", float("nan")),
                    metrics.get("train/entropy", float("nan")),
                    metrics.get("train/approx_kl", float("nan")),
                )
            if self.cfg.checkpoint_interval and (update + 1) % self.cfg.checkpoint_interval == 0:
                self.save_checkpoint(self.run_dir / "checkpoint.pt")
            if progress_callback is not None:
                progress_callback(update, metrics)

        self.save_checkpoint(self.run_dir / "checkpoint.pt")
        self.run_logger.close()
        return metrics

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str | Path) -> Path:
        """Persist model, optimizer, normalisers and the configs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "obs_rms": None if self.obs_rms is None else self.obs_rms.state_dict(),
                "reward_norm": None if self.reward_norm is None else self.reward_norm.state_dict(),
                "env_config": self.env_cfg.to_dict(),
                "ppo_config": self.cfg.to_dict(),
                "global_step": self.global_step,
                "seed": self.seeds.base,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore a checkpoint saved by :meth:`save_checkpoint`."""
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        if ckpt.get("env_config", {}).get("rate_split_mode", "simplex_scaled") != self.env_cfg.rate_split_mode:
            raise ValueError("checkpoint action parameterization differs")
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.obs_rms is not None and ckpt.get("obs_rms") is not None:
            self.obs_rms.load_state_dict(ckpt["obs_rms"])
        if self.reward_norm is not None and ckpt.get("reward_norm") is not None:
            self.reward_norm.load_state_dict(ckpt["reward_norm"])
        self.global_step = int(ckpt.get("global_step", 0))

    # ------------------------------------------------------------------
    def make_policy(self, deterministic: bool = True) -> PPOPolicy:
        """Inference wrapper around the current weights."""
        return PPOPolicy(
            self.model,
            self.obs_rms,
            device=self.device,
            deterministic=deterministic,
            generator=self.torch_gen,
        )

    def close(self) -> None:
        self.envs.close()
        self.run_logger.close()
