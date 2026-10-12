"""TD3 baseline for the continuous resource-allocation problem.

Scope
-----
This is the off-policy *continuous-control* reference corresponding to the
direct prior work (Kaya, 2025).  It performs **no discrete scheduling**: it is
trained on an environment with ``scheduling_enabled: false``, so every user is
active in every slot and the agent only decides the power split, the
common-rate split and ``rho_r`` - exactly the problem the continuous-only PPO
agent solves.

Constraint handling is shared, not re-implemented: the actor emits a bounded
latent vector and
:func:`rsma_aoi.agents.distributions.latent_to_feasible_action` maps it to the
environment action, i.e. the *same* masked-softmax / sigmoid transform the PPO
policy uses.  A comparison between the two therefore differs in the learning
algorithm only.

Algorithm (Fujimoto et al., 2018): twin critics with a clipped double-Q target,
target policy smoothing, delayed actor and target updates, Polyak-averaged
target networks, a uniform-random warm-up phase and Gaussian exploration noise.

Fair-comparison note: TD3 is off-policy and reuses a replay buffer, so it must
be compared with PPO at an equal **environment-interaction** budget, never at an
equal number of gradient steps or episodes.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from rsma_aoi.agents.distributions import latent_to_feasible_action
from rsma_aoi.agents.networks import DeterministicActor, TwinCritic
from rsma_aoi.agents.ppo import resolve_device
from rsma_aoi.agents.replay_buffer import ReplayBuffer
from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.utils.config import EnvConfig, TD3Config, dump_config_snapshot
from rsma_aoi.utils.logging import RunLogger, get_logger, git_commit_hash
from rsma_aoi.utils.running_stats import RunningMeanStd
from rsma_aoi.utils.seeding import SeedBundle, make_torch_generator, seed_everything, spawn_seeds
from rsma_aoi.utils.vec_env import SyncVecEnv

#: TD3 emits scalars as pre-sigmoid latents, i.e. the squashed-Gaussian mapping.
SCALAR_KIND = "squashed_gaussian"


def split_latent(
    latent: torch.Tensor, num_users: int, power_scale_action: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Slice a flat latent into the power / rate / scalar blocks.

    Layout: ``[power (K+1) | rate (K) | rho_r (1) | rho_P (1, optional)]``.
    """
    k = num_users
    power = latent[:, : k + 1]
    rate = latent[:, k + 1 : 2 * k + 1]
    rho_r = latent[:, 2 * k + 1 : 2 * k + 2]
    rho_p = latent[:, 2 * k + 2 : 2 * k + 3] if power_scale_action else None
    return power, rate, rho_r, rho_p


def latent_to_env_actions(
    latent: torch.Tensor, num_users: int, power_scale_action: bool
) -> list[dict[str, np.ndarray]]:
    """Transform a batch of TD3 latents into per-environment action dicts.

    Every user is active (TD3 runs in the continuous-only setting), so the
    scheduling mask handed to the shared transform is all ones.
    """
    with torch.no_grad():
        power, rate, rho_r, rho_p = split_latent(latent, num_users, power_scale_action)
        active = torch.ones(latent.shape[0], num_users, dtype=torch.bool, device=latent.device)
        batched = latent_to_feasible_action(
            power, rate, rho_r, active, SCALAR_KIND, rho_p
        )
    actions: list[dict[str, np.ndarray]] = []
    for i in range(latent.shape[0]):
        action = {
            "schedule": np.ones(num_users, dtype=np.int8),
            "power_fractions": batched["power_fractions"][i].cpu().numpy().astype(np.float64),
            "rate_fractions": batched["rate_fractions"][i].cpu().numpy().astype(np.float64),
            "common_payload_scale": batched["common_payload_scale"][i]
            .cpu()
            .numpy()
            .astype(np.float64),
        }
        if "power_scale" in batched:
            action["power_scale"] = batched["power_scale"][i].cpu().numpy().astype(np.float64)
        actions.append(action)
    return actions


class TD3Policy:
    """Inference-only wrapper with the same ``act`` contract as :class:`PPOPolicy`."""

    name = "td3"

    def __init__(
        self,
        actor: DeterministicActor,
        obs_rms: RunningMeanStd | None,
        num_users: int,
        power_scale_action: bool,
        device: torch.device | str = "cpu",
        deterministic: bool = True,
        exploration_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> None:
        self.actor = actor
        self.obs_rms = obs_rms
        self.num_users = num_users
        self.power_scale_action = power_scale_action
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.exploration_noise = float(exploration_noise)
        self.generator = generator

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms is None:
            return obs.astype(np.float32)
        return self.obs_rms.normalize(obs)

    def act(
        self, obs: np.ndarray, states: list[dict[str, Any]] | None = None
    ) -> list[dict[str, np.ndarray]]:
        obs_t = torch.as_tensor(self.normalize(obs), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            latent = self.actor(obs_t)
            if not self.deterministic and self.exploration_noise > 0.0:
                noise = torch.randn(
                    latent.shape, generator=self.generator, device=latent.device
                )
                latent = torch.clamp(
                    latent + self.exploration_noise * self.actor.latent_scale * noise,
                    -self.actor.latent_scale,
                    self.actor.latent_scale,
                )
        return latent_to_env_actions(latent, self.num_users, self.power_scale_action)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str = "cpu", deterministic: bool = True
    ) -> tuple["TD3Policy", EnvConfig, TD3Config]:
        """Rebuild a policy (and its configs) from a TD3 checkpoint."""
        device = str(resolve_device(device))
        ckpt = torch.load(Path(path), map_location=device, weights_only=False)
        env_cfg = EnvConfig(**ckpt["env_config"])
        td3_cfg = TD3Config(**ckpt["td3_config"])
        probe = AoIRSMAEnv(env_cfg)
        obs_dim = int(np.prod(probe.observation_space.shape))
        actor = DeterministicActor(
            obs_dim, env_cfg.num_users, td3_cfg.hidden_sizes, td3_cfg.activation,
            td3_cfg.latent_scale, env_cfg.power_scale_action,
        ).to(device)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        obs_rms = None
        if ckpt.get("obs_rms") is not None:
            obs_rms = RunningMeanStd(shape=probe.observation_space.shape)
            obs_rms.load_state_dict(ckpt["obs_rms"])
        probe.close()
        return (
            cls(actor, obs_rms, env_cfg.num_users, env_cfg.power_scale_action,
                device=device, deterministic=deterministic),
            env_cfg,
            td3_cfg,
        )


class TD3Trainer:
    """TD3 training loop mirroring :class:`~rsma_aoi.agents.ppo.PPOTrainer`."""

    def __init__(
        self,
        env_cfg: EnvConfig,
        td3_cfg: TD3Config,
        run_dir: str | Path,
        seed: int = 0,
        env_factory: Callable[[EnvConfig], AoIRSMAEnv] | None = None,
    ) -> None:
        if env_cfg.scheduling_enabled:
            raise ValueError(
                "the TD3 baseline is scoped to continuous allocation only; use an "
                "environment config with scheduling_enabled: false "
                "(e.g. configs/m2_continuous.yaml or configs/m1_rsma.yaml)"
            )
        if env_cfg.rate_split_mode != "simplex_scaled":
            raise ValueError("TD3 direct-fraction actions are not implemented; use PPO")
        self.env_cfg = env_cfg
        self.cfg = td3_cfg
        self.seeds = SeedBundle(seed)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(td3_cfg.device)
        self.logger = get_logger()

        seed_everything(self.seeds.agent)
        self.torch_gen = make_torch_generator(self.seeds.agent, device="cpu")
        self.np_rng = np.random.default_rng(self.seeds.agent)

        factory = env_factory or (lambda cfg: AoIRSMAEnv(cfg))
        self.envs = SyncVecEnv([lambda: factory(env_cfg) for _ in range(td3_cfg.num_envs)])
        self.obs_dim = self.envs.obs_dim
        self.num_users = env_cfg.num_users

        self.actor = DeterministicActor(
            self.obs_dim, self.num_users, td3_cfg.hidden_sizes, td3_cfg.activation,
            td3_cfg.latent_scale, env_cfg.power_scale_action,
        ).to(self.device)
        self.action_dim = self.actor.action_dim
        self.critic = TwinCritic(
            self.obs_dim, self.action_dim, td3_cfg.hidden_sizes, td3_cfg.activation
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).eval()
        self.critic_target = copy.deepcopy(self.critic).eval()
        for param in list(self.actor_target.parameters()) + list(self.critic_target.parameters()):
            param.requires_grad_(False)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=td3_cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=td3_cfg.critic_lr)
        self.buffer = ReplayBuffer(
            td3_cfg.buffer_size, self.obs_dim, self.action_dim, device=self.device
        )
        self.obs_rms = (
            RunningMeanStd(shape=(self.obs_dim,)) if td3_cfg.normalize_observation else None
        )
        self.run_logger = RunLogger(self.run_dir)
        self.global_step = 0
        self.updates = 0
        self._episode_stats: list[dict[str, float]] = []

        self._obs, _ = self.envs.reset(spawn_seeds(self.seeds.env, td3_cfg.num_envs))
        dump_config_snapshot(self.run_dir / "config_snapshot.yaml", env=env_cfg, td3=td3_cfg)
        self.run_logger.log_meta(
            {
                "algorithm": "td3",
                "seed": seed,
                "git_commit": git_commit_hash(Path(__file__).resolve().parents[3]),
                "env_config": env_cfg.to_dict(),
                "td3_config": td3_cfg.to_dict(),
            }
        )

    # ------------------------------------------------------------------
    def _normalize_obs(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if self.obs_rms is None:
            return obs.astype(np.float32)
        if update:
            self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs)

    def _select_latent(self, norm_obs: np.ndarray) -> torch.Tensor:
        """Warm-up uniform exploration, then the actor plus Gaussian noise."""
        scale = self.cfg.latent_scale
        if self.global_step < self.cfg.learning_starts:
            uniform = torch.rand(
                (norm_obs.shape[0], self.action_dim), generator=self.torch_gen
            )
            return (2.0 * uniform - 1.0) * scale
        obs_t = torch.as_tensor(norm_obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            latent = self.actor(obs_t)
        noise = torch.randn(latent.shape, generator=self.torch_gen, device=latent.device)
        return torch.clamp(
            latent + self.cfg.exploration_noise * scale * noise, -scale, scale
        )

    # ------------------------------------------------------------------
    def collect_step(self) -> dict[str, float]:
        """Take one vectorised environment step and store it in the replay buffer."""
        norm_obs = self._normalize_obs(self._obs, update=True)
        latent = self._select_latent(norm_obs)
        actions = latent_to_env_actions(latent, self.num_users, self.env_cfg.power_scale_action)
        next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)

        # the replay buffer needs the *terminal* observation, not the reset one
        stored_next = next_obs.copy()
        done = np.logical_or(terminated, truncated)
        if done.any():
            for i, info in enumerate(infos):
                if done[i] and "final_observation" in info:
                    stored_next[i] = info["final_observation"]
        self.buffer.add_batch(
            norm_obs,
            latent.cpu().numpy(),
            rewards,
            self._normalize_obs(stored_next, update=False),
            terminated,  # time-limit truncation is bootstrapped, not terminal
        )

        metrics: dict[str, list[float]] = {"weighted_aoi": [], "mean_bler": [],
                                           "constraint_violation": []}
        for info in infos:
            source = info.get("final_info", info)
            for key in metrics:
                if key in source:
                    metrics[key].append(float(np.mean(source[key])))
            if "episode" in info:
                self._episode_stats.append(dict(info["episode"]))

        self._obs = next_obs
        self.global_step += self.cfg.num_envs
        return {f"env/{k}": float(np.mean(v)) for k, v in metrics.items() if v}

    # ------------------------------------------------------------------
    def update(self) -> dict[str, float]:
        """One TD3 gradient step (actor and targets updated every ``policy_delay``)."""
        cfg = self.cfg
        batch = self.buffer.sample(cfg.batch_size, self.np_rng)
        scale = cfg.latent_scale

        with torch.no_grad():
            noise = torch.randn(batch.action.shape, generator=self.torch_gen)
            noise = torch.clamp(noise * cfg.target_noise * scale, -cfg.noise_clip * scale,
                                cfg.noise_clip * scale)
            next_action = torch.clamp(self.actor_target(batch.next_obs) + noise, -scale, scale)
            target_q1, target_q2 = self.critic_target(batch.next_obs, next_action)
            target = batch.reward + cfg.gamma * batch.not_done * torch.min(target_q1, target_q2)

        q1, q2 = self.critic(batch.obs, batch.action)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("TD3 critic loss became non-finite")
        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
        self.critic_optim.step()

        stats = {
            "train/critic_loss": critic_loss.detach().item(),
            "train/critic_grad_norm": float(critic_grad.detach()),
            "train/q1_mean": q1.detach().mean().item(),
            "train/target_q_mean": target.mean().item(),
        }

        self.updates += 1
        if self.updates % cfg.policy_delay == 0:
            actor_loss = -self.critic.q1(batch.obs, self.actor(batch.obs)).mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("TD3 actor loss became non-finite")
            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
            self.actor_optim.step()
            self._polyak(self.actor, self.actor_target)
            self._polyak(self.critic, self.critic_target)
            stats["train/actor_loss"] = actor_loss.detach().item()
            stats["train/actor_grad_norm"] = float(actor_grad.detach())
        return stats

    def _polyak(self, online: nn.Module, target: nn.Module) -> None:
        tau = self.cfg.tau
        with torch.no_grad():
            for p, p_target in zip(online.parameters(), target.parameters()):
                p_target.mul_(1.0 - tau).add_(tau * p)

    # ------------------------------------------------------------------
    def train(self) -> dict[str, float]:
        """Run the full interaction budget; returns the final metric dictionary."""
        cfg = self.cfg
        start = time.time()
        iterations = max(1, cfg.total_timesteps // cfg.num_envs)
        metrics: dict[str, float] = {}
        for iteration in range(iterations):
            env_metrics = self.collect_step()
            update_metrics: dict[str, float] = {}
            if len(self.buffer) >= cfg.batch_size and self.global_step >= cfg.learning_starts:
                for _ in range(cfg.gradient_steps_per_iter):
                    update_metrics = self.update()
            metrics = {**env_metrics, **update_metrics}
            metrics["perf/global_step"] = float(self.global_step)
            metrics["perf/sps"] = self.global_step / max(time.time() - start, 1e-9)
            metrics["perf/buffer_size"] = float(len(self.buffer))
            if self._episode_stats:
                recent = self._episode_stats[-50:]
                for key in ("r", "weighted_aoi", "success_rate", "violation_rate"):
                    metrics[f"episode/{key}"] = float(np.mean([e[key] for e in recent]))

            if cfg.log_interval and iteration % cfg.log_interval == 0:
                self.run_logger.log_scalars(self.global_step, metrics)
                self.logger.info(
                    "iter %d/%d step %d | AoI %.3f | critic %.4f | actor %.4f | buf %d",
                    iteration + 1, iterations, self.global_step,
                    metrics.get("env/weighted_aoi", float("nan")),
                    metrics.get("train/critic_loss", float("nan")),
                    metrics.get("train/actor_loss", float("nan")),
                    len(self.buffer),
                )
            if cfg.checkpoint_interval and (iteration + 1) % cfg.checkpoint_interval == 0:
                self.save_checkpoint(self.run_dir / "checkpoint.pt")

        self.save_checkpoint(self.run_dir / "checkpoint.pt")
        self.run_logger.close()
        return metrics

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str | Path) -> Path:
        """Persist actor, critics, targets, optimizers, normaliser and configs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "obs_rms": None if self.obs_rms is None else self.obs_rms.state_dict(),
                "env_config": self.env_cfg.to_dict(),
                "td3_config": self.cfg.to_dict(),
                "global_step": self.global_step,
                "seed": self.seeds.base,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore a checkpoint saved by :meth:`save_checkpoint`."""
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        if self.obs_rms is not None and ckpt.get("obs_rms") is not None:
            self.obs_rms.load_state_dict(ckpt["obs_rms"])
        self.global_step = int(ckpt.get("global_step", 0))

    def make_policy(self, deterministic: bool = True) -> TD3Policy:
        """Inference wrapper around the current actor."""
        return TD3Policy(
            self.actor, self.obs_rms, self.num_users, self.env_cfg.power_scale_action,
            device=self.device, deterministic=deterministic,
            exploration_noise=self.cfg.exploration_noise, generator=self.torch_gen,
        )

    def close(self) -> None:
        self.envs.close()
        self.run_logger.close()
