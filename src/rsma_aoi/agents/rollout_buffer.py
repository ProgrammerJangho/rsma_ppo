"""On-policy rollout storage and Generalized Advantage Estimation.

The buffer stores the *latent* action variables (Gaussian latents, the raw
Bernoulli schedule sample and the repaired active mask) rather than the
transformed environment action.  Re-scoring at update time therefore uses
exactly the same random variables and the same masks as the behaviour policy,
which is what makes the PPO ratio well defined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rsma_aoi.agents.distributions import LatentAction


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation with correct time-limit handling.

    ``terminated`` marks true MDP terminals (no bootstrap), ``truncated`` marks
    time-limit cuts (bootstrap with the value of the final observation, which the
    caller must supply through ``next_values``)::

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - terminated_t) - V(s_t)
        A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

    Args:
        rewards / values / next_values / terminated / truncated: ``(T, N)``.
        gamma: discount factor.
        gae_lambda: GAE lambda.

    Returns:
        ``(advantages, returns)`` each of shape ``(T, N)`` with
        ``returns = advantages + values``.
    """
    shapes = {a.shape for a in (rewards, values, next_values, terminated, truncated)}
    if len(shapes) != 1:
        raise ValueError(f"all GAE inputs must share a shape, got {shapes}")
    steps = rewards.shape[0]
    advantages = np.zeros_like(rewards, dtype=np.float64)
    last = np.zeros(rewards.shape[1:], dtype=np.float64)
    done = np.logical_or(terminated, truncated).astype(np.float64)
    not_terminated = 1.0 - terminated.astype(np.float64)
    for t in reversed(range(steps)):
        delta = rewards[t] + gamma * next_values[t] * not_terminated[t] - values[t]
        last = delta + gamma * gae_lambda * (1.0 - done[t]) * last
        advantages[t] = last
    returns = advantages + values
    return advantages, returns


@dataclass
class RolloutBatch:
    """A flattened minibatch handed to the PPO update."""

    obs: torch.Tensor
    latent: LatentAction
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


@dataclass
class RecurrentRolloutBatch(RolloutBatch):
    """Sequence minibatch with shape ``(L, B, ...)`` and initial memories."""

    episode_starts: torch.Tensor
    initial_actor_hidden: torch.Tensor
    initial_critic_hidden: torch.Tensor


class RolloutBuffer:
    """Fixed-size ``(T, N)`` on-policy buffer.

    Args:
        num_steps: rollout length per environment (T).
        num_envs: number of parallel environments (N).
        obs_dim: observation width.
        num_users: K (sets the latent widths).
        power_scale_action: whether the optional ``rho_P`` latent is stored.
        device: torch device for the produced minibatches.
        schedule_dim: width of the stored scheduling variable (``K`` for the
            Bernoulli head, 1 for the subset-categorical index).
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        num_users: int,
        power_scale_action: bool = False,
        device: str | torch.device = "cpu",
        schedule_dim: int | None = None,
        common_gate_enabled: bool = False,
    ) -> None:
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.num_users = int(num_users)
        self.power_scale_action = bool(power_scale_action)
        self.schedule_dim = int(schedule_dim if schedule_dim is not None else num_users)
        self.device = torch.device(device)
        self.common_gate_enabled = bool(common_gate_enabled)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        t, n, k = self.num_steps, self.num_envs, self.num_users
        f = np.float32
        self.obs = np.zeros((t, n, self.obs_dim), dtype=f)
        self.schedule_raw = np.zeros((t, n, self.schedule_dim), dtype=f)
        self.active_mask = np.zeros((t, n, k), dtype=bool)
        self.power_latent = np.zeros((t, n, k + 1), dtype=f)
        self.rate_latent = np.zeros((t, n, k), dtype=f)
        self.rho_r_latent = np.zeros((t, n, 1), dtype=f)
        self.rho_p_latent = np.zeros((t, n, 1), dtype=f) if self.power_scale_action else None
        self.common_gate_raw = np.zeros((t, n, 1), dtype=f) if self.common_gate_enabled else None
        self.log_probs = np.zeros((t, n), dtype=f)
        self.values = np.zeros((t, n), dtype=f)
        self.next_values = np.zeros((t, n), dtype=f)
        self.rewards = np.zeros((t, n), dtype=f)
        self.terminated = np.zeros((t, n), dtype=bool)
        self.truncated = np.zeros((t, n), dtype=bool)
        self.ptr = 0

    # ------------------------------------------------------------------
    def add(
        self,
        step: int,
        obs: np.ndarray,
        latent: LatentAction,
        log_prob: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> None:
        """Store one vectorised transition at index ``step``."""
        if not 0 <= step < self.num_steps:
            raise IndexError(f"step {step} out of range [0, {self.num_steps})")
        self.obs[step] = obs
        self.schedule_raw[step] = latent.schedule_raw.detach().cpu().numpy()
        self.active_mask[step] = latent.active_mask.detach().cpu().numpy().astype(bool)
        self.power_latent[step] = latent.power_latent.detach().cpu().numpy()
        self.rate_latent[step] = latent.rate_latent.detach().cpu().numpy()
        self.rho_r_latent[step] = latent.rho_r_latent.detach().cpu().numpy()
        if self.rho_p_latent is not None and latent.rho_p_latent is not None:
            self.rho_p_latent[step] = latent.rho_p_latent.detach().cpu().numpy()
        if self.common_gate_raw is not None and latent.common_gate_raw is not None:
            self.common_gate_raw[step] = latent.common_gate_raw.detach().cpu().numpy()
        self.log_probs[step] = log_prob
        self.values[step] = value
        self.rewards[step] = reward
        self.terminated[step] = terminated
        self.truncated[step] = truncated
        self.ptr = step + 1

    def set_next_values(self, step: int, values: np.ndarray) -> None:
        """Set ``V(s_{t+1})`` for step ``t`` (terminal observations included)."""
        self.next_values[step] = values

    # ------------------------------------------------------------------
    def compute_returns(self, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
        """Run GAE over the stored rollout; returns ``(advantages, returns)``."""
        adv, ret = compute_gae(
            self.rewards.astype(np.float64),
            self.values.astype(np.float64),
            self.next_values.astype(np.float64),
            self.terminated,
            self.truncated,
            gamma,
            gae_lambda,
        )
        self.advantages = adv.astype(np.float32)
        self.returns = ret.astype(np.float32)
        return self.advantages, self.returns

    # ------------------------------------------------------------------
    def _tensor(self, array: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype, device=self.device)

    def iter_minibatches(
        self, minibatch_size: int, rng: np.random.Generator
    ) -> list[RolloutBatch]:
        """Shuffle the flattened rollout and yield equally sized minibatches."""
        total = self.num_steps * self.num_envs
        if total % minibatch_size != 0:
            raise ValueError(
                f"batch size {total} is not divisible by minibatch size {minibatch_size}"
            )
        flat = lambda a: a.reshape(total, *a.shape[2:])  # noqa: E731
        obs = flat(self.obs)
        sched = flat(self.schedule_raw)
        mask = flat(self.active_mask)
        power = flat(self.power_latent)
        rate = flat(self.rate_latent)
        rho_r = flat(self.rho_r_latent)
        rho_p = flat(self.rho_p_latent) if self.rho_p_latent is not None else None
        common_gate = flat(self.common_gate_raw) if self.common_gate_raw is not None else None
        logp = self.log_probs.reshape(total)
        adv = self.advantages.reshape(total)
        ret = self.returns.reshape(total)
        val = self.values.reshape(total)

        order = rng.permutation(total)
        batches: list[RolloutBatch] = []
        for start in range(0, total, minibatch_size):
            idx = order[start : start + minibatch_size]
            latent = LatentAction(
                schedule_raw=self._tensor(sched[idx]),
                active_mask=self._tensor(mask[idx], dtype=torch.bool),
                power_latent=self._tensor(power[idx]),
                rate_latent=self._tensor(rate[idx]),
                rho_r_latent=self._tensor(rho_r[idx]),
                rho_p_latent=None if rho_p is None else self._tensor(rho_p[idx]),
                common_gate_raw=None if common_gate is None else self._tensor(common_gate[idx]),
            )
            batches.append(
                RolloutBatch(
                    obs=self._tensor(obs[idx]),
                    latent=latent,
                    log_probs=self._tensor(logp[idx]),
                    advantages=self._tensor(adv[idx]),
                    returns=self._tensor(ret[idx]),
                    values=self._tensor(val[idx]),
                )
            )
        return batches


class RecurrentRolloutBuffer(RolloutBuffer):
    """Rollout storage that emits intact chunks for truncated BPTT."""

    def __init__(self, *args, hidden_size: int, sequence_length: int, **kwargs) -> None:
        self.hidden_size = int(hidden_size)
        self.sequence_length = int(sequence_length)
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        t, n, h = self.num_steps, self.num_envs, self.hidden_size
        self.actor_hidden = np.zeros((t, n, h), dtype=np.float32)
        self.critic_hidden = np.zeros((t, n, h), dtype=np.float32)
        self.episode_starts = np.zeros((t, n), dtype=bool)

    def add(
        self,
        step: int,
        obs: np.ndarray,
        latent: LatentAction,
        log_prob: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        actor_hidden: np.ndarray,
        critic_hidden: np.ndarray,
        episode_starts: np.ndarray,
    ) -> None:
        super().add(step, obs, latent, log_prob, value, reward, terminated, truncated)
        self.actor_hidden[step] = actor_hidden
        self.critic_hidden[step] = critic_hidden
        self.episode_starts[step] = episode_starts

    def iter_sequence_minibatches(
        self, minibatch_size: int, rng: np.random.Generator
    ) -> list[RecurrentRolloutBatch]:
        """Shuffle sequence chunks while preserving order inside each chunk."""
        length = self.sequence_length
        if self.num_steps % length:
            raise ValueError("num_steps must be divisible by sequence_length")
        if minibatch_size % length:
            raise ValueError("minibatch_size must be divisible by sequence_length")
        chunks: list[tuple[int, int]] = [
            (env, start)
            for env in range(self.num_envs)
            for start in range(0, self.num_steps, length)
        ]
        sequences_per_batch = minibatch_size // length
        if len(chunks) % sequences_per_batch:
            raise ValueError("number of recurrent chunks must divide evenly into minibatches")
        order = rng.permutation(len(chunks))
        batches: list[RecurrentRolloutBatch] = []

        def gather(array: np.ndarray, chosen: list[tuple[int, int]]) -> np.ndarray:
            return np.stack([array[s : s + length, env] for env, s in chosen], axis=1)

        for offset in range(0, len(chunks), sequences_per_batch):
            chosen = [chunks[int(i)] for i in order[offset : offset + sequences_per_batch]]
            sched = gather(self.schedule_raw, chosen)
            mask = gather(self.active_mask, chosen)
            power = gather(self.power_latent, chosen)
            rate = gather(self.rate_latent, chosen)
            rho_r = gather(self.rho_r_latent, chosen)
            rho_p = None if self.rho_p_latent is None else gather(self.rho_p_latent, chosen)
            gate = None if self.common_gate_raw is None else gather(self.common_gate_raw, chosen)
            latent = LatentAction(
                schedule_raw=self._tensor(sched),
                active_mask=self._tensor(mask, dtype=torch.bool),
                power_latent=self._tensor(power),
                rate_latent=self._tensor(rate),
                rho_r_latent=self._tensor(rho_r),
                rho_p_latent=None if rho_p is None else self._tensor(rho_p),
                common_gate_raw=None if gate is None else self._tensor(gate),
            )
            initial_actor = np.stack(
                [self.actor_hidden[s, env] for env, s in chosen], axis=0
            )
            initial_critic = np.stack(
                [self.critic_hidden[s, env] for env, s in chosen], axis=0
            )
            batches.append(
                RecurrentRolloutBatch(
                    obs=self._tensor(gather(self.obs, chosen)),
                    latent=latent,
                    log_probs=self._tensor(gather(self.log_probs, chosen)),
                    advantages=self._tensor(gather(self.advantages, chosen)),
                    returns=self._tensor(gather(self.returns, chosen)),
                    values=self._tensor(gather(self.values, chosen)),
                    episode_starts=self._tensor(
                        gather(self.episode_starts, chosen), dtype=torch.bool
                    ),
                    initial_actor_hidden=self._tensor(initial_actor),
                    initial_critic_hidden=self._tensor(initial_critic),
                )
            )
        return batches
