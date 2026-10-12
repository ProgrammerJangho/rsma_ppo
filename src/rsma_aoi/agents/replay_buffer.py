"""Off-policy replay buffer for the TD3 baseline.

Stores the *latent* action (the unconstrained vector the actor emits), never the
transformed environment action, so the critic learns ``Q(s, z)`` on exactly the
variable the actor optimises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ReplayBatch:
    """One sampled minibatch."""

    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    not_done: torch.Tensor
    """1.0 when the next state should be bootstrapped (time-limit truncation
    counts as not-done, true termination as done)."""


class ReplayBuffer:
    """Fixed-capacity circular replay buffer.

    Args:
        capacity: maximum number of transitions retained.
        obs_dim: observation width.
        action_dim: latent action width.
        device: device the sampled tensors are placed on.
    """

    def __init__(
        self, capacity: int, obs_dim: int, action_dim: int, device: str | torch.device = "cpu"
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.not_done = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        terminated: np.ndarray,
    ) -> None:
        """Insert ``N`` transitions from a vectorised step.

        ``terminated`` must be the *true* MDP termination flag; time-limit
        truncations are bootstrapped and therefore have ``not_done = 1``.
        """
        n = obs.shape[0]
        if n > self.capacity:
            raise ValueError("batch larger than the buffer capacity")
        idx = (self.ptr + np.arange(n)) % self.capacity
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.action[idx] = action
        self.reward[idx] = reward
        self.not_done[idx] = 1.0 - terminated.astype(np.float32)
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = int(min(self.size + n, self.capacity))

    def sample(self, batch_size: int, rng: np.random.Generator) -> ReplayBatch:
        """Sample ``batch_size`` transitions uniformly with replacement."""
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = rng.integers(0, self.size, size=batch_size)
        to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)  # noqa: E731
        return ReplayBatch(
            obs=to(self.obs[idx]),
            action=to(self.action[idx]),
            reward=to(self.reward[idx]),
            next_obs=to(self.next_obs[idx]),
            not_done=to(self.not_done[idx]),
        )

    def __len__(self) -> int:
        return self.size
