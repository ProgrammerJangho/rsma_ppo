"""Minimal synchronous vector environment with explicit per-env seeding.

Gymnasium's own vector wrappers batch actions into arrays, which is awkward for
this project's ``Dict`` action space (each sub-environment needs its own dict).
This wrapper keeps a list of dicts, performs classic auto-reset and exposes the
terminal observation as ``info['final_observation']`` so that GAE can bootstrap
correctly on truncation.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from rsma_aoi.envs.aoi_env import AoIRSMAEnv


class SyncVecEnv:
    """Run ``N`` environments sequentially in the current process."""

    def __init__(self, env_fns: Sequence[Callable[[], AoIRSMAEnv]]) -> None:
        if not env_fns:
            raise ValueError("env_fns must be non-empty")
        self.envs: list[AoIRSMAEnv] = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.obs_dim = int(np.prod(self.observation_space.shape))

    # ------------------------------------------------------------------
    def reset(self, seeds: Sequence[int] | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Reset every sub-environment; ``seeds[i]`` seeds environment ``i``."""
        obs_list, infos = [], []
        for i, env in enumerate(self.envs):
            seed = None if seeds is None else int(seeds[i])
            obs, info = env.reset(seed=seed)
            obs_list.append(obs)
            infos.append(info)
        return np.stack(obs_list).astype(np.float32), infos

    def step(
        self, actions: Sequence[dict[str, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Step all environments; auto-resets on termination/truncation.

        Returns:
            ``(obs, rewards, terminated, truncated, infos)`` with array shapes
            ``(N, obs_dim)`` and ``(N,)``.  For finished episodes the info dict
            carries ``final_observation`` and ``final_info``.
        """
        if len(actions) != self.num_envs:
            raise ValueError("one action per sub-environment is required")
        obs_list, rewards, terms, truncs, infos = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                info = dict(info)
                info["final_observation"] = obs
                info["final_info"] = {k: v for k, v in info.items() if k != "final_observation"}
                obs, _ = env.reset()
            obs_list.append(obs)
            rewards.append(reward)
            terms.append(terminated)
            truncs.append(truncated)
            infos.append(info)
        return (
            np.stack(obs_list).astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terms, dtype=bool),
            np.asarray(truncs, dtype=bool),
            infos,
        )

    # ------------------------------------------------------------------
    def get_states(self) -> list[dict[str, Any]]:
        """Structured PHY/AoI state of every sub-environment (for baselines)."""
        return [env.get_state() for env in self.envs]

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def __len__(self) -> int:
        return self.num_envs
