"""Episode-level evaluation shared by PPO and every baseline.

Fair comparison
---------------
All algorithms are evaluated on a *single* environment instance reset with the
same seed sequence ``eval_seed + episode_index``.  The environment consumes a
fixed number of random draws per slot (channel evolution, CSIT error, ACK), so
the channel realisation sequence of episode ``i`` is identical for every
algorithm regardless of the actions it takes.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.evaluation.metrics import EpisodeMetrics, summarise_episode
from rsma_aoi.utils.config import EnvConfig


def apply_overrides(cfg: EnvConfig, overrides: dict[str, Any] | None) -> EnvConfig:
    """Return a copy of ``cfg`` with ``overrides`` applied and revalidated."""
    if not overrides:
        return replace(cfg)
    unknown = set(overrides) - set(cfg.to_dict())
    if unknown:
        raise KeyError(f"unknown environment override(s): {sorted(unknown)}")
    return replace(cfg, **overrides)


def evaluate_policy(
    policy: Any,
    env_cfg: EnvConfig,
    episodes: int = 20,
    eval_seed: int = 10_000,
    algorithm: str | None = None,
    env: AoIRSMAEnv | None = None,
) -> pd.DataFrame:
    """Run ``episodes`` episodes and return one row of metrics per episode.

    Args:
        policy: object exposing ``act(obs_batch, states) -> list[action dict]``.
        env_cfg: environment configuration (already including any overrides).
        episodes: number of evaluation episodes.
        eval_seed: base seed; episode ``i`` uses ``eval_seed + i``.
        algorithm: label written into the ``algorithm`` column.
        env: optional pre-built environment (otherwise one is created).

    Returns:
        DataFrame with the columns of :class:`EpisodeMetrics` plus
        ``algorithm``.
    """
    owns_env = env is None
    env = env or AoIRSMAEnv(env_cfg)
    label = algorithm or getattr(policy, "name", policy.__class__.__name__)
    weights = np.asarray(env_cfg.weight_vector, dtype=np.float64)
    rows: list[dict[str, float]] = []

    try:
        for episode in range(episodes):
            seed = int(eval_seed + episode)
            if hasattr(policy, "reset"):
                policy.reset(seed=seed)
            obs, _ = env.reset(seed=seed)
            aoi_trace, reward_trace, info_trace = [], [], []
            latency_total, latency_calls = 0.0, 0

            done = False
            while not done:
                start = time.perf_counter()
                action = policy.act(obs[None, :], [env.get_state()])[0]
                latency_total += time.perf_counter() - start
                latency_calls += 1
                obs, reward, terminated, truncated, info = env.step(action)
                aoi_trace.append(info["aoi"])
                reward_trace.append(reward)
                info_trace.append(info)
                done = bool(terminated or truncated)

            metrics: EpisodeMetrics = summarise_episode(
                episode_index=episode,
                seed=seed,
                aoi_trace=np.stack(aoi_trace),
                reward_trace=np.asarray(reward_trace, dtype=np.float64),
                info_trace=info_trace,
                weights=weights,
                tail_threshold=float(env_cfg.aoi_tail_threshold),
                total_power=env_cfg.total_power,
                latency_ms=1000.0 * latency_total / max(latency_calls, 1),
                classes=env_cfg.class_vector,
            )
            row = metrics.to_row()
            row["algorithm"] = label
            rows.append(row)
    finally:
        if owns_env:
            env.close()

    return pd.DataFrame(rows)


def save_results(
    df: pd.DataFrame, out_dir: str | Path, name: str = "episodes"
) -> Path:
    """Write per-episode results to ``out_dir/name.csv`` and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return path
