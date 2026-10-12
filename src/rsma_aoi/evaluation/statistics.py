"""Aggregation across episodes and seeds: mean, std, bootstrap 95% CI."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_METRICS = (
    "weighted_avg_aoi", "mean_aoi", "peak_aoi", "aoi_p95", "aoi_p99", "max_aoi",
    "aoi_jain", "avg_bler", "success_rate", "avg_power", "violation_rate",
    "episode_return", "inference_latency_ms", "avg_num_scheduled",
    "avg_common_power_ratio", "tail_excess", "avg_common_payload_ratio",
    "schedule_entropy", "sched_frac_0", "sched_frac_1", "sched_frac_2",
    "sched_frac_3", "sched_frac_4",
)


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    num_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean.

    Args:
        values: 1-D sample.
        confidence: coverage (0.95 -> 2.5%/97.5% percentiles).
        num_resamples: bootstrap replications.
        seed: seed of the resampling generator (explicit, never global).

    Returns:
        ``(low, high)``; a degenerate sample returns ``(mean, mean)``.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    if x.size == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(num_resamples, x.size))
    means = x[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))


def summarise(
    df: pd.DataFrame,
    group_keys: list[str] | None = None,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-group mean / std / n / bootstrap CI for the requested metrics.

    Args:
        df: raw per-episode results.
        group_keys: grouping columns (default ``['algorithm']`` if present).
        metrics: metric columns to aggregate (missing ones are skipped).
        seed: bootstrap seed.

    Returns:
        Long-format summary with one row per (group, metric).
    """
    if group_keys is None:
        group_keys = ["algorithm"] if "algorithm" in df.columns else []
    available = [m for m in metrics if m in df.columns]
    rows = []
    groups = df.groupby(group_keys, dropna=False) if group_keys else [((), df)]
    for key, sub in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        for metric in available:
            values = sub[metric].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            low, high = bootstrap_ci(finite, seed=seed)
            row = {k: v for k, v in zip(group_keys, key_tuple)}
            row.update(
                {
                    "metric": metric,
                    "mean": float(finite.mean()) if finite.size else float("nan"),
                    "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
                    "n": int(finite.size),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def pivot_summary(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Extract one metric from a long-format summary as a compact table."""
    sub = summary[summary["metric"] == metric]
    return sub.drop(columns=["metric"]).reset_index(drop=True)
