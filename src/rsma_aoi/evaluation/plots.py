"""Matplotlib figures for AoI results (headless 'Agg' backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_metric_bars(
    summary: pd.DataFrame, metric: str, out_path: str | Path, title: str | None = None
) -> Path:
    """Bar chart of one metric per algorithm with bootstrap CI error bars."""
    sub = summary[summary["metric"] == metric].copy()
    if sub.empty:
        raise ValueError(f"metric {metric!r} not present in the summary")
    sub = sub.sort_values("mean")
    labels = sub["algorithm"].astype(str).tolist()
    means = sub["mean"].to_numpy(dtype=float)
    lower = np.clip(means - sub["ci_low"].to_numpy(dtype=float), 0.0, None)
    upper = np.clip(sub["ci_high"].to_numpy(dtype=float) - means, 0.0, None)

    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2.0, 4.0))
    ax.bar(labels, means, yerr=np.vstack([lower, upper]), capsize=4, color="#4C72B0")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} (mean with bootstrap 95% CI)")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_aoi_cdf(
    df: pd.DataFrame, out_path: str | Path, metric: str = "weighted_avg_aoi"
) -> Path:
    """Empirical CDF of a per-episode metric, one curve per algorithm."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for name, sub in df.groupby("algorithm"):
        values = np.sort(sub[metric].to_numpy(dtype=float))
        if values.size == 0:
            continue
        ax.plot(values, np.linspace(0.0, 1.0, values.size, endpoint=True), label=str(name))
    ax.set_xlabel(metric)
    ax.set_ylabel("empirical CDF")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_learning_curve(
    scalars_csv: str | Path, out_path: str | Path, metric: str = "env/weighted_aoi"
) -> Path:
    """Plot one TensorBoard-mirrored scalar from a run's ``scalars.csv``."""
    return plot_learning_curves({"run": scalars_csv}, out_path, metric=metric)


def plot_learning_curves(
    runs: dict[str, str | Path],
    out_path: str | Path,
    metric: str = "env/weighted_aoi",
    baselines: dict[str, float] | None = None,
    smooth: int = 1,
) -> Path:
    """Compare several runs' learning curves on the same axes.

    Args:
        runs: mapping ``label -> path`` of each run's ``scalars.csv``.
        out_path: output image path.
        metric: column to plot (must exist in every run's CSV).
        baselines: optional ``label -> value`` horizontal reference lines.
        smooth: rolling-mean window in updates (1 disables smoothing).

    Returns:
        The written path.
    """
    if not runs:
        raise ValueError("runs must be non-empty")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for label, csv_path in runs.items():
        df = pd.read_csv(csv_path)
        if metric not in df.columns:
            raise ValueError(f"{metric!r} not in {Path(csv_path).name}: {sorted(df.columns)}")
        series = df[metric]
        if smooth > 1:
            series = series.rolling(smooth, min_periods=1).mean()
        ax.plot(df["step"], series, label=str(label), linewidth=1.6)
    for label, value in (baselines or {}).items():
        ax.axhline(value, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.annotate(f"{label} ({value:.2f})", xy=(0.01, value), xycoords=("axes fraction", "data"),
                    fontsize=7, va="bottom")
    ax.set_xlabel("environment interactions")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
