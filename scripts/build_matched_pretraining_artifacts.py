#!/usr/bin/env python3
"""Build artifacts for the objective-only RSMA/SDMA pretraining audit."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/matched_pretraining_target_v1"


def _ci(values: pd.Series) -> float:
    return float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def main() -> int:
    seeds = pd.read_csv(OUT / "target_seed_summary.csv")
    wide = seeds.pivot(index="training_seed", columns="label", values="mean_aoi")
    gain = 100.0 * (wide.objective_only_sdma - wide.objective_only_rsma) / wide.objective_only_sdma
    loss_change = 100.0 * (
        wide.action_objective_rsma - wide.objective_only_rsma
    ) / wide.action_objective_rsma
    difference = wide.objective_only_sdma - wide.objective_only_rsma
    trajectories = pd.read_csv(OUT / "paired_trajectory_comparison.csv")
    summary = {
        "training_seeds": int(len(wide)),
        "objective_only_rsma_mean_aoi": float(wide.objective_only_rsma.mean()),
        "objective_only_sdma_mean_aoi": float(wide.objective_only_sdma.mean()),
        "action_objective_rsma_mean_aoi": float(wide.action_objective_rsma.mean()),
        "matched_rsma_gain_vs_sdma_mean_pct": float(gain.mean()),
        "matched_rsma_gain_vs_sdma_ci95_halfwidth_pct": _ci(gain),
        "matched_absolute_aoi_difference_mean": float(difference.mean()),
        "matched_absolute_aoi_difference_ci95_halfwidth": _ci(difference),
        "objective_only_gain_vs_action_objective_rsma_mean_pct": float(loss_change.mean()),
        "objective_only_gain_vs_action_objective_rsma_ci95_halfwidth_pct": _ci(loss_change),
        "matched_trajectory_wins": int(
            (trajectories.objective_only_rsma < trajectories.objective_only_sdma).sum()
        ),
        "trajectory_count": int(len(trajectories)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    labels = ["RSMA\nobjective-only", "SDMA\nobjective-only", "RSMA\naction+objective"]
    columns = ["objective_only_rsma", "objective_only_sdma", "action_objective_rsma"]
    colors = ["#1f77b4", "#d97706", "#648fff"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    rng = np.random.default_rng(7)
    for index, (column, color) in enumerate(zip(columns, colors, strict=True)):
        values = wide[column].to_numpy()
        x = index + rng.uniform(-0.055, 0.055, size=len(values))
        ax.scatter(x, values, color=color, s=34, zorder=3)
        ax.errorbar(
            index, values.mean(), yerr=_ci(pd.Series(values)), fmt="s",
            color="black", capsize=5, zorder=4,
        )
    for _, row in wide.iterrows():
        ax.plot(
            [0, 1], [row.objective_only_rsma, row.objective_only_sdma],
            color="#9aa4ad", alpha=0.45, lw=0.9,
        )
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Mean AoI (slots)")
    ax.set_title(
        "Matched objective-only pretraining audit\n"
        f"RSMA gain over SDMA: {gain.mean():.2f}% ± {_ci(gain):.2f}%p",
        pad=8,
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"figure_matched_pretraining.{suffix}", dpi=300)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
