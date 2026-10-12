#!/usr/bin/env python3
"""Build the matched RSMA-PPO versus q-only PPO-SDMA thesis artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/q_only_sdma_target_v1"


def main() -> int:
    seeds = pd.read_csv(OUT / "target_seed_summary.csv")
    wide = seeds.pivot(index="training_seed", columns="label", values="mean_aoi")
    gain = 100.0 * (wide.q_only_ppo_sdma - wide.rsma_ppo) / wide.q_only_ppo_sdma
    difference = wide.q_only_ppo_sdma - wide.rsma_ppo
    critical = float(t.ppf(0.975, len(wide) - 1))
    summary = {
        "training_seeds": int(len(wide)),
        "rsma_mean_aoi": float(wide.rsma_ppo.mean()),
        "q_only_mean_aoi": float(wide.q_only_ppo_sdma.mean()),
        "rsma_gain_vs_q_only_mean_pct": float(gain.mean()),
        "rsma_gain_vs_q_only_seed_std_pct": float(gain.std(ddof=1)),
        "rsma_gain_vs_q_only_ci95_halfwidth_pct": float(
            critical * gain.std(ddof=1) / np.sqrt(len(gain))
        ),
        "absolute_aoi_difference_mean": float(difference.mean()),
        "absolute_aoi_difference_ci95_halfwidth": float(
            critical * difference.std(ddof=1) / np.sqrt(len(difference))
        ),
    }
    trajectories = pd.read_csv(OUT / "paired_trajectory_comparison.csv")
    summary["trajectory_wins"] = int((trajectories.rsma_ppo < trajectories.q_only_ppo_sdma).sum())
    summary["trajectory_count"] = int(len(trajectories))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = ["RSMA-PPO", "q-only\nPPO-SDMA"]
    groups = [wide.rsma_ppo.to_numpy(), wide.q_only_ppo_sdma.to_numpy()]
    colors = ["#1f77b4", "#d97706"]
    rng = np.random.default_rng(7)
    for index, (values, color) in enumerate(zip(groups, colors, strict=True)):
        x = index + rng.uniform(-0.055, 0.055, size=len(values))
        ax.scatter(x, values, color=color, s=34, zorder=3)
        mean = values.mean()
        ci = critical * values.std(ddof=1) / np.sqrt(len(values))
        ax.errorbar(index, mean, yerr=ci, fmt="s", color="black", capsize=5, zorder=4)
    for _, row in wide.iterrows():
        ax.plot([0, 1], [row.rsma_ppo, row.q_only_ppo_sdma], color="#9aa4ad", alpha=0.45, lw=0.9)
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel("Mean AoI (slots)")
    ax.set_title(
        "Matched target comparison\n"
        f"RSMA gain: {gain.mean():.2f}% ± "
        f"{summary['rsma_gain_vs_q_only_ci95_halfwidth_pct']:.2f}%p",
        pad=8,
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"figure_q_only_comparison.{suffix}", dpi=300)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
