#!/usr/bin/env python3
"""Build the target-condition comparison figure for the thesis rule baselines."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "thesis_rule_baselines_target_v1"


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, half


def main() -> None:
    ppo = pd.read_csv(
        ROOT / "results/ppo_k3_mixed_phy_seed_expansion_target_v1/target_trajectories.csv"
    )
    ppo = ppo[ppo["kind"] == "ppo"]
    ppo_seed = ppo.groupby("training_seed")["mean_aoi"].mean().to_numpy()

    q_only = pd.read_csv(ROOT / "results/q_only_sdma_target_v1/target_trajectories.csv")
    q_only = q_only[q_only["label"] == "q_only_ppo_sdma"]
    q_only_seed = q_only.groupby("training_seed")["mean_aoi"].mean().to_numpy()

    rules = pd.read_csv(OUT / "target_trajectories.csv")
    fixed = rules[rules["kind"] == "oldest_common_fixed_alpha"]["mean_aoi"].to_numpy()
    single = rules[rules["kind"] == "max_aoi_single_user"]["mean_aoi"].to_numpy()

    labels = ["RSMA-PPO", "Fixed-rule\nRSMA", "q-only\nPPO-SDMA", "Max-AoI\nsingle-user"]
    samples = [ppo_seed, fixed, q_only_seed, single]
    means = [float(x.mean()) for x in samples]
    # PPO policies use training-seed CIs. Rule policies have no training seed;
    # their whiskers show test-permutation SD and are labelled separately.
    ppo_ci = [mean_ci(ppo_seed)[1], 0.0, mean_ci(q_only_seed)[1], 0.0]
    rule_sd = [0.0, float(fixed.std(ddof=1)), 0.0, float(single.std(ddof=1))]

    fig, ax = plt.subplots(figsize=(8.2, 4.7), dpi=180)
    x = np.arange(len(labels))
    colors = ["#176B87", "#5FA8A2", "#C68B3C", "#6B7280"]
    ax.bar(x, means, width=0.64, color=colors, edgecolor="#263238", linewidth=0.6)
    for i in (0, 2):
        ax.errorbar(i, means[i], yerr=ppo_ci[i], fmt="none", ecolor="#111827", capsize=5, lw=1.3)
    for i in (1, 3):
        if rule_sd[i] > 0:
            ax.errorbar(i, means[i], yerr=rule_sd[i], fmt="none", ecolor="#374151", capsize=5, lw=1.1)
    ax.axhline(2.0, color="#B42318", linestyle="--", linewidth=1.3,
               label="Perfect one-success/slot lower bound: (K+1)/2 = 2")
    for i, value in enumerate(means):
        ax.text(i, value + 0.045, f"{value:.3f}", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean AoI")
    ax.set_ylim(0, 3.05)
    ax.set_title("Target condition: matched and AoI-aware baselines")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.text(
        0.01, -0.20,
        "Whiskers: 95% Student-t CI over 5 training seeds (learned policies);\n"
        "test-permutation SD for the validation-tuned fixed rule. Single-user is deterministic here.",
        transform=ax.transAxes, fontsize=7.7, color="#4B5563", va="top",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(OUT / "figure_target_rule_baselines.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
