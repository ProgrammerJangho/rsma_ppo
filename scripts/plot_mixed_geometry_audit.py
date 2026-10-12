#!/usr/bin/env python3
"""Plot the mixed-geometry policy's mode use and compression loss."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/thesis_mixed_geometry_audit_v1"

actions = pd.read_csv(RESULT / "action_stats.csv")
contrasts_a = pd.read_csv(RESULT / "paired_contrasts.csv")
contrasts_b = pd.read_csv(
    ROOT / "results/thesis_mixed_rate_argmax_v1/paired_contrasts.csv"
)

angles = np.array([5, 20, 60])
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
for metric, label, marker in (
    ("common_on_rate", "common-on probability", "o"),
    ("alpha_on_mean", r"common power $\alpha\mid$on", "s"),
    ("rho_on_mean", r"payload scale $\rho\mid$on", "^"),
):
    grouped = actions.groupby("angle_deg")[metric]
    mean = grouped.mean().reindex(angles)
    std = grouped.std().reindex(angles)
    axes[0].errorbar(angles, mean, yerr=std, marker=marker, capsize=3, label=label)
axes[0].set(xlabel="LoS angle separation (degrees)", ylabel="Fraction", ylim=(-0.03, 1.05))
axes[0].set_xticks(angles)
axes[0].grid(alpha=0.25)
axes[0].legend(fontsize=8)

labels = ["fixed α,ρ", "rate argmax", "both"]
tables = [
    contrasts_a[contrasts_a.candidate == "global_fixed_magnitude"],
    contrasts_b[contrasts_b.candidate == "rate_argmax"],
    contrasts_b[contrasts_b.candidate == "fixed_magnitude_rate_argmax"],
]
width = 0.24
for i, (label, table) in enumerate(zip(labels, tables)):
    values = table.set_index("angle_deg").reindex(angles)["mean_paired_percent"]
    axes[1].bar(np.arange(3) + (i - 1) * width, values, width, label=label)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set(
    xlabel="LoS angle separation",
    ylabel="AoI change from full PPO (%)",
    xticks=np.arange(3),
    xticklabels=[f"{x}°" for x in angles],
)
axes[1].grid(axis="y", alpha=0.25)
axes[1].legend(fontsize=8)
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(RESULT / f"mixed_geometry_policy_audit.{suffix}", dpi=220)
plt.close(fig)
