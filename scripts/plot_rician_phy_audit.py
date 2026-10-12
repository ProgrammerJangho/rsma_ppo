#!/usr/bin/env python3
"""Create the compact thesis figure for the controlled-Rician PHY audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = [
    "thesis_rician_alignment_5deg",
    "thesis_rician_alignment_20deg",
    "thesis_rician_alignment_60deg",
]
LABELS = ["5°", "20°", "60°"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir")
    args = parser.parse_args()
    root = Path(args.result_dir)
    trace = pd.read_csv(root / "optimized_trace.csv.gz")
    comparisons = pd.read_csv(root / "paired_comparisons.csv")
    summary = pd.read_csv(root / "scheme_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7), constrained_layout=True)

    unique = trace.drop_duplicates(["config", "state_id"])
    arrays = [unique.loc[unique.config == name, "realized_alignment"] for name in ORDER]
    axes[0].boxplot(arrays, tick_labels=LABELS, showfliers=False)
    axes[0].set(title="(a) Realized channel alignment", xlabel="LoS angle separation",
                ylabel=r"$|h_1^Hh_2|^2/(\|h_1\|^2\|h_2\|^2)$")
    axes[0].grid(axis="y", alpha=0.25)

    equal = comparisons[
        (comparisons.profile == "equal")
        & comparisons.comparison.isin([
            "optimized_principal_minus_sdma", "optimized_maxmin_minus_sdma"
        ])
    ]
    x = np.arange(3)
    for offset, (comparison, label, marker) in zip(
        [-0.08, 0.08],
        [
            ("optimized_principal_minus_sdma", "Principal", "o"),
            ("optimized_maxmin_minus_sdma", "Maxmin", "s"),
        ],
    ):
        rows = equal.set_index("config").loc[ORDER]
        rows = rows[rows.comparison == comparison] if "comparison" in rows.columns else rows
        # set_index above retains one row per config for this filtered comparison.
        mean = rows["mean"].to_numpy()
        low = mean - rows["ci95_low"].to_numpy()
        high = rows["ci95_high"].to_numpy() - mean
        axes[1].errorbar(x + offset, mean, yerr=[low, high], marker=marker,
                         capsize=3, linewidth=1.4, label=label)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(title="(b) One-step gain over SDMA", xlabel="LoS angle separation",
                ylabel=r"$J_{RSMA}-J_{SDMA}$", xticks=x, xticklabels=LABELS)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    equal_summary = summary[summary.profile == "equal"]
    for beam, label, marker in [("principal", "Principal", "o"), ("maxmin", "Maxmin", "s")]:
        rows = equal_summary[equal_summary.beam == beam].set_index("config").loc[ORDER]
        axes[2].plot(x, rows.mean_common_power, marker=marker, label=f"$f_c$: {label}")
        axes[2].plot(x, rows.mean_rho, marker=marker, linestyle="--", label=fr"$\rho$: {label}")
    axes[2].set(title="(c) CEM-selected allocation", xlabel="LoS angle separation",
                ylabel="Mean fraction", xticks=x, xticklabels=LABELS, ylim=(-0.03, 1.03))
    axes[2].legend(frameon=False, fontsize=8, ncol=2)
    axes[2].grid(axis="y", alpha=0.25)

    fig.savefig(root / "rician_phy_audit.png", dpi=220)
    fig.savefig(root / "rician_phy_audit.pdf")
    print(root / "rician_phy_audit.png")


if __name__ == "__main__":
    main()
