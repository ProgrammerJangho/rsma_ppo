#!/usr/bin/env python3
"""Create the compact table and figure for the final online-CEM audit."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SHORT_ORDER = (
    "objective_warm_joint_gate", "sdma_power_ppo", "cem_rsma_raw_960",
    "cem_sdma_960", "cem_rsma_safeguarded_1920",
)
LONG_262K_ORDER = (
    "rsma_262k", "sdma_262k", "cem_rsma_raw_960",
    "cem_sdma_960", "cem_rsma_safeguarded_1920",
)
LABELS = ("RSMA-PPO", "SDMA-PPO", "RSMA-CEM", "SDMA-CEM", "Safe RSMA-CEM")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/thesis_cem_reference_v2")
    parser.add_argument("--out-dir", default="results/thesis_cem_reference_summary_v1")
    parser.add_argument("--profile", choices=("short", "long_262k"), default="short")
    args = parser.parse_args()
    order = SHORT_ORDER if args.profile == "short" else LONG_262K_ORDER
    source, out = Path(args.input), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(source / "summary.csv")
    wide = summary.pivot(index="algorithm", columns="metric", values="mean").loc[list(order)]
    uncertainty = summary[summary.metric == "mean_aoi"].set_index("algorithm")
    table = wide[[
        "mean_aoi", "success_rate", "aoi_p95", "mean_common_power", "mean_rho",
        "common_on_rate", "decision_latency_ms", "candidate_evaluations_per_slot",
        "boundary_selection_rate",
    ]].copy()
    table.insert(1, "mean_aoi_std", uncertainty.loc[list(order), "std"])
    table.insert(2, "mean_aoi_ci_low", uncertainty.loc[list(order), "ci_low"])
    table.insert(3, "mean_aoi_ci_high", uncertainty.loc[list(order), "ci_high"])
    table["uncertainty_unit"] = uncertainty.loc[list(order), "uncertainty_unit"]
    table.reset_index().to_csv(out / "comparison.csv", index=False)

    x = np.arange(len(order))
    colors = ["#2563eb", "#f59e0b", "#60a5fa", "#fbbf24", "#16a34a"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.25), constrained_layout=True)
    aoi = table.mean_aoi.to_numpy()
    lo = table.mean_aoi_ci_low.to_numpy()
    hi = table.mean_aoi_ci_high.to_numpy()
    axes[0].bar(x, aoi, color=colors)
    axes[0].errorbar(x, aoi, yerr=np.vstack((aoi - lo, hi - aoi)), fmt="none",
                     ecolor="#111827", capsize=4)
    axes[0].set_ylabel("Mean AoI per user")
    axes[0].set_title(
        "Short matched-trajectory audit" if args.profile == "short"
        else "Frozen long-horizon protocol"
    )
    axes[0].grid(axis="y", alpha=.25)

    latency = table.decision_latency_ms.to_numpy()
    axes[1].bar(x, latency, color=colors)
    axes[1].set_ylabel("Decision latency (ms/slot, CPU)")
    axes[1].set_title("Measured implementation cost")
    axes[1].grid(axis="y", alpha=.25)
    for ax in axes:
        ax.set_xticks(x, LABELS, rotation=22, ha="right")
    fig.suptitle("20°, SNR 10 dB, relative CSIT error 0.2")
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"cem_reference.{suffix}", dpi=220)
    plt.close(fig)
    print(table.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
