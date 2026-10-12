#!/usr/bin/env python3
"""Summarize matched joint-gate RSMA versus learned-power SDMA pilots."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition", action="append", nargs=2, required=True,
        metavar=("LABEL", "RESULT_DIR"),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict] = []
    seed_rows: list[pd.DataFrame] = []
    for label, directory_text in args.condition:
        directory = Path(directory_text)
        data = pd.read_csv(directory / "training_seed_summary.csv")
        data = data[data.execution == "deterministic"].copy()
        gate = data[data.algorithm == "joint_gate"].set_index("training_seed")
        sdma = data[data.algorithm == "sdma_power"].set_index("training_seed")
        common = gate.index.intersection(sdma.index)
        if common.empty or len(common) != len(gate) or len(common) != len(sdma):
            raise ValueError(f"unmatched joint_gate/sdma_power training seeds in {directory}")
        paired = pd.DataFrame({
            "condition": label,
            "training_seed": common,
            "rsma_aoi": gate.loc[common, "mean_aoi"].to_numpy(),
            "sdma_aoi": sdma.loc[common, "mean_aoi"].to_numpy(),
        })
        paired["delta"] = paired.rsma_aoi - paired.sdma_aoi
        paired["percent_vs_sdma"] = 100.0 * paired.delta / paired.sdma_aoi
        seed_rows.append(paired)

        trace = pd.read_csv(directory / "allocation_trace.csv.gz")
        trace = trace[(trace.algorithm == "joint_gate") & (trace.execution == "deterministic")]
        on = trace.common_enabled > 0.5
        rows.append({
            "condition": label,
            "training_seeds": len(common),
            "rsma_mean_aoi": float(paired.rsma_aoi.mean()),
            "sdma_mean_aoi": float(paired.sdma_aoi.mean()),
            "mean_paired_delta": float(paired.delta.mean()),
            "seed_sd_paired_delta": float(paired.delta.std(ddof=1)),
            "mean_percent_vs_sdma": float(paired.percent_vs_sdma.mean()),
            "seed_sd_percent_vs_sdma": float(paired.percent_vs_sdma.std(ddof=1)),
            "common_on_rate": float(on.mean()),
            "common_bler_when_on": float(trace.loc[on, "common_bler"].mean()) if on.any() else np.nan,
        })

    summary = pd.DataFrame(rows)
    paired_all = pd.concat(seed_rows, ignore_index=True)
    summary.to_csv(args.out_dir / "factorial_summary.csv", index=False)
    paired_all.to_csv(args.out_dir / "factorial_seed_pairs.csv", index=False)

    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    axes[0].bar(x - 0.18, summary.sdma_mean_aoi, width=0.36, label="SDMA power PPO", color="#64748b")
    axes[0].bar(x + 0.18, summary.rsma_mean_aoi, width=0.36, label="RSMA with gate", color="#2563eb")
    axes[0].set(ylabel="Mean AoI (slots)", xticks=x, xticklabels=summary.condition)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(
        x, summary.mean_percent_vs_sdma,
        yerr=summary.seed_sd_percent_vs_sdma,
        capsize=4, color="#2563eb",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(ylabel="RSMA AoI difference vs SDMA (%)", xticks=x, xticklabels=summary.condition)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Matched allocation-policy pilot | lower AoI is better")
    fig.savefig(args.out_dir / "factorial_comparison.png", dpi=180)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
