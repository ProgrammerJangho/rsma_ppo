#!/usr/bin/env python3
"""Compare deterministic joint-PPO stages using training-seed summaries."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", nargs=2, metavar=("LABEL", "RESULT_DIR"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for label, directory in args.stage:
        data = pd.read_csv(Path(directory) / "training_seed_summary.csv")
        data = data[(data.algorithm == "joint") & (data.execution == "deterministic")]
        if data.empty:
            raise ValueError(f"no deterministic joint rows in {directory}")
        for _, row in data.iterrows():
            rows.append({"stage": label, "training_seed": row.training_seed,
                         "mean_aoi": row.mean_aoi, "common_bler": row.mean_common_bler,
                         "common_power": row.mean_common_power, "rho": row.mean_rho})
    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out.with_suffix(".csv"), index=False)
    order = [label for label, _ in args.stage]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    for ax, metric, ylabel in [(axes[0], "mean_aoi", "Mean AoI (slots)"),
                               (axes[1], "common_bler", "Mean common BLER")]:
        grouped = frame.groupby("stage", sort=False)[metric]
        means = grouped.mean().reindex(order)
        stds = grouped.std().reindex(order).fillna(0)
        ax.bar(np.arange(len(order)), means, yerr=stds, capsize=4, color="#2563eb")
        ax.set(xticks=np.arange(len(order)), xticklabels=order, ylabel=ylabel)
        ax.grid(axis="y", alpha=.2)
    axes[0].axhline(1.837782, color="#64748b", linestyle="--", label="SDMA fixed reference")
    axes[0].legend(fontsize=8)
    fig.suptitle("Controlled joint-PPO stages | error bars: training-seed SD")
    fig.savefig(out, dpi=180)


if __name__ == "__main__":
    main()
