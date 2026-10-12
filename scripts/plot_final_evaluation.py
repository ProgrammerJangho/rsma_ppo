#!/usr/bin/env python3
"""Create thesis-ready figures from run_final_evaluation.py outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "objective_warm_joint_gate": "Objective-warm\nRSMA-PPO",
    "sdma_power_ppo": "SDMA power\nPPO",
    "scratch_joint_gate": "Scratch\nRSMA-PPO",
    "sdma_equal": "Equal SDMA",
    "fixed_rsma": "Fixed RSMA",
}
COLORS = {
    "objective_warm_joint_gate": "#2563eb",
    "sdma_power_ppo": "#f59e0b",
    "scratch_joint_gate": "#64748b",
    "sdma_equal": "#10b981",
    "fixed_rsma": "#dc2626",
}


def metric_rows(summary: pd.DataFrame, metric: str, order: list[str]) -> pd.DataFrame:
    rows = summary[summary.metric == metric].set_index("algorithm").loc[order].reset_index()
    rows["err_low"] = rows["mean"] - rows["ci_low"]
    rows["err_high"] = rows["ci_high"] - rows["mean"]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="results/thesis_final_evaluation_v1")
    args = parser.parse_args()
    out = Path(args.result_dir)
    summary = pd.read_csv(out / "summary.csv")

    learned = ["objective_warm_joint_gate", "sdma_power_ppo", "scratch_joint_gate"]
    all_methods = [*learned, "sdma_equal", "fixed_rsma"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    for ax, order, title, log_scale in (
        (axes[0], learned, "Learned policies", False),
        (axes[1], all_methods, "All references", True),
    ):
        rows = metric_rows(summary, "mean_aoi", order)
        x = np.arange(len(rows))
        ax.bar(x, rows["mean"], color=[COLORS[v] for v in rows.algorithm], width=.68)
        ax.errorbar(x, rows["mean"], yerr=np.vstack([rows.err_low, rows.err_high]),
                    fmt="none", ecolor="#111827", capsize=4, linewidth=1.2)
        if log_scale:
            ax.set_xticks(x, [LABELS[v] for v in rows.algorithm], fontsize=7.5,
                          rotation=12, ha="right")
        else:
            ax.set_xticks(x, [LABELS[v] for v in rows.algorithm], fontsize=8.5)
        ax.set_ylabel("Mean AoI per user")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.25)
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Mean AoI per user (log scale)")
    fig.suptitle("20° separation, imperfect CSIT $\\sigma_e=0.2$, SNR 10 dB")
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"mean_aoi_comparison.{suffix}", dpi=220)
    plt.close(fig)

    action_methods = ["objective_warm_joint_gate", "scratch_joint_gate", "sdma_power_ppo"]
    action_metrics = [
        ("mean_common_power", "Common power fraction"),
        ("mean_rho", "Common payload scale $\\rho$"),
        ("common_on_rate", "Common-on fraction"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, action_metrics):
        rows = metric_rows(summary, metric, action_methods)
        x = np.arange(len(rows))
        ax.bar(x, rows["mean"], color=[COLORS[v] for v in rows.algorithm], width=.68)
        ax.errorbar(x, rows["mean"], yerr=np.vstack([rows.err_low, rows.err_high]),
                    fmt="none", ecolor="#111827", capsize=4, linewidth=1.2)
        ax.set_xticks(x, [LABELS[v] for v in rows.algorithm], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Learned common-stream actions")
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"action_statistics.{suffix}", dpi=220)
    plt.close(fig)
    print(f"WROTE figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
