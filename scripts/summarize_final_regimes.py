#!/usr/bin/env python3
"""Combine the frozen headline, spatial and near/far evaluations."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STUDIES = {
    "spatial_5deg": "thesis_spatial_5deg_longrun_v1",
    "spatial_20deg": "thesis_spatial_20deg_longrun_v1",
    "spatial_60deg": "thesis_spatial_60deg_longrun_v1",
    "headline_20deg_csit02": "thesis_final_evaluation_5seed_v2",
    "nearfar5_20deg_csit02": "thesis_nearfar5_longrun_v1",
}
RSMA = "objective_warm_joint_gate"
SDMA = "sdma_power_ppo"


def _metric(summary: pd.DataFrame, method: str, metric: str) -> pd.Series:
    rows = summary[(summary.algorithm == method) & (summary.metric == metric)]
    if len(rows) != 1:
        raise ValueError(f"expected one {method}/{metric} row, found {len(rows)}")
    return rows.iloc[0]


def collect(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons, actions = [], []
    for study, directory in STUDIES.items():
        root = results_root / directory
        summary = pd.read_csv(root / "summary.csv")
        units = pd.read_csv(root / "uncertainty_units.csv")
        paired = pd.read_csv(root / "paired_differences.csv")
        r, s = _metric(summary, RSMA, "mean_aoi"), _metric(summary, SDMA, "mean_aoi")
        seed_values = units[units.training_seed >= 0].pivot(
            index="training_seed", columns="algorithm", values="mean_aoi"
        )
        seed_delta = seed_values[RSMA] - seed_values[SDMA]
        trajectory = paired[(paired.algorithm == RSMA) & (paired.metric == "mean_aoi")].iloc[0]
        comparisons.append({
            "study": study,
            "rsma_mean_aoi": r["mean"], "rsma_std": r["std"],
            "rsma_ci_low": r["ci_low"], "rsma_ci_high": r["ci_high"],
            "sdma_mean_aoi": s["mean"], "sdma_std": s["std"],
            "sdma_ci_low": s["ci_low"], "sdma_ci_high": s["ci_high"],
            "rsma_minus_sdma": r["mean"] - s["mean"],
            "rsma_relative_gain_percent": 100.0 * (s["mean"] - r["mean"]) / s["mean"],
            "rsma_seedwise_wins": int((seed_delta < 0).sum()),
            "training_seed_pairs": int(seed_delta.notna().sum()),
            "rsma_trajectory_wins": int(trajectory["wins"]),
            "paired_trajectories": int(trajectory["paired_trajectories"]),
        })
        for metric in ("mean_common_power", "mean_rho", "common_on_rate"):
            row = _metric(summary, RSMA, metric)
            actions.append({"study": study, "metric": metric, "mean": row["mean"],
                            "ci_low": row["ci_low"], "ci_high": row["ci_high"]})
    return pd.DataFrame(comparisons), pd.DataFrame(actions)


def plot_spatial(comparisons: pd.DataFrame, actions: pd.DataFrame, out: Path) -> None:
    studies = ["spatial_5deg", "spatial_20deg", "spatial_60deg"]
    labels = ["5°", "20°", "60°"]
    rows = comparisons.set_index("study").loc[studies]
    x = np.arange(3)
    width = .36
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for offset, prefix, label, color in (
        (-width / 2, "rsma", "Objective-warm RSMA-PPO", "#2563eb"),
        (width / 2, "sdma", "SDMA-power PPO", "#f59e0b"),
    ):
        y = rows[f"{prefix}_mean_aoi"].to_numpy()
        err = np.vstack([y - rows[f"{prefix}_ci_low"], rows[f"{prefix}_ci_high"] - y])
        axes[0].bar(x + offset, y, width, label=label, color=color)
        axes[0].errorbar(x + offset, y, yerr=err, fmt="none", ecolor="#111827", capsize=4)
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("Angular separation")
    axes[0].set_ylabel("Mean AoI per user")
    axes[0].set_title("Matched learned policies")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=.25)

    action = actions[actions.study.isin(studies)].pivot(index="study", columns="metric", values="mean").loc[studies]
    axes[1].plot(x, action.mean_common_power, "o-", label="Common power fraction")
    axes[1].plot(x, action.mean_rho, "s-", label="Common payload scale $\\rho$")
    axes[1].plot(x, action.common_on_rate, "^-", label="Common-on fraction")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("Angular separation")
    axes[1].set_ylabel("Mean action value")
    axes[1].set_ylim(-.02, .58)
    axes[1].set_title("RSMA-PPO common-stream use")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.25)
    fig.suptitle("Directional Rician spatial regime, perfect CSIT, SNR 10 dB")
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"spatial_regime_comparison.{suffix}", dpi=220)
    plt.close(fig)


def plot_nearfar(comparisons: pd.DataFrame, out: Path) -> None:
    studies = ["headline_20deg_csit02", "nearfar5_20deg_csit02"]
    rows = comparisons.set_index("study").loc[studies]
    x = np.arange(2)
    width = .36
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for offset, prefix, label, color in (
        (-width / 2, "rsma", "Objective-warm RSMA-PPO", "#2563eb"),
        (width / 2, "sdma", "SDMA-power PPO", "#f59e0b"),
    ):
        y = rows[f"{prefix}_mean_aoi"].to_numpy()
        err = np.vstack([y - rows[f"{prefix}_ci_low"], rows[f"{prefix}_ci_high"] - y])
        ax.bar(x + offset, y, width, label=label, color=color)
        ax.errorbar(x + offset, y, yerr=err, fmt="none", ecolor="#111827", capsize=4)
    ax.set_xticks(x, ["Homogeneous", "5 dB near/far"])
    ax.set_ylabel("Mean AoI per user")
    ax.set_title("20° separation, relative CSIT error 0.2")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.25)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"nearfar_comparison.{suffix}", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out-dir", default="results/thesis_regime_summary_v1")
    args = parser.parse_args()
    root, out = Path(args.results_root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparisons, actions = collect(root)
    comparisons.to_csv(out / "comparisons.csv", index=False)
    actions.to_csv(out / "rsma_action_statistics.csv", index=False)
    plot_spatial(comparisons, actions, out)
    plot_nearfar(comparisons, out)
    print(comparisons.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
