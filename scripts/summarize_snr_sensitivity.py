#!/usr/bin/env python3
"""Consolidate the matched 5/10/15 dB long-run sensitivity experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STUDIES = {
    5: "thesis_snr05_longrun_v1",
    10: "thesis_snr10_longrun_v1",
    15: "thesis_snr15_longrun_v1",
}
RSMA = "objective_warm_joint_gate"
SDMA = "sdma_power_ppo"


def _metric(summary: pd.DataFrame, algorithm: str, metric: str) -> pd.Series:
    row = summary[(summary.algorithm == algorithm) & (summary.metric == metric)]
    if len(row) != 1:
        raise ValueError(f"expected one {algorithm}/{metric} row, found {len(row)}")
    return row.iloc[0]


def collect(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons: list[dict] = []
    actions: list[dict] = []
    for snr, directory in STUDIES.items():
        root = results_root / directory
        summary = pd.read_csv(root / "summary.csv")
        units = pd.read_csv(root / "uncertainty_units.csv")
        paired = pd.read_csv(root / "paired_differences.csv")
        rsma = _metric(summary, RSMA, "mean_aoi")
        sdma = _metric(summary, SDMA, "mean_aoi")
        seed_values = units[units.training_seed >= 0].pivot(
            index="training_seed", columns="algorithm", values="mean_aoi"
        )
        seed_delta = seed_values[RSMA] - seed_values[SDMA]
        trajectory = paired[
            (paired.algorithm == RSMA) & (paired.metric == "mean_aoi")
        ].iloc[0]
        comparisons.append({
            "snr_db": snr,
            "rsma_mean_aoi": rsma["mean"],
            "rsma_std": rsma["std"],
            "rsma_ci_low": rsma["ci_low"],
            "rsma_ci_high": rsma["ci_high"],
            "sdma_mean_aoi": sdma["mean"],
            "sdma_std": sdma["std"],
            "sdma_ci_low": sdma["ci_low"],
            "sdma_ci_high": sdma["ci_high"],
            "rsma_minus_sdma": rsma["mean"] - sdma["mean"],
            "rsma_relative_gain_percent": 100 * (sdma["mean"] - rsma["mean"]) / sdma["mean"],
            "rsma_seedwise_wins": int((seed_delta < 0).sum()),
            "training_seed_pairs": int(seed_delta.notna().sum()),
            "rsma_trajectory_wins": int(trajectory["wins"]),
            "paired_trajectories": int(trajectory["paired_trajectories"]),
        })
        for metric in (
            "mean_common_power", "mean_rho", "common_on_rate",
            "mean_common_bler", "mean_private_bler",
        ):
            row = _metric(summary, RSMA, metric)
            actions.append({
                "snr_db": snr, "metric": metric, "mean": row["mean"],
                "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            })
    return pd.DataFrame(comparisons), pd.DataFrame(actions)


def plot(comparisons: pd.DataFrame, actions: pd.DataFrame, out: Path) -> None:
    rows = comparisons.set_index("snr_db").loc[list(STUDIES)]
    x = np.arange(len(rows))
    width = .36
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    for offset, prefix, label, color in (
        (-width / 2, "rsma", "Objective-warm RSMA-PPO", "#2563eb"),
        (width / 2, "sdma", "SDMA-power PPO", "#f59e0b"),
    ):
        values = rows[f"{prefix}_mean_aoi"].to_numpy()
        error = np.vstack([
            values - rows[f"{prefix}_ci_low"].to_numpy(),
            rows[f"{prefix}_ci_high"].to_numpy() - values,
        ])
        axes[0].bar(x + offset, values, width, color=color, label=label)
        axes[0].errorbar(x + offset, values, yerr=error, fmt="none", ecolor="#111827", capsize=4)
    axes[0].set_xticks(x, [f"{snr} dB" for snr in STUDIES])
    axes[0].set_ylabel("Mean AoI per user")
    axes[0].set_title("Matched learned policies")
    axes[0].grid(axis="y", alpha=.25)
    axes[0].legend(fontsize=8)

    action = actions.pivot(index="snr_db", columns="metric", values="mean").loc[list(STUDIES)]
    axes[1].plot(x, action.mean_common_power, "o-", label="Common power fraction")
    axes[1].plot(x, action.mean_rho, "s-", label="Common payload scale $\\rho$")
    axes[1].plot(x, action.common_on_rate, "^-", label="Common-on fraction")
    axes[1].set_xticks(x, [f"{snr} dB" for snr in STUDIES])
    axes[1].set_ylim(-.02, .72)
    axes[1].set_ylabel("Mean action value")
    axes[1].set_title("RSMA-PPO common-stream use")
    axes[1].grid(alpha=.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Directional Rician, 20° separation, relative CSIT error 0.2")
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"snr_sensitivity.{suffix}", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out-dir", default="results/thesis_snr_sensitivity_v1")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparisons, actions = collect(Path(args.results_root))
    comparisons.to_csv(out / "comparisons.csv", index=False)
    actions.to_csv(out / "rsma_action_statistics.csv", index=False)
    plot(comparisons, actions, out)
    print(comparisons.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
