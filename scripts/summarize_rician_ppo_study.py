#!/usr/bin/env python3
"""Consolidate the controlled-Rician PPO, SDMA, and myopic-CEM study.

PPO uncertainty is computed across independent training seeds.  CEM uncertainty
is computed across evaluation trajectories and is deliberately labelled as
such; the two standard deviations must not be interpreted as the same source
of uncertainty.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANGLES = (5, 20, 60)
PPO_VARIANTS = {
    "low_rho": "rician_ppo_{angle}deg_3seed_v1",
    "broad_rho": "rician_ppo_{angle}deg_rho05broad_3seed_v1",
    "joint_init": "rician_ppo_{angle}deg_jointinit_3seed_v1",
    "objective_warmstart": "rician_ppo_{angle}deg_objective_warmstart_3seed_v1",
}


def _ppo_rows(results: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for angle in ANGLES:
        for variant, pattern in PPO_VARIANTS.items():
            directory = pattern.format(angle=angle)
            # The first 5° warm-start pilot used a geometry-specific mixed
            # loss.  The controlled study uses objective_only at every angle.
            if variant == "objective_warmstart" and angle == 5:
                directory = "rician_ppo_5deg_objective_warmstart_uniform_3seed_v1"
            path = results / directory / "training_seed_summary.csv"
            data = pd.read_csv(path)
            rows = data[(data.algorithm == "joint_gate") & (data.execution == "deterministic")].copy()
            rows["angle_deg"] = angle
            rows["method"] = f"ppo_{variant}"
            rows["uncertainty_source"] = "training_seed"
            frames.append(rows)
            if variant == "low_rho":
                sdma = data[(data.algorithm == "sdma_power") & (data.execution == "deterministic")].copy()
                sdma["angle_deg"] = angle
                sdma["method"] = "ppo_sdma_power"
                sdma["uncertainty_source"] = "training_seed"
                frames.append(sdma)
    return pd.concat(frames, ignore_index=True)


def _cem_rows(results: Path) -> pd.DataFrame:
    data = pd.read_csv(results / "rician_allocator_longrun_cem_full_v1" / "episodes.csv")
    data["angle_deg"] = data["config"].str.extract(r"_(\d+)deg$").astype(int)
    data["method"] = data.algorithm.map({
        "rsma_cem_full": "cem_rsma",
        "sdma_cem_full": "cem_sdma",
    })
    data["training_seed"] = data["seed"]
    data["uncertainty_source"] = "evaluation_trajectory"
    data["mean_common_power"] = data["avg_common_power_ratio"]
    data["mean_rho"] = data["avg_common_payload_ratio"]
    data["common_on_rate"] = np.nan
    data["common_bler_when_on"] = np.nan
    return data


def _summarize(seed_level: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_aoi", "success_rate", "mean_common_power", "mean_rho",
        "common_on_rate", "common_bler_when_on",
    ]
    rows = []
    for (angle, method, uncertainty), group in seed_level.groupby(
        ["angle_deg", "method", "uncertainty_source"], sort=True
    ):
        row = {
            "angle_deg": angle,
            "method": method,
            "uncertainty_source": uncertainty,
            "n": len(group),
        }
        for metric in metrics:
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _paired(ppo: pd.DataFrame, cem: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for angle in ANGLES:
        base = ppo[(ppo.angle_deg == angle) & (ppo.method == "ppo_sdma_power")].set_index("training_seed")
        for method in (
            "ppo_low_rho", "ppo_broad_rho", "ppo_joint_init",
            "ppo_objective_warmstart",
        ):
            candidate = ppo[(ppo.angle_deg == angle) & (ppo.method == method)].set_index("training_seed")
            joined = candidate[["mean_aoi"]].join(
                base[["mean_aoi"]], lsuffix="_candidate", rsuffix="_sdma", how="inner"
            )
            delta = joined.mean_aoi_candidate - joined.mean_aoi_sdma
            rows.append({
                "angle_deg": angle,
                "comparison": f"{method}_minus_ppo_sdma_power",
                "n_pairs": len(delta),
                "mean_delta_aoi": delta.mean(),
                "std_delta_aoi": delta.std(ddof=1),
                "candidate_wins": int((delta < 0).sum()),
            })
        low = ppo[(ppo.angle_deg == angle) & (ppo.method == "ppo_low_rho")].set_index(
            "training_seed"
        )
        warm = ppo[
            (ppo.angle_deg == angle) & (ppo.method == "ppo_objective_warmstart")
        ].set_index("training_seed")
        delta = warm.mean_aoi - low.mean_aoi
        rows.append({
            "angle_deg": angle,
            "comparison": "ppo_objective_warmstart_minus_ppo_low_rho",
            "n_pairs": len(delta),
            "mean_delta_aoi": delta.mean(),
            "std_delta_aoi": delta.std(ddof=1),
            "candidate_wins": int((delta < 0).sum()),
        })
        c = cem[cem.angle_deg == angle]
        r = c[c.method == "cem_rsma"].set_index("training_seed")
        s = c[c.method == "cem_sdma"].set_index("training_seed")
        delta = r.mean_aoi - s.mean_aoi
        rows.append({
            "angle_deg": angle,
            "comparison": "cem_rsma_minus_cem_sdma",
            "n_pairs": len(delta),
            "mean_delta_aoi": delta.mean(),
            "std_delta_aoi": delta.std(ddof=1),
            "candidate_wins": int((delta < 0).sum()),
        })
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, output: Path) -> None:
    labels = {
        "ppo_low_rho": "PPO: low $\\rho$ init",
        "ppo_broad_rho": "PPO: broad $\\rho$ init",
        "ppo_joint_init": "PPO: joint init",
        "ppo_objective_warmstart": "PPO: objective warm-start",
        "ppo_sdma_power": "PPO: SDMA power",
        "cem_rsma": "CEM: RSMA",
        "cem_sdma": "CEM: SDMA",
    }
    methods = list(labels)
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(methods)))
    color_by_method = dict(zip(methods, colors))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.1), constrained_layout=True)
    x = np.arange(len(ANGLES), dtype=float)
    for method, color in zip(methods, colors):
        q = summary[summary.method == method].set_index("angle_deg").reindex(ANGLES)
        axes[0].errorbar(
            x, q.mean_aoi_mean, yerr=q.mean_aoi_std, marker="o", capsize=3,
            linewidth=1.5, label=labels[method], color=color,
        )
    axes[0].set(
        title="(a) Long-run AoI by channel geometry", xlabel="LoS angle separation",
        ylabel="Mean AoI (lower is better)", xticks=x,
        xticklabels=[f"{a}°" for a in ANGLES],
    )
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    alloc_methods = [
        "ppo_low_rho", "ppo_broad_rho", "ppo_joint_init",
        "ppo_objective_warmstart", "cem_rsma",
    ]
    for method in alloc_methods:
        color = color_by_method[method]
        q = summary[summary.method == method].set_index("angle_deg").reindex(ANGLES)
        axes[1].plot(x, q.mean_common_power_mean, marker="o", color=color,
                     label=f"power: {labels[method]}")
        axes[1].plot(x, q.mean_rho_mean, marker="s", linestyle="--", color=color,
                     label=f"payload: {labels[method]}")
    axes[1].set(
        title="(b) Learned/selected common allocation", xlabel="LoS angle separation",
        ylabel="Mean fraction", xticks=x, xticklabels=[f"{a}°" for a in ANGLES],
        ylim=(-0.03, 0.72),
    )
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(output / "rician_ppo_study.png", dpi=220)
    fig.savefig(output / "rician_ppo_study.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out-dir", default="results/rician_ppo_study_v1")
    args = parser.parse_args()
    results = Path(args.results_root)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    ppo = _ppo_rows(results)
    cem = _cem_rows(results)
    seed_level = pd.concat([ppo, cem], ignore_index=True, sort=False)
    summary = _summarize(seed_level)
    paired = _paired(ppo, cem)
    seed_level.to_csv(output / "seed_level.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    paired.to_csv(output / "paired_comparisons.csv", index=False)
    _plot(summary, output)
    print(summary[["angle_deg", "method", "mean_aoi_mean", "mean_aoi_std"]].to_string(index=False))
    print("\nPaired comparisons (negative favours RSMA/PPO candidate):")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
