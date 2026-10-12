#!/usr/bin/env python3
"""Evaluate RSMA and SDMA PPO under matched objective-only pretraining."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_mixed_phy_sinr_ppo import _job  # noqa: E402


LABELS = {
    "objective_only_rsma": (
        "runs/matched_objective_rsma_ppo_k3_20deg_from_stage1_seed{seed}_524k_v1/checkpoint.pt"
    ),
    "objective_only_sdma": (
        "runs/q_only_ppo_k3_20deg_from_stage1_seed{seed}_524k_v1/checkpoint.pt"
    ),
    "action_objective_rsma": (
        "runs/ppo_k3_20deg_sinr_from_stage1_seed{seed}_524k_v1/checkpoint.pt"
    ),
}


def _ci(values: pd.Series) -> float:
    if len(values) < 2:
        return float("nan")
    return float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--config", default="configs/thesis_rician_k3_nt2_mixed_phy_sinr.yaml")
    ap.add_argument("--steps", type=int, default=768)
    ap.add_argument("--burn-in", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)

    angle_sets = list(permutations((-10.0, 0.0, 10.0)))
    jobs: list[dict] = []
    for seed in args.seeds:
        for label, template in LABELS.items():
            checkpoint = template.format(seed=seed)
            if not Path(checkpoint).is_file():
                raise FileNotFoundError(checkpoint)
            for permutation, angles in enumerate(angle_sets):
                jobs.append({
                    "kind": "ppo", "label": label, "checkpoint": checkpoint,
                    "training_seed": seed, "config": args.config,
                    "geometry_deg": 20, "angles": angles, "permutation": permutation,
                    "snr_db": 15.0, "blocklength": 128,
                    "steps": args.steps, "burn_in": args.burn_in,
                    "eval_seed": 990278 + 100 * permutation,
                })
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_job, jobs))
    for row, job in zip(rows, jobs, strict=True):
        row["label"] = job["label"]
        row["training_seed"] = job["training_seed"]
    trajectories = pd.DataFrame(rows)
    trajectories.to_csv(out / "target_trajectories.csv", index=False)

    seed_summary = trajectories.groupby(
        ["label", "training_seed"], as_index=False
    ).agg(
        mean_aoi=("mean_aoi", "mean"), p95_aoi=("p95_aoi", "mean"),
        success_probability=("success_probability", "mean"),
        alpha_mean=("alpha_mean", "mean"),
        common_payload_mean=("common_payload_mean", "mean"),
    )
    seed_summary.to_csv(out / "target_seed_summary.csv", index=False)

    wide = seed_summary.pivot(index="training_seed", columns="label", values="mean_aoi")
    wide["matched_rsma_gain_vs_sdma_pct"] = 100.0 * (
        wide.objective_only_sdma - wide.objective_only_rsma
    ) / wide.objective_only_sdma
    wide["objective_only_gain_vs_action_objective_rsma_pct"] = 100.0 * (
        wide.action_objective_rsma - wide.objective_only_rsma
    ) / wide.action_objective_rsma
    wide.reset_index().to_csv(out / "paired_training_seed_comparison.csv", index=False)

    trajectory_wide = trajectories.pivot(
        index=["training_seed", "permutation", "eval_seed"],
        columns="label", values="mean_aoi",
    ).reset_index()
    trajectory_wide["matched_rsma_gain_vs_sdma_pct"] = 100.0 * (
        trajectory_wide.objective_only_sdma - trajectory_wide.objective_only_rsma
    ) / trajectory_wide.objective_only_sdma
    trajectory_wide.to_csv(out / "paired_trajectory_comparison.csv", index=False)

    aggregate = seed_summary.groupby("label", as_index=False).agg(
        training_seeds=("training_seed", "count"), mean_aoi=("mean_aoi", "mean"),
        std_aoi=("mean_aoi", "std"), mean_p95_aoi=("p95_aoi", "mean"),
        mean_success_probability=("success_probability", "mean"),
        mean_alpha=("alpha_mean", "mean"),
        mean_common_payload=("common_payload_mean", "mean"),
    )
    aggregate["aoi_ci95_halfwidth"] = aggregate.apply(
        lambda row: _ci(seed_summary[
            seed_summary.label == row.label
        ].mean_aoi), axis=1,
    )
    aggregate.to_csv(out / "target_aggregate.csv", index=False)

    matched_gain = wide.matched_rsma_gain_vs_sdma_pct
    summary = pd.DataFrame([{
        "training_seeds": len(matched_gain),
        "matched_rsma_gain_vs_sdma_mean_pct": matched_gain.mean(),
        "matched_rsma_gain_vs_sdma_ci95_halfwidth_pct": _ci(matched_gain),
        "matched_rsma_trajectory_wins": int(
            (trajectory_wide.objective_only_rsma < trajectory_wide.objective_only_sdma).sum()
        ),
        "trajectory_count": len(trajectory_wide),
    }])
    summary.to_csv(out / "matched_summary.csv", index=False)
    print("Aggregate:\n", aggregate.to_string(index=False))
    print("\nPaired seeds:\n", wide.reset_index().to_string(index=False))
    print("\nMatched summary:\n", summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
