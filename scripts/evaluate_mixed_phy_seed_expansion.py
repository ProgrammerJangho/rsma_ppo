#!/usr/bin/env python3
"""Frozen target evaluation for independently trained mixed-PHY PPO seeds."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_mixed_phy_sinr_ppo import _job  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--config", default="configs/thesis_rician_k3_nt2_mixed_phy_sinr.yaml")
    parser.add_argument("--steps", type=int, default=768)
    parser.add_argument("--burn-in", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    angle_permutations = list(permutations((-10.0, 0.0, 10.0)))
    jobs: list[dict] = []
    for training_seed in args.seeds:
        checkpoints = {
            "warm": (
                f"runs/action_objective_actor_k3_mixed_phy_sinr_seed{training_seed}_v1/"
                "checkpoint.pt"
            ),
            "ppo": (
                f"runs/ppo_k3_20deg_sinr_from_stage1_seed{training_seed}_524k_v1/"
                "checkpoint.pt"
            ),
        }
        for label, checkpoint in checkpoints.items():
            for permutation, angles in enumerate(angle_permutations):
                jobs.append({
                    "kind": label,
                    "label": label,
                    "training_seed": training_seed,
                    "checkpoint": checkpoint,
                    "config": args.config,
                    "geometry_deg": 20,
                    "angles": angles,
                    "permutation": permutation,
                    "snr_db": 15.0,
                    "blocklength": 128,
                    "steps": args.steps,
                    "burn_in": args.burn_in,
                    "eval_seed": 990278 + 100 * permutation,
                })
    for permutation, angles in enumerate(angle_permutations):
        jobs.append({
            "kind": "sdma",
            "label": "sdma",
            "training_seed": -1,
            "checkpoint": "",
            "config": args.config,
            "geometry_deg": 20,
            "angles": angles,
            "permutation": permutation,
            "snr_db": 15.0,
            "blocklength": 128,
            "steps": args.steps,
            "burn_in": args.burn_in,
            "eval_seed": 990278 + 100 * permutation,
        })

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_job, jobs))
    for row, job in zip(rows, jobs, strict=True):
        row["label"] = job["label"]
        row["training_seed"] = job["training_seed"]
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "target_trajectories.csv", index=False)

    learned = frame[frame["training_seed"] >= 0]
    seed_summary = learned.groupby(["label", "training_seed"], as_index=False).agg(
        mean_aoi=("mean_aoi", "mean"),
        p95_aoi=("p95_aoi", "mean"),
        success_probability=("success_probability", "mean"),
        alpha_mean=("alpha_mean", "mean"),
        alpha_within_std=("alpha_std", "mean"),
        common_payload_mean=("common_payload_mean", "mean"),
    )
    sdma_mean = float(frame.loc[frame.label == "sdma", "mean_aoi"].mean())
    seed_summary["gain_vs_sdma_pct"] = 100.0 * (
        sdma_mean - seed_summary["mean_aoi"]
    ) / sdma_mean
    seed_summary.to_csv(out / "target_seed_summary.csv", index=False)

    paired = seed_summary.pivot(
        index="training_seed", columns="label", values="mean_aoi"
    ).reset_index()
    paired = paired.rename(columns={"warm": "warm_aoi", "ppo": "ppo_aoi"})
    paired["ppo_gain_vs_own_warm_pct"] = 100.0 * (
        paired["warm_aoi"] - paired["ppo_aoi"]
    ) / paired["warm_aoi"]
    paired.to_csv(out / "target_paired_training_seed_comparison.csv", index=False)

    aggregate = seed_summary.groupby("label", as_index=False).agg(
        training_seeds=("training_seed", "count"),
        mean_aoi=("mean_aoi", "mean"),
        std_aoi=("mean_aoi", "std"),
        mean_p95_aoi=("p95_aoi", "mean"),
        mean_alpha=("alpha_mean", "mean"),
        std_alpha=("alpha_mean", "std"),
        mean_gain_vs_sdma_pct=("gain_vs_sdma_pct", "mean"),
        std_gain_vs_sdma_pct=("gain_vs_sdma_pct", "std"),
    )
    # Student-t critical value for n=5 (df=4); keep the script dependency-light.
    critical = 2.7764451051977987 if len(args.seeds) == 5 else np.nan
    aggregate["aoi_ci95_halfwidth"] = critical * aggregate["std_aoi"] / np.sqrt(
        aggregate["training_seeds"]
    )
    aggregate["sdma_mean_aoi"] = sdma_mean
    aggregate.to_csv(out / "target_aggregate.csv", index=False)
    print(seed_summary.to_string(index=False))
    print("\nTraining-seed aggregate:")
    print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
