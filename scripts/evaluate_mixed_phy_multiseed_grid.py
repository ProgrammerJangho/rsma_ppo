#!/usr/bin/env python3
"""Evaluate independently trained PPO seeds on the frozen 36-regime grid."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_mixed_phy_sinr_ppo import GEOMETRIES, _job  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--config", default="configs/thesis_rician_k3_nt2_mixed_phy_sinr.yaml")
    parser.add_argument("--steps", type=int, default=768)
    parser.add_argument("--burn-in", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--checkpoint-template",
        default=(
            "runs/ppo_k3_20deg_sinr_from_stage1_seed{seed}_524k_v1/"
            "checkpoint.pt"
        ),
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    jobs: list[dict] = []
    for training_seed in args.seeds:
        checkpoint = args.checkpoint_template.format(seed=training_seed)
        for geometry_deg, values in GEOMETRIES.items():
            for permutation, angles in enumerate(permutations(values)):
                for snr_db in (5.0, 10.0, 15.0, 20.0):
                    for blocklength in (64, 128, 256):
                        jobs.append({
                            "kind": "ppo",
                            "checkpoint": checkpoint,
                            "training_seed": training_seed,
                            "config": args.config,
                            "geometry_deg": geometry_deg,
                            "angles": angles,
                            "permutation": permutation,
                            "snr_db": snr_db,
                            "blocklength": blocklength,
                            "steps": args.steps,
                            "burn_in": args.burn_in,
                            "eval_seed": 970000 + 1000 * geometry_deg
                            + 100 * permutation + 10 * int(snr_db) + blocklength,
                        })
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_job, jobs))
    for row, job in zip(rows, jobs, strict=True):
        row["training_seed"] = job["training_seed"]
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "trajectories.csv", index=False)

    regime = frame.groupby(
        ["training_seed", "geometry_deg", "snr_db", "blocklength"], as_index=False
    ).agg(
        mean_aoi=("mean_aoi", "mean"),
        p95_aoi=("p95_aoi", "mean"),
        success_probability=("success_probability", "mean"),
        alpha_mean=("alpha_mean", "mean"),
        alpha_within_std=("alpha_std", "mean"),
        common_payload_mean=("common_payload_mean", "mean"),
    )
    regime.to_csv(out / "regime_by_training_seed.csv", index=False)

    marginal_rows = []
    for axis in ("geometry_deg", "snr_db", "blocklength"):
        grouped = regime.groupby(["training_seed", axis], as_index=False).agg(
            alpha_mean=("alpha_mean", "mean"), mean_aoi=("mean_aoi", "mean")
        )
        grouped["axis"] = axis
        grouped["level"] = grouped[axis]
        marginal_rows.append(grouped[["training_seed", "axis", "level", "alpha_mean", "mean_aoi"]])
    marginal = pd.concat(marginal_rows, ignore_index=True)
    marginal.to_csv(out / "alpha_marginals_by_training_seed.csv", index=False)

    aggregate = marginal.groupby(["axis", "level"], as_index=False).agg(
        training_seeds=("training_seed", "count"),
        alpha_mean=("alpha_mean", "mean"),
        alpha_std=("alpha_mean", "std"),
        mean_aoi=("mean_aoi", "mean"),
        aoi_std=("mean_aoi", "std"),
    )
    aggregate.to_csv(out / "alpha_marginal_aggregate.csv", index=False)
    print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
