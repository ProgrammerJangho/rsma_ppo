#!/usr/bin/env python3
"""Paired target evaluation of RSMA-PPO, q-only PPO-SDMA and heuristic SDMA."""
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

    angles = list(permutations((-10.0, 0.0, 10.0)))
    jobs = []
    for seed in args.seeds:
        checkpoints = {
            "rsma_ppo": f"runs/ppo_k3_20deg_sinr_from_stage1_seed{seed}_524k_v1/checkpoint.pt",
            "q_only_ppo_sdma": f"runs/q_only_ppo_k3_20deg_from_stage1_seed{seed}_524k_v1/checkpoint.pt",
        }
        for label, checkpoint in checkpoints.items():
            for perm, values in enumerate(angles):
                jobs.append({
                    "kind": "ppo", "label": label, "checkpoint": checkpoint,
                    "training_seed": seed, "config": args.config,
                    "geometry_deg": 20, "angles": values, "permutation": perm,
                    "snr_db": 15.0, "blocklength": 128,
                    "steps": args.steps, "burn_in": args.burn_in,
                    "eval_seed": 990278 + 100 * perm,
                })
    for perm, values in enumerate(angles):
        jobs.append({
            "kind": "sdma", "label": "max_aoi_top2_sdma", "checkpoint": "",
            "training_seed": -1, "config": args.config,
            "geometry_deg": 20, "angles": values, "permutation": perm,
            "snr_db": 15.0, "blocklength": 128,
            "steps": args.steps, "burn_in": args.burn_in,
            "eval_seed": 990278 + 100 * perm,
        })
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_job, jobs))
    for row, job in zip(rows, jobs, strict=True):
        row["label"] = job["label"]
        row["training_seed"] = job["training_seed"]
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "target_trajectories.csv", index=False)

    learned = frame[frame.training_seed >= 0]
    seeds = learned.groupby(["label", "training_seed"], as_index=False).agg(
        mean_aoi=("mean_aoi", "mean"), p95_aoi=("p95_aoi", "mean"),
        success_probability=("success_probability", "mean"),
        alpha_mean=("alpha_mean", "mean"), common_payload_mean=("common_payload_mean", "mean"),
    )
    seeds.to_csv(out / "target_seed_summary.csv", index=False)
    wide = seeds.pivot(index="training_seed", columns="label", values="mean_aoi").reset_index()
    wide["rsma_gain_vs_q_only_pct"] = 100.0 * (
        wide["q_only_ppo_sdma"] - wide["rsma_ppo"]
    ) / wide["q_only_ppo_sdma"]
    wide.to_csv(out / "paired_training_seed_comparison.csv", index=False)

    aggregate = seeds.groupby("label", as_index=False).agg(
        training_seeds=("training_seed", "count"), mean_aoi=("mean_aoi", "mean"),
        std_aoi=("mean_aoi", "std"), mean_p95_aoi=("p95_aoi", "mean"),
        mean_success_probability=("success_probability", "mean"),
        mean_alpha=("alpha_mean", "mean"), mean_common_payload=("common_payload_mean", "mean"),
    )
    critical = 2.7764451051977987 if len(args.seeds) == 5 else np.nan
    aggregate["aoi_ci95_halfwidth"] = critical * aggregate.std_aoi / np.sqrt(aggregate.training_seeds)
    aggregate.to_csv(out / "target_aggregate.csv", index=False)
    print(seeds.to_string(index=False))
    print("\nAggregate:\n", aggregate.to_string(index=False))
    print("\nRSMA gain vs q-only PPO-SDMA:\n", wide.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

