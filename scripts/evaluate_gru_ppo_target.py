#!/usr/bin/env python3
"""Paired frozen-trajectory comparison of feed-forward PPO and GRU-PPO."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_mixed_phy_sinr_ppo import _job  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedforward", required=True)
    parser.add_argument("--gru", required=True)
    parser.add_argument(
        "--config", default="configs/thesis_rician_k3_nt2_mixed_phy_sinr.yaml"
    )
    parser.add_argument("--steps", type=int, default=768)
    parser.add_argument("--burn-in", type=int, default=256)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--memory-ablation", action="store_true")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)

    jobs = []
    methods = [("feedforward_ppo", args.feedforward, False), ("gru_ppo", args.gru, False)]
    if args.memory_ablation:
        methods.append(("gru_reset_each_slot", args.gru, True))
    for label, checkpoint, reset_memory in methods:
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        for permutation, angles in enumerate(permutations((-10.0, 0.0, 10.0))):
            jobs.append({
                "kind": "ppo", "label": label, "checkpoint": checkpoint,
                "config": args.config, "geometry_deg": 20, "angles": angles,
                "permutation": permutation, "snr_db": 15.0, "blocklength": 128,
                "steps": args.steps, "burn_in": args.burn_in,
                "eval_seed": 990278 + 100 * permutation,
                "reset_memory_each_slot": reset_memory,
            })
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_job, jobs))
    for row, job in zip(rows, jobs, strict=True):
        row["label"] = job["label"]
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "target_trajectories.csv", index=False)
    summary = frame.groupby("label", as_index=False).agg(
        trajectories=("mean_aoi", "count"), mean_aoi=("mean_aoi", "mean"),
        std_aoi=("mean_aoi", "std"), mean_p95_aoi=("p95_aoi", "mean"),
        success_probability=("success_probability", "mean"),
        alpha_mean=("alpha_mean", "mean"), alpha_within_std=("alpha_std", "mean"),
        common_payload_mean=("common_payload_mean", "mean"),
    )
    summary.to_csv(out / "target_summary.csv", index=False)
    wide = frame.pivot(
        index=["permutation", "eval_seed"], columns="label", values="mean_aoi"
    ).reset_index()
    wide["gru_gain_vs_feedforward_pct"] = 100.0 * (
        wide.feedforward_ppo - wide.gru_ppo
    ) / wide.feedforward_ppo
    if "gru_reset_each_slot" in wide:
        wide["memory_gain_pct"] = 100.0 * (
            wide.gru_reset_each_slot - wide.gru_ppo
        ) / wide.gru_reset_each_slot
    wide.to_csv(out / "paired_comparison.csv", index=False)
    print(summary.to_string(index=False))
    print("\nPaired GRU gain (%):", float(wide.gru_gain_vs_feedforward_pct.mean()))
    print("GRU wins:", int((wide.gru_ppo < wide.feedforward_ppo).sum()), "/", len(wide))
    if "memory_gain_pct" in wide:
        print("Memory gain vs per-slot reset (%):", float(wide.memory_gain_pct.mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
