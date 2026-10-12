#!/usr/bin/env python3
"""Calibrate and evaluate the max-AoI single-user and fixed-rule RSMA baselines."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsma_aoi.baselines.thesis_rules import (  # noqa: E402
    MaxAoISingleUserPolicy, OldestCommonFixedAlphaPolicy,
)
from rsma_aoi.envs.aoi_env import AoIRSMAEnv  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402


def evaluate(job: dict) -> dict:
    base = EnvConfig.from_yaml(job["config"])
    single = job["kind"] == "max_aoi_single_user"
    cfg = replace(
        base, user_angles_deg=list(job["angles"]), user_angle_sets_deg=None,
        snr_db=float(job["snr_db"]), snr_db_values=None,
        blocklength=int(job["blocklength"]), blocklength_values=None,
        episode_length=int(job["steps"]),
        scheduling_enabled=single,
        multiple_access=("sdma" if single else "rsma"),
    )
    policy = (
        MaxAoISingleUserPolicy(cfg.num_users) if single
        else OldestCommonFixedAlphaPolicy(cfg.num_users, float(job["alpha"]))
    )
    env = AoIRSMAEnv(cfg)
    obs, _ = env.reset(seed=int(job["eval_seed"]))
    ages, successes, common_bits = [], [], []
    for slot in range(job["steps"]):
        action = policy.act(obs[None], [env.get_state()])[0]
        obs, _, terminated, truncated, info = env.step(action)
        if slot >= job["burn_in"]:
            ages.extend(np.asarray(info["aoi"], float).tolist())
            successes.extend(np.asarray(info["success_probabilities"], float).tolist())
            common_bits.append(float(info["common_payload"]))
        if terminated or truncated:
            break
    env.close()
    age = np.asarray(ages)
    return {
        "kind": job["kind"], "phase": job["phase"], "alpha": float(job["alpha"]),
        "permutation": int(job["permutation"]), "eval_seed": int(job["eval_seed"]),
        "geometry_deg": int(job["geometry_deg"]), "snr_db": float(job["snr_db"]),
        "blocklength": int(job["blocklength"]), "mean_aoi": float(age.mean()),
        "p95_aoi": float(np.quantile(age, 0.95)),
        "success_probability": float(np.mean(successes)),
        "common_payload_mean": float(np.mean(common_bits)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/thesis_rician_k3_nt2_mixed_phy_sinr.yaml")
    ap.add_argument("--steps", type=int, default=768)
    ap.add_argument("--burn-in", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    angle_sets = list(permutations((-10.0, 0.0, 10.0)))
    alpha_grid = np.linspace(0.0, 1.0, 21)

    calibration_jobs = []
    for alpha in alpha_grid:
        for repeat in range(3):
            for perm, angles in enumerate(angle_sets):
                calibration_jobs.append({
                    "kind": "oldest_common_fixed_alpha", "phase": "validation",
                    "alpha": float(alpha), "permutation": perm, "angles": angles,
                    "geometry_deg": 20, "snr_db": 15.0, "blocklength": 128,
                    "steps": args.steps, "burn_in": args.burn_in, "config": args.config,
                    "eval_seed": 940000 + 10000 * repeat + 100 * perm,
                })
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        calibration = list(pool.map(evaluate, calibration_jobs))
    calibration = pd.DataFrame(calibration)
    calibration.to_csv(out / "fixed_rsma_validation_grid.csv", index=False)
    grid = calibration.groupby("alpha", as_index=False).agg(
        mean_aoi=("mean_aoi", "mean"), std_aoi=("mean_aoi", "std"),
        mean_p95_aoi=("p95_aoi", "mean"),
    )
    grid.to_csv(out / "fixed_rsma_validation_summary.csv", index=False)
    chosen_alpha = float(grid.loc[grid.mean_aoi.idxmin(), "alpha"])

    test_jobs = []
    for perm, angles in enumerate(angle_sets):
        common = {
            "phase": "test", "permutation": perm, "angles": angles,
            "geometry_deg": 20, "snr_db": 15.0, "blocklength": 128,
            "steps": args.steps, "burn_in": args.burn_in, "config": args.config,
            "eval_seed": 990278 + 100 * perm,
        }
        test_jobs.append({**common, "kind": "max_aoi_single_user", "alpha": 0.0})
        test_jobs.append({
            **common, "kind": "oldest_common_fixed_alpha", "alpha": chosen_alpha
        })
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        test = list(pool.map(evaluate, test_jobs))
    test = pd.DataFrame(test)
    test.to_csv(out / "target_trajectories.csv", index=False)
    summary = test.groupby("kind", as_index=False).agg(
        trajectories=("mean_aoi", "count"), mean_aoi=("mean_aoi", "mean"),
        trajectory_std=("mean_aoi", "std"), mean_p95_aoi=("p95_aoi", "mean"),
        success_probability=("success_probability", "mean"),
        common_payload_mean=("common_payload_mean", "mean"), alpha=("alpha", "first"),
    )
    summary.to_csv(out / "target_summary.csv", index=False)
    print("chosen alpha", chosen_alpha)
    print(grid.sort_values("mean_aoi").head(8).to_string(index=False))
    print("\nTest:\n", summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

