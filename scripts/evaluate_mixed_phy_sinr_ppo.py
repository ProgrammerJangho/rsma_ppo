#!/usr/bin/env python3
"""Paired regime-grid evaluation for the mixed-PHY direct-action PPO policy."""
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

from rsma_aoi.agents.ppo import PPOPolicy  # noqa: E402
from rsma_aoi.baselines import build_baseline  # noqa: E402
from rsma_aoi.envs.aoi_env import AoIRSMAEnv  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402


GEOMETRIES = {5: (-2.5, 0.0, 2.5), 20: (-10.0, 0.0, 10.0), 60: (-30.0, 0.0, 30.0)}


def _policy(kind: str, checkpoint: str, cfg: EnvConfig):
    if kind in ("warm", "ppo"):
        return PPOPolicy.from_checkpoint(checkpoint, deterministic=True, device="cpu")[0]
    policy, _ = build_baseline("sdma", cfg, seed=0)
    return policy


def _job(job: dict) -> dict:
    base = EnvConfig.from_yaml(job["config"])
    cfg = replace(
        base,
        user_angles_deg=list(job["angles"]), user_angle_sets_deg=None,
        snr_db=float(job["snr_db"]), snr_db_values=None,
        blocklength=int(job["blocklength"]), blocklength_values=None,
        multiple_access=("sdma" if job["kind"] == "sdma" else "rsma"),
        episode_length=int(job["steps"]),
    )
    policy = _policy(job["kind"], job["checkpoint"], cfg)
    if hasattr(policy, "reset"):
        policy.reset(seed=job["eval_seed"])
    env = AoIRSMAEnv(cfg)
    obs, _ = env.reset(seed=job["eval_seed"])
    ages, alphas, common_bits, success = [], [], [], []
    for slot in range(job["steps"]):
        state = env.get_state()
        if job.get("reset_memory_each_slot", False) and hasattr(policy, "reset"):
            policy.reset(seed=job["eval_seed"] + slot)
        action = policy.act(obs[None, :], [state])[0]
        obs, _, terminated, truncated, info = env.step(action)
        if slot >= job["burn_in"]:
            ages.extend(np.asarray(info["aoi"], dtype=float).tolist())
            success.extend(np.asarray(info["success_probabilities"], dtype=float).tolist())
            common_bits.append(float(info["common_payload"]))
            if "common_power_fraction" in action:
                alphas.append(float(np.asarray(action["common_power_fraction"]).reshape(-1)[0]))
        if terminated or truncated:
            break
    env.close()
    age = np.asarray(ages)
    return {
        "kind": job["kind"], "geometry_deg": job["geometry_deg"],
        "snr_db": job["snr_db"], "blocklength": job["blocklength"],
        "permutation": job["permutation"], "eval_seed": job["eval_seed"],
        "mean_aoi": float(age.mean()), "p95_aoi": float(np.quantile(age, .95)),
        "success_probability": float(np.mean(success)),
        "alpha_mean": float(np.mean(alphas)) if alphas else 0.0,
        "alpha_std": float(np.std(alphas)) if alphas else 0.0,
        "common_payload_mean": float(np.mean(common_bits)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/thesis_rician_k3_nt2_mixed_phy_sinr.yaml")
    ap.add_argument("--warm", required=True)
    ap.add_argument("--ppo", required=True)
    ap.add_argument("--steps", type=int, default=768)
    ap.add_argument("--burn-in", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    checkpoints = {"warm": args.warm, "ppo": args.ppo, "sdma": ""}
    jobs = []
    for kind, checkpoint in checkpoints.items():
        for geometry_deg, values in GEOMETRIES.items():
            for permutation, angles in enumerate(permutations(values)):
                for snr_db in (5.0, 10.0, 15.0, 20.0):
                    for blocklength in (64, 128, 256):
                        jobs.append({
                            "kind": kind, "checkpoint": checkpoint, "config": args.config,
                            "geometry_deg": geometry_deg, "angles": angles,
                            "permutation": permutation, "snr_db": snr_db,
                            "blocklength": blocklength, "steps": args.steps,
                            "burn_in": args.burn_in,
                            "eval_seed": 970000 + 1000 * geometry_deg + 100 * permutation
                                         + 10 * int(snr_db) + blocklength,
                        })
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_job, jobs))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "trajectories.csv", index=False)
    summary = frame.groupby(
        ["kind", "geometry_deg", "snr_db", "blocklength"], as_index=False
    ).agg(
        mean_aoi=("mean_aoi", "mean"), p95_aoi=("p95_aoi", "mean"),
        success_probability=("success_probability", "mean"),
        alpha_mean=("alpha_mean", "mean"), alpha_between_trajectory_std=("alpha_mean", "std"),
        alpha_within_trajectory_std=("alpha_std", "mean"),
        common_payload_mean=("common_payload_mean", "mean"),
    )
    summary.to_csv(out / "regime_summary.csv", index=False)
    pivot = frame.pivot_table(
        index=["geometry_deg", "snr_db", "blocklength", "permutation", "eval_seed"],
        columns="kind", values="mean_aoi",
    ).reset_index()
    pivot["ppo_vs_warm_pct"] = 100.0 * (pivot["warm"] - pivot["ppo"]) / pivot["warm"]
    pivot["ppo_vs_sdma_pct"] = 100.0 * (pivot["sdma"] - pivot["ppo"]) / pivot["sdma"]
    pivot.to_csv(out / "paired_comparisons.csv", index=False)
    print(summary.to_string(index=False))
    print("\nOverall paired gains (%):")
    print(pivot[["ppo_vs_warm_pct", "ppo_vs_sdma_pct"]].mean().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
