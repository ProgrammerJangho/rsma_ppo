#!/usr/bin/env python3
"""Audit K=3 allocation-only PPO over every ordering of one geometry."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import permutations
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsma_aoi.agents.ppo import PPOPolicy  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402
from run_final_evaluation import evaluate_trajectory  # noqa: E402


def _job(job: dict) -> dict:
    base = EnvConfig.from_yaml(ROOT / job["config"])
    angles = list(job["angles"])
    cfg = replace(
        base, user_angles_deg=angles, user_angle_sets_deg=None,
        snr_db=15.0, snr_db_values=None, scheduling_enabled=False,
        episode_length=job["steps"],
    )
    checkpoint = ROOT / (
        "results/rician_ppo_k3_nt2_all3_262k_v1/runs/"
        f"rsma_joint_gate_seed{job['training_seed']}/checkpoint.pt"
    )
    actor, _, _ = PPOPolicy.from_checkpoint(checkpoint, deterministic=True, device="cpu")
    row, _ = evaluate_trajectory(
        cfg, {"burn_in": job["burn_in"]}, "k3_all3_262k",
        job["training_seed"], job["eval_seed"], actor,
    )
    row["angle_order"] = ",".join(str(int(x)) for x in angles)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/thesis_rician_k3_nt2_all3_ppo_csit02.yaml")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[981000, 981001, 981002])
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--burn-in", type=int, default=512)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--out-dir", default="results/k3_nt2_all3_ppo_permutation_audit_v1")
    args = parser.parse_args()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    orders = list(permutations((-10.0, 0.0, 10.0)))
    jobs = [dict(config=args.config, angles=a, training_seed=t, eval_seed=e,
                 steps=args.steps, burn_in=args.burn_in)
            for a in orders for t in args.training_seeds for e in args.eval_seeds]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        episodes = pd.DataFrame(pool.map(_job, jobs))
    episodes.to_csv(out / "episodes.csv", index=False)
    summary = episodes.groupby("angle_order", as_index=False).agg(
        mean_aoi=("mean_aoi", "mean"), std_aoi=("mean_aoi", "std"),
        success_rate=("success_rate", "mean"), common_on_rate=("common_on_rate", "mean"),
        mean_common_power=("mean_common_power", "mean"), mean_rho=("mean_rho", "mean"),
        trajectories=("mean_aoi", "size"),
    )
    summary.to_csv(out / "summary.csv", index=False)
    spread = float(summary.mean_aoi.max() - summary.mean_aoi.min())
    (out / "README.md").write_text(
        "# K=3 user-order permutation audit\n\n"
        f"All six permutations, {len(args.training_seeds)} training seeds, "
        f"{len(args.eval_seeds)} evaluation seeds, {args.steps} slots, "
        f"burn-in {args.burn_in}.\n\nMean-AoI max-minus-min spread: {spread:.6f}.\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\nAoI spread={spread:.6f}\nCOMPLETE: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
