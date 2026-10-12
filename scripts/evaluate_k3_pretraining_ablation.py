#!/usr/bin/env python3
"""Long-run comparison of K=3 zero-step actor pretraining variants."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsma_aoi.agents.ppo import PPOPolicy  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402
from run_final_evaluation import evaluate_trajectory  # noqa: E402


def _checkpoint(variant: str, seed: int) -> Path:
    if variant == "action_objective_128":
        return ROOT / f"runs/action_objective_actor_k3_nt2_all3_seed{seed}_v1/checkpoint.pt"
    if variant == "objective_only_128_ppo65k":
        return ROOT / (
            "results/rician_ppo_k3_nt2_all3_obj128_65k_v1/runs/"
            f"rsma_joint_gate_seed{seed}/checkpoint.pt"
        )
    short = {"objective_only_128": "obj128", "objective_only_256": "obj256"}[variant]
    return ROOT / f"runs/k3_nt2_all3_{short}_seed{seed}_v1/checkpoint.pt"


def _job(job: dict) -> dict:
    base = EnvConfig.from_yaml(ROOT / job["config"])
    cfg = replace(
        base, user_angles_deg=[-10.0, 0.0, 10.0], user_angle_sets_deg=None,
        snr_db=15.0, snr_db_values=None, scheduling_enabled=False,
        episode_length=job["steps"],
    )
    actor, _, _ = PPOPolicy.from_checkpoint(
        _checkpoint(job["variant"], job["training_seed"]), deterministic=True, device="cpu"
    )
    row, _ = evaluate_trajectory(
        cfg, {"burn_in": job["burn_in"]}, job["variant"],
        job["training_seed"], job["eval_seed"], actor,
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/thesis_rician_k3_nt2_all3_ppo_csit02.yaml")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[980000, 980001, 980002, 980003, 980004, 980005])
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--burn-in", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--out-dir", default="results/k3_nt2_all3_pretraining_ablation_v1")
    args = parser.parse_args()
    variants = (
        "action_objective_128", "objective_only_128",
        "objective_only_128_ppo65k", "objective_only_256",
    )
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    jobs = [dict(config=args.config, variant=v, training_seed=t, eval_seed=e,
                 steps=args.steps, burn_in=args.burn_in)
            for v in variants for t in args.training_seeds for e in args.eval_seeds]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        episodes = pd.DataFrame(pool.map(_job, jobs))
    episodes.to_csv(out / "episodes.csv", index=False)
    summary = episodes.groupby("algorithm", as_index=False).agg(
        mean_aoi=("mean_aoi", "mean"), std_aoi=("mean_aoi", "std"),
        success_rate=("success_rate", "mean"), common_on_rate=("common_on_rate", "mean"),
        common_power=("mean_common_power", "mean"), rho=("mean_rho", "mean"),
        trajectories=("mean_aoi", "size"),
    )
    summary.to_csv(out / "summary.csv", index=False)
    seed_summary = episodes.groupby(["algorithm", "training_seed"], as_index=False).agg(
        mean_aoi=("mean_aoi", "mean"), std_eval=("mean_aoi", "std")
    )
    seed_summary.to_csv(out / "training_seed_summary.csv", index=False)
    print(summary.sort_values("mean_aoi").to_string(index=False))
    print("\nBY TRAINING SEED")
    print(seed_summary.to_string(index=False))
    print(f"\nCOMPLETE: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
