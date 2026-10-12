#!/usr/bin/env python3
"""Paired held-out evaluation of sampled- versus expected-AoI PPO rewards."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from run_final_evaluation import evaluate_trajectory
from rsma_aoi.agents.ppo import PPOPolicy


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--eval-seeds", nargs="+", type=int,
                   default=[992000, 992001, 992002, 992003, 992004, 992005])
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--burn-in", type=int, default=512)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    protocol = {"burn_in": args.burn_in, "require_all_users_active": True}
    episodes, users, curves = [], [], []
    for mode in ("sampled_aoi", "expected_aoi"):
        for seed in range(3):
            run = ROOT / f"runs/ppo_k3_direct_{mode}_seed{seed}_65k_v1"
            checkpoint = run / "checkpoint.pt"
            policy, cfg, _ = PPOPolicy.from_checkpoint(checkpoint, deterministic=True)
            cfg = replace(cfg, episode_length=args.steps)
            for eval_seed in args.eval_seeds:
                row, user_rows = evaluate_trajectory(
                    cfg, protocol, mode, seed, eval_seed, policy
                )
                episodes.append(row); users.extend(user_rows)
            scalars = pd.read_csv(run / "scalars.csv")
            tail = scalars.iloc[int(.75*len(scalars)):]
            curves.append({
                "reward_mode": mode, "training_seed": seed,
                "initial_rollout_aoi": scalars["env/mean_aoi"].iloc[:13].mean(),
                "final_rollout_aoi": scalars["env/mean_aoi"].iloc[-13:].mean(),
                "tail_explained_variance": tail["train/explained_variance"].mean(),
                "tail_value_loss": tail["train/value_loss"].mean(),
                "tail_kl": tail["train/approx_kl"].mean(),
            })
    frame = pd.DataFrame(episodes)
    frame.to_csv(out / "episodes.csv", index=False)
    pd.DataFrame(users).to_csv(out / "users.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "learning_summary.csv", index=False)
    summary = frame.groupby(["algorithm", "training_seed"], as_index=False).agg(
        mean_aoi=("mean_aoi", "mean"), std_eval=("mean_aoi", "std"),
        p95=("aoi_p95", "mean"), success_rate=("success_rate", "mean"),
        alpha=("mean_common_power", "mean"), phi=("mean_common_fraction", "mean"),
    )
    summary.to_csv(out / "summary.csv", index=False)
    rows = []
    for seed in range(3):
        pivot = frame[frame.training_seed == seed].pivot(
            index="eval_seed", columns="algorithm", values="mean_aoi"
        )
        delta = pivot.expected_aoi - pivot.sampled_aoi
        rows.append({
            "training_seed": seed,
            "sampled_mean_aoi": pivot.sampled_aoi.mean(),
            "expected_mean_aoi": pivot.expected_aoi.mean(),
            "expected_minus_sampled": delta.mean(),
            "expected_gain_pct": 100*(pivot.sampled_aoi.mean()-pivot.expected_aoi.mean())/pivot.sampled_aoi.mean(),
            "expected_wins": int((delta < 0).sum()),
            "paired_trajectories": len(delta),
        })
    paired = pd.DataFrame(rows)
    paired.to_csv(out / "paired_comparisons.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "status": "complete", "purpose": "PPO reward-mode pilot without CEM",
        "steps": args.steps, "burn_in": args.burn_in, "eval_seeds": args.eval_seeds,
        "training_steps": 65536,
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nPAIRED EXPECTED - SAMPLED")
    print(paired.to_string(index=False))
    print("\nLEARNING")
    print(pd.DataFrame(curves).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
