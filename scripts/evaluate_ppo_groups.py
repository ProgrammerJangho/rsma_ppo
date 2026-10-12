#!/usr/bin/env python3
"""Evaluate named PPO checkpoint groups on identical held-out trajectories."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from run_final_evaluation import evaluate_trajectory
from rsma_aoi.agents.ppo import PPOPolicy


def parse_group(value: str) -> tuple[str, list[Path]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("group must be LABEL=checkpoint,checkpoint,...")
    label, raw = value.split("=", 1)
    paths = [Path(x) for x in raw.split(",") if x]
    if not label or not paths:
        raise argparse.ArgumentTypeError("group label and checkpoint list cannot be empty")
    return label, paths


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", action="append", type=parse_group, required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--eval-seeds", nargs="+", type=int,
                   default=[993000, 993001, 993002, 993003, 993004, 993005])
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--burn-in", type=int, default=512)
    args = p.parse_args()
    labels = [label for label, _ in args.group]
    if len(labels) != len(set(labels)) or args.reference not in labels:
        p.error("group labels must be unique and include --reference")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    protocol = {"burn_in": args.burn_in, "require_all_users_active": True}
    episodes, users = [], []
    manifest_groups = {}
    for label, paths in args.group:
        manifest_groups[label] = [str(path) for path in paths]
        for seed, checkpoint in enumerate(paths):
            policy, cfg, _ = PPOPolicy.from_checkpoint(checkpoint, deterministic=True)
            cfg = replace(cfg, episode_length=args.steps)
            for eval_seed in args.eval_seeds:
                row, user_rows = evaluate_trajectory(
                    cfg, protocol, label, seed, eval_seed, policy
                )
                episodes.append(row); users.extend(user_rows)
    frame = pd.DataFrame(episodes)
    frame.to_csv(out / "episodes.csv", index=False)
    pd.DataFrame(users).to_csv(out / "users.csv", index=False)
    summary = frame.groupby(["algorithm", "training_seed"], as_index=False).agg(
        mean_aoi=("mean_aoi", "mean"), std_eval=("mean_aoi", "std"),
        p95=("aoi_p95", "mean"), p99=("aoi_p99", "mean"),
        success_rate=("success_rate", "mean"),
        alpha=("mean_common_power", "mean"), phi=("mean_common_fraction", "mean"),
    )
    summary.to_csv(out / "summary.csv", index=False)
    reference = frame[frame.algorithm == args.reference].groupby("eval_seed").mean_aoi.mean()
    comparisons = []
    for label in labels:
        if label == args.reference:
            continue
        candidate = frame[frame.algorithm == label].groupby("eval_seed").mean_aoi.mean()
        aligned = pd.concat([candidate.rename("candidate"), reference.rename("reference")], axis=1)
        delta = aligned.candidate - aligned.reference
        comparisons.append({
            "candidate": label, "reference": args.reference,
            "candidate_seed_mean_aoi": aligned.candidate.mean(),
            "reference_seed_mean_aoi": aligned.reference.mean(),
            "gain_pct": 100*(aligned.reference.mean()-aligned.candidate.mean())/aligned.reference.mean(),
            "wins": int((delta < 0).sum()), "paired_trajectories": len(delta),
        })
    pd.DataFrame(comparisons).to_csv(out / "comparisons.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "status": "complete", "groups": manifest_groups,
        "reference": args.reference, "eval_seeds": args.eval_seeds,
        "steps": args.steps, "burn_in": args.burn_in,
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nGROUP COMPARISONS")
    print(pd.DataFrame(comparisons).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
