#!/usr/bin/env python3
"""Train three all-user K=3, Nt=2 allocation-only PPO policies."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_matched_budget_audit import _train_one  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", default="configs/thesis_rician_k3_nt2_all3_ppo_csit02.yaml"
    )
    p.add_argument("--steps", type=int, default=262_144)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument(
        "--pretrained-template",
        default="runs/action_objective_actor_k3_nt2_all3_seed{seed}_v1/checkpoint.pt",
        help="checkpoint path relative to the project root; supports {seed}",
    )
    p.add_argument("--out-dir", default="results/rician_ppo_k3_nt2_all3_262k_v1")
    args = p.parse_args()

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    jobs = []
    for seed in args.seeds:
        pretrained = ROOT / args.pretrained_template.format(seed=seed)
        if not pretrained.is_file():
            raise FileNotFoundError(pretrained)
        jobs.append({
            "config": str(ROOT / args.config), "method": "rsma_joint_gate",
            "seed": seed, "steps": args.steps, "threads": args.threads,
            "pretrained_actor": str(pretrained),
            "run_dir": str(out / "runs" / f"rsma_joint_gate_seed{seed}"),
        })
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_train_one, jobs))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "K=3 Nt=2 all-user allocation-only action-objective PPO",
        "config": args.config, "steps": args.steps, "seeds": args.seeds,
        "runs": results,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for result in results:
        print(f"seed={result['seed']} checkpoint={result['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
