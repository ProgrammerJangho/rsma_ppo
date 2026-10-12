#!/usr/bin/env python3
"""Train one RSMA PPO policy over episode-randomized 5/20/60-degree geometry."""

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/thesis_rician_mixed_geometry_csit02.yaml"
    )
    parser.add_argument("--steps", type=int, default=262_144)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--out-dir", default="results/rician_ppo_mixed_geometry_csit02_262k_v1"
    )
    args = parser.parse_args(argv)

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    jobs = []
    for seed in args.seeds:
        pretrained = (
            ROOT
            / f"runs/objective_actor_rician_mixed_geometry_csit02_seed{seed}_v1/checkpoint.pt"
        )
        if not pretrained.is_file():
            raise FileNotFoundError(pretrained)
        jobs.append(
            {
                "config": str(ROOT / args.config),
                "method": "rsma_joint_gate",
                "seed": int(seed),
                "steps": int(args.steps),
                "threads": int(args.threads),
                "pretrained_actor": str(pretrained),
                "run_dir": str(out / "runs" / f"rsma_joint_gate_seed{seed}"),
            }
        )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_train_one, jobs))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "single-policy mixed 5/20/60-degree geometry audit",
        "config": args.config,
        "steps": int(args.steps),
        "seeds": list(args.seeds),
        "geometry_sampling": "uniform per episode; fixed within episode",
        "runs": results,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for result in results:
        print(f"seed={result['seed']} checkpoint={result['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
