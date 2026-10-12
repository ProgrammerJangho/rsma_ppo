#!/usr/bin/env python3
"""Train matched RSMA/SDMA PPO policies over a mixed geometry/SNR grid."""

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
        "--config", default="configs/thesis_rician_mixed_geometry_snr_csit02.yaml"
    )
    parser.add_argument("--steps", type=int, default=262_144)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--out-dir", default="results/rician_ppo_mixed_geometry_snr_csit02_262k_v1"
    )
    args = parser.parse_args(argv)

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    jobs = []
    for method, short in (("rsma_joint_gate", "rsma"), ("sdma_power", "sdma")):
        for seed in args.seeds:
            pretrained = ROOT / (
                "runs/objective_actor_rician_mixed_geometry_snr_csit02_"
                f"{short}_seed{seed}_v1/checkpoint.pt"
            )
            if not pretrained.is_file():
                raise FileNotFoundError(pretrained)
            jobs.append({
                "config": str(ROOT / args.config),
                "method": method,
                "seed": int(seed),
                "steps": int(args.steps),
                "threads": int(args.threads),
                "pretrained_actor": str(pretrained),
                "run_dir": str(out / "runs" / f"{method}_seed{seed}"),
            })
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_train_one, jobs))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "matched single-policy 3x3 mixed geometry/SNR pilot",
        "config": args.config,
        "steps": int(args.steps),
        "seeds": list(args.seeds),
        "sampling": "geometry and SNR independent uniform per episode",
        "runs": results,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for result in results:
        print(f"{result['method']} seed={result['seed']} checkpoint={result['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
