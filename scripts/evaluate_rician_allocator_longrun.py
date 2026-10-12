#!/usr/bin/env python3
"""Long-run test of the myopic CEM allocator on controlled Rician channels.

The experiment separates two hypotheses: if one-step CEM also improves sampled
long-run AoI, PPO failed to learn the available allocation; if it does not, the
one-step expected-AoI surrogate itself is the limiting approximation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from rsma_aoi.baselines.max_aoi_first import MaxAoIFirstPolicy
from rsma_aoi.evaluation.evaluator import evaluate_policy
from rsma_aoi.evaluation.statistics import summarise
from rsma_aoi.utils.config import EnvConfig, dump_config_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--eval-seed", type=int, default=910000)
    parser.add_argument("--allocator", default="cem_full", choices=["cem_fast", "cem_full"])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 128:
        parser.error("episodes must be positive and steps must be >= 128")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    frames: list[pd.DataFrame] = []
    configs: dict[str, EnvConfig] = {}
    for path_text in args.config:
        path = Path(path_text)
        base = EnvConfig.from_yaml(path)
        if (
            base.num_users != 2
            or base.small_scale_model != "rician_ula"
            or base.link_model != "fbl"
            or base.csit_error_std != 0.0
            or base.scheduling_enabled
        ):
            raise ValueError("requires two-user all-active perfect-CSIT Rician FBL configs")
        label = path.stem
        configs[label] = base
        policy = MaxAoIFirstPolicy(base.num_users, base.num_antennas)
        for ma in ("rsma", "sdma"):
            cfg = replace(
                base,
                multiple_access=ma,
                allocator_mode=args.allocator,
                allocator_common_gate=False,
                episode_length=args.steps,
            )
            frame = evaluate_policy(
                policy,
                cfg,
                episodes=args.episodes,
                eval_seed=args.eval_seed,
                algorithm=f"{ma}_{args.allocator}",
            )
            frame["config"] = label
            frames.append(frame)
            print(
                f"{label} {ma}: mean AoI={frame.mean_aoi.mean():.6f}, "
                f"success={frame.success_rate.mean():.6f}",
                flush=True,
            )

    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(out / "episodes.csv", index=False)
    summary = summarise(raw, group_keys=["config", "algorithm"], seed=0)
    summary.to_csv(out / "summary.csv", index=False)
    for label, cfg in configs.items():
        dump_config_snapshot(out / f"{label}_config.yaml", env=cfg)

    paths = [Path(p).resolve() for p in args.config]
    manifest = {
        "purpose": "long-run sampled-AoI test of a one-step CEM allocation policy",
        "configs": [str(p) for p in paths],
        "config_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "episodes": args.episodes,
        "steps_per_episode": args.steps,
        "eval_seeds": list(range(args.eval_seed, args.eval_seed + args.episodes)),
        "allocator": args.allocator,
        "selection_information": "perfect CSIT; current AoI and h_hat only",
        "comparison": "RSMA and SDMA use the same channel and ACK seeds",
        "caveat": "myopic approximate CEM reference, not a global or long-horizon optimum",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    table = summary[summary.metric == "mean_aoi"]
    print("\n" + table[["config", "algorithm", "mean", "ci_low", "ci_high", "n"]].to_string(index=False))
    print(f"PASS: results written to {out}")


if __name__ == "__main__":
    main()
