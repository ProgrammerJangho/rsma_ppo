#!/usr/bin/env python3
"""Evaluate the non-learning arms of an operating-point comparison.

Produces the long-run AoI of OMA, SDMA (equal and optimiser-allocated),
max-AoI-first scheduling with an optimiser allocator, and equal-allocation RSMA
- on the same environment and the same evaluation seeds as the RL sweeps, so the
two sets of numbers are directly comparable.

Every arm here is a *long-run* measurement (full episodes with sampled ACKs),
unlike the myopic one-step screening used by the phase map.

Examples:
    python scripts/evaluate_operating_point.py --config configs/op1_rsma.yaml --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsma_aoi.baselines import build_baseline  # noqa: E402
from rsma_aoi.evaluation.evaluator import evaluate_policy, save_results  # noqa: E402
from rsma_aoi.evaluation.fairness import build_manifest, rsma_gain  # noqa: E402
from rsma_aoi.evaluation.statistics import summarise  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402
from rsma_aoi.utils.logging import get_logger  # noqa: E402

#: label -> (baseline policy, environment overrides)
ARMS = {
    "oma": ("oma", {"multiple_access": "oma"}),
    "sdma_equal": ("max_aoi_first", {"multiple_access": "sdma"}),
    "sdma_optimised": ("max_aoi_first", {"multiple_access": "sdma",
                                         "allocator_mode": "cem_fast"}),
    "rsma_equal": ("max_aoi_rsma", {}),
    "heuristic_sched_rsma_optimised": ("max_aoi_first", {"allocator_mode": "cem_fast"}),
    "random": ("random", {}),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    logger = get_logger()
    base_cfg = EnvConfig.from_yaml(args.config)
    out_dir = Path(args.out_dir or f"results/baselines_{Path(args.config).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for label, (policy_name, overrides) in ARMS.items():
        cfg = dataclasses.replace(base_cfg, **overrides)
        for seed in args.seeds:
            policy, extra = build_baseline(policy_name, cfg, seed=seed)
            cfg_arm = dataclasses.replace(cfg, **extra) if extra else cfg
            df = evaluate_policy(
                policy, cfg_arm, episodes=args.episodes,
                eval_seed=100_000 + 1_000 * seed, algorithm=label,
            )
            df["seed"] = seed
            frames.append(df)
        logger.info("%-32s AoI %.4f", label,
                    pd.concat(frames[-len(args.seeds):])["weighted_avg_aoi"].mean())

    raw = pd.concat(frames, ignore_index=True)
    save_results(raw, out_dir, name="episodes")
    summary = summarise(raw, group_keys=["algorithm"], seed=0)
    summary.to_csv(out_dir / "summary.csv", index=False)

    means = raw.groupby("algorithm")["weighted_avg_aoi"].mean()
    gains = {
        "gain_vs_sdma": rsma_gain(means["sdma_optimised"],
                                  means["heuristic_sched_rsma_optimised"]),
        "gain_vs_oma": rsma_gain(means["oma"], means["heuristic_sched_rsma_optimised"]),
    }
    manifest, report = build_manifest(
        f"operating_point_{Path(args.config).stem}", base_cfg,
        seeds=args.seeds, episodes=args.episodes, long_run_gains=gains,
        metric="long-run weighted average AoI (sampled ACKs, full episodes)",
    )
    manifest.save(out_dir / "manifest.json")

    pd.set_option("display.width", 200)
    table = summary[summary["metric"] == "weighted_avg_aoi"].sort_values("mean")
    print(f"\n=== {Path(args.config).stem}: long-run weighted AoI (non-learning arms) ===")
    print(table[["algorithm", "mean", "std", "ci_low", "ci_high", "n"]].round(4).to_string(index=False))
    print("\nLong-run RSMA gains (heuristic scheduling + CEM allocation):")
    for key, value in gains.items():
        print(f"  {key}: {value:+.4f}")
    print(f"parity audit: {report.passed}")
    logger.info("results written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
