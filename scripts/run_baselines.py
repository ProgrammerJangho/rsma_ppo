#!/usr/bin/env python3
"""Evaluate every baseline (and optionally a PPO checkpoint) on one scenario.

All algorithms use the same environment configuration, the same evaluation
seeds and the same number of episodes; OMA/SDMA/NOMA additionally apply their
documented ``multiple_access`` override so that the *physics* differ, not just
the label.

Examples:
    python scripts/run_baselines.py --config configs/m2_hybrid.yaml --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsma_aoi.baselines import BASELINE_REGISTRY, build_baseline  # noqa: E402
from rsma_aoi.evaluation.evaluator import apply_overrides, evaluate_policy, save_results  # noqa: E402
from rsma_aoi.evaluation.plots import plot_aoi_cdf, plot_metric_bars  # noqa: E402
from rsma_aoi.evaluation.statistics import summarise  # noqa: E402
from rsma_aoi.utils.config import EnvConfig, dump_config_snapshot  # noqa: E402
from rsma_aoi.utils.logging import get_logger, git_commit_hash  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="environment YAML")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="independent evaluation seeds (>= 10 recommended for papers)")
    parser.add_argument("--episodes", type=int, default=20, help="episodes per seed")
    parser.add_argument("--baselines", nargs="+", default=sorted(BASELINE_REGISTRY),
                        help=f"subset of {sorted(BASELINE_REGISTRY)}")
    parser.add_argument("--checkpoint", nargs="+", default=None,
                        help="optional PPO checkpoint(s) to evaluate alongside the baselines")
    parser.add_argument("--checkpoint-labels", nargs="+", default=None,
                        help="labels for --checkpoint (default: ppo_hybrid / ppo_continuous)")
    parser.add_argument("--out-dir", default="results/baselines")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = get_logger()
    base_cfg = EnvConfig.from_yaml(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for name in args.baselines:
        for seed in args.seeds:
            policy, overrides = build_baseline(name, base_cfg, seed=seed)
            cfg = apply_overrides(base_cfg, overrides)
            df = evaluate_policy(
                policy, cfg, episodes=args.episodes,
                eval_seed=100_000 + 1_000 * seed, algorithm=name,
            )
            df["seed"] = seed
            frames.append(df)
            logger.info(
                "%-14s seed %d | weighted AoI %.3f | success %.3f | violation %.3f",
                name, seed, df["weighted_avg_aoi"].mean(), df["success_rate"].mean(),
                df["violation_rate"].mean(),
            )

    if args.checkpoint:
        from rsma_aoi.agents.ppo import PPOPolicy

        labels = args.checkpoint_labels or [None] * len(args.checkpoint)
        if len(labels) != len(args.checkpoint):
            raise SystemExit("--checkpoint-labels must have one entry per --checkpoint")
        for path, given_label in zip(args.checkpoint, labels):
            policy, ckpt_cfg, _ = PPOPolicy.from_checkpoint(path, deterministic=True)
            label = given_label or (
                "ppo_hybrid" if ckpt_cfg.scheduling_enabled else "ppo_continuous"
            )
            # the checkpoint's own scheduling mode decides whether the discrete
            # head is used, so evaluate it in the matching environment
            cfg = apply_overrides(base_cfg, {"scheduling_enabled": ckpt_cfg.scheduling_enabled})
            for seed in args.seeds:
                df = evaluate_policy(
                    policy, cfg, episodes=args.episodes,
                    eval_seed=100_000 + 1_000 * seed, algorithm=label,
                )
                df["seed"] = seed
                frames.append(df)
                logger.info("%-14s seed %d | weighted AoI %.3f", label, seed,
                            df["weighted_avg_aoi"].mean())

    raw = pd.concat(frames, ignore_index=True)
    save_results(raw, out_dir, name="episodes")
    summary = summarise(raw, group_keys=["algorithm"], seed=0)
    summary.to_csv(out_dir / "summary.csv", index=False)
    dump_config_snapshot(
        out_dir / "config_snapshot.yaml", env=base_cfg,
        experiment={"seeds": args.seeds, "episodes": args.episodes,
                    "git_commit": git_commit_hash(Path.cwd())},
    )
    if not args.no_plots:
        plot_metric_bars(summary, "weighted_avg_aoi", out_dir / "weighted_aoi.png")
        plot_aoi_cdf(raw, out_dir / "aoi_cdf.png")

    table = summary[summary["metric"] == "weighted_avg_aoi"].sort_values("mean")
    logger.info("\n%s", table[["algorithm", "mean", "std", "ci_low", "ci_high", "n"]]
                .to_string(index=False))
    logger.info("results written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
