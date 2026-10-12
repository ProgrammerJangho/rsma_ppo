#!/usr/bin/env python3
"""Evaluate a trained PPO checkpoint.

Examples:
    python scripts/evaluate.py --checkpoint runs/<run>/checkpoint.pt \
        --config configs/m2_hybrid.yaml --episodes 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsma_aoi.agents.ppo import PPOPolicy  # noqa: E402
from rsma_aoi.evaluation.evaluator import evaluate_policy, save_results  # noqa: E402
from rsma_aoi.evaluation.plots import plot_aoi_cdf  # noqa: E402
from rsma_aoi.evaluation.statistics import summarise  # noqa: E402
from rsma_aoi.utils.config import EnvConfig, dump_config_snapshot  # noqa: E402
from rsma_aoi.utils.logging import get_logger, git_commit_hash  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None,
                        help="environment YAML (default: the config stored in the checkpoint)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=100_000)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--stochastic", action="store_true", help="sample instead of using the mode")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = get_logger()
    policy, ckpt_env_cfg, _ = PPOPolicy.from_checkpoint(
        args.checkpoint, device=args.device, deterministic=not args.stochastic
    )
    env_cfg = EnvConfig.from_yaml(args.config) if args.config else ckpt_env_cfg
    if env_cfg.num_users != ckpt_env_cfg.num_users:
        raise SystemExit(
            f"config has K={env_cfg.num_users} but the checkpoint was trained with "
            f"K={ckpt_env_cfg.num_users}; the policy heads would not match"
        )

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = evaluate_policy(
        policy, env_cfg, episodes=args.episodes, eval_seed=args.eval_seed, algorithm="ppo"
    )
    save_results(df, out_dir, name="episodes")
    summary = summarise(df, seed=0)
    summary.to_csv(out_dir / "summary.csv", index=False)
    dump_config_snapshot(out_dir / "config_snapshot.yaml", env=env_cfg)
    with (out_dir / "provenance.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"checkpoint={args.checkpoint}\neval_seed={args.eval_seed}\n")
        fh.write(f"episodes={args.episodes}\ngit_commit={git_commit_hash(Path.cwd())}\n")

    if not args.no_plots:
        plot_aoi_cdf(df, out_dir / "aoi_cdf.png")

    for metric in ("weighted_avg_aoi", "mean_aoi", "peak_aoi", "aoi_p95", "aoi_p99",
                   "success_rate", "avg_power", "violation_rate", "inference_latency_ms"):
        row = summary[summary["metric"] == metric]
        if not row.empty:
            r = row.iloc[0]
            logger.info("%-22s %.4f +/- %.4f  CI [%.4f, %.4f]",
                        metric, r["mean"], r["std"], r["ci_low"], r["ci_high"])
    logger.info("results written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
