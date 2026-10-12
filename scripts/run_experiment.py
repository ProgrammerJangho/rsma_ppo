#!/usr/bin/env python3
"""Multi-seed experiment runner: train PPO over N seeds, evaluate, aggregate.

Trains one PPO agent per seed, evaluates every seed's final policy on the same
evaluation seeds as the baselines, and writes raw + aggregated results with
bootstrap confidence intervals.

Examples:
    python scripts/run_experiment.py --env-config configs/m2_hybrid.yaml \
        --agent-config configs/ppo_default.yaml --seeds 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsma_aoi.agents.ppo import PPOTrainer  # noqa: E402
from rsma_aoi.baselines import BASELINE_REGISTRY, build_baseline  # noqa: E402
from rsma_aoi.evaluation.evaluator import apply_overrides, evaluate_policy, save_results  # noqa: E402
from rsma_aoi.evaluation.plots import plot_aoi_cdf, plot_metric_bars  # noqa: E402
from rsma_aoi.evaluation.statistics import summarise  # noqa: E402
from rsma_aoi.utils.config import dump_config_snapshot, load_run_configs  # noqa: E402
from rsma_aoi.utils.logging import get_logger, git_commit_hash  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--agent-config", default="configs/ppo_default.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--episodes", type=int, default=50, help="evaluation episodes per seed")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--baselines", nargs="+", default=sorted(BASELINE_REGISTRY))
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = get_logger()
    env_cfg, ppo_cfg = load_run_configs(args.env_config, args.agent_config)
    if args.total_timesteps is not None:
        ppo_cfg.total_timesteps = int(args.total_timesteps)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or f"results/{Path(args.env_config).stem}-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    label = "ppo_hybrid" if env_cfg.scheduling_enabled else "ppo_continuous"
    frames: list[pd.DataFrame] = []

    for seed in args.seeds:
        run_dir = out_dir / f"ppo_seed{seed}"
        logger.info("=== training %s seed %d (%d steps) ===", label, seed, ppo_cfg.total_timesteps)
        trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=run_dir, seed=seed)
        trainer.train()
        policy = trainer.make_policy(deterministic=True)
        df = evaluate_policy(
            policy, env_cfg, episodes=args.episodes,
            eval_seed=100_000 + 1_000 * seed, algorithm=label,
        )
        df["seed"] = seed
        frames.append(df)
        logger.info("seed %d weighted AoI %.4f", seed, df["weighted_avg_aoi"].mean())
        trainer.close()

    if not args.skip_baselines:
        for name in args.baselines:
            for seed in args.seeds:
                policy, overrides = build_baseline(name, env_cfg, seed=seed)
                cfg = apply_overrides(env_cfg, overrides)
                df = evaluate_policy(
                    policy, cfg, episodes=args.episodes,
                    eval_seed=100_000 + 1_000 * seed, algorithm=name,
                )
                df["seed"] = seed
                frames.append(df)
            logger.info("baseline %s done", name)

    raw = pd.concat(frames, ignore_index=True)
    save_results(raw, out_dir, name="episodes")
    summary = summarise(raw, group_keys=["algorithm"], seed=0)
    summary.to_csv(out_dir / "summary.csv", index=False)
    dump_config_snapshot(
        out_dir / "config_snapshot.yaml", env=env_cfg, ppo=ppo_cfg,
        experiment={
            "seeds": args.seeds, "episodes_per_seed": args.episodes,
            "interaction_budget_per_seed": ppo_cfg.total_timesteps,
            "git_commit": git_commit_hash(Path.cwd()),
        },
    )
    plot_metric_bars(summary, "weighted_avg_aoi", out_dir / "weighted_aoi.png")
    plot_aoi_cdf(raw, out_dir / "aoi_cdf.png")

    table = summary[summary["metric"] == "weighted_avg_aoi"].sort_values("mean")
    logger.info("\n%s", table[["algorithm", "mean", "std", "ci_low", "ci_high", "n"]]
                .to_string(index=False))
    logger.info("results written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
