#!/usr/bin/env python3
"""Train the hybrid (or continuous-only) PPO agent.

Examples:
    python scripts/train_ppo.py --env-config configs/m1_rsma.yaml \
        --agent-config configs/ppo_default.yaml --seed 0
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsma_aoi.agents.ppo import PPOTrainer  # noqa: E402
from rsma_aoi.evaluation.evaluator import evaluate_policy, save_results  # noqa: E402
from rsma_aoi.evaluation.statistics import summarise  # noqa: E402
from rsma_aoi.utils.config import apply_overrides_from_strings, load_run_configs  # noqa: E402
from rsma_aoi.utils.logging import get_logger  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-config", required=True, help="environment YAML")
    parser.add_argument("--agent-config", default=None, help="PPO YAML (optional)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", default=None, help="output directory (default: runs/<tag>)")
    parser.add_argument("--tag", default=None, help="run name suffix")
    parser.add_argument("--total-timesteps", type=int, default=None, help="override the budget")
    parser.add_argument("--device", default=None, help="cpu | cuda")
    parser.add_argument(
        "--resume", default=None,
        help="checkpoint to restore before running the requested additional training budget",
    )
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="deterministic evaluation episodes after training (0 to skip)")
    parser.add_argument("--set", nargs="*", default=None, metavar="KEY=VALUE",
                        help="override PPO config fields, e.g. --set entropy_coef=0.02 anneal_lr=false")
    parser.add_argument("--set-env", nargs="*", default=None, metavar="KEY=VALUE",
                        help="override environment config fields")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = get_logger()
    env_cfg, ppo_cfg = load_run_configs(args.env_config, args.agent_config)
    if args.total_timesteps is not None:
        ppo_cfg.total_timesteps = int(args.total_timesteps)
    if args.device is not None:
        ppo_cfg.device = args.device
    if args.set:
        ppo_cfg = apply_overrides_from_strings(ppo_cfg, args.set)
    if args.set_env:
        env_cfg = apply_overrides_from_strings(env_cfg, args.set_env)
    ppo_cfg.seed = int(args.seed)
    ppo_cfg.validate()
    env_cfg.validate()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = args.tag or f"{Path(args.env_config).stem}-seed{args.seed}-{stamp}"
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / tag

    logger.info("training %s -> %s", tag, run_dir)
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=run_dir, seed=args.seed)
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
        logger.info(
            "resumed %s at cumulative step %d; running %d additional interactions",
            args.resume, trainer.global_step, ppo_cfg.total_timesteps,
        )
    metrics = trainer.train()
    checkpoint = trainer.save_checkpoint(run_dir / "checkpoint.pt")
    logger.info("final metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    if args.eval_episodes > 0:
        policy = trainer.make_policy(deterministic=True)
        df = evaluate_policy(
            policy, env_cfg, episodes=args.eval_episodes,
            eval_seed=100_000 + args.seed, algorithm="ppo",
        )
        save_results(df, run_dir, name="eval_episodes")
        summary = summarise(df, seed=args.seed)
        summary.to_csv(run_dir / "eval_summary.csv", index=False)
        row = summary[summary["metric"] == "weighted_avg_aoi"].iloc[0]
        logger.info(
            "eval weighted AoI: %.4f (95%% CI [%.4f, %.4f], n=%d)",
            row["mean"], row["ci_low"], row["ci_high"], int(row["n"]),
        )
    trainer.close()
    logger.info("checkpoint: %s", checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
