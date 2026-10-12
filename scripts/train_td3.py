#!/usr/bin/env python3
"""Train the TD3 continuous-control baseline.

TD3 is scoped to continuous resource allocation, so the environment config must
set ``scheduling_enabled: false`` (e.g. ``configs/m2_continuous.yaml`` or
``configs/m1_rsma.yaml``).

Fair-comparison reminder: compare against PPO at an equal ``total_timesteps``
(environment interactions), never at an equal number of episodes or gradient
steps - TD3 reuses a replay buffer.

Examples:
    python scripts/train_td3.py --env-config configs/m2_continuous.yaml \
        --agent-config configs/td3_default.yaml --seed 0
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsma_aoi.agents.td3 import TD3Trainer  # noqa: E402
from rsma_aoi.evaluation.evaluator import evaluate_policy, save_results  # noqa: E402
from rsma_aoi.evaluation.statistics import summarise  # noqa: E402
from rsma_aoi.utils.config import apply_overrides_from_strings, load_td3_configs  # noqa: E402
from rsma_aoi.utils.logging import get_logger  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--agent-config", default="configs/td3_default.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--set", nargs="*", default=None, metavar="KEY=VALUE")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = get_logger()
    env_cfg, td3_cfg = load_td3_configs(args.env_config, args.agent_config)
    if args.total_timesteps is not None:
        td3_cfg.total_timesteps = int(args.total_timesteps)
    if args.device is not None:
        td3_cfg.device = args.device
    if args.set:
        td3_cfg = apply_overrides_from_strings(td3_cfg, args.set)
    td3_cfg.seed = int(args.seed)
    td3_cfg.validate()

    if env_cfg.scheduling_enabled:
        raise SystemExit(
            f"{args.env_config} has scheduling_enabled: true, but the TD3 baseline is "
            "scoped to continuous allocation; use configs/m2_continuous.yaml or pass "
            "an environment config with scheduling disabled"
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = args.tag or f"td3-{Path(args.env_config).stem}-seed{args.seed}-{stamp}"
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / tag

    logger.info("training %s -> %s", tag, run_dir)
    trainer = TD3Trainer(env_cfg, td3_cfg, run_dir=run_dir, seed=args.seed)
    metrics = trainer.train()
    checkpoint = trainer.save_checkpoint(run_dir / "checkpoint.pt")
    logger.info("final metrics: %s", {k: round(v, 5) for k, v in metrics.items()})

    if args.eval_episodes > 0:
        for mode, deterministic in (("deterministic", True), ("stochastic", False)):
            df = evaluate_policy(
                trainer.make_policy(deterministic=deterministic), env_cfg,
                episodes=args.eval_episodes, eval_seed=100_000 + 1_000 * args.seed,
                algorithm=f"td3_{mode}" if mode == "stochastic" else "td3",
            )
            save_results(df, run_dir, name=f"eval_episodes_{mode}")
            summary = summarise(df, seed=args.seed)
            summary.to_csv(run_dir / f"eval_summary_{mode}.csv", index=False)
            row = summary[summary["metric"] == "weighted_avg_aoi"].iloc[0]
            logger.info("%-14s weighted AoI: %.4f (95%% CI [%.4f, %.4f])",
                        mode, row["mean"], row["ci_low"], row["ci_high"])
    trainer.close()
    logger.info("checkpoint: %s", checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
