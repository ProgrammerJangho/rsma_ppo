#!/usr/bin/env python3
"""End-to-end smoke test: env sanity -> short PPO training -> checkpoint -> eval.

Examples:
    python scripts/smoke_test.py --config configs/m0_debug.yaml
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsma_aoi.agents.ppo import PPOPolicy, PPOTrainer  # noqa: E402
from rsma_aoi.baselines import build_baseline  # noqa: E402
from rsma_aoi.envs.aoi_env import AoIRSMAEnv  # noqa: E402
from rsma_aoi.evaluation.evaluator import apply_overrides, evaluate_policy  # noqa: E402
from rsma_aoi.utils.config import load_run_configs  # noqa: E402
from rsma_aoi.utils.logging import get_logger  # noqa: E402


def check_environment(env_cfg, logger) -> None:
    """Gymnasium API check + determinism check under a fixed action sequence."""
    from gymnasium.utils.env_checker import check_env

    env = AoIRSMAEnv(env_cfg)
    check_env(env, skip_render_check=True)
    logger.info("gymnasium env checker: OK (obs dim %d)", env.observation_space.shape[0])

    actions = [env.action_space.sample() for _ in range(10)]
    traces = []
    for _ in range(2):
        obs, _ = env.reset(seed=1234)
        trace = [obs.copy()]
        for action in actions:
            obs, reward, term, trunc, info = env.step(action)
            trace.append(obs.copy())
            if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
                raise SystemExit("non-finite observation/reward encountered")
        traces.append(np.stack(trace))
    if not np.allclose(traces[0], traces[1]):
        raise SystemExit("environment is not reproducible under a fixed seed")
    logger.info("seed reproducibility: OK")
    env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/m0_debug.yaml")
    parser.add_argument("--agent-config", default=None)
    parser.add_argument("--timesteps", type=int, default=None, help="override the tiny budget")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args(argv)

    logger = get_logger()
    env_cfg, ppo_cfg = load_run_configs(args.config, args.agent_config)
    if args.timesteps is not None:
        ppo_cfg.total_timesteps = int(args.timesteps)
    ppo_cfg.total_timesteps = min(ppo_cfg.total_timesteps, 4 * ppo_cfg.batch_size)
    ppo_cfg.validate()

    check_environment(env_cfg, logger)

    tmp = None
    if args.run_dir is None:
        tmp = tempfile.TemporaryDirectory()
        run_dir = Path(tmp.name) / "smoke"
    else:
        run_dir = Path(args.run_dir)

    try:
        trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=run_dir, seed=0)
        metrics = trainer.train()
        for key in ("train/policy_loss", "train/value_loss", "train/entropy"):
            if not np.isfinite(metrics.get(key, np.nan)):
                raise SystemExit(f"non-finite training metric: {key}")
        logger.info(
            "training: OK (%d steps, policy_loss %.4f, value_loss %.4f, entropy %.3f)",
            trainer.global_step, metrics["train/policy_loss"],
            metrics["train/value_loss"], metrics["train/entropy"],
        )

        ckpt = trainer.save_checkpoint(run_dir / "checkpoint.pt")
        policy, loaded_cfg, _ = PPOPolicy.from_checkpoint(ckpt, deterministic=True)
        logger.info("checkpoint save/load: OK (%s)", ckpt)

        df = evaluate_policy(policy, env_cfg, episodes=args.episodes,
                             eval_seed=999, algorithm="ppo")
        logger.info("deterministic evaluation: weighted AoI %.3f over %d episodes",
                    df["weighted_avg_aoi"].mean(), len(df))

        for name in ("random", "max_aoi_first", "sdma", "oma", "noma", "equal_power"):
            base_policy, overrides = build_baseline(name, env_cfg, seed=0)
            cfg = apply_overrides(env_cfg, overrides)
            bdf = evaluate_policy(base_policy, cfg, episodes=1, eval_seed=999, algorithm=name)
            logger.info("baseline %-14s weighted AoI %.3f | violation %.3f",
                        name, bdf["weighted_avg_aoi"].mean(), bdf["violation_rate"].mean())
        trainer.close()
    finally:
        if tmp is not None:
            tmp.cleanup()

    logger.info("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
