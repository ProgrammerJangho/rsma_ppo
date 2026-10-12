#!/usr/bin/env python3
"""Run a short hyper-parameter sweep defined by a YAML file.

Every (variant, seed) pair is trained from scratch with the same environment and
the same interaction budget, then evaluated deterministically on the **same**
evaluation channel seeds as every other pair, so the comparison is paired.

Selection criteria (documented):
    1. weighted average AoI      (primary)
    2. AoI p95                   (tie-break)
    3. constraint violation == 0 (hard requirement)
The scheduled-user count is *reported* but never used as a selection criterion.

A variant may override PPO fields, environment fields, or both::

    variants:
      hp_only:   {entropy_coef: 0.02}                 # flat keys = PPO fields
      phy_only:  {env: {common_beam: mrt_sum}}
      both:      {ppo: {gamma: 0.95}, env: {link_model: shannon}}

Environment overrides may not change the quantities that define the comparison
(blocklength, payload, power/SNR, slot count, K, Nt) unless
``--allow-unfair-env`` is passed - see :data:`PROTECTED_ENV_FIELDS`.

Examples:
    python scripts/run_sweep.py --sweep configs/sweep_m2.yaml --workers 5
    python scripts/run_sweep.py --sweep configs/ablations_m2.yaml --workers 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsma_aoi.evaluation.evaluator import evaluate_policy, save_results  # noqa: E402
from rsma_aoi.evaluation.statistics import summarise  # noqa: E402
from rsma_aoi.utils.config import (  # noqa: E402
    apply_overrides_from_strings, dump_config_snapshot, load_run_configs, load_yaml,
)
from rsma_aoi.utils.logging import get_logger  # noqa: E402

#: environment fields an ablation must not change: they define the resource
#: budget every arm is compared under (same slot duration, power, blocklength,
#: payload and network size).
PROTECTED_ENV_FIELDS = frozenset({
    "blocklength", "payload_bits", "snr_db", "p_max", "noise_power",
    "episode_length", "num_users", "num_antennas", "aoi_max", "aoi_weights",
    "aoi_tail_threshold", "reward_scale", "lambda_power", "lambda_tail",
    "lambda_violation",
})

#: training scalars copied from the end of each run into the results table
TRAIL_METRICS = (
    "train/explained_variance", "train/approx_kl", "train/clip_fraction",
    "train/entropy", "train/learning_rate", "env/num_scheduled", "env/violation_rate",
)


def _overrides_to_strings(mapping: dict) -> list[str]:
    return [f"{k}={json.dumps(v) if isinstance(v, list) else v}" for k, v in mapping.items()]


def split_variant(spec: dict | None) -> tuple[dict, dict]:
    """Split a variant entry into ``(ppo_overrides, env_overrides)``.

    Flat keys are treated as PPO overrides for backwards compatibility.
    """
    spec = spec or {}
    if "ppo" in spec or "env" in spec:
        unknown = set(spec) - {"ppo", "env"}
        if unknown:
            raise ValueError(f"variant may only contain 'ppo'/'env' keys, got {sorted(unknown)}")
        return dict(spec.get("ppo") or {}), dict(spec.get("env") or {})
    return dict(spec), {}


def check_env_overrides(env_overrides: dict, allow_unfair: bool) -> None:
    """Reject ablations that would change the shared resource budget."""
    violations = sorted(set(env_overrides) & PROTECTED_ENV_FIELDS)
    if violations and not allow_unfair:
        raise SystemExit(
            f"environment override(s) {violations} change the comparison budget "
            "(slot duration / power / blocklength / payload / network size). "
            "Every ablation arm must share these. Pass --allow-unfair-env to "
            "override deliberately."
        )


def _train_one(payload: tuple) -> dict:
    """Worker: train one (variant, seed) pair and evaluate it."""
    spec, run_dir, threads = payload
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)
    import torch

    torch.set_num_threads(threads)

    from rsma_aoi.agents.ppo import PPOTrainer

    env_cfg, ppo_cfg = load_run_configs(spec["base_env"], spec["base_agent"])
    ppo_cfg.total_timesteps = int(spec["total_timesteps"])
    if spec["overrides"]:
        ppo_cfg = apply_overrides_from_strings(ppo_cfg, _overrides_to_strings(spec["overrides"]))
    if spec["env_overrides"]:
        env_cfg = apply_overrides_from_strings(
            env_cfg, _overrides_to_strings(spec["env_overrides"])
        )
    ppo_cfg.seed = int(spec["seed"])
    ppo_cfg.checkpoint_interval = max(1, ppo_cfg.total_timesteps // ppo_cfg.batch_size // 5)

    run_dir = Path(run_dir)
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=run_dir, seed=spec["seed"])
    final = trainer.train()
    checkpoint = trainer.save_checkpoint(run_dir / "checkpoint.pt")

    stochastic = evaluate_policy(
        trainer.make_policy(deterministic=False), env_cfg,
        episodes=spec["eval_episodes"], eval_seed=100_000 + 1_000 * spec["seed"],
        algorithm=f"{spec['variant']}_stochastic",
    )
    deterministic = evaluate_policy(
        trainer.make_policy(deterministic=True), env_cfg,
        episodes=spec["eval_episodes"], eval_seed=100_000 + 1_000 * spec["seed"],
        algorithm=spec["variant"],
    )
    trainer.close()

    for frame, mode in ((deterministic, "deterministic"), (stochastic, "stochastic")):
        frame["variant"] = spec["variant"]
        frame["seed"] = spec["seed"]
        frame["eval_mode"] = mode
    combined = pd.concat([deterministic, stochastic], ignore_index=True)
    combined.to_csv(run_dir / "eval_episodes.csv", index=False)

    return {
        "variant": spec["variant"],
        "seed": spec["seed"],
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "episodes": combined,
        "train": {k: float(final.get(k, float("nan"))) for k in TRAIL_METRICS},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sweep", required=True, help="sweep YAML")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--workers", type=int, default=4, help="parallel training runs")
    parser.add_argument("--threads", type=int, default=3, help="torch threads per run")
    parser.add_argument("--variants", nargs="+", default=None, help="subset of variants")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--allow-unfair-env", action="store_true",
                        help="permit environment overrides that change the resource budget")
    args = parser.parse_args(argv)

    logger = get_logger()
    spec = load_yaml(args.sweep)["sweep"]
    seeds = args.seeds or spec["seeds"]
    total = args.total_timesteps or spec["total_timesteps"]
    variants = {
        name: overrides
        for name, overrides in spec["variants"].items()
        if args.variants is None or name in args.variants
    }
    if not variants:
        raise SystemExit("no variants selected")

    out_dir = Path(args.out_dir or f"results/sweep-{Path(args.sweep).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for name, raw in variants.items():
        ppo_overrides, env_overrides = split_variant(raw)
        check_env_overrides(env_overrides, args.allow_unfair_env)
        for seed in seeds:
            jobs.append(
                (
                    {
                        "variant": name, "seed": int(seed), "overrides": ppo_overrides,
                        "env_overrides": env_overrides,
                        "base_env": spec["base_env"], "base_agent": spec["base_agent"],
                        "total_timesteps": int(total),
                        "eval_episodes": int(spec.get("eval_episodes", 20)),
                    },
                    str(out_dir / f"{name}_seed{seed}"),
                    args.threads,
                )
            )
    logger.info("sweep: %d variants x %d seeds = %d runs at %d steps (%d workers)",
                len(variants), len(seeds), len(jobs), total, args.workers)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(_train_one, jobs):
            results.append(result)
            det = result["episodes"]
            det = det[det["eval_mode"] == "deterministic"]
            logger.info(
                "done %-9s seed %d | AoI %.3f | p95 %.3f | sched %.2f | viol %.3f",
                result["variant"], result["seed"], det["weighted_avg_aoi"].mean(),
                det["aoi_p95"].mean(), det["avg_num_scheduled"].mean(),
                det["violation_rate"].mean(),
            )

    raw = pd.concat([r["episodes"] for r in results], ignore_index=True)
    save_results(raw, out_dir, name="episodes")
    train_df = pd.DataFrame(
        [{"variant": r["variant"], "seed": r["seed"], **r["train"]} for r in results]
    )
    train_df.to_csv(out_dir / "train_metrics.csv", index=False)

    det = raw[raw["eval_mode"] == "deterministic"]
    summary = summarise(det, group_keys=["variant"], seed=0)
    summary.to_csv(out_dir / "summary.csv", index=False)
    summarise(raw[raw["eval_mode"] == "stochastic"], group_keys=["variant"], seed=0).to_csv(
        out_dir / "summary_stochastic.csv", index=False
    )
    dump_config_snapshot(
        out_dir / "config_snapshot.yaml",
        experiment={"sweep": args.sweep, "seeds": list(seeds),
                    "total_timesteps": int(total),
                    "variants": {n: split_variant(v) for n, v in variants.items()}},
    )

    ranking = (
        det.groupby("variant")
        .agg(weighted_avg_aoi=("weighted_avg_aoi", "mean"),
             aoi_p95=("aoi_p95", "mean"),
             aoi_p99=("aoi_p99", "mean"),
             max_aoi=("max_aoi", "mean"),
             success_rate=("success_rate", "mean"),
             avg_num_scheduled=("avg_num_scheduled", "mean"),
             schedule_entropy=("schedule_entropy", "mean"),
             common_power=("avg_common_power_ratio", "mean"),
             common_payload=("avg_common_payload_ratio", "mean"),
             violation=("violation_rate", "mean"))
        .sort_values(["weighted_avg_aoi", "aoi_p95"])
    )
    feasible = ranking[ranking["violation"] == 0.0]
    ranking.to_csv(out_dir / "ranking.csv")

    pd.set_option("display.width", 220)
    print("\n=== deterministic evaluation ranking (primary: weighted AoI, tie: p95) ===")
    print(ranking.round(4).to_string())
    print("\n=== end-of-training metrics ===")
    print(train_df.groupby("variant").mean(numeric_only=True).round(5).to_string())
    print("\n=== active-count histogram (fraction of slots) ===")
    hist_cols = [c for c in det.columns if c.startswith("sched_frac_")]
    print(det.groupby("variant")[hist_cols].mean().round(4).to_string())
    if not feasible.empty:
        print(f"\nTop-2 feasible variants: {list(feasible.index[:2])}")
    logger.info("results written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
