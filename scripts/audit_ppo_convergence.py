#!/usr/bin/env python3
"""Controlled PPO convergence audit with immutable intermediate checkpoints.

The training rollout is deliberately kept separate from evaluation.  Every
checkpoint is evaluated deterministically on the same channel/ACK seeds, which
makes changes along a curve attributable to the policy rather than to a new
evaluation trajectory.  The default variants form a small critic diagnostic;
they are not intended as a hyperparameter sweep.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from rsma_aoi.agents.ppo import PPOPolicy, PPOTrainer
from rsma_aoi.evaluation.evaluator import evaluate_policy
from rsma_aoi.utils.config import apply_overrides_from_strings, load_run_configs


VARIANTS: dict[str, list[str]] = {
    "reference": [],
    "critic_lr_1e3": ["critic_lr=0.001"],
    "value_unclipped": ["clip_value_loss=false"],
    "no_reward_norm": ["normalize_reward=false"],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _train_one(payload: tuple[Any, ...]) -> dict[str, Any]:
    config, out_dir, variant, overrides, seed, steps, checkpoint_updates, threads = payload
    torch.set_num_threads(threads)
    env_cfg, ppo_cfg = load_run_configs(config)
    common = [
        f"total_timesteps={steps}",
        f"seed={seed}",
        "allocation_control=joint",
        "common_gate_enabled=true",
        "checkpoint_interval=0",
        "log_interval=1",
    ]
    ppo_cfg = apply_overrides_from_strings(ppo_cfg, [*common, *overrides])
    run_dir = Path(out_dir) / "runs" / f"{variant}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=run_dir, seed=seed)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()
    trainer.save_checkpoint(checkpoints / "step_000000.pt")

    def save_progress(update: int, _metrics: dict[str, float]) -> None:
        update_number = update + 1
        if update_number % checkpoint_updates == 0 or update_number == trainer.num_updates:
            trainer.save_checkpoint(checkpoints / f"step_{trainer.global_step:06d}.pt")

    final = trainer.train(progress_callback=save_progress)
    trainer.close()
    return {
        "variant": variant,
        "seed": seed,
        "run_dir": str(run_dir),
        "final": {k: float(v) for k, v in final.items() if np.isscalar(v)},
    }


def _evaluate_one_checkpoint(payload: tuple[str, str, list[int], int]) -> pd.DataFrame:
    raw_checkpoint, config, eval_seeds, eval_steps = payload
    checkpoint = Path(raw_checkpoint)
    env_cfg, _ = load_run_configs(config)
    eval_cfg = replace(env_cfg, episode_length=eval_steps)
    run_name = checkpoint.parents[1].name
    variant, raw_seed = run_name.rsplit("_seed", 1)
    policy, trained_env, _ = PPOPolicy.from_checkpoint(checkpoint, deterministic=True)
    if trained_env.to_dict() != env_cfg.to_dict():
        raise AssertionError(f"environment mismatch in {checkpoint}")
    step = int(checkpoint.stem.split("_")[-1])
    rows: list[pd.DataFrame] = []
    for eval_seed in eval_seeds:
        frame = evaluate_policy(
            policy,
            eval_cfg,
            episodes=1,
            eval_seed=eval_seed,
            algorithm=variant,
        )
        frame["variant"] = variant
        frame["training_seed"] = int(raw_seed)
        frame["global_step"] = step
        frame["eval_seed"] = eval_seed
        frame["checkpoint"] = str(checkpoint.resolve())
        frame["checkpoint_sha256"] = _sha256(checkpoint)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _evaluate_checkpoints(
    out: Path,
    config: str,
    eval_seeds: list[int],
    eval_steps: int,
    workers: int,
) -> pd.DataFrame:
    checkpoints = sorted((out / "runs").glob("*_seed*/checkpoints/step_*.pt"))
    payloads = [(str(path), config, eval_seeds, eval_steps) for path in checkpoints]
    frames: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as pool:
        for frame in pool.map(_evaluate_one_checkpoint, payloads):
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _summarise(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["weighted_avg_aoi", "mean_aoi", "aoi_p95", "peak_aoi", "success_rate"]
    per_seed = (
        episodes.groupby(["variant", "training_seed", "global_step"], as_index=False)[metrics]
        .mean()
    )
    summary = (
        per_seed.groupby(["variant", "global_step"], as_index=False)
        .agg(
            mean_aoi=("weighted_avg_aoi", "mean"),
            mean_per_user_aoi=("mean_aoi", "mean"),
            seed_std=("weighted_avg_aoi", "std"),
            seed_std_per_user=("mean_aoi", "std"),
            seed_count=("training_seed", "nunique"),
            mean_p95=("aoi_p95", "mean"),
            mean_peak=("peak_aoi", "mean"),
            success_rate=("success_rate", "mean"),
        )
    )
    return per_seed, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/thesis_rician_alignment_20deg_csit02.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--checkpoint-updates", type=int, default=8)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[970000, 970001, 970002])
    parser.add_argument("--eval-steps", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.steps <= 0 or args.checkpoint_updates <= 0 or args.eval_steps <= 0:
        parser.error("steps, checkpoint-updates, and eval-steps must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.eval_seeds)) != len(args.eval_seeds):
        parser.error("training and evaluation seeds must not contain duplicates")

    jobs = []
    for variant in args.variants:
        overrides = [*args.set, *VARIANTS[variant]]
        for seed in args.seeds:
            jobs.append((
                str(Path(args.config).resolve()), str(out), variant, overrides,
                seed, args.steps, args.checkpoint_updates, args.threads,
            ))

    manifest = {
        "status": "training",
        "args": vars(args),
        "variants": {name: [*args.set, *VARIANTS[name]] for name in args.variants},
        "evaluation": (
            "deterministic policy; identical held-out channel/CSIT/ACK seeds at every checkpoint"
        ),
        "interpretation": "critic diagnostic and convergence audit; not a broad hyperparameter search",
        "source_sha256": {str(Path(__file__).relative_to(ROOT)): _sha256(Path(__file__))},
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    trained: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = [pool.submit(_train_one, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            trained.append(result)
            print(f"TRAINED {result['variant']} seed={result['seed']}", flush=True)

    manifest["status"] = "evaluating"
    manifest["training_results"] = trained
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    episodes = _evaluate_checkpoints(
        out, args.config, args.eval_seeds, args.eval_steps, args.workers
    )
    episodes.to_csv(out / "checkpoint_episodes.csv", index=False)
    per_seed, summary = _summarise(episodes)
    per_seed.to_csv(out / "checkpoint_training_seed_summary.csv", index=False)
    summary.to_csv(out / "checkpoint_summary.csv", index=False)
    manifest["status"] = "complete"
    manifest["checkpoint_count"] = int(episodes.checkpoint.nunique())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
