#!/usr/bin/env python3
"""Controlled multi-seed pilot for RSMA allocation and its SDMA boundary.

Training and evaluation seeds are separate. Fixed references are evaluated once,
not duplicated as if they were independent trained policies. Both sampled and
deterministic PPO are reported; action-path assertions run during evaluation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from audit_allocation import audit_policy
from rsma_aoi.agents.ppo import PPOPolicy
from rsma_aoi.utils.config import (
    apply_overrides_from_strings,
    dump_config_snapshot,
    load_run_configs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/thesis_ablation.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-steps", type=int, default=2048)
    parser.add_argument("--burn-in", type=int, default=512)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=["power_only", "split_only", "joint"],
                        choices=["power_only", "split_only", "joint", "joint_gate", "sdma_power"])
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="additional PPO overrides applied to every run")
    parser.add_argument("--set-env", nargs="*", default=[], metavar="KEY=VALUE",
                        help="environment overrides applied to training and evaluation")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    cfg, agent = load_run_configs(args.config)
    if args.set_env:
        cfg = apply_overrides_from_strings(cfg, args.set_env)
    if (cfg.scheduling_enabled or cfg.allocator_mode != "none" or cfg.aoi_cap_enabled
            or cfg.multiple_access != "rsma" or cfg.rate_split_mode != "simplex_scaled"
            or cfg.power_scale_action or cfg.ack_mode != "sampled" or cfg.link_model != "fbl"):
        parser.error("requires all-user, uncapped, direct simplex_scaled FBL-RSMA")
    if (args.steps < agent.batch_size or args.steps % agent.batch_size
            or args.burn_in < 0 or args.eval_steps <= args.burn_in
            or args.eval_episodes < 1 or args.workers < 1 or len(set(args.seeds)) != len(args.seeds)):
        parser.error("invalid budget, evaluation lengths, workers, or repeated training seeds")
    torch.set_num_threads(1)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    modes = list(dict.fromkeys(args.modes))

    def mode_overrides(mode: str) -> list[str]:
        if mode == "joint_gate":
            return [*args.set, "allocation_control=joint", "common_gate_enabled=true"]
        if mode == "sdma_power":
            return [*args.set, "allocation_control=sdma_power", "common_gate_enabled=false"]
        return [*args.set, f"allocation_control={mode}", "common_gate_enabled=false"]

    effective_ppo = {
        mode: apply_overrides_from_strings(agent, mode_overrides(mode)).to_dict()
        for mode in modes
    }
    eval_seeds = list(range(910000, 910000 + args.eval_episodes))
    sources = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")) + [Path(args.config).resolve()]
    manifest = {
        "args": vars(args), "eval_seeds": eval_seeds,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
        "status": "running", "interpretation": "pilot, not converged/optimality evidence",
        "fixed_reference": {"common_power_fraction": 0.25, "rho": 0.3},
        "evaluation": "same channel/ACK seeds; fresh per-episode policy RNG; frozen observation normalization",
        "python": sys.version, "torch": torch.__version__,
        "effective_ppo_by_mode": effective_ppo,
    }
    def save_manifest():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save_manifest()
    dump_config_snapshot(out / "config_snapshot.yaml", env=cfg, ppo=agent)

    def train(mode, seed):
        run = out / "runs" / f"{mode}_seed{seed}"
        run.mkdir(parents=True)
        command = [sys.executable, str(ROOT / "scripts/train_ppo.py"),
                   "--env-config", str(Path(args.config).resolve()), "--total-timesteps", str(args.steps),
                   "--seed", str(seed), "--run-dir", str(run), "--eval-episodes", "0",
                   "--set", *mode_overrides(mode)]
        if args.set_env:
            command.extend(["--set-env", *args.set_env])
        with (run / "console.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        return mode, seed, run / "checkpoint.pt"

    checkpoints = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(train, mode, seed) for mode in modes for seed in args.seeds]
        for future in as_completed(futures):
            item = future.result()
            checkpoints.append(item)
            print(f"TRAINED {item[0]} seed={item[1]} steps={args.steps}", flush=True)

    eval_cfg = replace(cfg, episode_length=args.eval_steps)
    frames, episode_rows, user_rows = [], [], []

    def evaluate(label, policy, training_seed, execution):
        for eval_seed in eval_seeds:
            if policy is not None:
                policy.generator = torch.Generator(device="cpu").manual_seed(eval_seed + 1000000)
            frame = pd.DataFrame(audit_policy(policy, eval_cfg, [eval_seed], label))
            frame = frame[frame.slot > args.burn_in].copy()
            frame["training_seed"] = training_seed
            frame["execution"] = execution
            frames.append(frame)
            common_on = frame.get(
                "common_enabled",
                (frame.common_power_fraction > 1e-12) & (frame.rho > 1e-12),
            ).astype(bool)
            common = {"algorithm": label, "execution": execution, "training_seed": training_seed,
                      "eval_seed": eval_seed}
            episode_rows.append({**common, "mean_aoi": frame.aoi.mean(), "pooled_p95": frame.aoi.quantile(.95),
                                 "max_aoi": frame.aoi.max(), "success_rate": frame.ack.mean(),
                                 "mean_common_power": frame.common_power_fraction.mean(),
                                 "common_on_rate": common_on.mean(),
                                 "mean_rho": frame.rho.mean(),
                                 "mean_common_code_rate": frame.common_code_rate.mean(),
                                 "mean_common_bler": frame.common_bler.mean(),
                                 "common_bler_when_on": (
                                     frame.loc[common_on, "common_bler"].mean()
                                     if common_on.any() else np.nan
                                 ),
                                 "mean_private_bler": frame.private_bler.mean()})
            for user, group in frame.groupby("user"):
                ack_slots = group.loc[group.ack > .5, "slot"].to_numpy()
                intervals = np.diff(ack_slots)  # completed intervals only, no invented peak for censored runs
                user_rows.append({**common, "user": user, "mean_aoi": group.aoi.mean(),
                                  "p95": group.aoi.quantile(.95), "p99": group.aoi.quantile(.99),
                                  "success_rate": group.ack.mean(), "complete_intervals": len(intervals),
                                  "mean_interdelivery": intervals.mean() if len(intervals) else np.nan,
                                  "max_interdelivery": intervals.max() if len(intervals) else np.nan,
                                  "left_censored_slots": ack_slots[0] - args.burn_in - 1 if len(ack_slots) else len(group),
                                  "right_censored_slots": args.eval_steps - ack_slots[-1] if len(ack_slots) else len(group)})
        print(f"EVALUATED {label} seed={training_seed} {execution}", flush=True)

    for label in ("fixed_rsma", "sdma_equal"):
        evaluate(label, None, -1, "fixed")
    for mode, seed, path in sorted(checkpoints):
        for deterministic in (True, False):
            policy, trained_cfg, loaded = PPOPolicy.from_checkpoint(path, deterministic=deterministic)
            expected = effective_ppo[mode]
            if trained_cfg.to_dict() != cfg.to_dict() or loaded.to_dict() != {
                **expected, "total_timesteps": args.steps, "seed": seed
            }:
                raise AssertionError("checkpoint does not match controlled experiment")
            evaluate(mode, policy, seed, "deterministic" if deterministic else "sampled")
    pd.concat(frames, ignore_index=True).to_csv(out / "allocation_trace.csv.gz", index=False)
    episodes = pd.DataFrame(episode_rows)
    episodes.to_csv(out / "episodes.csv", index=False)
    pd.DataFrame(user_rows).to_csv(out / "users.csv", index=False)
    metrics = ["mean_aoi", "pooled_p95", "max_aoi", "success_rate", "mean_common_power", "common_on_rate", "mean_rho",
               "mean_common_code_rate", "mean_common_bler", "common_bler_when_on", "mean_private_bler"]
    seeds = episodes.groupby(["algorithm", "execution", "training_seed"])[metrics].mean().reset_index()
    seeds.to_csv(out / "training_seed_summary.csv", index=False)
    summary = seeds.groupby(["algorithm", "execution"]).agg(
        mean_aoi=("mean_aoi", "mean"), training_seed_std=("mean_aoi", "std"),
        training_seed_count=("training_seed", "count"), mean_episode_p95=("pooled_p95", "mean"),
        success_rate=("success_rate", "mean"), common_power=("mean_common_power", "mean"), rho=("mean_rho", "mean"),
        common_on_rate=("common_on_rate", "mean"),
        common_code_rate=("mean_common_code_rate", "mean"), common_bler=("mean_common_bler", "mean"),
        common_bler_when_on=("common_bler_when_on", "mean"),
    )
    summary.to_csv(out / "summary.csv")
    manifest["status"] = "complete"
    manifest["checkpoint_sha256"] = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for _, _, p in checkpoints}
    save_manifest()
    print(summary.to_string(), flush=True)
    print("COMPLETE: fixed references have no training-seed uncertainty; seed=-1 is a sentinel.", flush=True)


if __name__ == "__main__":
    main()
