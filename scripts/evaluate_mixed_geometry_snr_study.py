#!/usr/bin/env python3
"""Evaluate one mixed-geometry/SNR PPO family on seen and unseen regimes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsma_aoi.agents.ppo import PPOPolicy  # noqa: E402
from rsma_aoi.baselines.cem_allocation import CEMAllocationPolicy  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402
from run_final_evaluation import evaluate_trajectory  # noqa: E402


SEEN = tuple((g, s) for g in (5, 20, 60) for s in (5.0, 10.0, 15.0))
UNSEEN = tuple((g, s) for g in (10, 40) for s in (7.5, 12.5))


def _angles(separation: int) -> list[float]:
    return [-separation / 2.0, separation / 2.0]


def _fixed_cfg(base: EnvConfig, geometry: int, snr_db: float, steps: int) -> EnvConfig:
    return replace(
        base,
        user_angles_deg=_angles(geometry),
        user_angle_sets_deg=None,
        snr_db=float(snr_db),
        snr_db_values=None,
        episode_length=int(steps),
    )


def _checkpoint(method: str, seed: int) -> Path:
    return ROOT / (
        "results/rician_ppo_mixed_geometry_snr_csit02_262k_v1/runs/"
        f"{method}_seed{seed}/checkpoint.pt"
    )


def _ppo_condition(job: dict) -> list[dict]:
    base = EnvConfig.from_yaml(ROOT / job["config"])
    cfg = _fixed_cfg(base, job["geometry_deg"], job["snr_db"], job["steps"])
    protocol = {"burn_in": job["burn_in"]}
    rows = []
    for method in ("rsma_joint_gate", "sdma_power"):
        for training_seed in job["training_seeds"]:
            policy, _, _ = PPOPolicy.from_checkpoint(
                _checkpoint(method, training_seed), deterministic=True, device="cpu"
            )
            for eval_seed in job["eval_seeds"]:
                row, _ = evaluate_trajectory(
                    cfg, protocol, method, training_seed, eval_seed, policy
                )
                row.update(
                    geometry_deg=job["geometry_deg"], snr_db=job["snr_db"],
                    split=job["split"], horizon=job["steps"], burn_in=job["burn_in"],
                )
                rows.append(row)
    return rows


def _cem_condition(job: dict) -> list[dict]:
    base = EnvConfig.from_yaml(ROOT / job["config"])
    cfg = _fixed_cfg(base, job["geometry_deg"], job["snr_db"], job["steps"])
    protocol = {"burn_in": job["burn_in"]}
    methods = []
    for method in ("rsma_joint_gate", "sdma_power"):
        for training_seed in job["training_seeds"]:
            policy, _, _ = PPOPolicy.from_checkpoint(
                _checkpoint(method, training_seed), deterministic=True, device="cpu"
            )
            methods.append((method, training_seed, policy))
    methods.extend([
        ("cem_rsma_raw_960", -1, CEMAllocationPolicy(cfg, "rsma_raw", iterations=5, population=192)),
        ("cem_sdma_960", -1, CEMAllocationPolicy(cfg, "sdma", iterations=5, population=192)),
        ("cem_rsma_safeguarded_1920", -1,
         CEMAllocationPolicy(cfg, "rsma_safeguarded", iterations=5, population=192)),
    ])
    rows = []
    for label, training_seed, policy in methods:
        for eval_seed in job["eval_seeds"]:
            row, _ = evaluate_trajectory(
                cfg, protocol, label, training_seed, eval_seed, policy
            )
            row.update(
                geometry_deg=job["geometry_deg"], snr_db=job["snr_db"], split="seen",
                horizon=job["steps"], burn_in=job["burn_in"],
            )
            rows.append(row)
    return rows


def _condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(
        ["split", "geometry_deg", "snr_db", "algorithm"], as_index=False
    ).agg(
        mean_aoi=("mean_aoi", "mean"),
        std_aoi=("mean_aoi", "std"),
        success_rate=("success_rate", "mean"),
        common_on_rate=("common_on_rate", "mean"),
        mean_common_power=("mean_common_power", "mean"),
        mean_rho=("mean_rho", "mean"),
        samples=("mean_aoi", "size"),
    )


def _macro(frame: pd.DataFrame) -> pd.DataFrame:
    # One mean per training seed (learned) or evaluation seed (numerical), with
    # every condition equally weighted before uncertainty is taken.
    key = np.where(frame.training_seed >= 0, frame.training_seed, frame.eval_seed)
    tagged = frame.assign(uncertainty_id=key)
    units = tagged.groupby(["split", "algorithm", "uncertainty_id"], as_index=False).agg(
        mean_aoi=("mean_aoi", "mean"), success_rate=("success_rate", "mean"),
        common_on_rate=("common_on_rate", "mean"),
    )
    return units.groupby(["split", "algorithm"], as_index=False).agg(
        mean_aoi=("mean_aoi", "mean"), std_aoi=("mean_aoi", "std"),
        success_rate=("success_rate", "mean"), common_on_rate=("common_on_rate", "mean"),
        uncertainty_units=("uncertainty_id", "size"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/thesis_rician_mixed_geometry_snr_csit02.yaml"
    )
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[970000, 970001, 970002, 970003, 970004, 970005])
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--burn-in", type=int, default=1024)
    parser.add_argument("--cem-seeds", type=int, nargs="+", default=[970000, 970001, 970002])
    parser.add_argument("--cem-steps", type=int, default=2048)
    parser.add_argument("--cem-burn-in", type=int, default=512)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--out-dir", default="results/thesis_mixed_geometry_snr_v1")
    args = parser.parse_args(argv)

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    for method in ("rsma_joint_gate", "sdma_power"):
        for seed in args.training_seeds:
            if not _checkpoint(method, seed).is_file():
                raise FileNotFoundError(_checkpoint(method, seed))

    ppo_jobs = [
        dict(config=args.config, geometry_deg=g, snr_db=s, split=split,
             training_seeds=args.training_seeds, eval_seeds=args.eval_seeds,
             steps=args.steps, burn_in=args.burn_in)
        for split, grid in (("seen", SEEN), ("unseen", UNSEEN)) for g, s in grid
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(ppo_jobs))) as pool:
        ppo_rows = [row for part in pool.map(_ppo_condition, ppo_jobs) for row in part]
    ppo = pd.DataFrame(ppo_rows)
    ppo.to_csv(out / "ppo_episodes.csv", index=False)
    _condition_summary(ppo).to_csv(out / "ppo_condition_summary.csv", index=False)
    _macro(ppo).to_csv(out / "ppo_macro_summary.csv", index=False)

    cem_jobs = [
        dict(config=args.config, geometry_deg=g, snr_db=s,
             training_seeds=args.training_seeds, eval_seeds=args.cem_seeds,
             steps=args.cem_steps, burn_in=args.cem_burn_in)
        for g, s in SEEN
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(cem_jobs))) as pool:
        cem_rows = [row for part in pool.map(_cem_condition, cem_jobs) for row in part]
    cem = pd.DataFrame(cem_rows)
    cem.to_csv(out / "cem_matched_episodes.csv", index=False)
    _condition_summary(cem).to_csv(out / "cem_condition_summary.csv", index=False)
    _macro(cem).to_csv(out / "cem_macro_summary.csv", index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "training_seeds": args.training_seeds,
        "ppo_eval_seeds": args.eval_seeds,
        "ppo_horizon": args.steps,
        "ppo_burn_in": args.burn_in,
        "cem_eval_seeds": args.cem_seeds,
        "cem_horizon": args.cem_steps,
        "cem_burn_in": args.cem_burn_in,
        "seen_conditions": list(SEEN),
        "unseen_conditions": list(UNSEEN),
        "caveat": "CEM is a finite-budget one-step numerical reference",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("PPO MACRO")
    print(_macro(ppo).to_string(index=False))
    print("CEM MATCHED MACRO")
    print(_macro(cem).to_string(index=False))
    print(f"COMPLETE: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
