#!/usr/bin/env python3
"""Evaluate K=3, Nt=2 all-user allocation-only actors before/after PPO.

The common-off projection preserves each actor's private-power proportions.
This gives a paired estimate of the realised common-stream contribution without
training a different SDMA policy.
"""

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
from rsma_aoi.utils.config import EnvConfig  # noqa: E402
from run_final_evaluation import evaluate_trajectory  # noqa: E402


class CommonOffProjection:
    """Set common power/payload to zero and renormalize private power."""

    def __init__(self, actor: PPOPolicy):
        self.actor = actor

    def reset(self, seed: int | None = None) -> None:
        if hasattr(self.actor, "reset"):
            self.actor.reset(seed=seed)

    def act(self, observations, states=None):
        actions = self.actor.act(observations, states)
        projected = []
        for action in actions:
            out = {key: np.array(value, copy=True) for key, value in action.items()}
            private = np.asarray(out["power_fractions"], dtype=np.float64)[1:]
            total = float(private.sum())
            if total <= 1e-12:
                private = np.full_like(private, 1.0 / private.size)
            else:
                private = private / total
            out["power_fractions"] = np.concatenate(([0.0], private))
            out["common_payload_scale"] = np.zeros_like(out["common_payload_scale"])
            projected.append(out)
        return projected


def _checkpoint(stage: str, seed: int) -> Path:
    if stage == "0step":
        return ROOT / f"runs/action_objective_actor_k3_nt2_all3_seed{seed}_v1/checkpoint.pt"
    if stage == "262k":
        return ROOT / (
            "results/rician_ppo_k3_nt2_all3_262k_v1/runs/"
            f"rsma_joint_gate_seed{seed}/checkpoint.pt"
        )
    raise ValueError(stage)


def _job(job: dict) -> dict:
    base = EnvConfig.from_yaml(ROOT / job["config"])
    cfg = replace(
        base,
        user_angles_deg=[-10.0, 0.0, 10.0],
        user_angle_sets_deg=None,
        snr_db=15.0,
        snr_db_values=None,
        scheduling_enabled=False,
        episode_length=job["steps"],
    )
    actor, _, _ = PPOPolicy.from_checkpoint(
        _checkpoint(job["stage"], job["training_seed"]),
        deterministic=True,
        device="cpu",
    )
    policy = CommonOffProjection(actor) if job["common_off"] else actor
    algorithm = f"k3_all3_{job['stage']}" + ("_common_off" if job["common_off"] else "")
    row, _ = evaluate_trajectory(
        cfg,
        {"burn_in": job["burn_in"]},
        algorithm,
        job["training_seed"],
        job["eval_seed"],
        policy,
    )
    row.update(
        geometry_span_deg=20,
        snr_db=15.0,
        horizon=job["steps"],
        burn_in=job["burn_in"],
        stage=job["stage"],
        projected_common_off=job["common_off"],
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/thesis_rician_k3_nt2_all3_ppo_csit02.yaml"
    )
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+",
        default=[980000, 980001, 980002, 980003, 980004, 980005],
    )
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--burn-in", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--out-dir", default="results/thesis_k3_nt2_all3_ppo_long_v1")
    args = parser.parse_args()

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    for stage in ("0step", "262k"):
        for seed in args.training_seeds:
            if not _checkpoint(stage, seed).is_file():
                raise FileNotFoundError(_checkpoint(stage, seed))

    jobs = [
        {
            "config": args.config,
            "stage": stage,
            "training_seed": training_seed,
            "eval_seed": eval_seed,
            "common_off": common_off,
            "steps": args.steps,
            "burn_in": args.burn_in,
        }
        for stage in ("0step", "262k")
        for training_seed in args.training_seeds
        for eval_seed in args.eval_seeds
        for common_off in (False, True)
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        rows = list(pool.map(_job, jobs))
    episodes = pd.DataFrame(rows)
    episodes.to_csv(out / "episodes.csv", index=False)

    summary = episodes.groupby("algorithm", as_index=False).agg(
        mean_aoi=("mean_aoi", "mean"),
        std_aoi=("mean_aoi", "std"),
        success_rate=("success_rate", "mean"),
        common_on_rate=("common_on_rate", "mean"),
        common_power_fraction=("mean_common_power", "mean"),
        rho=("mean_rho", "mean"),
        trajectories=("mean_aoi", "size"),
    )
    summary.to_csv(out / "summary.csv", index=False)

    keys = ["training_seed", "eval_seed"]
    wide = episodes.pivot(index=keys, columns="algorithm", values="mean_aoi").reset_index()
    paired_rows = []
    for stage in ("0step", "262k"):
        on = f"k3_all3_{stage}"
        off = f"{on}_common_off"
        delta = wide[on] - wide[off]
        paired_rows.append({
            "stage": stage,
            "mean_delta_aoi_on_minus_off": float(delta.mean()),
            "common_on_wins": int((delta < 0).sum()),
            "common_off_wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "paired_units": int(delta.size),
            "aggregate_common_gain_pct": float(
                100.0 * (episodes.loc[episodes.algorithm == off, "mean_aoi"].mean()
                         - episodes.loc[episodes.algorithm == on, "mean_aoi"].mean())
                / episodes.loc[episodes.algorithm == off, "mean_aoi"].mean()
            ),
        })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out / "paired_common_contribution.csv", index=False)

    seed_summary = episodes.groupby(["algorithm", "training_seed"], as_index=False).agg(
        mean_aoi=("mean_aoi", "mean"),
        std_over_eval_seeds=("mean_aoi", "std"),
        eval_seeds=("eval_seed", "size"),
    )
    seed_summary.to_csv(out / "training_seed_summary.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "paired fixed-condition evaluation of K=3 Nt=2 all-user allocation-only PPO",
        "config": args.config,
        "geometry_angles_deg": [-10.0, 0.0, 10.0],
        "snr_db": 15.0,
        "training_seeds": args.training_seeds,
        "eval_seeds": args.eval_seeds,
        "steps": args.steps,
        "burn_in": args.burn_in,
        "comparison": "deterministic actor and its paired common-off projection",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.sort_values("mean_aoi").to_string(index=False))
    print("\nPAIRED COMMON CONTRIBUTION")
    print(paired.to_string(index=False))
    print(f"\nCOMPLETE: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
