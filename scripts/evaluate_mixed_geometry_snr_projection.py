#!/usr/bin/env python3
"""Project the mixed RSMA actor onto the SDMA boundary and re-evaluate it."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsma_aoi.agents.ppo import PPOPolicy  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402
from evaluate_mixed_geometry_snr_study import (  # noqa: E402
    SEEN,
    UNSEEN,
    _checkpoint,
    _condition_summary,
    _fixed_cfg,
    _macro,
)
from run_final_evaluation import evaluate_trajectory  # noqa: E402


class CommonOffProjection:
    """Keep the actor's private-power ratio while forcing the common stream off."""

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
            power = np.asarray(out["power_fractions"], dtype=np.float64)
            private = power[1:]
            total = float(private.sum())
            if total <= 1e-12:
                private = np.full_like(private, 1.0 / private.size)
            else:
                private = private / total
            out["power_fractions"] = np.concatenate(([0.0], private))
            out["common_payload_scale"] = np.zeros_like(out["common_payload_scale"])
            projected.append(out)
        return projected


def _job(job: dict) -> list[dict]:
    base = EnvConfig.from_yaml(ROOT / job["config"])
    cfg = _fixed_cfg(base, job["geometry_deg"], job["snr_db"], job["steps"])
    rows = []
    for seed in job["training_seeds"]:
        actor, _, _ = PPOPolicy.from_checkpoint(
            _checkpoint("rsma_joint_gate", seed), deterministic=True, device="cpu"
        )
        policy = CommonOffProjection(actor)
        for eval_seed in job["eval_seeds"]:
            row, _ = evaluate_trajectory(
                cfg, {"burn_in": job["burn_in"]}, "rsma_projected_common_off",
                seed, eval_seed, policy,
            )
            row.update(
                geometry_deg=job["geometry_deg"], snr_db=job["snr_db"],
                split=job["split"], horizon=job["steps"], burn_in=job["burn_in"],
            )
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/thesis_rician_mixed_geometry_snr_csit02.yaml"
    )
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+",
        default=[970000, 970001, 970002, 970003, 970004, 970005],
    )
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--burn-in", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--result-dir", default="results/thesis_mixed_geometry_snr_v1")
    args = parser.parse_args()

    jobs = [
        dict(
            config=args.config, geometry_deg=g, snr_db=s, split=split,
            training_seeds=args.training_seeds, eval_seeds=args.eval_seeds,
            steps=args.steps, burn_in=args.burn_in,
        )
        for split, grid in (("seen", SEEN), ("unseen", UNSEEN))
        for g, s in grid
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        rows = [row for part in pool.map(_job, jobs) for row in part]
    projected = pd.DataFrame(rows)
    out = ROOT / args.result_dir
    projected.to_csv(out / "common_off_projection_episodes.csv", index=False)
    _condition_summary(projected).to_csv(
        out / "common_off_projection_condition_summary.csv", index=False
    )
    _macro(projected).to_csv(out / "common_off_projection_macro_summary.csv", index=False)

    reference = pd.read_csv(out / "ppo_episodes.csv")
    combined = pd.concat([reference, projected], ignore_index=True)
    summary = _condition_summary(combined)
    wide = summary.pivot_table(
        index=["split", "geometry_deg", "snr_db"],
        columns="algorithm", values="mean_aoi",
    )
    wide["common_stream_gain_pct"] = 100.0 * (
        wide["rsma_projected_common_off"] - wide["rsma_joint_gate"]
    ) / wide["rsma_projected_common_off"]
    wide["projected_gain_over_sdma_pct"] = 100.0 * (
        wide["sdma_power"] - wide["rsma_projected_common_off"]
    ) / wide["sdma_power"]
    wide.reset_index().to_csv(out / "common_off_projection_comparison.csv", index=False)
    print(_macro(projected).to_string(index=False))
    print("\n", wide.reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
