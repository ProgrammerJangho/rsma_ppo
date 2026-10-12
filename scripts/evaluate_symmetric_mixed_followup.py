#!/usr/bin/env python3
"""Evaluate the symmetry-aware mixed-regime actor before and after PPO.

The same learned actor is also projected onto the common-off boundary.  This
paired comparison isolates the realised contribution of the common stream
without introducing a separately trained SDMA actor.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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
from evaluate_mixed_geometry_snr_study import (  # noqa: E402
    SEEN,
    UNSEEN,
    _condition_summary,
    _fixed_cfg,
    _macro,
)
from run_final_evaluation import evaluate_trajectory  # noqa: E402


class CommonOffProjection:
    """Preserve private-power proportions while setting common power/bits to zero."""

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
            private = (np.full_like(private, 1.0 / private.size)
                       if total <= 1e-12 else private / total)
            out["power_fractions"] = np.concatenate(([0.0], private))
            out["common_payload_scale"] = np.zeros_like(out["common_payload_scale"])
            projected.append(out)
        return projected


def _checkpoint(stage: str, seed: int) -> Path:
    if stage == "0step":
        return ROOT / (
            "runs/action_objective_actor_rician_mixed_geometry_snr_"
            f"symmetric_seed{seed}_v1/checkpoint.pt"
        )
    if stage == "262k":
        return ROOT / (
            "results/rician_ppo_mixed_geometry_snr_symmetric_262k_v1/runs/"
            f"rsma_joint_gate_seed{seed}/checkpoint.pt"
        )
    raise ValueError(stage)


def _condition_job(job: dict) -> list[dict]:
    base = EnvConfig.from_yaml(ROOT / job["config"])
    cfg = _fixed_cfg(base, job["geometry_deg"], job["snr_db"], job["steps"])
    protocol = {"burn_in": job["burn_in"]}
    rows: list[dict] = []
    for stage in ("0step", "262k"):
        for seed in job["training_seeds"]:
            for projected in (False, True):
                # Load separately so reset/state in one arm cannot affect its pair.
                actor, _, _ = PPOPolicy.from_checkpoint(
                    _checkpoint(stage, seed), deterministic=True, device="cpu"
                )
                policy = CommonOffProjection(actor) if projected else actor
                label = f"symmetric_{stage}" + ("_common_off" if projected else "")
                for eval_seed in job["eval_seeds"]:
                    row, _ = evaluate_trajectory(
                        cfg, protocol, label, seed, eval_seed, policy
                    )
                    row.update(
                        geometry_deg=job["geometry_deg"],
                        snr_db=job["snr_db"],
                        split=job["split"],
                        horizon=job["steps"],
                        burn_in=job["burn_in"],
                    )
                    rows.append(row)
    return rows


def _paired_table(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["split", "geometry_deg", "snr_db", "training_seed", "eval_seed"]
    wide = frame.pivot(index=keys, columns="algorithm", values="mean_aoi").reset_index()
    for stage in ("0step", "262k"):
        rsma = f"symmetric_{stage}"
        off = f"{rsma}_common_off"
        wide[f"{stage}_common_gain_pct"] = 100.0 * (wide[off] - wide[rsma]) / wide[off]
    return wide


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="configs/thesis_rician_mixed_geometry_snr_symmetric_csit02.yaml",
    )
    p.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--eval-seeds", type=int, nargs="+",
        default=[970000, 970001, 970002, 970003, 970004, 970005],
    )
    p.add_argument("--steps", type=int, default=4096)
    p.add_argument("--burn-in", type=int, default=1024)
    p.add_argument("--workers", type=int, default=9)
    p.add_argument(
        "--splits", nargs="+", choices=["seen", "unseen"],
        default=["seen", "unseen"],
    )
    p.add_argument(
        "--out-dir", default="results/thesis_symmetric_mixed_geometry_snr_v1"
    )
    args = p.parse_args()

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    for stage in ("0step", "262k"):
        for seed in args.training_seeds:
            if not _checkpoint(stage, seed).is_file():
                raise FileNotFoundError(_checkpoint(stage, seed))

    grids = {"seen": SEEN, "unseen": UNSEEN}
    jobs = [
        dict(
            config=args.config, geometry_deg=g, snr_db=s, split=split,
            training_seeds=args.training_seeds, eval_seeds=args.eval_seeds,
            steps=args.steps, burn_in=args.burn_in,
        )
        for split in args.splits
        for grid in (grids[split],)
        for g, s in grid
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        rows = [row for part in pool.map(_condition_job, jobs) for row in part]
    episodes = pd.DataFrame(rows)
    episodes.to_csv(out / "episodes.csv", index=False)
    conditions = _condition_summary(episodes)
    macro = _macro(episodes)
    paired = _paired_table(episodes)
    conditions.to_csv(out / "condition_summary.csv", index=False)
    macro.to_csv(out / "macro_summary.csv", index=False)
    paired.to_csv(out / "paired_common_contribution.csv", index=False)

    # Ratio of aggregate means is the stable headline estimand.  Averaging
    # per-trajectory percentages can be distorted by small denominators.
    by_split = episodes.groupby(["split", "algorithm"], as_index=False).agg(
        mean_aoi=("mean_aoi", "mean")
    ).pivot(index="split", columns="algorithm", values="mean_aoi")
    paired_macro = by_split.reset_index()[["split"]].copy()
    for stage in ("0step", "262k"):
        rsma = f"symmetric_{stage}"
        off = f"{rsma}_common_off"
        paired_macro[f"{stage}_common_gain_pct"] = (
            100.0 * (by_split[off] - by_split[rsma]) / by_split[off]
        ).to_numpy()
    paired_macro["paired_units"] = paired.groupby("split").size().to_numpy()
    paired_macro.to_csv(out / "paired_macro_summary.csv", index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "training_seeds": args.training_seeds,
        "eval_seeds": args.eval_seeds,
        "horizon": args.steps,
        "burn_in": args.burn_in,
        "splits": args.splits,
        "seen_conditions": list(SEEN) if "seen" in args.splits else [],
        "unseen_conditions": list(UNSEEN) if "unseen" in args.splits else [],
        "comparison": "actor versus paired common-off projection; lower AoI is better",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(macro.to_string(index=False))
    print("\nPAIRED COMMON CONTRIBUTION")
    print(paired_macro.to_string(index=False))
    print(f"\nCOMPLETE: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
