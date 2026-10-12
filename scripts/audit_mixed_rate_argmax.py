#!/usr/bin/env python3
"""Test whether mixed-geometry PPO needs continuous common-rate proportions."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_mixed_geometry_policy import ANGLE_SETS  # noqa: E402
from audit_two_mode_policy import DistilledGatePolicy  # noqa: E402
from rsma_aoi.agents.ppo import PPOPolicy  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402
from run_final_evaluation import evaluate_trajectory  # noqa: E402

EVAL_SEEDS = [953000, 953001, 953002, 953003, 953004, 953005]


def _worker(seed: int):
    cfg_mixed = EnvConfig.from_yaml(
        ROOT / "configs/thesis_rician_mixed_geometry_csit02.yaml"
    )
    checkpoint = (
        ROOT
        / "results/rician_ppo_mixed_geometry_csit02_262k_v1/runs"
        / f"rsma_joint_gate_seed{seed}/checkpoint.pt"
    )
    policy, _, _ = PPOPolicy.from_checkpoint(checkpoint, deterministic=True, device="cpu")
    constants = json.loads(
        (ROOT / "results/thesis_mixed_geometry_audit_v1/global_calibration_constants.json")
        .read_text(encoding="utf-8")
    )[seed]
    candidates = {
        "rate_argmax": DistilledGatePolicy(policy, constants, "rate_argmax"),
        "fixed_magnitude_rate_argmax": DistilledGatePolicy(
            policy, constants, "fixed_magnitude_rate_argmax"
        ),
    }
    rows = []
    for angle, angles in ANGLE_SETS.items():
        cfg = replace(
            cfg_mixed,
            user_angles_deg=list(angles),
            user_angle_sets_deg=None,
            episode_length=8192,
        )
        protocol = {"eval_seeds": EVAL_SEEDS, "burn_in": 2048}
        for label, candidate in candidates.items():
            for eval_seed in EVAL_SEEDS:
                episode, _ = evaluate_trajectory(
                    cfg, protocol, label, seed, eval_seed, candidate
                )
                episode["angle_deg"] = angle
                rows.append(episode)
    return rows


def main() -> int:
    out = ROOT / "results/thesis_mixed_rate_argmax_v1"
    out.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=5) as pool:
        candidate_rows = [row for part in pool.map(_worker, range(5)) for row in part]
    candidate = pd.DataFrame(candidate_rows)
    reference = pd.read_csv(ROOT / "results/thesis_mixed_geometry_audit_v1/episodes.csv")
    reference = reference[reference.algorithm == "ppo_original"]
    candidate.to_csv(out / "episodes.csv", index=False)

    cand_seed = candidate.groupby(
        ["angle_deg", "algorithm", "training_seed"]
    ).mean(numeric_only=True)
    ref_seed = reference.groupby(
        ["angle_deg", "training_seed"]
    ).mean(numeric_only=True)
    contrasts = []
    for angle in ANGLE_SETS:
        ref = ref_seed.loc[angle, "mean_aoi"]
        for label in ("rate_argmax", "fixed_magnitude_rate_argmax"):
            cand = cand_seed.loc[(angle, label), "mean_aoi"]
            delta = cand - ref
            contrasts.append({
                "angle_deg": angle,
                "candidate": label,
                "candidate_mean_aoi": cand.mean(),
                "reference_mean_aoi": ref.mean(),
                "mean_paired_delta": delta.mean(),
                "mean_paired_percent": 100.0 * delta.mean() / ref.mean(),
                "candidate_wins": int((delta < 0).sum()),
                "reference_wins": int((delta > 0).sum()),
            })
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(out / "paired_contrasts.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "reference": "results/thesis_mixed_geometry_audit_v1/episodes.csv",
        "meaning": "replace common-rate simplex by the actor's per-slot argmax",
        "eval_seeds": EVAL_SEEDS,
        "episode_steps": 8192,
        "burn_in": 2048,
    }, indent=2), encoding="utf-8")
    print(contrasts.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
