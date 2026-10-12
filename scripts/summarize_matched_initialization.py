#!/usr/bin/env python3
"""Summarize paired action-class, pretraining, and PPO effects."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONTRASTS = {
    "rsma_vs_sdma_zero_step": (
        "objective_pretrain_joint_gate", "objective_pretrain_sdma_power"
    ),
    "rsma_vs_sdma_after_ppo": (
        "objective_warm_joint_gate", "objective_warm_sdma_power"
    ),
    "ppo_increment_rsma": (
        "objective_warm_joint_gate", "objective_pretrain_joint_gate"
    ),
    "ppo_increment_sdma": (
        "objective_warm_sdma_power", "objective_pretrain_sdma_power"
    ),
    "pretraining_effect_rsma_final": (
        "objective_warm_joint_gate", "scratch_joint_gate"
    ),
    "pretraining_effect_sdma_final": (
        "objective_warm_sdma_power", "scratch_sdma_power"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    units = pd.read_csv(args.result_dir / "uncertainty_units.csv")
    units = units[units.uncertainty_unit == "training_seed"]
    rows = []
    for name, (candidate, reference) in CONTRASTS.items():
        left = units[units.algorithm == candidate].set_index("training_seed")
        right = units[units.algorithm == reference].set_index("training_seed")
        paired = left[["mean_aoi"]].join(
            right[["mean_aoi"]], lsuffix="_candidate", rsuffix="_reference",
            how="inner",
        )
        delta = paired.mean_aoi_candidate - paired.mean_aoi_reference
        percent = 100.0 * delta / paired.mean_aoi_reference
        rows.append({
            "contrast": name,
            "candidate": candidate,
            "reference": reference,
            "candidate_mean_aoi": float(paired.mean_aoi_candidate.mean()),
            "reference_mean_aoi": float(paired.mean_aoi_reference.mean()),
            "mean_paired_delta": float(delta.mean()),
            "paired_delta_std": float(delta.std(ddof=1)),
            "mean_paired_percent": float(percent.mean()),
            "candidate_wins": int((delta < 0).sum()),
            "paired_seeds": int(len(paired)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(args.result_dir / "matched_contrasts.csv", index=False)
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
