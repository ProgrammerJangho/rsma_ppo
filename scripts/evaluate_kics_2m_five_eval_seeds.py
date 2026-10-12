#!/usr/bin/env python3
"""Evaluate the final seed-0 2.10M policy on five independent test seeds.

Every learned/rule baseline sees exactly the same 5 evaluation repeats x 6
user-angle permutations. Confidence intervals use the five repeat means, not
the 30 trajectories as independent samples.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_mixed_phy_sinr_ppo import _job as evaluate_ppo  # noqa: E402
from scripts.evaluate_thesis_rule_baselines import evaluate as evaluate_rule  # noqa: E402


CONFIG = "configs/thesis_rician_k3_nt2_mixed_phy_sinr.yaml"
CHECKPOINTS = {
    "rsma_ppo_2m": "runs/matched_objective_rsma_ppo_k3_20deg_seed0_continue_to_2097k_v1/checkpoint.pt",
    "rsma_ppo_524k_diagnostic": "runs/matched_objective_rsma_ppo_k3_20deg_from_stage1_seed0_524k_v1/checkpoint.pt",
    "sdma_ppo": "runs/q_only_ppo_k3_20deg_from_stage1_seed0_524k_v1/checkpoint.pt",
}


def _ci95(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) < 2:
        return float("nan")
    return float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def _dispatch(job: dict) -> dict:
    if job["family"] == "ppo":
        result = evaluate_ppo(job)
    else:
        result = evaluate_rule(job)
        result["alpha_mean"] = float(result["alpha"])
        result["alpha_std"] = 0.0
    result["label"] = job["label"]
    result["eval_repeat"] = int(job["eval_repeat"])
    return result


def main() -> int:
    out = ROOT / "results/kics_2m_five_eval_seeds_v1"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    for checkpoint in CHECKPOINTS.values():
        if not (ROOT / checkpoint).is_file():
            raise FileNotFoundError(checkpoint)

    jobs: list[dict] = []
    angle_sets = list(permutations((-10.0, 0.0, 10.0)))
    for repeat in range(5):
        for permutation, angles in enumerate(angle_sets):
            eval_seed = 990278 + 10000 * repeat + 100 * permutation
            common = {
                "config": CONFIG,
                "geometry_deg": 20,
                "angles": angles,
                "permutation": permutation,
                "snr_db": 15.0,
                "blocklength": 128,
                "steps": 768,
                "burn_in": 256,
                "eval_seed": eval_seed,
                "eval_repeat": repeat,
            }
            for label, checkpoint in CHECKPOINTS.items():
                jobs.append({
                    **common, "family": "ppo", "kind": "ppo",
                    "label": label, "checkpoint": checkpoint,
                })
            jobs.append({
                **common, "family": "rule", "kind": "oldest_common_fixed_alpha",
                "label": "fixed_rule_rsma", "phase": "test", "alpha": 0.70,
            })
            jobs.append({
                **common, "family": "rule", "kind": "max_aoi_single_user",
                "label": "max_aoi_single_user", "phase": "test", "alpha": 0.0,
            })

    with ProcessPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(_dispatch, jobs))
    trajectories = pd.DataFrame(rows)
    trajectories.to_csv(out / "target_trajectories.csv", index=False)

    repeat_means = trajectories.groupby(["label", "eval_repeat"], as_index=False).agg(
        mean_aoi=("mean_aoi", "mean"),
        p95_aoi=("p95_aoi", "mean"),
        success_probability=("success_probability", "mean"),
        alpha_mean=("alpha_mean", "mean"),
        common_payload_mean=("common_payload_mean", "mean"),
    )
    repeat_means.to_csv(out / "evaluation_seed_summary.csv", index=False)

    aggregate = repeat_means.groupby("label", as_index=False).agg(
        evaluation_seeds=("eval_repeat", "count"),
        mean_aoi=("mean_aoi", "mean"),
        std_over_evaluation_seeds=("mean_aoi", "std"),
        mean_p95_aoi=("p95_aoi", "mean"),
        success_probability=("success_probability", "mean"),
        alpha_mean=("alpha_mean", "mean"),
        common_payload_mean=("common_payload_mean", "mean"),
    )
    aggregate["aoi_ci95_halfwidth"] = aggregate["label"].map(
        lambda label: _ci95(repeat_means.loc[repeat_means.label == label, "mean_aoi"])
    )
    aggregate.to_csv(out / "target_aggregate.csv", index=False)

    wide = repeat_means.pivot(index="eval_repeat", columns="label", values="mean_aoi")
    for baseline in ("sdma_ppo", "fixed_rule_rsma", "max_aoi_single_user"):
        wide[f"rsma_2m_gain_vs_{baseline}_pct"] = 100.0 * (
            wide[baseline] - wide["rsma_ppo_2m"]
        ) / wide[baseline]
    wide["rsma_2m_gain_vs_524k_pct"] = 100.0 * (
        wide["rsma_ppo_524k_diagnostic"] - wide["rsma_ppo_2m"]
    ) / wide["rsma_ppo_524k_diagnostic"]
    wide.reset_index().to_csv(out / "paired_evaluation_seed_comparison.csv", index=False)

    paired_rows = []
    for column in [c for c in wide.columns if c.endswith("_pct")]:
        paired_rows.append({
            "comparison": column,
            "mean_gain_pct": float(wide[column].mean()),
            "ci95_halfwidth_pct": _ci95(wide[column]),
            "evaluation_seed_wins": int((wide[column] > 0).sum()),
            "evaluation_seeds": int(len(wide)),
        })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out / "paired_summary.csv", index=False)

    print("Aggregate over five evaluation-seed means:\n", aggregate.to_string(index=False))
    print("\nPaired gains:\n", paired.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
