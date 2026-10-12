#!/usr/bin/env python3
"""Create thesis-facing comparisons for the canonical K=3 direct-action run."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _comparison(candidate, reference, candidate_name, reference_name, training_seed):
    aligned = pd.concat(
        [candidate.rename("candidate"), reference.rename("reference")], axis=1
    ).dropna()
    delta = aligned.candidate - aligned.reference
    return {
        "candidate": candidate_name,
        "reference": reference_name,
        "training_seed": training_seed,
        "candidate_mean_aoi": float(aligned.candidate.mean()),
        "reference_mean_aoi": float(aligned.reference.mean()),
        "mean_delta_aoi": float(delta.mean()),
        "gain_pct": float(
            100.0 * (aligned.reference.mean() - aligned.candidate.mean())
            / aligned.reference.mean()
        ),
        "wins": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
        "paired_trajectories": int(len(delta)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir", default="results/k3_nt2_all3_direct_cem_matched_v4"
    )
    args = parser.parse_args()
    out = Path(args.result_dir)
    episodes = pd.read_csv(out / "episodes.csv")
    users = pd.read_csv(out / "users.csv")

    # Explicit helper kept local so every comparison is paired by eval_seed.
    def by_eval(algorithm: str, training_seed: int | None = None) -> pd.Series:
        frame = episodes[episodes.algorithm == algorithm]
        if training_seed is not None:
            frame = frame[frame.training_seed == training_seed]
        return frame.groupby("eval_seed").mean_aoi.mean()

    sdma = by_eval("cem_sdma_960")
    rows = []
    for algorithm in ("cem_rsma_raw_960", "cem_rsma_safeguarded_1920"):
        rows.append(_comparison(by_eval(algorithm), sdma, algorithm, "cem_sdma_960", -1))
    for seed in sorted(episodes.loc[episodes.algorithm == "ppo_direct", "training_seed"].unique()):
        seed = int(seed)
        ppo = by_eval("ppo_direct", seed)
        warm = by_eval("objective_warm_direct", seed)
        rows.append(_comparison(ppo, sdma, "ppo_direct", "cem_sdma_960", seed))
        rows.append(
            _comparison(ppo, warm, "ppo_direct", "objective_warm_direct", seed)
        )
    rows.append(
        _comparison(
            by_eval("ppo_direct"), sdma,
            "ppo_direct_seed_average", "cem_sdma_960", "all",
        )
    )
    rows.append(
        _comparison(
            by_eval("ppo_direct"), by_eval("objective_warm_direct"),
            "ppo_direct_seed_average", "objective_warm_direct_seed_average", "all",
        )
    )
    pd.DataFrame(rows).to_csv(out / "key_comparisons.csv", index=False)

    action = episodes.groupby(["algorithm", "training_seed"], as_index=False).agg(
        mean_aoi=("mean_aoi", "mean"),
        std_eval_aoi=("mean_aoi", "std"),
        aoi_p95=("aoi_p95", "mean"),
        aoi_p99=("aoi_p99", "mean"),
        success_rate=("success_rate", "mean"),
        alpha=("mean_common_power", "mean"),
        common_on_rate=("common_on_rate", "mean"),
        phi_mean=("mean_common_fraction", "mean"),
        common_bler=("mean_common_bler", "mean"),
        private_bler=("mean_private_bler", "mean"),
        latency_ms=("decision_latency_ms", "mean"),
        boundary_selection_rate=("boundary_selection_rate", "mean"),
    )
    action.to_csv(out / "action_summary.csv", index=False)
    per_user = users.groupby(
        ["algorithm", "training_seed", "user"], as_index=False
    ).agg(
        phi=("mean_common_fraction", "mean"),
        q=("mean_private_power_share", "mean"),
        mean_aoi=("mean_aoi", "mean"),
        success_rate=("success_rate", "mean"),
    )
    per_user.to_csv(out / "per_user_action_summary.csv", index=False)
    q_sums = per_user.groupby(["algorithm", "training_seed"]).q.sum()
    if not ((q_sums - 1.0).abs() < 1e-9).all():
        raise AssertionError("mean q_k values do not sum to one")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
