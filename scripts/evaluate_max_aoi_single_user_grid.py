#!/usr/bin/env python3
"""Evaluate the max-AoI single-user MRT baseline on the frozen 36-regime grid."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_mixed_phy_sinr_ppo import GEOMETRIES  # noqa: E402
from scripts.evaluate_thesis_rule_baselines import evaluate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/thesis_rician_k3_nt2_mixed_phy_sinr.yaml")
    ap.add_argument("--ppo-trajectories", default="results/ppo_k3_mixed_phy_seed_expansion_grid_v1/trajectories.csv")
    ap.add_argument("--steps", type=int, default=768)
    ap.add_argument("--burn-in", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    jobs = []
    for geometry, values in GEOMETRIES.items():
        for perm, angles in enumerate(permutations(values)):
            for snr in (5.0, 10.0, 15.0, 20.0):
                for n in (64, 128, 256):
                    jobs.append({
                        "kind": "max_aoi_single_user", "phase": "test", "alpha": 0.0,
                        "permutation": perm, "angles": angles, "geometry_deg": geometry,
                        "snr_db": snr, "blocklength": n, "steps": args.steps,
                        "burn_in": args.burn_in, "config": args.config,
                        "eval_seed": 970000 + 1000 * geometry + 100 * perm + 10 * int(snr) + n,
                    })
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(evaluate, jobs))
    baseline = pd.DataFrame(rows)
    baseline.to_csv(out / "trajectories.csv", index=False)
    baseline.groupby(["geometry_deg", "snr_db", "blocklength"], as_index=False).agg(
        mean_aoi=("mean_aoi", "mean"), p95_aoi=("p95_aoi", "mean"),
        success_probability=("success_probability", "mean"),
    ).to_csv(out / "regime_summary.csv", index=False)

    ppo = pd.read_csv(args.ppo_trajectories)
    keys = ["geometry_deg", "snr_db", "blocklength", "permutation", "eval_seed"]
    paired = ppo.merge(
        baseline[keys + ["mean_aoi"]].rename(columns={"mean_aoi": "single_user_aoi"}),
        on=keys, how="inner", validate="many_to_one",
    )
    paired["gain_vs_single_user_pct"] = 100.0 * (
        paired.single_user_aoi - paired.mean_aoi
    ) / paired.single_user_aoi
    paired["stratum"] = np.where(
        (paired.snr_db >= 10.0) & (paired.blocklength >= 128), "feasible", "stress"
    )
    paired.to_csv(out / "ppo_paired_comparison.csv", index=False)
    by_seed = paired.groupby(["stratum", "training_seed"], as_index=False).agg(
        ppo_mean_aoi=("mean_aoi", "mean"), single_user_mean_aoi=("single_user_aoi", "mean"),
        gain_pct=("gain_vs_single_user_pct", "mean"),
    )
    by_seed.to_csv(out / "stratum_by_training_seed.csv", index=False)
    records = []
    for stratum, group in by_seed.groupby("stratum"):
        gain = group.gain_pct.to_numpy()
        critical = float(t.ppf(0.975, len(gain)-1))
        records.append({
            "stratum": stratum, "training_seeds": len(gain),
            "ppo_mean_aoi": group.ppo_mean_aoi.mean(),
            "single_user_mean_aoi": group.single_user_mean_aoi.mean(),
            "gain_mean_pct": gain.mean(), "gain_seed_std_pct": gain.std(ddof=1),
            "gain_ci95_halfwidth_pct": critical * gain.std(ddof=1) / np.sqrt(len(gain)),
        })
    summary = pd.DataFrame(records)
    summary.to_csv(out / "stratum_aggregate.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

