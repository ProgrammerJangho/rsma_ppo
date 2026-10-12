#!/usr/bin/env python3
"""Measure how well the myopic phase-map screen predicts long-run AoI.

The screen ranks cells by the expected *next-slot* weighted AoI.  This script
takes screened cells and re-measures them with full episodes and sampled ACKs,
so the screen's predictive value can be quantified rather than assumed.

For each cell it reports the screened gain, the long-run gain, and whether the
sign agrees - the only property a screening tool really needs.

Examples:
    python scripts/validate_screening.py --csv results/hetero_map/coarse.csv --top 12
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsma_aoi.baselines import build_baseline  # noqa: E402
from rsma_aoi.evaluation.evaluator import evaluate_policy  # noqa: E402
from rsma_aoi.evaluation.fairness import rsma_gain  # noqa: E402
from rsma_aoi.utils.logging import get_logger  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", default="results/hetero_map/coarse.csv")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--rank-by", default="gain_vs_sdma")
    parser.add_argument("--sample", type=int, default=None,
                        help="randomly sample this many cells instead of taking the top")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--heterogeneous", action="store_true", default=True)
    parser.add_argument("--out", default="results/hetero_map/screening_validation.csv")
    args = parser.parse_args(argv)

    logger = get_logger()
    if args.heterogeneous:
        from map_heterogeneous_regimes import make_config
        keys = ("num_users", "num_antennas", "disparity_db", "snr_db", "csit_error",
                "rate", "correlation", "weighting")
    else:
        from map_operating_regimes import make_config
        keys = ("num_users", "num_antennas", "snr_db", "csit_error", "blocklength",
                "payload_bits", "correlation")

    frame = pd.read_csv(args.csv)
    if args.sample:
        table = frame.sample(n=min(args.sample, len(frame)), random_state=args.sample_seed)
    else:
        table = frame.sort_values(args.rank_by, ascending=False).head(args.top)
    rows = []
    for _, record in table.iterrows():
        point = {k: record[k] for k in keys}
        cfg = make_config(point)
        long_run: dict[str, float] = {}
        saturated: dict[str, bool] = {}
        for label, overrides in (
            ("rsma", {"allocator_mode": "cem_fast"}),
            ("sdma", {"multiple_access": "sdma", "allocator_mode": "cem_fast"}),
            ("oma", {"multiple_access": "oma"}),
        ):
            arm_cfg = dataclasses.replace(cfg, **overrides)
            policy_name = "oma" if label == "oma" else "max_aoi_first"
            frames = []
            for seed in args.seeds:
                policy, extra = build_baseline(policy_name, arm_cfg, seed=seed)
                frames.append(evaluate_policy(
                    policy, arm_cfg, episodes=args.episodes,
                    eval_seed=100_000 + 1_000 * seed, algorithm=label,
                ))
            merged = pd.concat(frames)
            long_run[label] = float(merged["weighted_avg_aoi"].mean())
            # A scheme sitting near the AoI cap has failed outright rather than
            # been "beaten"; a gain measured against it is vacuous.  0.8 of the
            # cap already means most users are pinned at Delta_max.
            cap = float(cfg.aoi_max) * float(np.sum(cfg.weight_vector))
            saturated[label] = long_run[label] > 0.8 * cap

        row = {k: record[k] for k in keys}
        row.update({
            "screened_gain_vs_sdma": float(record.get("gain_vs_sdma", np.nan)),
            "screened_gain_vs_oma": float(record.get("gain_vs_oma", np.nan)),
            "longrun_aoi_rsma": long_run["rsma"],
            "longrun_aoi_sdma": long_run["sdma"],
            "longrun_aoi_oma": long_run["oma"],
            "longrun_gain_vs_sdma": rsma_gain(long_run["sdma"], long_run["rsma"]),
            "longrun_gain_vs_oma": rsma_gain(long_run["oma"], long_run["rsma"]),
            "oma_saturated": saturated["oma"],
            "rsma_saturated": saturated["rsma"],
        })
        row["sign_agrees_sdma"] = bool(
            np.sign(row["screened_gain_vs_sdma"]) == np.sign(row["longrun_gain_vs_sdma"])
        )
        rows.append(row)
        logger.info("K=%d disp=%.0f rate=%.0f | screened %+.3f -> long-run %+.3f  %s",
                    row["num_users"], row["disparity_db"], row["rate"],
                    row["screened_gain_vs_sdma"], row["longrun_gain_vs_sdma"],
                    "OK" if row["sign_agrees_sdma"] else "SIGN FLIP")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 250)
    print("\n=== screened (myopic one-step) vs long-run RSMA gain over SDMA ===")
    show = ["num_users", "disparity_db", "snr_db", "csit_error", "rate",
            "screened_gain_vs_sdma", "longrun_gain_vs_sdma", "sign_agrees_sdma",
            "longrun_gain_vs_oma", "oma_saturated"]
    print(df[show].round(4).to_string(index=False))
    print(f"\nsign agreement on gain_vs_sdma : {df['sign_agrees_sdma'].mean():.1%}")
    print(f"cells with a long-run RSMA gain : {(df['longrun_gain_vs_sdma'] > 0).mean():.1%}")
    print(f"cells where OMA is saturated    : {df['oma_saturated'].mean():.1%} "
          f"(there 'beating OMA' is vacuous)")
    if len(df) > 2:
        corr = df[["screened_gain_vs_sdma", "longrun_gain_vs_sdma"]].corr().iloc[0, 1]
        print(f"screened vs long-run correlation: {corr:+.3f}")
    logger.info("results written to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
