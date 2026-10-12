#!/usr/bin/env python3
"""One-step CEM regime audit for the K=3, Nt=2 overloaded extension.

The audit compares simultaneous service of all three users against the best
two-user subset.  Within each schedule, raw 1-layer RSMA, its independently
optimised SDMA boundary, and a safeguarded selector receive matched CEM
budgets.  Candidate selection uses h_hat; final scoring uses h_true.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsma_aoi.envs.channel import ChannelModel  # noqa: E402
from rsma_aoi.evaluation.beam_audit import audit_state_allocations  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402


GEOMETRIES = {
    5: (-2.5, 0.0, 2.5),
    20: (-10.0, 0.0, 10.0),
    60: (-30.0, 0.0, 30.0),
}
AOI_PROFILES = {
    "equal": np.asarray([1.0, 1.0, 1.0]),
    "user0_urgent": np.asarray([8.0, 1.0, 1.0]),
    "user1_urgent": np.asarray([1.0, 8.0, 1.0]),
    "user2_urgent": np.asarray([1.0, 1.0, 8.0]),
}


def _pairwise_alignment(h: np.ndarray) -> tuple[float, float]:
    values = []
    for i, j in combinations(range(h.shape[0]), 2):
        denom = float(np.vdot(h[i], h[i]).real * np.vdot(h[j], h[j]).real)
        values.append(float(abs(np.vdot(h[i], h[j])) ** 2 / max(denom, 1e-300)))
    return float(np.mean(values)), float(np.max(values))


def _compact(row: dict) -> dict:
    return {
        "base": float(row["base_expected_aoi"]),
        "rsma_aoi": float(row["raw_expected_aoi_true"]),
        "sdma_aoi": float(row["base_expected_aoi"] - row["sdma_gain_true"]),
        "safe_aoi": float(row["safe_expected_aoi_true"]),
        "rsma_gain_est": float(row["raw_gain_estimated"]),
        "sdma_gain_est": float(row["sdma_gain_estimated"]),
        "safe_gain_est": float(max(row["raw_gain_estimated"], row["sdma_gain_estimated"])),
        "safe_boundary": bool(row["selected_boundary"]),
        "common_power": float(row["common_power_fraction"]),
        "rho": float(row["rho"]),
        "mean_success": float(row["mean_success_true"]),
    }


def _audit_schedule(
    cfg: EnvConfig,
    aoi: np.ndarray,
    h_true: np.ndarray,
    h_hat: np.ndarray,
    active: np.ndarray,
    *,
    seed: int,
    iterations: int,
    population: int,
    restarts: int,
) -> dict:
    rows = audit_state_allocations(
        cfg, aoi, h_true, h_hat, active,
        beam_modes=("principal",), csit_mode="estimated", seed=seed,
        iterations=iterations, population=population, restarts=restarts,
    )
    return _compact(rows[0])


def _best_scheduled_two(
    cfg: EnvConfig,
    aoi: np.ndarray,
    h_true: np.ndarray,
    h_hat: np.ndarray,
    *,
    seed: int,
    iterations: int,
    population: int,
    restarts: int,
) -> tuple[dict, dict, dict]:
    candidates = []
    for subset_index, subset in enumerate(combinations(range(cfg.num_users), 2)):
        active = np.zeros(cfg.num_users, dtype=bool)
        active[list(subset)] = True
        result = _audit_schedule(
            cfg, aoi, h_true, h_hat, active,
            seed=seed + 1009 * subset_index,
            iterations=iterations, population=population, restarts=restarts,
        )
        result["subset"] = "".join(str(x) for x in subset)
        candidates.append(result)

    # Each method chooses its subset using transmitter-available h_hat only.
    best_rsma = max(candidates, key=lambda x: x["rsma_gain_est"])
    best_sdma = max(candidates, key=lambda x: x["sdma_gain_est"])
    best_safe = max(candidates, key=lambda x: x["safe_gain_est"])
    return best_rsma, best_sdma, best_safe


def run(args: argparse.Namespace) -> pd.DataFrame:
    base = EnvConfig.from_yaml(args.config)
    if (
        base.num_users != 3 or base.num_antennas != 2
        or base.small_scale_model != "rician_ula"
        or base.multiple_access != "rsma" or base.link_model != "fbl"
        or base.scheduling_enabled
    ):
        raise ValueError("requires all-active K=3, Nt=2 Rician FBL RSMA config")

    rows = []
    for geometry in args.geometry_spans:
        angles = GEOMETRIES[geometry]
        for snr_db in args.snr_db:
            cfg = replace(
                base, user_angles_deg=list(angles), user_angle_sets_deg=None,
                snr_db=float(snr_db), snr_db_values=None,
            )
            cfg.validate()
            channel = ChannelModel.from_config(cfg)
            rng = np.random.default_rng(args.channel_seed)
            for state_id in range(args.states):
                h_true = channel.sample_initial(rng)
                h_hat = channel.estimate(h_true, rng)
                align_mean, align_max = _pairwise_alignment(h_true)
                for profile_id, (profile, aoi) in enumerate(AOI_PROFILES.items()):
                    seed = args.cem_seed + state_id * 1_000_003 + profile_id * 10_007
                    all3 = _audit_schedule(
                        cfg, aoi, h_true, h_hat, np.ones(3, dtype=bool),
                        seed=seed, iterations=args.iterations,
                        population=args.population, restarts=args.restarts,
                    )
                    s_rsma, s_sdma, s_safe = _best_scheduled_two(
                        cfg, aoi, h_true, h_hat, seed=seed + 70_001,
                        iterations=args.iterations, population=args.population,
                        restarts=args.restarts,
                    )
                    rows.append({
                        "geometry_span_deg": geometry,
                        "snr_db": float(snr_db),
                        "state_id": state_id,
                        "profile": profile,
                        "alignment_mean": align_mean,
                        "alignment_max": align_max,
                        "base_expected_aoi": all3["base"],
                        "all3_rsma_raw_aoi": all3["rsma_aoi"],
                        "all3_sdma_aoi": all3["sdma_aoi"],
                        "all3_safeguarded_aoi": all3["safe_aoi"],
                        "all3_common_power": all3["common_power"],
                        "all3_rho": all3["rho"],
                        "all3_safe_boundary": all3["safe_boundary"],
                        "scheduled2_rsma_raw_aoi": s_rsma["rsma_aoi"],
                        "scheduled2_sdma_aoi": s_sdma["sdma_aoi"],
                        "scheduled2_safeguarded_aoi": s_safe["safe_aoi"],
                        "scheduled2_rsma_subset": s_rsma["subset"],
                        "scheduled2_sdma_subset": s_sdma["subset"],
                        "scheduled2_safe_subset": s_safe["subset"],
                        "scheduled2_safe_boundary": s_safe["safe_boundary"],
                        "scheduled2_common_power": s_rsma["common_power"],
                        "scheduled2_rho": s_rsma["rho"],
                    })
    return pd.DataFrame(rows)


def summarise(trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_columns = [
        "all3_rsma_raw_aoi", "all3_sdma_aoi", "all3_safeguarded_aoi",
        "scheduled2_rsma_raw_aoi", "scheduled2_sdma_aoi",
        "scheduled2_safeguarded_aoi",
    ]
    long = trace.melt(
        id_vars=["geometry_span_deg", "snr_db", "state_id", "profile"],
        value_vars=value_columns, var_name="algorithm", value_name="expected_aoi",
    )
    condition = long.groupby(
        ["geometry_span_deg", "snr_db", "algorithm"], as_index=False
    ).agg(
        mean_expected_aoi=("expected_aoi", "mean"),
        std_expected_aoi=("expected_aoi", "std"),
        samples=("expected_aoi", "size"),
    )
    wide = condition.pivot(
        index=["geometry_span_deg", "snr_db"], columns="algorithm",
        values="mean_expected_aoi",
    )
    reference = wide["scheduled2_sdma_aoi"]
    for algorithm in value_columns:
        wide[f"{algorithm}_gain_vs_scheduled2_sdma_pct"] = (
            100.0 * (reference - wide[algorithm]) / reference
        )
    return condition, wide.reset_index()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", default="configs/thesis_rician_k3_nt2_overloaded_csit02.yaml"
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--states", type=int, default=16,
                   help="independent channel draws; each receives four AoI profiles")
    p.add_argument("--snr-db", type=float, nargs="+", default=[5.0, 10.0, 15.0])
    p.add_argument(
        "--geometry-spans", type=int, nargs="+", choices=sorted(GEOMETRIES),
        default=sorted(GEOMETRIES),
    )
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--population", type=int, default=256)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--channel-seed", type=int, default=20260916)
    p.add_argument("--cem-seed", type=int, default=3162026)
    args = p.parse_args()
    if min(args.states, args.iterations, args.population, args.restarts) < 1:
        p.error("CEM and state counts must be positive")

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    trace = run(args)
    condition, gains = summarise(trace)
    trace.to_csv(out / "trace.csv.gz", index=False)
    condition.to_csv(out / "condition_summary.csv", index=False)
    gains.to_csv(out / "gains_vs_scheduled2_sdma.csv", index=False)
    behaviour = trace.groupby(["geometry_span_deg", "snr_db"], as_index=False).agg(
        all3_boundary_rate=("all3_safe_boundary", "mean"),
        all3_common_power=("all3_common_power", "mean"),
        all3_rho=("all3_rho", "mean"),
        scheduled2_boundary_rate=("scheduled2_safe_boundary", "mean"),
        scheduled2_common_power=("scheduled2_common_power", "mean"),
        scheduled2_rho=("scheduled2_rho", "mean"),
        alignment_mean=("alignment_mean", "mean"),
        alignment_max=("alignment_max", "mean"),
    )
    behaviour.to_csv(out / "allocation_summary.csv", index=False)

    config_path = ROOT / args.config
    manifest = {
        "purpose": "K=3 Nt=2 overloaded one-step CEM feasibility/regime audit",
        "config": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "geometries": {str(k): list(GEOMETRIES[k]) for k in args.geometry_spans},
        "snr_db": args.snr_db,
        "aoi_profiles": {k: v.tolist() for k, v in AOI_PROFILES.items()},
        "states_per_condition": args.states,
        "samples_per_condition": args.states * len(AOI_PROFILES),
        "csit_rule": "select on h_hat; score on h_true",
        "cem": {
            "iterations": args.iterations,
            "population": args.population,
            "restarts": args.restarts,
            "evaluations_per_solve": args.iterations * args.population * args.restarts,
        },
        "schemes": {
            "all3": "all three users active simultaneously",
            "scheduled2": "enumerate all three two-user subsets and select on h_hat",
            "safeguarded": "choose independently optimised RSMA or exact SDMA boundary on h_hat",
        },
        "caveat": "finite-budget one-step numerical audit, not long-run AoI or global optimum",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(condition.to_string(index=False))
    print("\nGAINS VS SCHEDULED-2 SDMA")
    print(gains.to_string(index=False))
    print("\nALLOCATION")
    print(behaviour.to_string(index=False))
    print(f"\nCOMPLETE: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
