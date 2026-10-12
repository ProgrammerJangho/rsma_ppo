#!/usr/bin/env python3
"""Paired one-step FBL-AoI audit across controlled Rician angle regimes.

This is a PHY/allocator diagnostic, not a long-run policy comparison.  For
every channel and AoI state, principal and maxmin common beams receive the same
CEM budget and are scored beside an independently optimised SDMA boundary.
A fixed allocation is also evaluated for both beams to separate beam effects
from allocation re-optimisation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from rsma_aoi.envs.channel import ChannelModel
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer
from rsma_aoi.evaluation.beam_audit import audit_state_allocations
from rsma_aoi.utils.config import EnvConfig


AOI_PROFILES = {
    "equal": np.asarray([1.0, 1.0]),
    "user0_urgent": np.asarray([8.0, 1.0]),
    "user1_urgent": np.asarray([1.0, 8.0]),
}


def _alignment(h: np.ndarray) -> float:
    denom = float(np.vdot(h[0], h[0]).real * np.vdot(h[1], h[1]).real)
    return float(abs(np.vdot(h[0], h[1])) ** 2 / max(denom, 1e-300))


def _condition_number(h: np.ndarray) -> float:
    singular = np.linalg.svd(h, compute_uv=False)
    return float(singular[0] / max(singular[-1], 1e-15))


def _fixed_result(
    cfg: EnvConfig,
    h: np.ndarray,
    aoi: np.ndarray,
    beam: str,
    common_power: float,
    rho: float,
) -> dict[str, float | str]:
    beam_cfg = replace(cfg, common_beam=beam, multiple_access="rsma", allocator_mode="none")
    phy = RSMAPhysicalLayer(beam_cfg)
    active = np.ones(cfg.num_users, dtype=bool)
    beams = phy.rsma_beams(h, active, use_common=True)
    power = np.zeros((1, cfg.num_users + 1), dtype=float)
    power[0, 0] = common_power
    power[0, 1:] = (1.0 - common_power) / cfg.num_users
    rate = np.full((1, cfg.num_users), 1.0 / cfg.num_users)
    out = phy.evaluate_allocations(
        h, active, power, rate, np.asarray([rho]), beams=beams
    )
    urgency = np.asarray(cfg.weight_vector) * aoi
    gain = float(out["success_prob"][0] @ urgency)
    base = float(np.sum(np.asarray(cfg.weight_vector) * (aoi + 1.0)))
    return {
        "beam": beam,
        "fixed_gain": gain,
        "fixed_expected_aoi": base - gain,
        "fixed_mean_success": float(np.mean(out["success_prob"][0])),
        "fixed_worst_common_bler": float(np.max(out["common_bler"][0])),
    }


def run_audit(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    optimized_rows: list[dict] = []
    fixed_rows: list[dict] = []
    for config_index, config_path in enumerate(args.config):
        cfg = EnvConfig.from_yaml(config_path)
        if (
            cfg.num_users != 2
            or cfg.small_scale_model != "rician_ula"
            or cfg.multiple_access != "rsma"
            or cfg.link_model != "fbl"
            or cfg.csit_error_std != 0.0
        ):
            raise ValueError(
                "audit requires two-user, perfect-CSIT, FBL rician_ula RSMA configs"
            )
        channel = ChannelModel.from_config(cfg)
        # Reusing the seed pairs the NLoS draws across angle configs.  Each
        # algorithm within a state receives exactly the same h_true == h_hat.
        rng = np.random.default_rng(args.channel_seed)
        for state_id in range(args.states):
            h = channel.sample_initial(rng)
            alignment = _alignment(h)
            condition = _condition_number(h)
            norms = np.sum(np.abs(h) ** 2, axis=1)
            gain_gap_db = float(10.0 * np.log10(max(norms) / max(min(norms), 1e-300)))
            configured_los_alignment = _alignment(channel.mean)
            for profile_index, (profile, aoi) in enumerate(AOI_PROFILES.items()):
                audit_seed = (
                    args.cem_seed
                    + state_id * 1_000_003
                    + profile_index * 10_007
                )
                rows = audit_state_allocations(
                    cfg,
                    aoi,
                    h,
                    h,
                    np.ones(cfg.num_users, dtype=bool),
                    beam_modes=args.beams,
                    csit_mode="estimated",
                    seed=audit_seed,
                    iterations=args.iterations,
                    population=args.population,
                    restarts=args.restarts,
                )
                common = {
                    "config": Path(config_path).stem,
                    "config_index": config_index,
                    "state_id": state_id,
                    "profile": profile,
                    "aoi_user0": float(aoi[0]),
                    "aoi_user1": float(aoi[1]),
                    "realized_alignment": alignment,
                    "configured_los_alignment": configured_los_alignment,
                    "channel_condition_number": condition,
                    "realized_gain_gap_db": gain_gap_db,
                }
                for row in rows:
                    for key in (
                        "boundary_power_fractions", "boundary_rate_fractions",
                        "raw_power_fractions", "raw_rate_fractions",
                        "raw_success_probabilities", "safe_power_fractions",
                        "safe_rate_fractions", "safe_success_probabilities",
                    ):
                        row.pop(key)
                    optimized_rows.append({**common, **row})
                    fixed_rows.append({
                        **common,
                        **_fixed_result(
                            cfg, h, aoi, str(row["beam"]),
                            args.fixed_common_power, args.fixed_rho,
                        ),
                    })
    return pd.DataFrame(optimized_rows), pd.DataFrame(fixed_rows)


def _distribution(values: pd.Series) -> dict[str, float]:
    standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
    mean = float(values.mean())
    return {
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
        "median": float(values.median()),
        "p05": float(values.quantile(0.05)),
        "p95": float(values.quantile(0.95)),
        "improve_rate": float((values > 1e-10).mean()),
        "worsen_rate": float((values < -1e-10).mean()),
    }


def summarise(
    optimized: pd.DataFrame, fixed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by = ["config", "profile", "beam"]
    summary = optimized.groupby(by, sort=False).agg(
        states=("state_id", "count"),
        mean_raw_gain=("raw_gain_true", "mean"),
        mean_safe_gain=("safe_gain_true", "mean"),
        mean_sdma_gain=("sdma_gain_true", "mean"),
        mean_raw_expected_aoi=("raw_expected_aoi_true", "mean"),
        mean_safe_expected_aoi=("safe_expected_aoi_true", "mean"),
        boundary_selection_rate=("selected_boundary", "mean"),
        mean_common_power=("common_power_fraction", "mean"),
        mean_rho=("rho", "mean"),
        effective_common_rate=("rho", lambda x: (x > 0.01).mean()),
        mean_success=("mean_success_true", "mean"),
        mean_runtime_ms=("runtime_s", lambda x: 1000.0 * x.mean()),
    ).reset_index()
    fixed_summary = fixed.groupby(by, sort=False).agg(
        states=("state_id", "count"),
        configured_los_alignment=("configured_los_alignment", "first"),
        mean_fixed_gain=("fixed_gain", "mean"),
        mean_fixed_expected_aoi=("fixed_expected_aoi", "mean"),
        mean_fixed_success=("fixed_mean_success", "mean"),
        mean_fixed_worst_common_bler=("fixed_worst_common_bler", "mean"),
    ).reset_index()
    summary = summary.merge(
        fixed_summary, on=by + ["states"], how="left", validate="one_to_one"
    )

    comparisons: list[dict] = []
    group_keys = ["config", "profile"]
    for keys, group in optimized.groupby(group_keys, sort=False):
        raw = group.pivot(index="state_id", columns="beam", values="raw_gain_true")
        safe = group.pivot(index="state_id", columns="beam", values="safe_gain_true")
        sdma = group.groupby("state_id")["sdma_gain_true"].first()
        fixed_group = fixed[
            (fixed["config"] == keys[0]) & (fixed["profile"] == keys[1])
        ].pivot(index="state_id", columns="beam", values="fixed_gain")
        comparisons_to_make = [
            ("optimized_raw_maxmin_minus_principal", raw["maxmin"] - raw["principal"]),
            ("optimized_safe_maxmin_minus_principal", safe["maxmin"] - safe["principal"]),
            ("optimized_principal_minus_sdma", raw["principal"] - sdma),
            ("optimized_maxmin_minus_sdma", raw["maxmin"] - sdma),
            ("fixed_maxmin_minus_principal", fixed_group["maxmin"] - fixed_group["principal"]),
        ]
        for comparison, delta in comparisons_to_make:
            comparisons.append({
                "config": keys[0], "profile": keys[1], "comparison": comparison,
                "states": int(delta.size), **_distribution(delta),
            })

    channel_summary = optimized.drop_duplicates(["config", "state_id"]).groupby(
        "config", sort=False
    ).agg(
        states=("state_id", "count"),
        alignment_mean=("realized_alignment", "mean"),
        alignment_median=("realized_alignment", "median"),
        alignment_p05=("realized_alignment", lambda x: x.quantile(0.05)),
        alignment_p95=("realized_alignment", lambda x: x.quantile(0.95)),
        condition_number_mean=("channel_condition_number", "mean"),
        gain_gap_db_mean=("realized_gain_gap_db", "mean"),
    ).reset_index()
    return summary, pd.DataFrame(comparisons), channel_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--states", type=int, default=192)
    parser.add_argument("--beams", nargs="+", default=["principal", "maxmin"])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--population", type=int, default=192)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--channel-seed", type=int, default=20260907)
    parser.add_argument("--cem-seed", type=int, default=3302026)
    parser.add_argument("--fixed-common-power", type=float, default=0.25)
    parser.add_argument("--fixed-rho", type=float, default=0.10)
    args = parser.parse_args()
    if min(args.states, args.iterations, args.population, args.restarts) < 1:
        parser.error("state and CEM counts must be positive")
    if set(args.beams) != {"principal", "maxmin"}:
        parser.error("this paired report requires exactly principal and maxmin")
    if not 0.0 <= args.fixed_common_power <= 1.0 or not 0.0 <= args.fixed_rho <= 1.0:
        parser.error("fixed allocation values must lie in [0, 1]")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    optimized, fixed = run_audit(args)
    summary, comparisons, channel_summary = summarise(optimized, fixed)
    optimized.to_csv(out / "optimized_trace.csv.gz", index=False)
    fixed.to_csv(out / "fixed_trace.csv.gz", index=False)
    summary.to_csv(out / "scheme_summary.csv", index=False)
    comparisons.to_csv(out / "paired_comparisons.csv", index=False)
    channel_summary.to_csv(out / "channel_summary.csv", index=False)

    paths = [Path(p).resolve() for p in args.config]
    manifest = {
        "purpose": "paired one-step Rician PHY/allocator audit; not a long-run PPO result",
        "configs": [str(p) for p in paths],
        "config_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "aoi_profiles": {key: value.tolist() for key, value in AOI_PROFILES.items()},
        "same_trace_rule": "same NLoS random draws across angle configs and all schemes",
        "csit": "perfect; h_hat equals h_true",
        "cem": {
            "iterations": args.iterations,
            "population": args.population,
            "restarts": args.restarts,
            "evaluations_per_scheme_state": args.iterations * args.population * args.restarts,
            "seed": args.cem_seed,
            "status": "approximate reference, not a global optimum",
        },
        "fixed_allocation": {
            "common_power_fraction": args.fixed_common_power,
            "rho": args.fixed_rho,
            "private_power": "equal",
            "common_rate_allocation": "equal",
        },
        "states_per_config": args.states,
        "uncertainty": "paired mean differences use normal 95% confidence intervals over independent stationary channel draws",
        "channel_seed": args.channel_seed,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Channel distribution:\n" + channel_summary.to_string(index=False))
    print("\nPaired comparisons:\n" + comparisons.to_string(index=False))
    print(f"\nPASS: {len(optimized)} optimized rows written to {out}")


if __name__ == "__main__":
    main()
