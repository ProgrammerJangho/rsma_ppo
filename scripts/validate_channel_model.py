#!/usr/bin/env python3
"""Statistically validate Rayleigh/Rician power, mean, time and CSIT moments."""

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
from rsma_aoi.utils.config import EnvConfig


def _complex_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = x.reshape(-1)
    y = y.reshape(-1)
    return float(np.real(np.vdot(x, y)) / np.sqrt(np.vdot(x, x).real * np.vdot(y, y).real))


def validate_config(
    cfg: EnvConfig, trials: int, seed: int, csit_error: float
) -> dict[str, float | str | int]:
    cfg = replace(cfg, csit_error_std=float(csit_error))
    channel = ChannelModel.from_config(cfg)
    rng = np.random.default_rng(seed)
    initial = np.stack([channel.sample_initial(rng) for _ in range(trials)])
    empirical_mean = initial.mean(axis=0)
    empirical_power = np.mean(np.abs(initial) ** 2, axis=(0, 2))
    power_relative_error = np.max(np.abs(empirical_power - channel.path_gain) / channel.path_gain)
    mean_rmse = float(np.sqrt(np.mean(np.abs(empirical_mean - channel.mean) ** 2)))

    successors = np.stack([channel.evolve(h, rng) for h in initial])
    temporal_correlation = _complex_correlation(
        initial - channel.mean[None, :, :],
        successors - channel.mean[None, :, :],
    )

    estimates = np.stack([channel.estimate(h, rng) for h in initial])
    csit_correlation = _complex_correlation(
        initial - channel.mean[None, :, :],
        estimates - channel.mean[None, :, :],
    )
    estimate_mean_rmse = float(
        np.sqrt(np.mean(np.abs(estimates.mean(axis=0) - channel.mean) ** 2))
    )
    estimate_power = np.mean(np.abs(estimates) ** 2, axis=(0, 2))
    estimate_power_relative_error = np.max(
        np.abs(estimate_power - channel.path_gain) / channel.path_gain
    )

    mean_correlation = np.nan
    if cfg.num_users == 2 and np.all(np.linalg.norm(channel.mean, axis=1) > 0):
        mean_correlation = float(
            abs(np.vdot(channel.mean[0], channel.mean[1])) ** 2
            / (
                np.vdot(channel.mean[0], channel.mean[0]).real
                * np.vdot(channel.mean[1], channel.mean[1]).real
            )
        )
    return {
        "config": cfg.small_scale_model,
        "angles_deg": str(cfg.user_angles_deg),
        "rician_k_db": float(cfg.rician_k_db),
        "csit_error_std": float(csit_error),
        "trials": int(trials),
        "max_power_relative_error": float(power_relative_error),
        "channel_mean_rmse": mean_rmse,
        "temporal_correlation": temporal_correlation,
        "target_temporal_correlation": float(cfg.alpha),
        "csit_centered_correlation": csit_correlation,
        "target_csit_centered_correlation": float(np.sqrt(1.0 - csit_error**2)),
        "estimate_mean_rmse": estimate_mean_rmse,
        "max_estimate_power_relative_error": float(estimate_power_relative_error),
        "los_mean_correlation": mean_correlation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--csit-errors", type=float, nargs="+", default=[0.0, 0.3])
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.trials < 100:
        parser.error("--trials must be >= 100")
    if any(not 0.0 <= value <= 1.0 for value in args.csit_errors):
        parser.error("--csit-errors values must lie in [0, 1]")

    rows = []
    for config_index, path in enumerate(args.config):
        cfg = EnvConfig.from_yaml(path)
        for error_index, error in enumerate(args.csit_errors):
            row = validate_config(
                cfg, args.trials,
                args.seed + 10_000 * config_index + error_index,
                error,
            )
            row["config_path"] = str(Path(path).resolve())
            rows.append(row)
    frame = pd.DataFrame(rows)

    # Loose Monte-Carlo gates catch normalization/conjugation/model errors while
    # remaining stable across supported platforms.
    if (frame.max_power_relative_error > 0.03).any():
        raise AssertionError("channel power moment failed")
    if (frame.max_estimate_power_relative_error > 0.03).any():
        raise AssertionError("CSIT power moment failed")
    if (abs(frame.temporal_correlation - frame.target_temporal_correlation) > 0.02).any():
        raise AssertionError("lag-one centered correlation failed")
    if (abs(frame.csit_centered_correlation - frame.target_csit_centered_correlation) > 0.02).any():
        raise AssertionError("CSIT centered correlation failed")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(out / "channel_moments.csv", index=False)
    manifest = {
        "purpose": "statistical channel-model validation",
        "configs": [str(Path(p).resolve()) for p in args.config],
        "trials": args.trials,
        "csit_errors": args.csit_errors,
        "seed": args.seed,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(f"PASS: channel moments written to {out}")


if __name__ == "__main__":
    main()

