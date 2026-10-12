#!/usr/bin/env python3
"""Compare joint RSMA, its exact common-off gate, and learned-power SDMA.

The script keeps the original slot traces intact and writes a compact,
reproducible mechanism diagnostic.  Inter-delivery moments use completed ACK
intervals only; measured AoI remains the primary outcome because finite-window
censoring makes renewal estimates approximate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_case(directory: Path, algorithm: str, label: str) -> pd.DataFrame:
    frame = pd.read_csv(directory / "allocation_trace.csv.gz")
    frame = frame[(frame.algorithm == algorithm) & (frame.execution == "deterministic")].copy()
    if frame.empty:
        raise ValueError(f"no deterministic {algorithm!r} rows in {directory}")
    frame["method"] = label
    return frame


def _diagnose(frame: pd.DataFrame) -> dict[str, float | str]:
    intervals: list[int] = []
    ack_corr: list[float] = []
    both_success: list[float] = []
    neither_success: list[float] = []
    for _, episode in frame.groupby(["training_seed", "eval_seed"]):
        ack = episode.pivot(index="slot", columns="user", values="ack")
        if ack.shape[1] == 2:
            ack_corr.append(float(ack.corr().iloc[0, 1]))
            both_success.append(float((ack.eq(1).all(axis=1)).mean()))
            neither_success.append(float((ack.eq(0).all(axis=1)).mean()))
        for _, user in episode.groupby("user"):
            slots = user.loc[user.ack > 0.5, "slot"].to_numpy()
            intervals.extend(np.diff(slots).astype(int).tolist())
    x = np.asarray(intervals, dtype=float)
    enabled = frame.common_enabled > 0.5 if "common_enabled" in frame else pd.Series(False, index=frame.index)
    result: dict[str, float | str] = {
        "method": str(frame.method.iloc[0]),
        "mean_aoi": float(frame.aoi.mean()),
        "mean_success_rate": float(frame.ack.mean()),
        "common_on_rate": float(enabled.mean()),
        "common_bler_when_on": float(frame.loc[enabled, "common_bler"].mean()) if enabled.any() else np.nan,
        "ack_correlation": float(np.nanmean(ack_corr)) if ack_corr else np.nan,
        "both_success_rate": float(np.mean(both_success)) if both_success else np.nan,
        "neither_success_rate": float(np.mean(neither_success)) if neither_success else np.nan,
        "completed_intervals": int(x.size),
        "mean_interdelivery": float(x.mean()),
        "second_moment_interdelivery": float(np.mean(x**2)),
        "interdelivery_cv": float(x.std() / x.mean()),
        "p95_interdelivery": float(np.quantile(x, 0.95)),
        "max_interdelivery": float(x.max()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-sdma-dir", required=True, type=Path)
    parser.add_argument("--gate-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)

    cases = [
        _load_case(args.joint_sdma_dir, "sdma_power", "SDMA power PPO"),
        _load_case(args.joint_sdma_dir, "joint", "Joint RSMA (common forced on)"),
        _load_case(args.gate_dir, "joint", "Joint RSMA (common gate)"),
    ]
    summary = pd.DataFrame([_diagnose(case) for case in cases])
    sdma_aoi = float(summary.loc[summary.method == "SDMA power PPO", "mean_aoi"].iloc[0])
    summary["aoi_delta_vs_sdma"] = summary.mean_aoi - sdma_aoi
    summary["aoi_percent_vs_sdma"] = 100.0 * summary.aoi_delta_vs_sdma / sdma_aoi
    summary.to_csv(args.out_dir / "mechanism_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    labels = ["SDMA\npower PPO", "RSMA\nforced on", "RSMA\nwith gate"]
    axes[0].bar(labels, summary.mean_aoi, color=["#64748b", "#dc2626", "#2563eb"])
    axes[0].set_ylabel("Mean AoI (slots)")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(labels, summary.interdelivery_cv, color=["#64748b", "#dc2626", "#2563eb"])
    axes[1].set_ylabel("Inter-delivery coefficient of variation")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Common-stream ablation at 131,072 training steps")
    fig.savefig(args.out_dir / "mechanism_comparison.png", dpi=180)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
