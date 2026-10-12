#!/usr/bin/env python3
"""Summarize the seed-0 feed-forward PPO extension through 2.10M target steps."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/feedforward_ppo_seed0_step_extension_to_2m_v1"
RUNS = [
    ROOT / "runs/matched_objective_rsma_ppo_k3_20deg_from_stage1_seed0_524k_v1",
    ROOT / "runs/matched_objective_rsma_ppo_k3_20deg_seed0_continue_524k_v1",
    ROOT / "runs/matched_objective_rsma_ppo_k3_20deg_seed0_continue_to_2097k_v1",
]


def _mean(directory: str, label: str) -> float:
    frame = pd.read_csv(ROOT / directory / "target_summary.csv")
    return float(frame.loc[frame.label == label, "mean_aoi"].iloc[0])


def _paired_stats(path: str, comparison: str) -> dict:
    values = pd.read_csv(ROOT / path).gru_gain_vs_feedforward_pct.to_numpy()
    halfwidth = t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return {
        "comparison": comparison, "trajectories": len(values),
        "mean_gain_pct": values.mean(), "ci95_halfwidth_pct": halfwidth,
        "extended_wins": int((values > 0).sum()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    original = _mean("results/feedforward_ppo_seed0_total2097k_vs524k_v1", "feedforward_ppo")
    definitions = [
        (524288, original),
        (655360, _mean("results/feedforward_ppo_seed0_total655k_pilot", "gru_ppo")),
        (786432, _mean("results/feedforward_ppo_seed0_total786k_pilot", "gru_ppo")),
        (917504, _mean("results/feedforward_ppo_seed0_total917k_pilot", "gru_ppo")),
        (1048576, _mean("results/feedforward_ppo_seed0_total1048k_v1", "gru_ppo")),
        (1310720, _mean("results/feedforward_ppo_seed0_total1310k_pilot", "gru_ppo")),
        (1572864, _mean("results/feedforward_ppo_seed0_total1572k_pilot", "gru_ppo")),
        (1835008, _mean("results/feedforward_ppo_seed0_total1835k_pilot", "gru_ppo")),
        (2097152, _mean("results/feedforward_ppo_seed0_total2097k_v1", "gru_ppo")),
    ]
    checkpoints = pd.DataFrame(definitions, columns=["total_target_steps", "mean_aoi"])
    checkpoints["gain_vs_524k_pct"] = 100.0 * (original - checkpoints.mean_aoi) / original
    checkpoints.to_csv(OUT / "checkpoint_aoi.csv", index=False)
    pd.DataFrame([
        _paired_stats(
            "results/feedforward_ppo_seed0_total2097k_v1/paired_comparison.csv",
            "2097k_vs_1048k",
        ),
        _paired_stats(
            "results/feedforward_ppo_seed0_total2097k_vs524k_v1/paired_comparison.csv",
            "2097k_vs_524k",
        ),
    ]).to_csv(OUT / "paired_statistics.csv", index=False)

    final = pd.read_csv(ROOT / "results/feedforward_ppo_seed0_total2097k_vs524k_v1/target_summary.csv")
    final["label"] = final.label.replace({
        "feedforward_ppo": "feedforward_ppo_524k",
        "gru_ppo": "feedforward_ppo_2097k",
    })
    final.to_csv(OUT / "final_summary.csv", index=False)

    frames = [pd.read_csv(run / "scalars.csv") for run in RUNS]
    training = pd.concat(frames, ignore_index=True).sort_values("step")
    diagnostics = []
    for name, data in zip(("0_to_524k", "524k_to_1048k", "1048k_to_2097k"), frames):
        diagnostics.append({
            "phase": name, "logged_updates": len(data),
            "sampled_mean_aoi": data["env/mean_aoi"].mean(),
            "last_logged_16_mean_aoi": data.tail(16)["env/mean_aoi"].mean(),
            "mean_approx_kl": data["train/approx_kl"].mean(),
            "nonfinite_core": int((~np.isfinite(data[[
                "env/mean_aoi", "train/value_loss", "train/approx_kl", "train/grad_norm"
            ]].to_numpy())).sum()),
        })
    pd.DataFrame(diagnostics).to_csv(OUT / "training_diagnostics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.7))
    axes[0].plot(checkpoints.total_target_steps, checkpoints.mean_aoi,
                 "o-", color="#2563eb")
    axes[0].set_xlabel("Total target interactions")
    axes[0].set_ylabel("Held-out mean AoI")
    axes[0].grid(alpha=.25)
    smooth = training["env/mean_aoi"].rolling(16, min_periods=1).mean()
    axes[1].plot(training.step, smooth, color="#2563eb")
    for boundary in (524288, 1048576):
        axes[1].axvline(boundary, color="#dc2626", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Total target interactions")
    axes[1].set_ylabel("Sampled-policy mean AoI")
    axes[1].grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "feedforward_step_extension_to_2m.png", dpi=180)
    fig.savefig(OUT / "feedforward_step_extension_to_2m.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
