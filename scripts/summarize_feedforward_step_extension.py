#!/usr/bin/env python3
"""Summarize the seed-0 feed-forward PPO target-budget extension."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/feedforward_ppo_seed0_step_extension_v1"
BASE_RUN = ROOT / "runs/matched_objective_rsma_ppo_k3_20deg_from_stage1_seed0_524k_v1"
CONT_RUN = ROOT / "runs/matched_objective_rsma_ppo_k3_20deg_seed0_continue_524k_v1"


def _extended_mean(directory: str) -> float:
    frame = pd.read_csv(ROOT / directory / "target_summary.csv")
    return float(frame.loc[frame.label == "gru_ppo", "mean_aoi"].iloc[0])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(ROOT / "results/feedforward_ppo_seed0_total1048k_v1/target_summary.csv")
    base_aoi = float(base.loc[base.label == "feedforward_ppo", "mean_aoi"].iloc[0])
    checkpoints = pd.DataFrame([
        {"total_target_steps": 524288, "mean_aoi": base_aoi},
        {"total_target_steps": 655360, "mean_aoi": _extended_mean("results/feedforward_ppo_seed0_total655k_pilot")},
        {"total_target_steps": 786432, "mean_aoi": _extended_mean("results/feedforward_ppo_seed0_total786k_pilot")},
        {"total_target_steps": 917504, "mean_aoi": _extended_mean("results/feedforward_ppo_seed0_total917k_pilot")},
        {"total_target_steps": 1048576, "mean_aoi": _extended_mean("results/feedforward_ppo_seed0_total1048k_v1")},
    ])
    checkpoints["gain_vs_524k_pct"] = 100.0 * (base_aoi - checkpoints.mean_aoi) / base_aoi
    checkpoints.to_csv(OUT / "checkpoint_aoi.csv", index=False)

    final_summary = base.rename(columns={"gru_ppo": "extended_feedforward_ppo"})
    final_summary["label"] = final_summary.label.replace({
        "feedforward_ppo": "feedforward_ppo_524k",
        "gru_ppo": "feedforward_ppo_1048k",
    })
    final_summary.to_csv(OUT / "final_summary.csv", index=False)
    paired = pd.read_csv(ROOT / "results/feedforward_ppo_seed0_total1048k_v1/paired_comparison.csv")
    paired = paired.rename(columns={
        "feedforward_ppo": "feedforward_ppo_524k",
        "gru_ppo": "feedforward_ppo_1048k",
        "gru_gain_vs_feedforward_pct": "gain_1048k_vs_524k_pct",
    })
    paired.to_csv(OUT / "paired_comparison.csv", index=False)
    values = paired.gain_1048k_vs_524k_pct.to_numpy()
    halfwidth = t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    pd.DataFrame([{
        "trajectories": len(values), "mean_gain_pct": values.mean(),
        "ci95_halfwidth_pct": halfwidth, "extended_wins": int((values > 0).sum()),
    }]).to_csv(OUT / "paired_statistics.csv", index=False)

    first = pd.read_csv(BASE_RUN / "scalars.csv")
    second = pd.read_csv(CONT_RUN / "scalars.csv")
    training = pd.concat([first, second], ignore_index=True).sort_values("step")
    diagnostics = []
    for phase, data in (("first_524k", first), ("additional_524k", second)):
        diagnostics.append({
            "phase": phase, "updates": len(data),
            "sampled_mean_aoi": data["env/mean_aoi"].mean(),
            "last64_sampled_mean_aoi": data.tail(64)["env/mean_aoi"].mean(),
            "mean_approx_kl": data["train/approx_kl"].mean(),
            "mean_clip_fraction": data["train/clip_fraction"].mean(),
            "nonfinite_core": int((~np.isfinite(data[[
                "env/mean_aoi", "train/value_loss", "train/approx_kl", "train/grad_norm"
            ]].to_numpy())).sum()),
        })
    pd.DataFrame(diagnostics).to_csv(OUT / "training_diagnostics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    axes[0].plot(checkpoints.total_target_steps, checkpoints.mean_aoi,
                 "o-", color="#2563eb")
    axes[0].set_xlabel("Total target interactions")
    axes[0].set_ylabel("Held-out mean AoI")
    axes[0].grid(alpha=.25)
    smooth = training["env/mean_aoi"].rolling(24, min_periods=1).mean()
    axes[1].plot(training.step, smooth, color="#2563eb")
    axes[1].axvline(524288, color="#dc2626", linestyle="--", linewidth=1,
                    label="continuation begins")
    axes[1].set_xlabel("Total target interactions")
    axes[1].set_ylabel("Sampled-policy mean AoI")
    axes[1].grid(alpha=.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "feedforward_step_extension.png", dpi=180)
    fig.savefig(OUT / "feedforward_step_extension.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
