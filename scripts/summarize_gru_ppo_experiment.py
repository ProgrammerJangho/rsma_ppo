#!/usr/bin/env python3
"""Build the compact tables and figure for the seed-0 GRU-PPO pilot."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/gru_ppo_target_seed0_summary_v1"
RUN = ROOT / "runs/gru_objective_rsma_ppo_k3_20deg_from_stage1_seed0_524k_v1"
FF_RUN = ROOT / "runs/matched_objective_rsma_ppo_k3_20deg_from_stage1_seed0_524k_v1"


def _mean(result_dir: str, label: str) -> float:
    frame = pd.read_csv(ROOT / result_dir / "target_summary.csv")
    return float(frame.loc[frame.label == label, "mean_aoi"].iloc[0])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stage1 = _mean("results/gru_ppo_target_seed0_131k_vs_stage1", "feedforward_ppo")
    rows = [
        {"target_steps": 0, "mean_aoi": stage1},
        {"target_steps": 131072, "mean_aoi": _mean("results/gru_ppo_target_seed0_131k_pilot", "gru_ppo")},
        {"target_steps": 262144, "mean_aoi": _mean("results/gru_ppo_target_seed0_262k_pilot", "gru_ppo")},
        {"target_steps": 393216, "mean_aoi": _mean("results/gru_ppo_target_seed0_393k_pilot", "gru_ppo")},
        {"target_steps": 524288, "mean_aoi": _mean("results/gru_ppo_target_seed0_524k_v1", "gru_ppo")},
    ]
    checkpoints = pd.DataFrame(rows)
    checkpoints.to_csv(OUT / "checkpoint_aoi.csv", index=False)

    final = pd.read_csv(
        ROOT / "results/gru_ppo_target_seed0_524k_memory_ablation_v1/target_summary.csv"
    )
    final.to_csv(OUT / "final_methods.csv", index=False)
    paired = pd.read_csv(
        ROOT / "results/gru_ppo_target_seed0_524k_memory_ablation_v1/paired_comparison.csv"
    )
    stats = []
    for column in ("gru_gain_vs_feedforward_pct", "memory_gain_pct"):
        values = paired[column].to_numpy()
        halfwidth = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
        stats.append({
            "comparison": column, "trajectories": len(values),
            "mean_pct": values.mean(), "ci95_halfwidth_pct": halfwidth,
            "positive_count": int((values > 0).sum()),
        })
    pd.DataFrame(stats).to_csv(OUT / "paired_statistics.csv", index=False)

    gru_log = pd.read_csv(RUN / "scalars.csv")
    ff_log = pd.read_csv(FF_RUN / "scalars.csv")
    diagnostics = []
    for name, data in (("feedforward_ppo", ff_log), ("gru_ppo", gru_log)):
        for quarter, indices in enumerate(np.array_split(np.arange(len(data)), 4), 1):
            part = data.iloc[indices]
            diagnostics.append({
                "method": name, "quarter": quarter,
                "sampled_mean_aoi": part["env/mean_aoi"].mean(),
                "approx_kl": part["train/approx_kl"].mean(),
                "clip_fraction": part["train/clip_fraction"].mean(),
                "explained_variance": part["train/explained_variance"].mean(),
                "nonfinite_core": int((~np.isfinite(part[[
                    "env/mean_aoi", "train/value_loss", "train/approx_kl",
                    "train/grad_norm",
                ]].to_numpy())).sum()),
            })
    pd.DataFrame(diagnostics).to_csv(OUT / "training_diagnostics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    axes[0].plot(checkpoints.target_steps, checkpoints.mean_aoi, "o-", color="#7c3aed", label="GRU-PPO")
    ff_aoi = float(final.loc[final.label == "feedforward_ppo", "mean_aoi"].iloc[0])
    axes[0].axhline(ff_aoi, color="#2563eb", linestyle="--", label="Feed-forward PPO (524k)")
    axes[0].set_xlabel("Target-environment interactions")
    axes[0].set_ylabel("Held-out mean AoI")
    axes[0].grid(alpha=.25)
    axes[0].legend(fontsize=8)

    colors = {"feedforward_ppo": "#2563eb", "gru_ppo": "#7c3aed"}
    for name, data in (("feedforward_ppo", ff_log), ("gru_ppo", gru_log)):
        smooth = data["env/mean_aoi"].rolling(24, min_periods=1).mean()
        axes[1].plot(data.step, smooth, color=colors[name], label=name.replace("_", " "))
    axes[1].set_xlabel("Target-environment interactions")
    axes[1].set_ylabel("Sampled-policy mean AoI")
    axes[1].grid(alpha=.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "gru_ppo_pilot.png", dpi=180)
    fig.savefig(OUT / "gru_ppo_pilot.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
