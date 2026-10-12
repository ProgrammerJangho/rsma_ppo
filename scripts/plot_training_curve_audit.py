#!/usr/bin/env python3
"""Plot and summarize valid sigma_e=0.2 PPO learning curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "rician_ppo_20deg_csit02_training_curve_audit_v1"
RUNS = {
    "Joint + gate warm": lambda seed: RESULTS
    / f"rician_ppo_20deg_csit02_single_fair_endtoend_seed{seed}_v2"
    / "runs"
    / f"joint_gate_seed{seed}"
    / "scalars.csv",
    "Joint warm": lambda seed: RESULTS
    / f"rician_ppo_20deg_csit02_joint_objective_seed{seed}_v2"
    / "runs"
    / f"joint_seed{seed}"
    / "scalars.csv",
    "Scratch joint + gate": lambda seed: RESULTS
    / "rician_ppo_20deg_csit02_controls_3seed_v1"
    / "runs"
    / f"joint_gate_seed{seed}"
    / "scalars.csv",
}


def main() -> None:
    frames = []
    diagnostics = []
    for method, path_for_seed in RUNS.items():
        for seed in (0, 1, 2):
            data = pd.read_csv(path_for_seed(seed))
            data["method"] = method
            data["training_seed"] = seed
            frames.append(data)
            for quarter, indices in enumerate(np.array_split(np.arange(len(data)), 4), 1):
                part = data.iloc[indices]
                diagnostics.append({
                    "method": method,
                    "training_seed": seed,
                    "quarter": quarter,
                    "mean_aoi": part["env/mean_aoi"].mean(),
                    "approx_kl": part["train/approx_kl"].mean(),
                    "clip_fraction": part["train/clip_fraction"].mean(),
                    "explained_variance": part["train/explained_variance"].mean(),
                    "value_loss": part["train/value_loss"].mean(),
                    "grad_norm": part["train/grad_norm"].mean(),
                })
    raw = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUT / "curve_points.csv.gz", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "quarter_diagnostics.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    panels = [
        ("env/mean_aoi", "Sampled-policy mean AoI"),
        ("train/approx_kl", "Approximate KL"),
        ("train/clip_fraction", "PPO clip fraction"),
        ("train/explained_variance", "Critic explained variance"),
    ]
    colors = ["#2563eb", "#059669", "#dc2626"]
    for ax, (column, ylabel) in zip(axes.flat, panels):
        for (method, group), color in zip(raw.groupby("method", sort=False), colors):
            pivot = group.pivot(index="step", columns="training_seed", values=column)
            smooth = pivot.rolling(8, min_periods=1).mean()
            mean = smooth.mean(axis=1)
            std = smooth.std(axis=1)
            ax.plot(mean.index, mean, label=method, color=color)
            ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("PPO interactions")
    axes[1, 1].set_xlabel("PPO interactions")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(r"PPO training audit: $20^\circ$, $\sigma_e=0.2$ (mean $\pm$ seed SD)")
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=180)
    fig.savefig(OUT / "training_curves.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
