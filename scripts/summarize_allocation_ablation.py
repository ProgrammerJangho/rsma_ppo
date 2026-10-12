#!/usr/bin/env python3
"""Plot pilot means/std across training seeds, without treating episodes as seeds."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir")
    args = parser.parse_args()
    root = Path(args.result_dir)
    summary = pd.read_csv(root / "summary.csv").set_index(["algorithm", "execution"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    order = ["sdma_equal", "fixed_rsma", "power_only", "split_only", "joint"]
    labels = ["SDMA\nequal power", "RSMA\nfixed", "PPO\npower only", "PPO\nsplit only", "PPO\njoint"]
    records = [summary.loc[(name, "fixed" if i < 2 else "deterministic")] for i, name in enumerate(order)]
    means = [r.mean_aoi for r in records]
    std = [0 if i < 2 else r.training_seed_std for i, r in enumerate(records)]
    bars = axes[0].bar(np.arange(5), means, yerr=std, capsize=4,
                       color=["#64748b", "#94a3b8", "#2563eb", "#0891b2", "#0d9488"])
    for bar, mean in zip(bars, means):
        axes[0].text(bar.get_x()+bar.get_width()/2, mean+0.6, f"{mean:.2f}", ha="center", fontsize=9)
    axes[0].set(xticks=np.arange(5), xticklabels=labels, ylabel="Mean AoI (slots)",
                title="Deterministic PPO vs fixed references", ylim=(0, max(means)*1.16))
    x = np.arange(3)
    for j, execution in enumerate(["deterministic", "sampled"]):
        rows = [summary.loc[(mode, execution)] for mode in order[2:]]
        axes[1].bar(x+(j-.5)*.36, [r.mean_aoi for r in rows], .36,
                     yerr=[r.training_seed_std for r in rows], capsize=4,
                     label=execution, color=["#2563eb", "#f59e0b"][j])
    axes[1].axhline(means[0], color="#64748b", linestyle="--", label="SDMA equal power")
    axes[1].set(xticks=x, xticklabels=["Power only", "Split only", "Joint"],
                ylabel="Mean AoI (slots)", title="Execution policy matters")
    axes[1].legend(fontsize=8)
    fig.suptitle("32,768-step pilot | 3 training seeds | uncapped age\nError bars: training-seed SD, not confidence intervals", fontsize=12)
    fig.savefig(root / "aoi_comparison.png", dpi=180)
    plt.close(fig)
    # Logged environment AoI is stochastic training rollout performance, not test performance.
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for mode, color in zip(order[2:], ["#2563eb", "#0891b2", "#0d9488"]):
        paths = sorted((root / "runs").glob(f"{mode}_seed*/scalars.csv"))
        frames = [pd.read_csv(path) for path in paths]
        if "env/mean_aoi" not in frames[0]:
            continue
        values = np.stack([frame["env/mean_aoi"].to_numpy() for frame in frames])
        steps = frames[0]["step"].to_numpy()
        avg, sd = values.mean(0), values.std(0, ddof=1)
        ax.plot(steps, avg, label=mode, color=color)
        ax.fill_between(steps, avg-sd, avg+sd, color=color, alpha=.14)
    ax.set(xlabel="Environment interactions", ylabel="Training rollout mean AoI (slots)",
           title="Training diagnostics (not evaluation returns)")
    ax.legend()
    fig.savefig(root / "training_aoi.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
