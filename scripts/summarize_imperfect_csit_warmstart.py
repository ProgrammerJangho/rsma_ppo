#!/usr/bin/env python3
"""Summarize the controlled 20-degree, sigma_e=0.2 PPO comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCES = {
    "scratch_rsma": ("rician_ppo_20deg_csit02_controls_3seed_v1", "joint_gate"),
    "sdma_power": ("rician_ppo_20deg_csit02_controls_3seed_v1", "sdma_power"),
}


def main() -> None:
    out = RESULTS / "rician_ppo_20deg_csit02_comparison_v2"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for method, (directory, algorithm) in SOURCES.items():
        data = pd.read_csv(RESULTS / directory / "training_seed_summary.csv")
        rows = data[(data.algorithm == algorithm) & (data.execution == "deterministic")].copy()
        rows["method"] = method
        frames.append(rows)
    # Fair end-to-end seeds: PPO seed i uses pretraining seed i for both
    # objective variants.  SAA additionally averages its posterior sample axis
    # inside each state loss.
    for method, directory_stem in (
        ("single_fair_endtoend", "single"),
        ("saa8_fair_endtoend", "saa8"),
    ):
        end_to_end = []
        for training_seed in (0, 1, 2):
            data = pd.read_csv(
                RESULTS
                / f"rician_ppo_20deg_csit02_{directory_stem}_fair_endtoend_seed{training_seed}_v2"
                / "training_seed_summary.csv"
            )
            end_to_end.append(
                data[
                    (data.algorithm == "joint_gate")
                    & (data.execution == "deterministic")
                ].copy()
            )
        end_to_end = pd.concat(end_to_end, ignore_index=True)
        end_to_end["method"] = method
        frames.append(end_to_end)
    seed = pd.concat(frames, ignore_index=True)
    seed.to_csv(out / "seed_level.csv", index=False)

    metrics = [
        "mean_aoi", "pooled_p95", "success_rate", "mean_common_power",
        "mean_rho", "common_on_rate", "common_bler_when_on",
    ]
    aggregate = seed.groupby("method")[metrics].agg(["mean", "std"])
    aggregate.columns = ["_".join(c) for c in aggregate.columns]
    aggregate.reset_index().to_csv(out / "summary.csv", index=False)

    pivot = seed.pivot(index="training_seed", columns="method", values="mean_aoi")
    comparisons = []
    for candidate in ("saa8_fair_endtoend",):
        for reference in ("single_fair_endtoend", "scratch_rsma", "sdma_power"):
            delta = pivot[candidate] - pivot[reference]
            comparisons.append({
                "comparison": f"{candidate}_minus_{reference}",
                "n_pairs": len(delta),
                "mean_delta_aoi": delta.mean(),
                "std_delta_aoi": delta.std(ddof=1),
                "saa8_wins": int((delta < 0).sum()),
            })
    pd.DataFrame(comparisons).to_csv(out / "paired_comparisons.csv", index=False)

    order = ["scratch_rsma", "single_fair_endtoend", "saa8_fair_endtoend", "sdma_power"]
    means = seed.groupby("method").mean(numeric_only=True).loc[order, "mean_aoi"]
    stds = seed.groupby("method").std(numeric_only=True).loc[order, "mean_aoi"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(range(len(order)), means, yerr=stds, capsize=4)
    ax.set_xticks(
        range(len(order)),
        ["Scratch RSMA", "Single warm-start", "SAA-8 fair end-to-end", "SDMA power"],
        rotation=12,
    )
    ax.set_ylabel("Long-run average AoI")
    ax.set_title(r"Directional Rician, $20^\circ$, $\sigma_e=0.2$ (3 training seeds)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=180)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
