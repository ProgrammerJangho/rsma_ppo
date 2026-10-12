#!/usr/bin/env python3
"""Summarize the seed-matched objective-aware PPO action ablation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "rician_ppo_20deg_csit02_objective_action_ablation_v2"


def select(path: Path, algorithm: str) -> pd.DataFrame:
    data = pd.read_csv(path / "training_seed_summary.csv")
    return data[
        (data.algorithm == algorithm) & (data.execution == "deterministic")
    ].copy()


def main() -> None:
    frames: list[pd.DataFrame] = []
    for mode in ("power_only", "split_only", "joint"):
        for seed in (0, 1, 2):
            directory = RESULTS / (
                f"rician_ppo_20deg_csit02_{mode}_objective_seed{seed}_v2"
            )
            rows = select(directory, mode)
            rows["method"] = mode
            frames.append(rows)

    for seed in (0, 1, 2):
        directory = RESULTS / (
            f"rician_ppo_20deg_csit02_single_fair_endtoend_seed{seed}_v2"
        )
        rows = select(directory, "joint_gate")
        rows["method"] = "joint_gate"
        frames.append(rows)

    sdma = select(
        RESULTS / "rician_ppo_20deg_csit02_controls_3seed_v1", "sdma_power"
    )
    sdma["method"] = "sdma_power"
    frames.append(sdma)

    seed_level = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    seed_level.to_csv(OUT / "seed_level.csv", index=False)

    power_rho = seed_level.groupby("method").agg(
        common_power=("mean_common_power", "mean"),
        rho=("mean_rho", "mean"),
        common_on=("common_on_rate", "mean"),
    )
    if not np.isclose(power_rho.loc["power_only", "rho"], 0.3, atol=1e-6):
        raise AssertionError("power_only did not keep rho fixed at 0.3")
    if not np.isclose(
        power_rho.loc["split_only", "common_power"], 0.25, atol=1e-6
    ):
        raise AssertionError("split_only did not keep common power at 0.25")
    if not np.isclose(power_rho.loc["joint", "common_on"], 1.0, atol=1e-6):
        raise AssertionError("joint no-gate arm unexpectedly switched common off")
    if not np.isclose(power_rho.loc["sdma_power", "common_on"], 0.0, atol=1e-9):
        raise AssertionError("SDMA arm used a common stream")

    metrics = [
        "mean_aoi", "pooled_p95", "max_aoi", "success_rate",
        "mean_common_power", "common_on_rate", "mean_rho",
        "common_bler_when_on", "mean_private_bler",
    ]
    summary = seed_level.groupby("method")[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(OUT / "summary.csv", index=False)

    pivot = seed_level.pivot(
        index="training_seed", columns="method", values="mean_aoi"
    )
    comparisons = []
    pairs = [
        ("joint", "power_only"),
        ("joint", "split_only"),
        ("joint_gate", "joint"),
        ("joint_gate", "power_only"),
        ("joint_gate", "split_only"),
        ("joint_gate", "sdma_power"),
    ]
    for candidate, reference in pairs:
        delta = pivot[candidate] - pivot[reference]
        comparisons.append({
            "candidate": candidate,
            "reference": reference,
            "mean_delta_aoi": delta.mean(),
            "std_delta_aoi": delta.std(ddof=1),
            "candidate_wins": int((delta < 0).sum()),
            "n_pairs": len(delta),
        })
    pd.DataFrame(comparisons).to_csv(
        OUT / "paired_comparisons.csv", index=False
    )

    order = ["power_only", "split_only", "joint", "joint_gate", "sdma_power"]
    grouped = seed_level.groupby("method").mean(numeric_only=True).loc[order]
    errors = seed_level.groupby("method").std(numeric_only=True).loc[order]
    labels = ["Power only", "Rate only", "Joint", "Joint + gate", "SDMA"]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.bar(labels, grouped.mean_aoi, yerr=errors.mean_aoi, capsize=4)
    ax.set_ylabel("Long-run average AoI")
    ax.set_title(r"Action ablation: $20^\circ$, $\sigma_e=0.2$")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "action_ablation.png", dpi=180)
    fig.savefig(OUT / "action_ablation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
