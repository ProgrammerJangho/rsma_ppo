#!/usr/bin/env python3
"""Summarize fixed-rho and fixed-common-power PPO sensitivity curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "rician_ppo_20deg_csit02_fixed_action_sensitivity_v1"

POWER_ONLY = {
    0.05: "rho005",
    0.10: "rho010",
    0.20: "rho020",
    0.30: None,
}
SPLIT_ONLY = {
    0.10: "pc010",
    0.25: None,
    0.40: "pc040",
    0.60: "pc060",
}


def deterministic(path: Path, algorithm: str) -> pd.DataFrame:
    data = pd.read_csv(path / "training_seed_summary.csv")
    return data[
        (data.algorithm == algorithm) & (data.execution == "deterministic")
    ].copy()


def restricted_directory(mode: str, label: str | None, seed: int) -> Path:
    if label is None:
        return RESULTS / (
            f"rician_ppo_20deg_csit02_{mode}_objective_seed{seed}_v2"
        )
    return RESULTS / (
        f"rician_ppo_20deg_csit02_{mode}_{label}_objective_seed{seed}_v1"
    )


def main() -> None:
    frames: list[pd.DataFrame] = []
    for fixed_rho, label in POWER_ONLY.items():
        for seed in (0, 1, 2):
            rows = deterministic(
                restricted_directory("power_only", label, seed), "power_only"
            )
            rows["family"] = "power_only"
            rows["fixed_value"] = fixed_rho
            rows["setting"] = f"rho={fixed_rho:.2f}"
            frames.append(rows)
    for fixed_power, label in SPLIT_ONLY.items():
        for seed in (0, 1, 2):
            rows = deterministic(
                restricted_directory("split_only", label, seed), "split_only"
            )
            rows["family"] = "split_only"
            rows["fixed_value"] = fixed_power
            rows["setting"] = f"pc={fixed_power:.2f}"
            frames.append(rows)

    for mode in ("joint",):
        for seed in (0, 1, 2):
            rows = deterministic(
                RESULTS
                / f"rician_ppo_20deg_csit02_{mode}_objective_seed{seed}_v2",
                mode,
            )
            rows["family"] = mode
            rows["fixed_value"] = np.nan
            rows["setting"] = mode
            frames.append(rows)
    for seed in (0, 1, 2):
        rows = deterministic(
            RESULTS
            / f"rician_ppo_20deg_csit02_single_fair_endtoend_seed{seed}_v2",
            "joint_gate",
        )
        rows["family"] = "joint_gate"
        rows["fixed_value"] = np.nan
        rows["setting"] = "joint_gate"
        frames.append(rows)

    seed_level = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    seed_level.to_csv(OUT / "seed_level.csv", index=False)

    for fixed_rho in POWER_ONLY:
        observed = seed_level[
            (seed_level.family == "power_only")
            & (seed_level.fixed_value == fixed_rho)
        ].mean_rho
        if not np.allclose(observed, fixed_rho, atol=1e-6):
            raise AssertionError(f"rho={fixed_rho} was not fixed in the trace")
    for fixed_power in SPLIT_ONLY:
        observed = seed_level[
            (seed_level.family == "split_only")
            & (seed_level.fixed_value == fixed_power)
        ].mean_common_power
        if not np.allclose(observed, fixed_power, atol=1e-6):
            raise AssertionError(
                f"common power={fixed_power} was not fixed in the trace"
            )

    metrics = [
        "mean_aoi", "pooled_p95", "max_aoi", "success_rate",
        "mean_common_power", "mean_rho", "common_bler_when_on",
        "mean_private_bler",
    ]
    summary = seed_level.groupby(
        ["family", "fixed_value", "setting"], dropna=False
    )[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(OUT / "summary.csv", index=False)

    pivot = seed_level.pivot(
        index="training_seed", columns="setting", values="mean_aoi"
    )
    comparisons = []
    for setting in [
        *(f"rho={value:.2f}" for value in POWER_ONLY),
        *(f"pc={value:.2f}" for value in SPLIT_ONLY),
    ]:
        for reference in ("joint", "joint_gate"):
            delta = pivot[setting] - pivot[reference]
            comparisons.append({
                "setting": setting,
                "reference": reference,
                "mean_delta_aoi": delta.mean(),
                "std_delta_aoi": delta.std(ddof=1),
                "setting_wins": int((delta < 0).sum()),
                "n_pairs": len(delta),
            })
    pd.DataFrame(comparisons).to_csv(
        OUT / "paired_comparisons.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, family, label in (
        (axes[0], "power_only", r"Fixed common-payload fraction $\rho$"),
        (axes[1], "split_only", "Fixed common-power fraction"),
    ):
        data = seed_level[seed_level.family == family]
        means = data.groupby("fixed_value").mean(numeric_only=True)
        stds = data.groupby("fixed_value").std(numeric_only=True)
        ax.errorbar(
            means.index, means.mean_aoi, yerr=stds.mean_aoi,
            marker="o", capsize=4, label=family,
        )
        joint = seed_level[seed_level.family == "joint"].mean_aoi.mean()
        gated = seed_level[seed_level.family == "joint_gate"].mean_aoi.mean()
        ax.axhline(joint, color="#475569", linestyle="--", label="Joint")
        ax.axhline(gated, color="#2563eb", linestyle=":", label="Joint + gate")
        ax.set_xlabel(label)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Long-run average AoI")
    axes[1].legend(fontsize=8)
    fig.suptitle(r"Restricted-action sensitivity: $20^\circ$, $\sigma_e=0.2$")
    fig.tight_layout()
    fig.savefig(OUT / "fixed_action_sensitivity.png", dpi=180)
    fig.savefig(OUT / "fixed_action_sensitivity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
