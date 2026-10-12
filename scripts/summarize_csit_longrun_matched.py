#!/usr/bin/env python3
"""Aggregate completed seed-matched long-run CSIT experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ERROR_LEVELS = {
    "00": (0.0, "rician_ppo_20deg_csit00_controls_3seed_v2"),
    "01": (0.1, "rician_ppo_20deg_csit01_controls_3seed_v2"),
    "02": (0.2, "rician_ppo_20deg_csit02_controls_3seed_v1"),
    "03": (0.3, "rician_ppo_20deg_csit03_controls_3seed_v2"),
}


def deterministic_row(path: Path, algorithm: str) -> pd.DataFrame:
    data = pd.read_csv(path / "training_seed_summary.csv")
    return data[
        (data.algorithm == algorithm) & (data.execution == "deterministic")
    ].copy()


def main() -> None:
    frames: list[pd.DataFrame] = []
    for token, (sigma_e, control_dir) in ERROR_LEVELS.items():
        controls = RESULTS / control_dir
        for algorithm, method in (
            ("joint_gate", "scratch_rsma"),
            ("sdma_power", "sdma_power"),
        ):
            rows = deterministic_row(controls, algorithm)
            rows["method"] = method
            rows["sigma_e"] = sigma_e
            frames.append(rows)

        for stem, method in (("single", "single"), ("saa8", "saa8")):
            for seed in (0, 1, 2):
                directory = RESULTS / (
                    f"rician_ppo_20deg_csit{token}_{stem}_"
                    f"fair_endtoend_seed{seed}_v2"
                )
                rows = deterministic_row(directory, "joint_gate")
                rows["method"] = method
                rows["sigma_e"] = sigma_e
                frames.append(rows)

    seed_level = pd.concat(frames, ignore_index=True)
    out = RESULTS / "rician_ppo_20deg_csit_matched_sweep_v2"
    out.mkdir(parents=True, exist_ok=True)
    seed_level.to_csv(out / "seed_level.csv", index=False)

    metrics = [
        "mean_aoi", "pooled_p95", "max_aoi", "success_rate",
        "mean_common_power", "common_on_rate", "mean_rho",
    ]
    summary = seed_level.groupby(["sigma_e", "method"])[metrics].agg(
        ["mean", "std"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(out / "summary.csv", index=False)

    paired = []
    for sigma_e, group in seed_level.groupby("sigma_e"):
        pivot = group.pivot(
            index="training_seed", columns="method", values="mean_aoi"
        )
        for candidate in ("single", "saa8"):
            for reference in ("scratch_rsma", "sdma_power"):
                delta = pivot[candidate] - pivot[reference]
                paired.append({
                    "sigma_e": sigma_e,
                    "candidate": candidate,
                    "reference": reference,
                    "mean_delta_aoi": delta.mean(),
                    "std_delta_aoi": delta.std(ddof=1),
                    "wins": int((delta < 0).sum()),
                    "n_pairs": len(delta),
                })
        delta = pivot["saa8"] - pivot["single"]
        paired.append({
            "sigma_e": sigma_e,
            "candidate": "saa8",
            "reference": "single",
            "mean_delta_aoi": delta.mean(),
            "std_delta_aoi": delta.std(ddof=1),
            "wins": int((delta < 0).sum()),
            "n_pairs": len(delta),
        })
    pd.DataFrame(paired).to_csv(out / "paired_comparisons.csv", index=False)

    plot = seed_level.groupby(["sigma_e", "method"]).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = {
        "scratch_rsma": "Scratch RSMA",
        "single": "Single warm-start",
        "saa8": "Posterior-SAA-8",
        "sdma_power": "SDMA power",
    }
    for method, label in labels.items():
        values = plot.xs(method, level="method")
        ax.plot(values.index, values.mean_aoi, marker="o", label=label)
    ax.set_xlabel(r"Relative CSIT error $\sigma_e$")
    ax.set_ylabel("Long-run average AoI")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "csit_longrun.png", dpi=180)
    fig.savefig(out / "csit_longrun.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
