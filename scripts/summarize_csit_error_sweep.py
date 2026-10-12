#!/usr/bin/env python3
"""Summarize fair-compute SAA-8 actor pretraining across CSIT errors."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TAGS = {0.0: "00", 0.1: "01", 0.2: "02", 0.3: "03"}


def main() -> None:
    rows = []
    for sigma, tag in TAGS.items():
        for seed in (0, 1, 2):
            run = ROOT / "runs" / f"objective_actor_rician_20deg_csit{tag}_saa8_seed{seed}_v2"
            data = pd.read_csv(run / "scalars.csv")
            test = data[data["test/predicted_expected_aoi"].notna()].iloc[-1]
            rows.append({
                "csit_error_std": sigma,
                "actor_seed": seed,
                "test_expected_aoi": test["test/predicted_expected_aoi"],
                "oracle_expected_aoi": test["test/oracle_expected_aoi"],
                "objective_gap": test["test/objective_gap"],
                "gate_accuracy": test["test/gate_accuracy"],
                "run": str(run.relative_to(ROOT)),
            })
    seed_level = pd.DataFrame(rows)
    out = ROOT / "results" / "rician_20deg_csit_saa8_actor_sweep_v2"
    out.mkdir(parents=True, exist_ok=True)
    seed_level.to_csv(out / "seed_level.csv", index=False)
    metrics = ["test_expected_aoi", "oracle_expected_aoi", "objective_gap", "gate_accuracy"]
    summary = seed_level.groupby("csit_error_std")[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(out / "summary.csv", index=False)

    mean = seed_level.groupby("csit_error_std").test_expected_aoi.mean()
    std = seed_level.groupby("csit_error_std").test_expected_aoi.std(ddof=1)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.errorbar(mean.index, mean, yerr=std, marker="o", capsize=4, label="SAA-8 actor")
    oracle = seed_level.groupby("csit_error_std").oracle_expected_aoi.mean()
    ax.plot(oracle.index, oracle, marker="s", label="plug-in CEM label")
    ax.set_xlabel(r"CSIT error $\sigma_e$")
    ax.set_ylabel("Held-out one-step expected AoI")
    ax.set_title(r"Directional Rician, $20^\circ$: fair SAA-8 actor")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "csit_sweep.png", dpi=180)
    fig.savefig(out / "csit_sweep.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
