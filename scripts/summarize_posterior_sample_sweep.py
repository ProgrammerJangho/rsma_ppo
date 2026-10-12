#!/usr/bin/env python3
"""Summarize the imperfect-CSIT posterior sample-count ablation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = []
    for samples in (1, 4, 8, 16):
        for seed in (0, 1, 2):
            run = ROOT / "runs" / (
                f"objective_actor_rician_20deg_csit02_saa{samples}_seed{seed}_v2"
            )
            data = pd.read_csv(run / "scalars.csv")
            test = data[data["test/predicted_expected_aoi"].notna()].iloc[-1]
            rows.append({
                "posterior_samples": samples,
                "actor_seed": seed,
                "posterior_seed": 700 + seed,
                "test_expected_aoi": test["test/predicted_expected_aoi"],
                "oracle_expected_aoi": test["test/oracle_expected_aoi"],
                "objective_gap": test["test/objective_gap"],
                "gate_accuracy": test["test/gate_accuracy"],
                "power_l1": test["test/power_l1"],
                "rate_l1": test["test/rate_l1"],
                "rho_mae": test["test/rho_mae"],
                "run": str(run.relative_to(ROOT)),
            })
    seed_level = pd.DataFrame(rows)
    out = ROOT / "results" / "rician_20deg_csit02_posterior_sample_sweep_v2"
    out.mkdir(parents=True, exist_ok=True)
    seed_level.to_csv(out / "seed_level.csv", index=False)
    metrics = ["test_expected_aoi", "objective_gap", "gate_accuracy"]
    summary = seed_level.groupby("posterior_samples")[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(out / "summary.csv", index=False)

    pivot = seed_level.pivot(
        index="actor_seed", columns="posterior_samples", values="test_expected_aoi"
    )
    paired = []
    for candidate, reference in ((4, 1), (8, 4), (16, 8)):
        delta = pivot[candidate] - pivot[reference]
        paired.append({
            "comparison": f"S{candidate}_minus_S{reference}",
            "mean_delta_aoi": delta.mean(),
            "std_delta_aoi": delta.std(ddof=1),
            "candidate_wins": int((delta < 0).sum()),
            "n_pairs": len(delta),
        })
    pd.DataFrame(paired).to_csv(out / "paired_comparisons.csv", index=False)

    means = seed_level.groupby("posterior_samples").test_expected_aoi.mean()
    stds = seed_level.groupby("posterior_samples").test_expected_aoi.std(ddof=1)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.errorbar(means.index, means, yerr=stds, marker="o", capsize=4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(means.index, [str(x) for x in means.index])
    ax.set_xlabel("Posterior samples per observed state")
    ax.set_ylabel("Held-out expected AoI")
    ax.set_title(r"Posterior-SAA ablation: $20^\circ$, $\sigma_e=0.2$")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "sample_sweep.png", dpi=180)
    fig.savefig(out / "sample_sweep.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
