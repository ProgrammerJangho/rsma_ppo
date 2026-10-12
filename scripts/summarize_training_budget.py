#!/usr/bin/env python3
"""Summarize the matched 0/65k/262k PPO budget audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPARISONS = (
    ("rsma_65k_vs_0step", "rsma_65k", "rsma_0step"),
    ("rsma_262k_vs_65k", "rsma_262k", "rsma_65k"),
    ("rsma_262k_vs_0step", "rsma_262k", "rsma_0step"),
    ("sdma_65k_vs_0step", "sdma_65k", "sdma_0step"),
    ("sdma_262k_vs_65k", "sdma_262k", "sdma_65k"),
    ("sdma_262k_vs_0step", "sdma_262k", "sdma_0step"),
    ("rsma_vs_sdma_0step", "rsma_0step", "sdma_0step"),
    ("rsma_vs_sdma_65k", "rsma_65k", "sdma_65k"),
    ("rsma_vs_sdma_262k", "rsma_262k", "sdma_262k"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", default="results/thesis_training_budget_v1")
    args = parser.parse_args(argv)
    out = Path(args.results_dir)
    episodes = pd.read_csv(out / "episodes.csv")
    seed_means = (
        episodes.groupby(["algorithm", "training_seed"], as_index=False)["mean_aoi"]
        .mean()
    )

    rows: list[dict] = []
    for name, candidate, reference in COMPARISONS:
        left = seed_means[seed_means.algorithm == candidate].set_index("training_seed")["mean_aoi"]
        right = seed_means[seed_means.algorithm == reference].set_index("training_seed")["mean_aoi"]
        common = left.index.intersection(right.index)
        delta = left.loc[common] - right.loc[common]
        rows.append(
            {
                "contrast": name,
                "candidate": candidate,
                "reference": reference,
                "candidate_mean_aoi": left.loc[common].mean(),
                "reference_mean_aoi": right.loc[common].mean(),
                "mean_paired_delta": delta.mean(),
                "paired_delta_std": delta.std(ddof=1),
                "mean_paired_percent": 100.0 * delta.mean() / right.loc[common].mean(),
                "candidate_wins": int((delta < 0).sum()),
                "paired_seeds": len(common),
            }
        )
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(out / "budget_contrasts.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.arange(3)
    for prefix, label, color, marker in (
        ("rsma", "RSMA joint+gate", "#2563eb", "o"),
        ("sdma", "SDMA power", "#dc2626", "s"),
    ):
        means, errors = [], []
        for suffix in ("0step", "65k", "262k"):
            values = seed_means[seed_means.algorithm == f"{prefix}_{suffix}"].mean_aoi
            means.append(values.mean())
            errors.append(values.std(ddof=1))
        ax.errorbar(x, means, yerr=errors, marker=marker, capsize=4, lw=2, label=label, color=color)
    ax.set_xticks(x, ["0 PPO steps", "65,536", "262,144"])
    ax.set_ylabel("Long-run mean AoI per user")
    ax.set_xlabel("PPO interaction budget")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "training_budget_aoi.png", dpi=220)
    fig.savefig(out / "training_budget_aoi.pdf")
    plt.close(fig)

    print(contrasts.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
