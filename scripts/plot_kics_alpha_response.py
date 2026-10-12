#!/usr/bin/env python3
"""Figure 2 of the KICS two-page paper: learned common-power fraction alpha.

Feasible regimes only (SNR >= 10 dB, n >= 128). Each point averages the other
two axes per training seed first; whiskers are 95% Student-t intervals over the
five training seeds. Source: the frozen 36-regime grid evaluation of the final
(target-fine-tuned) checkpoints.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "results/ppo_k3_mixed_phy_seed_expansion_grid_v1/regime_by_training_seed.csv"

SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID_INK = "#dcdbd5"


def marginal(frame: pd.DataFrame, axis: str) -> pd.DataFrame:
    per_seed = frame.groupby(["training_seed", axis]).alpha_mean.mean()
    out = per_seed.groupby(axis).agg(["mean", "std", "count"])
    out["hw"] = stats.t.ppf(0.975, out["count"] - 1) * out["std"] / np.sqrt(out["count"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default=str(GRID))
    ap.add_argument("--out", default=str(ROOT / "output/kics/figure_alpha_feasible.png"))
    args = ap.parse_args()
    grid = pd.read_csv(args.grid)
    feasible = grid[(grid.snr_db >= 10) & (grid.blocklength >= 128)]
    panels = [
        ("snr_db", "SNR (dB)", "(a)"),
        ("geometry_deg", "Angular span (deg)", "(b)"),
    ]
    plt.rcParams.update({
        "font.size": 7, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.linewidth": 0.6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.35), dpi=300, sharey=True)
    for ax, (axis, label, tag) in zip(axes, panels):
        m = marginal(feasible, axis)
        x = m.index.to_numpy(dtype=float)
        ax.errorbar(x, m["mean"], yerr=m["hw"], color=SERIES, lw=1.2, marker="o",
                    ms=3.2, capsize=2, elinewidth=0.8, markeredgecolor="white",
                    markeredgewidth=0.6)
        for xi, yi, hi in zip(x, m["mean"], m["hw"]):
            ax.annotate(f"{yi:.2f}", (xi, yi + hi), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6, color=INK)
        ax.set_xticks(x)
        ax.set_xlabel(f"{tag} {label}")
        ax.set_ylim(0.3, 0.95)
        ax.grid(axis="y", color=GRID_INK, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        margin = 0.08 * (x.max() - x.min())
        ax.set_xlim(x.min() - margin, x.max() + margin)
    axes[0].set_ylabel(r"Common power $\alpha$")
    fig.tight_layout(pad=0.3, w_pad=0.8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
