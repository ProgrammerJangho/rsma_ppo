#!/usr/bin/env python3
"""Figure of the KICS two-page paper: target fine-tuning return curves.

Same data and smoothing as ``build_thesis_mixed_phy_artifacts.py`` panel (b):
the logged environment episode return (sum over a 512-slot episode of the
reward -(1/K) sum_k Delta_k(t+1)), 8-point moving average per training seed,
with the seed mean and +-1 standard deviation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
RUN = "runs/ppo_k3_20deg_sinr_from_stage1_seed{seed}_524k_v1/scalars.csv"

SEED_INK = "#9fb8d9"
MEAN_INK = "#0b0b0b"
BAND = "#c9c8c1"
INK_2 = "#52514e"
GRID_INK = "#dcdbd5"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--run-template", default=RUN)
    ap.add_argument("--out", default=str(ROOT / "output/kics/figure_return_curve.png"))
    args = ap.parse_args()
    curves = []
    for seed in args.seeds:
        scalars = pd.read_csv(ROOT / args.run_template.format(seed=seed))
        ret = scalars[["step", "episode/r"]].dropna()
        smooth = ret["episode/r"].rolling(8, min_periods=1).mean()
        curves.append(pd.Series(smooth.to_numpy(), index=ret.step))
    frame = pd.concat(curves, axis=1)
    mean, std = frame.mean(axis=1), frame.std(axis=1, ddof=1)

    plt.rcParams.update({
        "font.size": 7, "axes.edgecolor": INK_2, "axes.labelcolor": MEAN_INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.linewidth": 0.6,
    })
    fig, ax = plt.subplots(figsize=(3.3, 1.45), dpi=300)
    ax.fill_between(frame.index, mean - std, mean + std, color=BAND, alpha=0.6, lw=0,
                    label="Mean ± 1 SD")
    for i, col in enumerate(frame.columns):
        ax.plot(frame.index, frame[col], color=SEED_INK, lw=0.7,
                label="Training seed (×5)" if i == 0 else None)
    ax.plot(frame.index, mean, color=MEAN_INK, lw=1.4, label="Seed mean")
    ax.set_xlabel("Target fine-tuning step")
    ax.set_ylabel("Episode return")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x/1000:.0f}k"))
    ax.set_xlim(0, frame.index.max())
    ax.grid(axis="y", color=GRID_INK, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=6, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print("saved", args.out)
    print("start mean %.1f, end mean %.1f" % (mean.iloc[0], mean.iloc[-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
