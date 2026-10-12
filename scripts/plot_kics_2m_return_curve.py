#!/usr/bin/env python3
"""Stitched 2.10M-step target fine-tuning return curves for the KICS paper.

Each training seed is trained in three resumed stages (0-0.52M, 0.52-1.05M,
1.05-2.10M cumulative steps). Every stage restores actor, critic, optimizer and
normalisers but restarts learning-rate annealing over its own budget, so the
stage boundaries are marked. Logging intervals differ by stage (4,096 / 8,192 /
16,384 steps), so smoothing uses a fixed step window inside each stage instead
of a fixed number of log points.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "RSMA-PPO": (
        "runs/matched_objective_rsma_ppo_k3_20deg_from_stage1_seed{seed}_524k_v1",
        "runs/matched_objective_rsma_ppo_k3_20deg_seed{seed}_continue_524k_v1",
        "runs/matched_objective_rsma_ppo_k3_20deg_seed{seed}_continue_to_2097k_v1",
    ),
    "SDMA-PPO": (
        "runs/q_only_ppo_k3_20deg_from_stage1_seed{seed}_524k_v1",
        "runs/q_only_ppo_k3_20deg_seed{seed}_continue_524k_v1",
        "runs/q_only_ppo_k3_20deg_seed{seed}_continue_to_2097k_v1",
    ),
}
COLORS = {"RSMA-PPO": "#2a78d6", "SDMA-PPO": "#eb6834"}
INK, INK_2, GRID_INK = "#0b0b0b", "#52514e", "#dcdbd5"
STAGE_ENDS = (524_288, 1_048_576, 2_097_152)


def smoothed_stage(path: Path, window: int) -> pd.Series:
    scalars = pd.read_csv(path / "scalars.csv")
    ret = scalars[["step", "episode/r"]].dropna()
    steps = ret["step"].to_numpy(dtype=float)
    values = ret["episode/r"].to_numpy(dtype=float)
    smooth = np.array([values[np.abs(steps - s) <= window / 2].mean() for s in steps])
    return pd.Series(smooth, index=steps.astype(int))


def family_curves(templates, seeds, window) -> pd.DataFrame:
    columns = []
    for seed in seeds:
        parts = [smoothed_stage(ROOT / template.format(seed=seed), window) for template in templates]
        columns.append(pd.concat(parts).rename(seed))
    return pd.concat(columns, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--window", type=int, default=65_536, help="smoothing window in steps")
    ap.add_argument("--out", default=str(ROOT / "output/kics/figure_return_curve_2m.png"))
    ap.add_argument("--summary", default=str(ROOT / "output/kics/figure_return_curve_2m_summary.csv"))
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": 7, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.linewidth": 0.6,
    })
    fig, ax = plt.subplots(figsize=(3.3, 1.42), dpi=300)
    rows = []
    for name, templates in FAMILIES.items():
        frame = family_curves(templates, args.seeds, args.window)
        if frame.isna().any().any():
            raise ValueError(f"{name}: seeds logged at different steps")
        mean, std = frame.mean(axis=1), frame.std(axis=1, ddof=1)
        for end in STAGE_ENDS:
            stage = frame[frame.index <= end]
            rows.append({"family": name, "stage_end": end,
                         "seed_mean_last": float(stage.iloc[-1].mean()),
                         "seed_min_last": float(stage.iloc[-1].min()),
                         "seed_max_last": float(stage.iloc[-1].max())})
        rows.append({"family": name, "stage_end": 0,
                     "seed_mean_last": float(frame.iloc[0].mean()),
                     "seed_min_last": float(frame.iloc[0].min()),
                     "seed_max_last": float(frame.iloc[0].max())})
        color = COLORS[name]
        ax.fill_between(frame.index, mean - std, mean + std, color=color, alpha=0.18, lw=0)
        ax.plot(frame.index, mean, color=color, lw=1.2, label=f"{name} (5 seeds)")
    for boundary in STAGE_ENDS[:2]:
        ax.axvline(boundary, color=INK_2, lw=0.6, ls=(0, (2, 2)))
    ax.text(STAGE_ENDS[0], 0.98, " LR restart", transform=ax.get_xaxis_transform(),
            fontsize=5.5, color=INK_2, va="top", ha="left")
    ax.text(STAGE_ENDS[1], 0.98, " LR restart", transform=ax.get_xaxis_transform(),
            fontsize=5.5, color=INK_2, va="top", ha="left")
    ax.set_xlabel("Cumulative target fine-tuning step")
    ax.set_ylabel("Episode return")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x/1e6:.2f}M"))
    ax.set_xticks([0, 524_288, 1_048_576, 1_572_864, 2_097_152])
    ax.set_xlim(0, STAGE_ENDS[-1])
    ax.grid(axis="y", color=GRID_INK, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=6, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    pd.DataFrame(rows).sort_values(["family", "stage_end"]).to_csv(args.summary, index=False)
    print("saved", args.out)
    print(pd.DataFrame(rows).sort_values(["family", "stage_end"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
