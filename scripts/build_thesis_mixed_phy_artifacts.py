#!/usr/bin/env python3
"""Build frozen thesis tables, figures and a manifest from the five-seed study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "results/ppo_k3_mixed_phy_seed_expansion_target_v1"
GRID = ROOT / "results/ppo_k3_mixed_phy_seed_expansion_grid_v1"
OUT = ROOT / "results/thesis_mixed_phy_final_v1"
T_CRIT_DF4 = 2.7764451051977987


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_latex(frame: pd.DataFrame, name: str, float_format: str = "%.3f") -> None:
    (OUT / name).write_text(
        frame.to_latex(
            index=False, float_format=float_format, escape=True, na_rep="--"
        ),
        encoding="utf-8",
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    seed = pd.read_csv(TARGET / "target_seed_summary.csv")
    trajectories = pd.read_csv(TARGET / "target_trajectories.csv")
    paired = pd.read_csv(TARGET / "target_paired_training_seed_comparison.csv")
    alpha = pd.read_csv(GRID / "alpha_marginal_aggregate.csv")
    regime = pd.read_csv(GRID / "performance_vs_sdma_aggregate.csv")

    rows = []
    for label, display in (("warm", "Objective-pretrained actor"), ("ppo", "Final PPO")):
        block = seed[seed.label == label]
        n = len(block)
        rows.append({
            "method": display,
            "training_seeds": n,
            "mean_aoi": block.mean_aoi.mean(),
            "aoi_seed_std": block.mean_aoi.std(ddof=1),
            "aoi_ci95_halfwidth": T_CRIT_DF4 * block.mean_aoi.std(ddof=1) / np.sqrt(n),
            "mean_p95_aoi": block.p95_aoi.mean(),
            "mean_success_probability": block.success_probability.mean(),
            "mean_alpha": block.alpha_mean.mean(),
            "alpha_seed_std": block.alpha_mean.std(ddof=1),
        })
    sdma = trajectories[trajectories.label == "sdma"]
    rows.append({
        "method": "SDMA",
        "training_seeds": 0,
        "mean_aoi": sdma.mean_aoi.mean(),
        "aoi_seed_std": np.nan,
        "aoi_ci95_halfwidth": np.nan,
        "mean_p95_aoi": sdma.p95_aoi.mean(),
        "mean_success_probability": sdma.success_probability.mean(),
        "mean_alpha": 0.0,
        "alpha_seed_std": np.nan,
    })
    target_table = pd.DataFrame(rows)
    target_table.to_csv(OUT / "table_1_target_performance.csv", index=False)
    write_latex(target_table, "table_1_target_performance.tex")

    target_seed_table = seed[seed.label == "ppo"].copy()
    target_seed_table = target_seed_table.merge(
        paired[["training_seed", "ppo_gain_vs_own_warm_pct"]], on="training_seed"
    )
    target_seed_table.to_csv(OUT / "table_2_target_by_training_seed.csv", index=False)
    write_latex(target_seed_table, "table_2_target_by_training_seed.tex")

    alpha_table = alpha.copy()
    alpha_table["alpha_ci95_halfwidth"] = (
        T_CRIT_DF4 * alpha_table.alpha_std / np.sqrt(alpha_table.training_seeds)
    )
    alpha_table.to_csv(OUT / "table_3_alpha_response.csv", index=False)
    write_latex(alpha_table, "table_3_alpha_response.tex")

    regime.to_csv(OUT / "table_4_regime_performance.csv", index=False)
    write_latex(regime, "table_4_regime_performance.tex")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    warm = seed[seed.label == "warm"].sort_values("training_seed")
    ppo = seed[seed.label == "ppo"].sort_values("training_seed")
    for x, block, color, label in (
        (0, warm, "tab:orange", "Pretrained actor"),
        (1, ppo, "tab:blue", "Final PPO"),
    ):
        jitter = np.linspace(-0.07, 0.07, len(block))
        axis.scatter(x + jitter, block.mean_aoi, color=color, s=32, alpha=0.85)
        axis.errorbar(
            x, block.mean_aoi.mean(), yerr=block.mean_aoi.std(ddof=1),
            fmt="s", color="black", capsize=5, zorder=4,
        )
    axis.scatter(2, sdma.mean_aoi.mean(), marker="D", color="black", s=55)
    axis.set(
        xticks=[0, 1, 2], xticklabels=["Pretrained", "PPO", "SDMA"],
        ylabel="Mean AoI (slots)", title="Target operating point",
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"figure_1_target_performance.{suffix}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    specifications = (
        ("geometry_deg", "Geometry separation (deg)"),
        ("snr_db", "SNR (dB)"),
        ("blocklength", "Blocklength"),
    )
    for ax, (name, xlabel) in zip(axes, specifications, strict=True):
        block = alpha_table[alpha_table.axis == name].sort_values("level")
        ax.errorbar(
            block.level, block.alpha_mean, yerr=block.alpha_ci95_halfwidth,
            marker="o", linewidth=2, capsize=4,
        )
        ax.set(xlabel=xlabel, ylabel="Common-power fraction alpha", ylim=(0, 1))
        ax.set_xticks(block.level)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"figure_2_alpha_response.{suffix}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8))
    convergence_rows = []
    smoothed_curves = []
    return_curves = []
    for training_seed in range(5):
        path = ROOT / f"runs/ppo_k3_20deg_sinr_from_stage1_seed{training_seed}_524k_v1/scalars.csv"
        scalar = pd.read_csv(path)
        curve = scalar[["step", "env/mean_aoi"]].dropna()
        smooth = curve["env/mean_aoi"].rolling(8, min_periods=1).mean()
        smoothed_curves.append(smooth.to_numpy())
        color = f"C{training_seed}"
        axes[0].plot(
            curve.step, smooth, linewidth=1.1, alpha=0.65,
            color=color, label=f"seed {training_seed}",
        )
        return_curve = scalar[["step", "episode/r"]].dropna()
        smooth_return = return_curve["episode/r"].rolling(8, min_periods=1).mean()
        return_curves.append(pd.Series(smooth_return.to_numpy(), index=return_curve.step))
        axes[1].plot(
            return_curve.step, smooth_return, linewidth=1.1, alpha=0.65,
            color=color,
        )
        convergence_rows.append({
            "training_seed": training_seed,
            "initial_smoothed_training_aoi": float(smooth.iloc[0]),
            "final_smoothed_training_aoi": float(smooth.iloc[-1]),
            "late_training_aoi_std": float(curve["env/mean_aoi"].tail(16).std(ddof=1)),
            "initial_smoothed_logged_return": float(smooth_return.iloc[0]),
            "final_smoothed_logged_return": float(smooth_return.iloc[-1]),
            "target_mean_aoi": float(
                ppo.loc[ppo.training_seed == training_seed, "mean_aoi"].iloc[0]
            ),
        })
    stacked = np.vstack(smoothed_curves)
    mean_curve = stacked.mean(axis=0)
    std_curve = stacked.std(axis=0, ddof=1)
    axes[0].plot(curve.step, mean_curve, color="black", linewidth=2.2, label="seed mean")
    axes[0].fill_between(
        curve.step, mean_curve - std_curve, mean_curve + std_curve,
        color="black", alpha=0.10, linewidth=0,
    )
    axes[0].set(
        xlabel="Target fine-tuning step",
        ylabel="On-policy mean AoI (8-point mean)",
        title="(a) On-policy AoI",
    )
    axes[0].legend(ncol=3, fontsize=7.3)

    return_frame = pd.concat(return_curves, axis=1)
    return_mean = return_frame.mean(axis=1)
    return_std = return_frame.std(axis=1, ddof=1)
    axes[1].plot(
        return_frame.index, return_mean, color="black", linewidth=2.2,
        label="seed mean",
    )
    axes[1].fill_between(
        return_frame.index.to_numpy(),
        (return_mean - return_std).to_numpy(),
        (return_mean + return_std).to_numpy(),
        color="black", alpha=0.10, linewidth=0,
    )
    axes[1].set(
        xlabel="Target fine-tuning step",
        ylabel="Logged episode return (8-point mean)",
        title="(b) Environment episodic return",
    )
    fig.suptitle("PPO fine-tuning convergence diagnostics", fontsize=13)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"figure_3_training_curves.{suffix}", dpi=300)
    plt.close(fig)
    convergence = pd.DataFrame(convergence_rows)
    convergence.to_csv(OUT / "table_5_training_convergence.csv", index=False)
    write_latex(convergence, "table_5_training_convergence.tex")

    checkpoints = []
    q_only_checkpoints = []
    for training_seed in range(5):
        path = ROOT / f"runs/ppo_k3_20deg_sinr_from_stage1_seed{training_seed}_524k_v1/checkpoint.pt"
        checkpoints.append({
            "training_seed": training_seed,
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        })
        q_path = ROOT / f"runs/q_only_ppo_k3_20deg_from_stage1_seed{training_seed}_524k_v1/checkpoint.pt"
        q_only_checkpoints.append({
            "training_seed": training_seed,
            "path": str(q_path.relative_to(ROOT)),
            "sha256": sha256(q_path),
        })
    frozen_files = [
        ROOT / "configs/thesis_rician_k3_nt2_mixed_phy_sinr.yaml",
        ROOT / "configs/thesis_rician_k3_nt2_mixed_phy_sinr_stage1.yaml",
        ROOT / "configs/thesis_rician_k3_nt2_20deg_sinr_finetune.yaml",
        ROOT / "scripts/run_mixed_phy_seed_expansion.sh",
        ROOT / "scripts/evaluate_mixed_phy_seed_expansion.py",
        ROOT / "scripts/evaluate_mixed_phy_multiseed_grid.py",
        ROOT / "scripts/build_thesis_mixed_phy_artifacts.py",
        ROOT / "scripts/run_q_only_sdma_seed_expansion.sh",
        ROOT / "scripts/evaluate_q_only_sdma_target.py",
        ROOT / "scripts/build_q_only_sdma_artifacts.py",
        ROOT / "scripts/evaluate_thesis_rule_baselines.py",
        ROOT / "scripts/evaluate_max_aoi_single_user_grid.py",
        ROOT / "scripts/build_thesis_rule_baseline_artifacts.py",
        ROOT / "scripts/audit_current_common_off_and_aoi_actions.py",
        ROOT / "src/rsma_aoi/baselines/thesis_rules.py",
        ROOT / "docs/canonical_action_definition.md",
        ROOT / "docs/thesis_results_chapter_2026-09-22.md",
        ROOT / "docs/thesis_manuscript_draft_2026-09-22.md",
        ROOT / "docs/advisor_results_summary_2026-09-22.md",
        ROOT / "docs/current_common_off_aoi_action_audit_2026-09-22.md",
        ROOT / "docs/q_only_ppo_sdma_results_2026-09-22.md",
        ROOT / "docs/thesis_rule_baselines_results_2026-09-22.md",
        ROOT / "docs/thesis_references.bib",
        ROOT / "results/thesis_current_common_off_aoi_audit_v1/aggregate.json",
        ROOT / "results/q_only_sdma_target_v1/summary.json",
        ROOT / "results/q_only_sdma_target_v1/target_aggregate.csv",
        ROOT / "results/q_only_sdma_target_v1/paired_training_seed_comparison.csv",
        ROOT / "results/thesis_rule_baselines_target_v1/target_summary.csv",
        ROOT / "results/thesis_rule_baselines_target_v1/fixed_rsma_validation_summary.csv",
        ROOT / "results/max_aoi_single_user_grid_v1/stratum_aggregate.csv",
    ]
    manifest = {
        "name": "thesis_mixed_phy_final_v1",
        "frozen_on": "2026-09-22",
        "training_seeds": [0, 1, 2, 3, 4],
        "target": {"geometry_deg": 20, "snr_db": 15, "blocklength": 128},
        "evaluation": {"steps": 768, "burn_in": 256, "permutations": 6},
        "checkpoints": checkpoints,
        "q_only_checkpoints": q_only_checkpoints,
        "frozen_files": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in frozen_files
        ],
        "source_results": [
            str(TARGET.relative_to(ROOT)), str(GRID.relative_to(ROOT))
        ],
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(target_table.to_string(index=False))
    print(f"\nWrote frozen thesis artifacts to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
