#!/usr/bin/env python3
"""Create the thesis-ready plot and diagnostics for a PPO convergence audit."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.result_dir.resolve()
    evaluation = pd.read_csv(out / "checkpoint_summary.csv")
    if "seed_std_per_user" not in evaluation:
        # Backward compatibility for audits produced before this column was
        # written.  Equal weights are used in the two-user thesis scenario.
        episode_rows = pd.read_csv(out / "checkpoint_episodes.csv")
        derived = (
            episode_rows.groupby(["variant", "training_seed", "global_step"])["mean_aoi"]
            .mean().groupby(["variant", "global_step"]).std().rename("seed_std_per_user")
            .reset_index()
        )
        evaluation = evaluation.merge(derived, on=["variant", "global_step"], how="left")

    training_frames = []
    diagnostic_rows = []
    for scalars in sorted((out / "runs").glob("*_seed*/scalars.csv")):
        run_name = scalars.parent.name
        variant, raw_seed = run_name.rsplit("_seed", 1)
        frame = pd.read_csv(scalars)
        frame["variant"] = variant
        frame["training_seed"] = int(raw_seed)
        training_frames.append(frame)
        last = frame.tail(min(16, len(frame)))
        diagnostic_rows.append({
            "variant": variant,
            "training_seed": int(raw_seed),
            "updates": len(frame),
            "nonfinite_core": int((~np.isfinite(frame[[
                "env/mean_aoi", "train/value_loss", "train/approx_kl",
                "train/explained_variance", "train/grad_norm",
            ]].to_numpy())).sum()),
            "early_stops": int(frame["train/early_stop"].sum()),
            "max_approx_kl": frame["train/approx_kl"].max(),
            "max_grad_norm_preclip": frame["train/grad_norm"].max(),
            "last16_sampled_mean_aoi": last["env/mean_aoi"].mean(),
            "last16_explained_variance": last["train/explained_variance"].mean(),
            "last16_value_loss": last["train/value_loss"].mean(),
        })
    training = pd.concat(training_frames, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(out / "training_diagnostics.csv", index=False)

    audit_rows = []
    for variant, group in evaluation.groupby("variant"):
        group = group.sort_values("global_step")
        first, last = group.iloc[0], group.iloc[-1]
        best = group.loc[group.mean_per_user_aoi.idxmin()]
        audit_rows.append({
            "variant": variant,
            "initial_mean_aoi": first.mean_per_user_aoi,
            "final_mean_aoi": last.mean_per_user_aoi,
            "absolute_improvement": first.mean_per_user_aoi - last.mean_per_user_aoi,
            "relative_improvement_pct": 100 * (
                first.mean_per_user_aoi - last.mean_per_user_aoi
            ) / first.mean_per_user_aoi,
            "best_mean_aoi": best.mean_per_user_aoi,
            "best_global_step": int(best.global_step),
            "final_minus_best": last.mean_per_user_aoi - best.mean_per_user_aoi,
            "step_aoi_spearman": group.global_step.corr(
                group.mean_per_user_aoi, method="spearman"
            ),
            "final_success_rate": last.success_rate,
            "final_training_seed_std": last.seed_std,
            "final_training_seed_std_per_user": last.seed_std_per_user,
            "last16_explained_variance": diagnostics.loc[
                diagnostics.variant == variant, "last16_explained_variance"
            ].mean(),
            "early_stops": diagnostics.loc[
                diagnostics.variant == variant, "early_stops"
            ].sum(),
            "nonfinite_core": diagnostics.loc[
                diagnostics.variant == variant, "nonfinite_core"
            ].sum(),
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out / "convergence_audit_summary.csv", index=False)

    # The independent replication unit is the training seed.  Average the
    # common-random-number evaluation episodes first, then pair variants within
    # each training seed; treating all episodes as independent would inflate n.
    episode_rows = pd.read_csv(out / "checkpoint_episodes.csv")
    final_step = int(episode_rows.global_step.max())
    final = episode_rows[episode_rows.global_step == final_step]
    seed_means = final.groupby(["variant", "training_seed"])["mean_aoi"].mean().unstack("variant")
    paired_rows = []
    if {"reference", "critic_lr_1e3"}.issubset(seed_means.columns):
        differences = seed_means["reference"] - seed_means["critic_lr_1e3"]
        rng = np.random.default_rng(20260909)
        boot = differences.to_numpy()[
            rng.integers(0, len(differences), size=(20_000, len(differences)))
        ].mean(axis=1)
        for training_seed, difference in differences.items():
            paired_rows.append({
                "row_type": "training_seed", "training_seed": training_seed,
                "reference_minus_critic_lr_1e3": difference,
            })
        paired_rows.append({
            "row_type": "aggregate", "training_seed": -1,
            "reference_minus_critic_lr_1e3": differences.mean(),
            "std_across_training_seeds": differences.std(ddof=1),
            "bootstrap_ci_low": np.quantile(boot, .025),
            "bootstrap_ci_high": np.quantile(boot, .975),
            "positive_seed_count": int((differences > 0).sum()),
            "training_seed_count": len(differences),
        })
    pd.DataFrame(paired_rows).to_csv(out / "paired_final_comparison.csv", index=False)

    colors = {"reference": "#2563eb", "critic_lr_1e3": "#dc2626"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    for variant, group in evaluation.groupby("variant", sort=False):
        group = group.sort_values("global_step")
        color = colors.get(variant)
        axes[0, 0].plot(group.global_step, group.mean_per_user_aoi, marker="o",
                        label=variant, color=color)
        spread = group.seed_std_per_user.fillna(0.0)
        axes[0, 0].fill_between(
            group.global_step, group.mean_per_user_aoi - spread,
            group.mean_per_user_aoi + spread, color=color, alpha=.14,
        )
        axes[0, 1].plot(group.global_step, group.success_rate, marker="o",
                        label=variant, color=color)

    for variant, group in training.groupby("variant", sort=False):
        color = colors.get(variant)
        for column, ax in (("env/mean_aoi", axes[1, 0]),
                           ("train/explained_variance", axes[1, 1])):
            pivot = group.pivot(index="step", columns="training_seed", values=column)
            smooth = pivot.rolling(8, min_periods=1).mean()
            mean, std = smooth.mean(axis=1), smooth.std(axis=1).fillna(0.0)
            ax.plot(mean.index, mean, label=variant, color=color)
            ax.fill_between(mean.index, mean - std, mean + std,
                            color=color, alpha=.14)

    labels = [
        "Held-out deterministic mean AoI/user",
        "Held-out deterministic success rate",
        "Training sampled-policy mean AoI/user",
        "Critic explained variance",
    ]
    for ax, label in zip(axes.flat, labels):
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    axes[1, 0].set_xlabel("Environment interactions")
    axes[1, 1].set_xlabel("Environment interactions")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(r"PPO convergence audit: $20^\circ$, $\sigma_e=0.2$ (mean $\pm$ seed SD)")
    fig.tight_layout()
    fig.savefig(out / "ppo_convergence_audit.png", dpi=180)
    fig.savefig(out / "ppo_convergence_audit.pdf")
    plt.close(fig)
    print(audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
