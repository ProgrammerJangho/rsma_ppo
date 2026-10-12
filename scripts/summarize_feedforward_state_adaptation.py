#!/usr/bin/env python3
"""Create compact tables and figures from the feed-forward action audit."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results/feedforward_state_adaptation_audit_v4"


def main() -> None:
    global_actions = pd.read_csv(AUDIT / "global_action_samples.csv")
    users = pd.read_csv(AUDIT / "user_action_samples.csv")
    one_step = pd.read_csv(AUDIT / "one_step_counterfactual_summary.csv")
    closed_loop = pd.read_csv(AUDIT / "counterfactual_summary.csv")

    alignment_rows = []
    for policy, group in global_actions.groupby("policy"):
        quartile = pd.qcut(group.alignment_mean, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        temp = group.assign(alignment_quartile=quartile)
        agg = temp.groupby("alignment_quartile", observed=True, as_index=False).agg(
            alpha_mean=("alpha", "mean"), phi_mean=("phi_mean", "mean"),
            q_entropy=("q_entropy", "mean"), samples=("alpha", "size"),
        )
        agg["policy"] = policy
        alignment_rows.append(agg)
    alignment = pd.concat(alignment_rows, ignore_index=True)
    alignment.to_csv(AUDIT / "alignment_quartile_actions.csv", index=False)

    global_actions["max_aoi_bin"] = pd.cut(
        global_actions.max_aoi, [0, 1, 2, np.inf],
        labels=["1", "2", "3+"], include_lowest=True,
    )
    max_age = global_actions.groupby(
        ["policy", "max_aoi_bin"], observed=True, as_index=False
    ).agg(
        alpha_mean=("alpha", "mean"), phi_mean=("phi_mean", "mean"),
        q_entropy=("q_entropy", "mean"), samples=("alpha", "size"),
    )
    max_age.to_csv(AUDIT / "max_aoi_bin_actions.csv", index=False)

    users["aoi_bin"] = pd.cut(
        users.aoi, [0, 1, 2, np.inf], labels=["1", "2", "3+"], include_lowest=True
    )
    aoi_value = users.groupby(["policy", "aoi_bin"], observed=True, as_index=False).agg(
        q_mean=("q", "mean"), phi_mean=("phi", "mean"), samples=("q", "size")
    )
    aoi_value.to_csv(AUDIT / "aoi_value_actions.csv", index=False)

    tie_rows = []
    for (policy, permutation, slot), group in users.groupby(
        ["policy", "permutation", "slot"], sort=False
    ):
        max_age = group.aoi.max()
        max_set = group[group.aoi == max_age]
        chosen = group.loc[group.q.idxmax()]
        tie_rows.append({
            "policy": policy, "permutation": permutation, "slot": slot,
            "max_tie_count": len(max_set), "q_mass_on_max_age": max_set.q.sum(),
            "q_argmax_is_max_age": float(chosen.aoi == max_age),
            "q_argmax_is_strongest_tied": float(
                chosen.user == max_set.loc[max_set.gain_db.idxmax(), "user"]
            ),
        })
    ties = pd.DataFrame(tie_rows).groupby(
        ["policy", "max_tie_count"], as_index=False
    ).agg(
        states=("q_mass_on_max_age", "size"),
        q_mass_on_max_age=("q_mass_on_max_age", "mean"),
        q_argmax_is_max_age=("q_argmax_is_max_age", "mean"),
        q_argmax_is_strongest_tied=("q_argmax_is_strongest_tied", "mean"),
    )
    ties.to_csv(AUDIT / "max_aoi_tie_actions.csv", index=False)

    policies = ["524k", "1048k", "2097k"]
    colors = {"524k": "#94a3b8", "1048k": "#2563eb", "2097k": "#7c3aed"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    x = np.arange(4)
    for policy in policies:
        data = alignment[alignment.policy == policy]
        axes[0, 0].plot(x, data.alpha_mean, "o-", color=colors[policy], label=policy)
        axes[0, 1].plot(x, data.phi_mean, "o-", color=colors[policy], label=policy)
    for ax, ylabel in ((axes[0, 0], r"Mean $\alpha$"), (axes[0, 1], r"Mean user $\phi_k$")):
        ax.set_xticks(x, ["Q1", "Q2", "Q3", "Q4"])
        ax.set_xlabel("Channel-alignment quartile")
        ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(fontsize=8)

    rank = pd.read_csv(AUDIT / "aoi_rank_actions.csv")
    for policy in policies:
        data = rank[rank.policy == policy]
        axes[1, 0].plot(data.aoi_rank, data.q_mean, "o-", color=colors[policy], label=policy)
    axes[1, 0].set_xticks([0, 1, 2], ["youngest", "middle", "oldest"])
    axes[1, 0].set_ylabel(r"Mean private-power share $q_k$")
    axes[1, 0].grid(alpha=.25); axes[1, 0].legend(fontsize=8)

    modes = ["fixed_alpha", "rank_q", "fixed_phi_level"]
    labels = [r"fixed $\alpha$", r"AoI-rank $q_k$", r"fixed $\phi$ level"]
    one = one_step[(one_step.policy == "2097k") & one_step["mode"].isin(modes)].set_index("mode")
    long = closed_loop[(closed_loop.policy == "2097k") & closed_loop["mode"].isin(modes)].set_index("mode")
    width = .36
    axes[1, 1].bar(np.arange(3)-width/2, [one.loc[m].aoi_change_vs_original_pct for m in modes],
                   width, label="one-step")
    axes[1, 1].bar(np.arange(3)+width/2, [long.loc[m].aoi_change_vs_original_pct for m in modes],
                   width, label="closed-loop")
    axes[1, 1].set_xticks(np.arange(3), labels, rotation=12)
    axes[1, 1].set_ylabel("AoI increase vs original (%)")
    axes[1, 1].grid(axis="y", alpha=.25); axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(AUDIT / "state_adaptation_audit.png", dpi=180)
    fig.savefig(AUDIT / "state_adaptation_audit.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
