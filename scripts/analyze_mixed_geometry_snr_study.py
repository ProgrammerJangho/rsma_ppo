#!/usr/bin/env python3
"""Create derived comparisons and action diagnostics for the mixed study."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsma_aoi.agents.ppo import PPOPolicy  # noqa: E402


def _checkpoint(seed: int) -> Path:
    return ROOT / (
        "results/rician_ppo_mixed_geometry_snr_csit02_262k_v1/runs/"
        f"rsma_joint_gate_seed{seed}/checkpoint.pt"
    )


def _condition_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["split", "geometry_deg", "snr_db"]
    wide = frame.pivot_table(index=keys, columns="algorithm", values="mean_aoi")
    wide["rsma_gain_over_sdma_pct"] = (
        100.0 * (wide["sdma_power"] - wide["rsma_joint_gate"])
        / wide["sdma_power"]
    )
    return wide.reset_index()


def _paired_seed_summary(frame: pd.DataFrame) -> pd.DataFrame:
    learned = frame[frame.training_seed >= 0]
    units = learned.groupby(
        ["split", "training_seed", "algorithm"], as_index=False
    ).mean(numeric_only=True)
    wide = units.pivot_table(
        index=["split", "training_seed"], columns="algorithm", values="mean_aoi"
    )
    wide["rsma_gain_over_sdma_pct"] = (
        100.0 * (wide["sdma_power"] - wide["rsma_joint_gate"])
        / wide["sdma_power"]
    )
    return wide.reset_index()


def _cem_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["geometry_deg", "snr_db"]
    wide = frame.pivot_table(index=keys, columns="algorithm", values="mean_aoi")
    wide["raw_rsma_gain_over_sdma_pct"] = (
        100.0 * (wide["cem_sdma_960"] - wide["cem_rsma_raw_960"])
        / wide["cem_sdma_960"]
    )
    wide["safe_rsma_gain_over_sdma_pct"] = (
        100.0 * (wide["cem_sdma_960"] - wide["cem_rsma_safeguarded_1920"])
        / wide["cem_sdma_960"]
    )
    wide["ppo_gain_over_raw_cem_pct"] = (
        100.0 * (wide["cem_rsma_raw_960"] - wide["rsma_joint_gate"])
        / wide["cem_rsma_raw_960"]
    )
    return wide.reset_index()


def _action_rows(dataset: Path, seeds: list[int]) -> pd.DataFrame:
    z = np.load(dataset, allow_pickle=False)
    obs = z["observation"].astype(np.float32)
    geometry = np.abs(np.diff(z["episode_user_angles_deg"], axis=1)).ravel()
    snr = z["episode_snr_db"]
    records: list[dict] = []

    policies: list[tuple[str, int, dict[str, np.ndarray]]] = [
        (
            "cem_one_step_label",
            -1,
            {
                "alpha": z["oracle_power_fractions"][:, 0],
                "rho": z["oracle_rho_r"].ravel(),
                "c1": z["oracle_rate_fractions"][:, 0],
            },
        )
    ]
    for seed in seeds:
        policy, _, _ = PPOPolicy.from_checkpoint(
            _checkpoint(seed), deterministic=True, device="cpu"
        )
        actions = policy.act(obs, None)
        policies.append(
            (
                "rsma_joint_gate",
                seed,
                {
                    "alpha": np.asarray(
                        [a["power_fractions"][0] for a in actions], dtype=float
                    ),
                    "rho": np.asarray(
                        [float(np.ravel(a["common_payload_scale"])[0]) for a in actions],
                        dtype=float,
                    ),
                    "c1": np.asarray(
                        [a["rate_fractions"][0] for a in actions], dtype=float
                    ),
                },
            )
        )

    for policy_name, seed, action in policies:
        for g in sorted(np.unique(geometry)):
            for s in sorted(np.unique(snr)):
                mask = np.isclose(geometry, g) & np.isclose(snr, s)
                on = mask & (action["alpha"] > 1e-3)
                row = {
                    "policy": policy_name,
                    "training_seed": seed,
                    "geometry_deg": float(g),
                    "snr_db": float(s),
                    "states": int(mask.sum()),
                    "common_on_rate": float(on.sum() / max(mask.sum(), 1)),
                    "mean_alpha_all": float(action["alpha"][mask].mean()),
                    "mean_rho_all": float(action["rho"][mask].mean()),
                    "mean_alpha_on": float(action["alpha"][on].mean()) if on.any() else np.nan,
                    "median_alpha_on": float(np.median(action["alpha"][on])) if on.any() else np.nan,
                    "mean_rho_on": float(action["rho"][on].mean()) if on.any() else np.nan,
                    "median_rho_on": float(np.median(action["rho"][on])) if on.any() else np.nan,
                    "mean_c1_on": float(action["c1"][on].mean()) if on.any() else np.nan,
                }
                records.append(row)
    return pd.DataFrame(records)


def _heatmap(ax, table: pd.DataFrame, value: str, title: str, cmap: str) -> None:
    p = table.pivot(index="geometry_deg", columns="snr_db", values=value)
    image = ax.imshow(p.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(p.columns)), [f"{x:g}" for x in p.columns])
    ax.set_yticks(range(len(p.index)), [f"{x:g}" for x in p.index])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Angular separation (deg)")
    ax.set_title(title)
    for i in range(len(p.index)):
        for j in range(len(p.columns)):
            ax.text(j, i, f"{p.iloc[i, j]:.1f}", ha="center", va="center", fontsize=9)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="results/thesis_mixed_geometry_snr_v1")
    parser.add_argument(
        "--dataset",
        default="datasets/rician_mixed_geometry_snr_csit02_oracle_v1/test.npz",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    out = ROOT / args.result_dir
    ppo = pd.read_csv(out / "ppo_episodes.csv")
    cem = pd.read_csv(out / "cem_matched_episodes.csv")
    condition = _condition_comparison(ppo)
    paired = _paired_seed_summary(ppo)
    cem_condition = _cem_comparison(cem)
    actions = _action_rows(ROOT / args.dataset, args.seeds)
    condition.to_csv(out / "ppo_condition_comparison.csv", index=False)
    paired.to_csv(out / "ppo_paired_training_seed_summary.csv", index=False)
    cem_condition.to_csv(out / "cem_condition_comparison.csv", index=False)
    actions.to_csv(out / "heldout_action_condition_summary.csv", index=False)

    seen = condition[condition.split == "seen"]
    actor = actions[actions.policy == "rsma_joint_gate"].groupby(
        ["geometry_deg", "snr_db"], as_index=False
    ).mean(numeric_only=True)
    trajectory_action = ppo[
        (ppo.algorithm == "rsma_joint_gate") & (ppo.split == "seen")
    ].groupby(
        ["geometry_deg", "snr_db"], as_index=False
    ).mean(numeric_only=True)
    projection_path = out / "common_off_projection_comparison.csv"
    projection = pd.read_csv(projection_path)
    projection = projection[projection.split == "seen"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.0), constrained_layout=True)
    _heatmap(
        axes[0], seen, "rsma_gain_over_sdma_pct",
        "PPO AoI gain over SDMA (%)", "RdYlGn",
    )
    _heatmap(
        axes[1], projection, "common_stream_gain_pct",
        "Common-stream AoI gain (%)", "RdYlGn",
    )
    _heatmap(
        axes[2], trajectory_action.assign(common_on_pct=100 * trajectory_action.common_on_rate),
        "common_on_pct", "RSMA-PPO common gate on (%)", "Blues",
    )
    fig.savefig(out / "mixed_geometry_snr_heatmaps.png", dpi=200)
    fig.savefig(out / "mixed_geometry_snr_heatmaps.pdf")
    plt.close(fig)

    print(paired.to_string(index=False))
    print("\nHeld-out action means across training seeds")
    print(actor.to_string(index=False))


if __name__ == "__main__":
    main()
