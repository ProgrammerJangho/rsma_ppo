#!/usr/bin/env python3
"""Audit common-stream contribution and AoI-conditioned actions of final PPOs.

The audit uses the frozen target protocol and five independently trained final
checkpoints.  For each trajectory it compares the original deterministic actor
with a counterfactual that keeps the actor's all-user private-power simplex
``q_k`` but sets ``alpha=phi_k=0``.  It also records how ``q_k`` and ``phi_k``
vary with each user's pre-action AoI rank.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsma_aoi.agents.ppo import PPOPolicy  # noqa: E402
from rsma_aoi.envs.aoi_env import AoIRSMAEnv  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402


def common_off(action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Keep schedule and q, while forcing the exact canonical SDMA boundary."""
    out = {key: np.array(value, copy=True) for key, value in action.items()}
    q = np.asarray(out["private_power_fractions"], dtype=np.float64).reshape(-1)
    active = np.asarray(out["schedule"], dtype=bool).reshape(-1)
    q = np.where(active, np.clip(q, 0.0, None), 0.0)
    if q.sum() <= 1e-12:
        q = active.astype(np.float64)
    q /= q.sum()
    out["common_power_fraction"] = np.zeros(1, dtype=np.float64)
    out["private_power_fractions"] = q
    out["common_payload_fractions"] = np.zeros_like(q)
    # Keep internal aliases coherent for diagnostics that inspect them.
    out["power_fractions"] = np.concatenate(([0.0], q))
    out["rate_fractions"] = np.zeros_like(q)
    return out


def rank_labels(aoi: np.ndarray) -> np.ndarray:
    """Return stable ranks 0=youngest, 1=middle, 2=oldest for K=3."""
    order = np.argsort(aoi, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return rank


def evaluate(job: dict) -> tuple[dict, list[dict], list[dict]]:
    base = EnvConfig.from_yaml(ROOT / job["config"])
    cfg = replace(
        base,
        user_angles_deg=list(job["angles"]),
        user_angle_sets_deg=None,
        snr_db=15.0,
        snr_db_values=None,
        blocklength=128,
        blocklength_values=None,
        multiple_access="rsma",
        episode_length=job["steps"],
    )
    actor, _, _ = PPOPolicy.from_checkpoint(
        ROOT / job["checkpoint"], deterministic=True, device="cpu"
    )
    if hasattr(actor, "reset"):
        actor.reset(seed=job["eval_seed"])
    env = AoIRSMAEnv(cfg)
    obs, _ = env.reset(seed=job["eval_seed"])
    ages: list[float] = []
    success: list[float] = []
    action_rows: list[dict] = []
    alpha_rows: list[dict] = []
    for slot in range(job["steps"]):
        state = env.get_state()
        pre_aoi = np.asarray(state["aoi"], dtype=np.float64)
        action = actor.act(obs[None, :], [state])[0]
        executed = common_off(action) if job["mode"] == "common_off" else action
        if slot >= job["burn_in"] and job["mode"] == "ppo":
            alpha = float(np.asarray(action["common_power_fraction"]).reshape(-1)[0])
            q = np.asarray(action["private_power_fractions"], dtype=np.float64)
            phi = np.asarray(action["common_payload_fractions"], dtype=np.float64)
            ranks = rank_labels(pre_aoi)
            mean_aoi = float(pre_aoi.mean())
            for user in range(pre_aoi.size):
                action_rows.append({
                    "training_seed": job["training_seed"],
                    "permutation": job["permutation"],
                    "eval_seed": job["eval_seed"],
                    "slot": slot,
                    "user": user,
                    "aoi": pre_aoi[user],
                    "relative_aoi": pre_aoi[user] / max(mean_aoi, 1e-12),
                    "aoi_rank": int(ranks[user]),
                    "q": q[user],
                    "phi": phi[user],
                    "alpha": alpha,
                })
            max_age = float(pre_aoi.max())
            alpha_rows.append({
                "training_seed": job["training_seed"],
                "permutation": job["permutation"],
                "eval_seed": job["eval_seed"],
                "slot": slot,
                "alpha": alpha,
                "mean_aoi": mean_aoi,
                "max_aoi": max_age,
                "aoi_std": float(pre_aoi.std()),
                "max_aoi_bin": "1" if max_age <= 1 else "2" if max_age <= 2 else "3+",
            })
        obs, _, terminated, truncated, info = env.step(executed)
        if slot >= job["burn_in"]:
            ages.extend(np.asarray(info["aoi"], dtype=np.float64).tolist())
            success.extend(
                np.asarray(info["success_probabilities"], dtype=np.float64).tolist()
            )
        if terminated or truncated:
            break
    env.close()
    arr = np.asarray(ages)
    summary = {
        "mode": job["mode"],
        "training_seed": job["training_seed"],
        "permutation": job["permutation"],
        "eval_seed": job["eval_seed"],
        "mean_aoi": float(arr.mean()),
        "p95_aoi": float(np.quantile(arr, 0.95)),
        "success_probability": float(np.mean(success)),
    }
    return summary, action_rows, alpha_rows


def ci95(std: pd.Series, count: pd.Series) -> pd.Series:
    # Five training seeds -> Student-t df=4.
    return 2.7764451051977987 * std / np.sqrt(count)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/thesis_rician_k3_nt2_mixed_phy_sinr.yaml"
    )
    parser.add_argument("--steps", type=int, default=768)
    parser.add_argument("--burn-in", type=int, default=256)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--checkpoint-template",
        default=(
            "runs/ppo_k3_20deg_sinr_from_stage1_seed{seed}_524k_v1/"
            "checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--out-dir", default="results/thesis_current_common_off_aoi_audit_v1"
    )
    args = parser.parse_args()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)

    jobs = []
    for training_seed in range(5):
        for permutation, angles in enumerate(permutations((-10.0, 0.0, 10.0))):
            for mode in ("ppo", "common_off"):
                jobs.append({
                    "config": args.config,
                    "checkpoint": args.checkpoint_template.format(seed=training_seed),
                    "training_seed": training_seed,
                    "permutation": permutation,
                    "angles": angles,
                    "eval_seed": 990278 + 100 * permutation,
                    "mode": mode,
                    "steps": args.steps,
                    "burn_in": args.burn_in,
                })
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate, jobs))
    trajectories = pd.DataFrame([item[0] for item in results])
    actions = pd.DataFrame([row for item in results for row in item[1]])
    alphas = pd.DataFrame([row for item in results for row in item[2]])
    trajectories.to_csv(out / "trajectory_summary.csv", index=False)
    actions.to_csv(out / "action_samples.csv", index=False)
    alphas.to_csv(out / "alpha_state_samples.csv", index=False)

    wide = trajectories.pivot_table(
        index=["training_seed", "permutation", "eval_seed"],
        columns="mode", values="mean_aoi",
    ).reset_index()
    wide["common_gain_pct"] = 100.0 * (
        wide["common_off"] - wide["ppo"]
    ) / wide["common_off"]
    wide.to_csv(out / "paired_common_contribution.csv", index=False)
    seed = wide.groupby("training_seed", as_index=False).agg(
        ppo_mean_aoi=("ppo", "mean"),
        common_off_mean_aoi=("common_off", "mean"),
        common_gain_mean_pct=("common_gain_pct", "mean"),
        common_gain_wins=("common_gain_pct", lambda x: int((x > 0).sum())),
    )
    seed.to_csv(out / "seed_summary.csv", index=False)

    rank_by_seed = actions.groupby(
        ["training_seed", "aoi_rank"], as_index=False
    ).agg(q_mean=("q", "mean"), phi_mean=("phi", "mean"), samples=("q", "size"))
    rank_agg = rank_by_seed.groupby("aoi_rank", as_index=False).agg(
        training_seeds=("training_seed", "count"),
        q_mean=("q_mean", "mean"), q_std=("q_mean", "std"),
        phi_mean=("phi_mean", "mean"), phi_std=("phi_mean", "std"),
    )
    rank_agg["q_ci95_halfwidth"] = ci95(rank_agg.q_std, rank_agg.training_seeds)
    rank_agg["phi_ci95_halfwidth"] = ci95(
        rank_agg.phi_std, rank_agg.training_seeds
    )
    rank_by_seed.to_csv(out / "aoi_rank_actions_by_seed.csv", index=False)
    rank_agg.to_csv(out / "aoi_rank_action_aggregate.csv", index=False)

    alpha_by_seed = alphas.groupby(
        ["training_seed", "max_aoi_bin"], as_index=False
    ).agg(alpha_mean=("alpha", "mean"), samples=("alpha", "size"))
    alpha_agg = alpha_by_seed.groupby("max_aoi_bin", as_index=False).agg(
        training_seeds=("training_seed", "count"),
        alpha_mean=("alpha_mean", "mean"), alpha_std=("alpha_mean", "std"),
    )
    alpha_agg["alpha_ci95_halfwidth"] = ci95(
        alpha_agg.alpha_std, alpha_agg.training_seeds
    )
    alpha_by_seed.to_csv(out / "alpha_by_max_aoi_seed.csv", index=False)
    alpha_agg.to_csv(out / "alpha_by_max_aoi_aggregate.csv", index=False)

    aggregate = {
        "protocol": {
            "training_seeds": list(range(5)), "permutations": 6,
            "steps": args.steps, "burn_in": args.burn_in,
            "geometry_deg": 20, "snr_db": 15, "blocklength": 128,
        },
        "ppo_mean_aoi": float(seed.ppo_mean_aoi.mean()),
        "common_off_mean_aoi": float(seed.common_off_mean_aoi.mean()),
        "common_gain_mean_pct": float(seed.common_gain_mean_pct.mean()),
        "common_gain_seed_std_pct": float(seed.common_gain_mean_pct.std(ddof=1)),
        "common_gain_ci95_halfwidth_pct": float(
            2.7764451051977987
            * seed.common_gain_mean_pct.std(ddof=1) / np.sqrt(len(seed))
        ),
        "trajectory_wins": int((wide.common_gain_pct > 0).sum()),
        "trajectory_count": int(len(wide)),
    }
    (out / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))
    print("\nAoI-rank actions:\n", rank_agg.to_string(index=False))
    print("\nAlpha by max AoI:\n", alpha_agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
