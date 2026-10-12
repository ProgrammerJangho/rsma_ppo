#!/usr/bin/env python3
"""Audit state-dependent alpha, q_k, and phi_k of feed-forward PPO checkpoints."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsma_aoi.agents.ppo import PPOPolicy  # noqa: E402
from rsma_aoi.envs.aoi_env import AoIRSMAEnv  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402


CHECKPOINTS = {
    "524k": "runs/matched_objective_rsma_ppo_k3_20deg_from_stage1_seed0_524k_v1/checkpoint.pt",
    "1048k": "runs/matched_objective_rsma_ppo_k3_20deg_seed0_continue_524k_v1/checkpoint.pt",
    "2097k": "runs/matched_objective_rsma_ppo_k3_20deg_seed0_continue_to_2097k_v1/checkpoint.pt",
}
MODES = (
    "original", "fixed_alpha", "fixed_q", "fixed_phi", "fixed_all",
    "rank_q", "rank_phi", "rank_q_phi", "rank_all",
    "fixed_phi_level", "rank_phi_shape",
)


def _cfg(config: str, angles: tuple[float, ...], steps: int) -> EnvConfig:
    base = EnvConfig.from_yaml(ROOT / config)
    return replace(
        base, user_angles_deg=list(angles), user_angle_sets_deg=None,
        snr_db=15.0, snr_db_values=None, blocklength=128,
        blocklength_values=None, episode_length=steps,
    )


def _features(state: dict) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    aoi = np.asarray(state["aoi"], dtype=np.float64)
    h = np.asarray(state["h_hat"], dtype=np.complex128)
    gains = np.sum(np.abs(h) ** 2, axis=1)
    gain_db = 10.0 * np.log10(np.maximum(gains, 1e-12))
    norms = np.linalg.norm(h, axis=1)
    correlations = []
    for i in range(len(aoi)):
        for j in range(i + 1, len(aoi)):
            correlations.append(
                abs(np.vdot(h[i], h[j])) / max(norms[i] * norms[j], 1e-12)
            )
    aoi_order = np.argsort(aoi, kind="stable")
    aoi_rank = np.empty_like(aoi_order)
    aoi_rank[aoi_order] = np.arange(len(aoi))
    gain_order = np.argsort(gain_db, kind="stable")
    gain_rank = np.empty_like(gain_order)
    gain_rank[gain_order] = np.arange(len(aoi))
    global_features = {
        "mean_aoi": float(aoi.mean()), "max_aoi": float(aoi.max()),
        "aoi_std": float(aoi.std()),
        "alignment_mean": float(np.mean(correlations)),
        "alignment_max": float(np.max(correlations)),
        "gain_spread_db": float(gain_db.max() - gain_db.min()),
    }
    per_user = np.column_stack((
        aoi, aoi / max(aoi.mean(), 1e-12), aoi_rank, gain_db, gain_rank,
    ))
    return global_features, per_user, np.asarray(correlations)


def _action_parts(action: dict) -> tuple[float, np.ndarray, np.ndarray]:
    alpha = float(np.asarray(action["common_power_fraction"]).reshape(-1)[0])
    q = np.asarray(action["private_power_fractions"], dtype=np.float64).copy()
    phi = np.asarray(action["common_payload_fractions"], dtype=np.float64).copy()
    return alpha, q, phi


def _replace(action: dict, mode: str, constants: dict, state: dict) -> dict:
    out = {key: np.array(value, copy=True) for key, value in action.items()}
    if mode in ("fixed_alpha", "fixed_all"):
        out["common_power_fraction"] = np.asarray([constants["alpha"]], dtype=np.float64)
    if mode in ("fixed_q", "fixed_all"):
        q = np.asarray(constants["q"], dtype=np.float64)
        out["private_power_fractions"] = q / q.sum()
    if mode in ("fixed_phi", "fixed_all"):
        out["common_payload_fractions"] = np.asarray(constants["phi"], dtype=np.float64)
    if mode in ("rank_q", "rank_q_phi", "rank_all"):
        aoi = np.asarray(state["aoi"], dtype=np.float64)
        order = np.argsort(aoi, kind="stable")
        rank = np.empty_like(order); rank[order] = np.arange(len(aoi))
        q = np.asarray(constants["q_by_rank"], dtype=np.float64)[rank]
        out["private_power_fractions"] = q / q.sum()
    if mode in ("rank_phi", "rank_q_phi", "rank_all"):
        aoi = np.asarray(state["aoi"], dtype=np.float64)
        order = np.argsort(aoi, kind="stable")
        rank = np.empty_like(order); rank[order] = np.arange(len(aoi))
        out["common_payload_fractions"] = np.asarray(
            constants["phi_by_rank"], dtype=np.float64
        )[rank]
    if mode == "fixed_phi_level":
        phi = np.asarray(out["common_payload_fractions"], dtype=np.float64)
        current = max(float(phi.mean()), 1e-12)
        out["common_payload_fractions"] = np.clip(
            phi * float(constants["phi_level"]) / current, 0.0, 1.0
        )
    if mode == "rank_phi_shape":
        phi = np.asarray(out["common_payload_fractions"], dtype=np.float64)
        aoi = np.asarray(state["aoi"], dtype=np.float64)
        order = np.argsort(aoi, kind="stable")
        rank = np.empty_like(order); rank[order] = np.arange(len(aoi))
        shape = np.asarray(constants["phi_by_rank"], dtype=np.float64)[rank]
        shape /= max(float(shape.mean()), 1e-12)
        out["common_payload_fractions"] = np.clip(shape * float(phi.mean()), 0.0, 1.0)
    if mode == "rank_all":
        out["common_power_fraction"] = np.asarray([constants["alpha"]], dtype=np.float64)
    return out


def calibrate(job: dict) -> dict:
    cfg = _cfg(job["config"], job["angles"], job["steps"])
    policy, _, _ = PPOPolicy.from_checkpoint(ROOT / job["checkpoint"])
    policy.reset(seed=job["seed"])
    env = AoIRSMAEnv(cfg)
    obs, _ = env.reset(seed=job["seed"])
    alphas, qs, phis = [], [], []
    q_by_rank: list[list[float]] = [[], [], []]
    phi_by_rank: list[list[float]] = [[], [], []]
    for slot in range(job["steps"]):
        action = policy.act(obs[None], [env.get_state()])[0]
        if slot >= job["burn_in"]:
            alpha, q, phi = _action_parts(action)
            alphas.append(alpha); qs.append(q); phis.append(phi)
            aoi = np.asarray(env.get_state()["aoi"], dtype=np.float64)
            order = np.argsort(aoi, kind="stable")
            rank = np.empty_like(order); rank[order] = np.arange(len(aoi))
            for user in range(len(aoi)):
                q_by_rank[int(rank[user])].append(float(q[user]))
                phi_by_rank[int(rank[user])].append(float(phi[user]))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()
    return {
        "policy": job["policy"], "permutation": job["permutation"],
        "alpha": float(np.mean(alphas)), "q": np.mean(qs, axis=0).tolist(),
        "phi": np.mean(phis, axis=0).tolist(),
        "phi_level": float(np.mean(phis)),
        "q_by_rank": [float(np.mean(values)) for values in q_by_rank],
        "phi_by_rank": [float(np.mean(values)) for values in phi_by_rank],
    }


def evaluate(job: dict) -> tuple[dict, list[dict], list[dict], list[dict]]:
    cfg = _cfg(job["config"], job["angles"], job["steps"])
    policy, _, _ = PPOPolicy.from_checkpoint(ROOT / job["checkpoint"])
    policy.reset(seed=job["seed"])
    env = AoIRSMAEnv(cfg)
    obs, _ = env.reset(seed=job["seed"])
    ages, success = [], []
    global_rows, user_rows, one_step_rows = [], [], []
    for slot in range(job["steps"]):
        state = env.get_state()
        action = policy.act(obs[None], [state])[0]
        executed = _replace(action, job["mode"], job["constants"], state)
        if slot >= job["burn_in"] and job["mode"] == "original":
            alpha, q, phi = _action_parts(action)
            global_features, per_user, _ = _features(state)
            global_rows.append({
                "policy": job["policy"], "permutation": job["permutation"],
                "slot": slot, "alpha": alpha,
                "q_entropy": float(-np.sum(np.where(q > 0, q * np.log(q), 0.0))),
                "phi_mean": float(phi.mean()), "phi_std_users": float(phi.std()),
                **global_features,
            })
            for user, values in enumerate(per_user):
                user_rows.append({
                    "policy": job["policy"], "permutation": job["permutation"],
                    "slot": slot, "user": user, "aoi": values[0],
                    "relative_aoi": values[1], "aoi_rank": int(values[2]),
                    "gain_db": values[3], "gain_rank": int(values[4]),
                    "alpha": alpha, "q": q[user], "phi": phi[user],
                })
            pre_aoi = np.asarray(state["aoi"], dtype=np.float64)
            for counterfactual_mode in MODES:
                candidate = _replace(
                    action, counterfactual_mode, job["constants"], state
                )
                slot_action, _ = env._parse_action(candidate)
                result = env.phy.transmit(
                    env.true_channel, state["h_hat"], slot_action
                )
                probability = np.clip(result.success_prob, 0.0, 1.0)
                expected_next = probability + (1.0 - probability) * (pre_aoi + 1.0)
                one_step_rows.append({
                    "policy": job["policy"], "permutation": job["permutation"],
                    "slot": slot, "mode": counterfactual_mode,
                    "expected_next_mean_aoi": float(expected_next.mean()),
                    "mean_success_probability": float(probability.mean()),
                })
        obs, _, terminated, truncated, info = env.step(executed)
        if slot >= job["burn_in"]:
            ages.extend(np.asarray(info["aoi"], dtype=float))
            success.extend(np.asarray(info["success_probabilities"], dtype=float))
        if terminated or truncated:
            break
    env.close()
    age = np.asarray(ages)
    summary = {
        "policy": job["policy"], "mode": job["mode"],
        "permutation": job["permutation"], "eval_seed": job["seed"],
        "mean_aoi": float(age.mean()), "p95_aoi": float(np.quantile(age, .95)),
        "success_probability": float(np.mean(success)),
    }
    return summary, global_rows, user_rows, one_step_rows


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(pearsonr(x, y).statistic if method == "pearson" else spearmanr(x, y).statistic)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/thesis_rician_k3_nt2_mixed_phy_sinr.yaml")
    parser.add_argument("--steps", type=int, default=768)
    parser.add_argument("--burn-in", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    angle_sets = list(permutations((-10.0, 0.0, 10.0)))

    calibration_jobs = []
    for policy, checkpoint in CHECKPOINTS.items():
        for permutation, angles in enumerate(angle_sets):
            calibration_jobs.append({
                "policy": policy, "checkpoint": checkpoint, "config": args.config,
                "permutation": permutation, "angles": angles,
                "steps": args.steps, "burn_in": args.burn_in,
                "seed": 980278 + 100 * permutation,
            })
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        calibration = list(pool.map(calibrate, calibration_jobs))
    calibration_frame = pd.DataFrame(calibration)
    calibration_frame.to_csv(out / "calibration_constants.csv", index=False)
    constants = {(r["policy"], r["permutation"]): r for r in calibration}

    jobs = []
    for policy, checkpoint in CHECKPOINTS.items():
        for permutation, angles in enumerate(angle_sets):
            for mode in MODES:
                jobs.append({
                    "policy": policy, "checkpoint": checkpoint, "config": args.config,
                    "permutation": permutation, "angles": angles, "mode": mode,
                    "constants": constants[(policy, permutation)],
                    "steps": args.steps, "burn_in": args.burn_in,
                    "seed": 990278 + 100 * permutation,
                })
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate, jobs))
    trajectories = pd.DataFrame([r[0] for r in results])
    global_actions = pd.DataFrame([row for r in results for row in r[1]])
    user_actions = pd.DataFrame([row for r in results for row in r[2]])
    one_step = pd.DataFrame([row for r in results for row in r[3]])
    trajectories.to_csv(out / "counterfactual_trajectories.csv", index=False)
    global_actions.to_csv(out / "global_action_samples.csv", index=False)
    user_actions.to_csv(out / "user_action_samples.csv", index=False)
    one_step.to_csv(out / "one_step_counterfactual_samples.csv", index=False)

    counter = trajectories.groupby(["policy", "mode"], as_index=False).agg(
        mean_aoi=("mean_aoi", "mean"), p95_aoi=("p95_aoi", "mean"),
        success_probability=("success_probability", "mean"),
    )
    original = counter[counter["mode"] == "original"][["policy", "mean_aoi"]].rename(
        columns={"mean_aoi": "original_aoi"}
    )
    counter = counter.merge(original, on="policy")
    counter["aoi_change_vs_original_pct"] = 100.0 * (
        counter.mean_aoi - counter.original_aoi
    ) / counter.original_aoi
    counter.to_csv(out / "counterfactual_summary.csv", index=False)

    one_step_summary = one_step.groupby(["policy", "mode"], as_index=False).agg(
        expected_next_mean_aoi=("expected_next_mean_aoi", "mean"),
        mean_success_probability=("mean_success_probability", "mean"),
    )
    one_step_original = one_step_summary[
        one_step_summary["mode"] == "original"
    ][["policy", "expected_next_mean_aoi"]].rename(
        columns={"expected_next_mean_aoi": "original_expected_next_aoi"}
    )
    one_step_summary = one_step_summary.merge(one_step_original, on="policy")
    one_step_summary["aoi_change_vs_original_pct"] = 100.0 * (
        one_step_summary.expected_next_mean_aoi
        - one_step_summary.original_expected_next_aoi
    ) / one_step_summary.original_expected_next_aoi
    one_step_summary.to_csv(out / "one_step_counterfactual_summary.csv", index=False)

    global_corr = []
    for policy, group in global_actions.groupby("policy"):
        for action_name in ("alpha", "q_entropy", "phi_mean", "phi_std_users"):
            for state_name in (
                "alignment_mean", "alignment_max", "gain_spread_db",
                "mean_aoi", "max_aoi", "aoi_std",
            ):
                global_corr.append({
                    "policy": policy, "action": action_name, "state": state_name,
                    "pearson": _safe_corr(group[action_name], group[state_name], "pearson"),
                    "spearman": _safe_corr(group[action_name], group[state_name], "spearman"),
                    "samples": len(group),
                })
    pd.DataFrame(global_corr).to_csv(out / "global_state_correlations.csv", index=False)

    user_corr = []
    for policy, group in user_actions.groupby("policy"):
        for action_name in ("q", "phi"):
            for state_name in ("aoi", "relative_aoi", "gain_db"):
                user_corr.append({
                    "policy": policy, "action": action_name, "state": state_name,
                    "pearson": _safe_corr(group[action_name], group[state_name], "pearson"),
                    "spearman": _safe_corr(group[action_name], group[state_name], "spearman"),
                    "samples": len(group),
                })
    pd.DataFrame(user_corr).to_csv(out / "user_state_correlations.csv", index=False)

    rank = user_actions.groupby(["policy", "aoi_rank"], as_index=False).agg(
        q_mean=("q", "mean"), q_std=("q", "std"),
        phi_mean=("phi", "mean"), phi_std=("phi", "std"), samples=("q", "size"),
    )
    rank.to_csv(out / "aoi_rank_actions.csv", index=False)
    gain_rank = user_actions.groupby(["policy", "gain_rank"], as_index=False).agg(
        q_mean=("q", "mean"), phi_mean=("phi", "mean"), samples=("q", "size"),
    )
    gain_rank.to_csv(out / "gain_rank_actions.csv", index=False)

    variation = global_actions.groupby("policy", as_index=False).agg(
        alpha_mean=("alpha", "mean"), alpha_std=("alpha", "std"),
        alpha_p05=("alpha", lambda x: x.quantile(.05)),
        alpha_p95=("alpha", lambda x: x.quantile(.95)),
        q_entropy_mean=("q_entropy", "mean"), phi_mean=("phi_mean", "mean"),
    )
    variation.to_csv(out / "action_variation.csv", index=False)

    print("Counterfactuals:\n", counter.to_string(index=False))
    print("\nOne-step counterfactuals:\n", one_step_summary.to_string(index=False))
    print("\nAoI-rank actions:\n", rank.to_string(index=False))
    print("\nVariation:\n", variation.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
