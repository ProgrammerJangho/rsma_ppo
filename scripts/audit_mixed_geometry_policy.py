#!/usr/bin/env python3
"""Audit one mixed-geometry PPO policy on fixed 5/20/60-degree regimes.

Each checkpoint is calibrated once on episode-randomized geometry. The same
constants are then used without per-geometry retuning on all three fixed test
regimes. This distinguishes genuine cross-geometry continuous adaptation from
post-hoc constants fitted separately to each geometry.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_two_mode_policy import DistilledGatePolicy, _calibrate, _is_on  # noqa: E402
from rsma_aoi.agents.ppo import PPOPolicy  # noqa: E402
from rsma_aoi.envs.aoi_env import AoIRSMAEnv  # noqa: E402
from rsma_aoi.evaluation.statistics import summarise  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402
from run_final_evaluation import (  # noqa: E402
    METRICS, evaluate_trajectory,
)

ANGLE_SETS = {
    5: [-2.5, 2.5],
    20: [-10.0, 10.0],
    60: [-30.0, 30.0],
}


def _corr(x: list[float], y: list[float]) -> float:
    a, b = np.asarray(x), np.asarray(y)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _policy_action_stats(policy, cfg: EnvConfig, seeds: list[int], burn_in: int) -> dict:
    alignment, gate, alpha_on, rho_on, rate0_on = [], [], [], [], []
    alignment_on = []
    private0_on, private0_off = [], []
    for seed in seeds:
        env = AoIRSMAEnv(cfg)
        obs, _ = env.reset(seed=seed)
        try:
            for slot in range(cfg.episode_length):
                state = env.get_state()
                action = policy.act(obs[None, :], [state])[0]
                if slot >= burn_in:
                    h = np.asarray(state["h_hat"])
                    den = np.linalg.norm(h[0]) * np.linalg.norm(h[1])
                    align = float(abs(np.vdot(h[0], h[1])) ** 2 / max(den**2, 1e-24))
                    on = _is_on(action)
                    power = np.asarray(action["power_fractions"], dtype=float)
                    private = power[1:] / max(power[1:].sum(), 1e-12)
                    alignment.append(align)
                    gate.append(float(on))
                    if on:
                        alignment_on.append(align)
                        alpha_on.append(float(power[0]))
                        rho_on.append(float(np.asarray(action["common_payload_scale"]).ravel()[0]))
                        rate0_on.append(float(np.asarray(action["rate_fractions"])[0]))
                        private0_on.append(float(private[0]))
                    else:
                        private0_off.append(float(private[0]))
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        finally:
            env.close()
    return {
        "common_on_rate": float(np.mean(gate)),
        "alpha_on_mean": float(np.mean(alpha_on)) if alpha_on else np.nan,
        "alpha_on_median": float(np.median(alpha_on)) if alpha_on else np.nan,
        "rho_on_mean": float(np.mean(rho_on)) if rho_on else np.nan,
        "rate0_on_mean": float(np.mean(rate0_on)) if rate0_on else np.nan,
        "private0_share_on_mean": float(np.mean(private0_on)) if private0_on else np.nan,
        "private0_share_off_mean": float(np.mean(private0_off)) if private0_off else np.nan,
        "r_alignment_gate": _corr(alignment, gate),
        "r_alignment_alpha_on": _corr(alignment_on, alpha_on),
        "r_alignment_rho_on": _corr(alignment_on, rho_on),
    }


def _worker(job: dict):
    mixed_cfg = EnvConfig.from_yaml(ROOT / job["config"])
    checkpoint = ROOT / job["checkpoint"]
    base, _, ppo = PPOPolicy.from_checkpoint(checkpoint, deterministic=True, device="cpu")
    if ppo.allocation_control != "joint" or not ppo.common_gate_enabled:
        raise ValueError(f"unexpected action heads: {checkpoint}")
    constants = _calibrate(
        base,
        mixed_cfg,
        job["calibration_seeds"],
        job["calibration_steps"],
        job["calibration_burn_in"],
    )
    policies = {
        "ppo_original": base,
        "global_fixed_magnitude": DistilledGatePolicy(base, constants, "fixed_magnitude"),
        "global_dynamic_private_only": DistilledGatePolicy(
            base, constants, "fixed_rate_and_magnitude"
        ),
        "global_fixed_full_action": DistilledGatePolicy(
            base, constants, "fixed_full_action"
        ),
    }
    episodes, users, action_rows = [], [], []
    for separation, angles in ANGLE_SETS.items():
        cfg = replace(
            mixed_cfg,
            user_angles_deg=list(angles),
            user_angle_sets_deg=None,
            episode_length=job["eval_steps"],
        )
        protocol = {"eval_seeds": job["eval_seeds"], "burn_in": job["eval_burn_in"]}
        stats = _policy_action_stats(base, cfg, job["eval_seeds"], job["eval_burn_in"])
        action_rows.append(
            {"training_seed": job["training_seed"], "angle_deg": separation, **stats}
        )
        for label, policy in policies.items():
            for eval_seed in job["eval_seeds"]:
                episode, user_rows = evaluate_trajectory(
                    cfg, protocol, label, job["training_seed"], eval_seed, policy
                )
                episode["angle_deg"] = separation
                for row in user_rows:
                    row["angle_deg"] = separation
                episodes.append(episode)
                users.extend(user_rows)
    return episodes, users, action_rows, {"training_seed": job["training_seed"], **constants}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/thesis_rician_mixed_geometry_csit02.yaml"
    )
    parser.add_argument(
        "--checkpoint-template",
        default=(
            "results/rician_ppo_mixed_geometry_csit02_262k_v1/runs/"
            "rsma_joint_gate_seed{seed}/checkpoint.pt"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--calibration-seeds", type=int, nargs="+", default=[952000, 952001, 952002])
    parser.add_argument("--calibration-steps", type=int, default=12288)
    parser.add_argument("--calibration-burn-in", type=int, default=1024)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[953000, 953001, 953002, 953003, 953004, 953005])
    parser.add_argument("--eval-steps", type=int, default=8192)
    parser.add_argument("--eval-burn-in", type=int, default=2048)
    parser.add_argument("--out-dir", default="results/thesis_mixed_geometry_audit_v1")
    args = parser.parse_args(argv)

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    jobs = [
        {
            "config": args.config,
            "checkpoint": args.checkpoint_template.format(seed=seed),
            "training_seed": seed,
            "calibration_seeds": args.calibration_seeds,
            "calibration_steps": args.calibration_steps,
            "calibration_burn_in": args.calibration_burn_in,
            "eval_seeds": args.eval_seeds,
            "eval_steps": args.eval_steps,
            "eval_burn_in": args.eval_burn_in,
        }
        for seed in args.seeds
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_worker, jobs))

    episodes = pd.DataFrame([x for r in results for x in r[0]])
    users = pd.DataFrame([x for r in results for x in r[1]])
    actions = pd.DataFrame([x for r in results for x in r[2]])
    constants = [r[3] for r in results]
    units = episodes.groupby(
        ["angle_deg", "algorithm", "training_seed"], as_index=False
    )[list(METRICS)].mean()
    summary = summarise(
        units, group_keys=["angle_deg", "algorithm"], metrics=METRICS, seed=20260910
    )
    summary["uncertainty_unit"] = "training_seed"

    seed_means = episodes.groupby(
        ["angle_deg", "algorithm", "training_seed"]
    ).mean(numeric_only=True)
    contrasts = []
    for angle in ANGLE_SETS:
        ref = seed_means.loc[(angle, "ppo_original"), "mean_aoi"]
        for label in policies_for_contrast():
            cand = seed_means.loc[(angle, label), "mean_aoi"]
            delta = cand - ref
            contrasts.append({
                "angle_deg": angle,
                "candidate": label,
                "candidate_mean_aoi": cand.mean(),
                "reference_mean_aoi": ref.mean(),
                "mean_paired_delta": delta.mean(),
                "mean_paired_percent": 100.0 * delta.mean() / ref.mean(),
                "candidate_wins": int((delta < 0).sum()),
                "reference_wins": int((delta > 0).sum()),
            })

    episodes.to_csv(out / "episodes.csv", index=False)
    users.to_csv(out / "users.csv", index=False)
    actions.to_csv(out / "action_stats.csv", index=False)
    units.to_csv(out / "uncertainty_units.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "paired_contrasts.csv", index=False)
    (out / "global_calibration_constants.json").write_text(
        json.dumps(constants, indent=2), encoding="utf-8"
    )
    (out / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "training_geometry": "uniform episode-level choice of 5/20/60 degrees",
        "global_calibration": {
            "seeds": args.calibration_seeds,
            "steps": args.calibration_steps,
            "burn_in": args.calibration_burn_in,
        },
        "fixed_evaluation": {
            "seeds": args.eval_seeds,
            "steps": args.eval_steps,
            "burn_in": args.eval_burn_in,
        },
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame(contrasts).to_string(index=False))
    return 0


def policies_for_contrast() -> tuple[str, ...]:
    return (
        "global_fixed_magnitude",
        "global_dynamic_private_only",
        "global_fixed_full_action",
    )


if __name__ == "__main__":
    raise SystemExit(main())
