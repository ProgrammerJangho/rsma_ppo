#!/usr/bin/env python3
"""Test whether the 262k RSMA actor needs continuous state-dependent allocation.

The actor's binary common gate is retained.  Constants are estimated on separate
calibration trajectories and evaluated on the frozen held-out trajectories.
Two nested controls are reported:

1. fixed_magnitude: fix common-power fraction and common-payload scale when on,
   while retaining state-dependent private/common user proportions;
2. fixed_full_action: replace every continuous allocation by one conditional
   mean action for gate-on and one for gate-off states.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsma_aoi.agents.ppo import PPOPolicy  # noqa: E402
from rsma_aoi.envs.aoi_env import AoIRSMAEnv  # noqa: E402
from rsma_aoi.evaluation.statistics import summarise  # noqa: E402
from run_final_evaluation import (  # noqa: E402
    METRICS, build_uncertainty_units, evaluate_trajectory, load_protocol,
)


def _is_on(action: dict[str, np.ndarray]) -> bool:
    power = np.asarray(action["power_fractions"], dtype=np.float64)
    rho = float(np.asarray(action["common_payload_scale"]).reshape(-1)[0])
    return bool(power[0] > 1e-12 and rho > 1e-12)


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(values.sum())
    return values / total if total > 0.0 else np.full_like(values, 1.0 / len(values))


class DistilledGatePolicy:
    """Keep a learned gate and optionally freeze its continuous allocations."""

    def __init__(self, base: PPOPolicy, constants: dict[str, Any], mode: str):
        self.base = base
        self.constants = constants
        self.mode = mode

    def reset(self, seed: int | None = None) -> None:
        if hasattr(self.base, "reset"):
            self.base.reset(seed=seed)

    def act(self, observations, states=None):
        actions = self.base.act(observations, states)
        transformed = []
        for original in actions:
            action = {key: np.asarray(value).copy() for key, value in original.items()}
            gate_on = _is_on(action)
            key = "on" if gate_on else "off"
            if self.mode == "fixed_magnitude":
                if gate_on:
                    alpha = float(self.constants["on"]["common_power_fraction"])
                    private = _normalise(action["power_fractions"][1:])
                    action["power_fractions"] = np.r_[alpha, (1.0 - alpha) * private]
                    action["common_payload_scale"] = np.asarray(
                        [self.constants["on"]["common_payload_scale"]], dtype=np.float32
                    )
                else:
                    private = _normalise(action["power_fractions"][1:])
                    action["power_fractions"] = np.r_[0.0, private]
                    action["common_payload_scale"] = np.asarray([0.0], dtype=np.float32)
            elif self.mode == "fixed_power_and_rho":
                action["power_fractions"] = np.asarray(
                    self.constants[key]["power_fractions"], dtype=np.float64
                )
                action["common_payload_scale"] = np.asarray(
                    [self.constants[key]["common_payload_scale"]], dtype=np.float32
                )
            elif self.mode == "fixed_rate_and_magnitude":
                if gate_on:
                    alpha = float(self.constants["on"]["common_power_fraction"])
                    private = _normalise(action["power_fractions"][1:])
                    action["power_fractions"] = np.r_[alpha, (1.0 - alpha) * private]
                    action["common_payload_scale"] = np.asarray(
                        [self.constants["on"]["common_payload_scale"]], dtype=np.float32
                    )
                else:
                    private = _normalise(action["power_fractions"][1:])
                    action["power_fractions"] = np.r_[0.0, private]
                    action["common_payload_scale"] = np.asarray([0.0], dtype=np.float32)
                action["rate_fractions"] = np.asarray(
                    self.constants[key]["rate_fractions"], dtype=np.float64
                )
            elif self.mode == "private_argmax":
                alpha = float(self.constants[key]["common_power_fraction"])
                winner = int(np.argmax(np.asarray(original["power_fractions"])[1:]))
                private = np.zeros_like(np.asarray(original["power_fractions"])[1:], dtype=np.float64)
                private[winner] = 1.0 - alpha
                action["power_fractions"] = np.r_[alpha, private]
                action["rate_fractions"] = np.asarray(
                    self.constants[key]["rate_fractions"], dtype=np.float64
                )
                action["common_payload_scale"] = np.asarray(
                    [self.constants[key]["common_payload_scale"]], dtype=np.float32
                )
            elif self.mode in ("rate_argmax", "fixed_magnitude_rate_argmax"):
                if self.mode == "fixed_magnitude_rate_argmax":
                    if gate_on:
                        alpha = float(self.constants["on"]["common_power_fraction"])
                        private = _normalise(action["power_fractions"][1:])
                        action["power_fractions"] = np.r_[
                            alpha, (1.0 - alpha) * private
                        ]
                        action["common_payload_scale"] = np.asarray(
                            [self.constants["on"]["common_payload_scale"]],
                            dtype=np.float32,
                        )
                    else:
                        private = _normalise(action["power_fractions"][1:])
                        action["power_fractions"] = np.r_[0.0, private]
                        action["common_payload_scale"] = np.asarray(
                            [0.0], dtype=np.float32
                        )
                rate = np.zeros_like(
                    np.asarray(original["rate_fractions"]), dtype=np.float64
                )
                rate[int(np.argmax(original["rate_fractions"]))] = 1.0
                action["rate_fractions"] = rate
            elif self.mode == "fixed_full_action":
                action["power_fractions"] = np.asarray(
                    self.constants[key]["power_fractions"], dtype=np.float64
                )
                action["rate_fractions"] = np.asarray(
                    self.constants[key]["rate_fractions"], dtype=np.float64
                )
                action["common_payload_scale"] = np.asarray(
                    [self.constants[key]["common_payload_scale"]], dtype=np.float32
                )
            else:  # pragma: no cover
                raise ValueError(self.mode)
            transformed.append(action)
        return transformed


def _calibrate(
    policy: PPOPolicy, cfg, seeds: list[int], steps: int, burn_in: int,
) -> dict[str, Any]:
    records: dict[str, list[dict[str, np.ndarray | float]]] = {"on": [], "off": []}
    for seed in seeds:
        env = AoIRSMAEnv(cfg)
        if hasattr(policy, "reset"):
            policy.reset(seed=seed)
        obs, _ = env.reset(seed=seed)
        try:
            for slot in range(steps):
                action = policy.act(obs[None, :], [env.get_state()])[0]
                if slot >= burn_in:
                    key = "on" if _is_on(action) else "off"
                    records[key].append(
                        {
                            "power": np.asarray(action["power_fractions"], dtype=np.float64),
                            "rate": np.asarray(action["rate_fractions"], dtype=np.float64),
                            "rho": float(np.asarray(action["common_payload_scale"]).reshape(-1)[0]),
                        }
                    )
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset()
        finally:
            env.close()

    constants: dict[str, Any] = {}
    counts = {key: len(value) for key, value in records.items()}
    if not counts["on"] or not counts["off"]:
        raise RuntimeError(f"calibration did not observe both gate modes: {counts}")
    for key, rows in records.items():
        power = _normalise(np.mean([row["power"] for row in rows], axis=0))
        rate = _normalise(np.mean([row["rate"] for row in rows], axis=0))
        rho = float(np.mean([row["rho"] for row in rows])) if key == "on" else 0.0
        if key == "off":
            power = np.r_[0.0, _normalise(power[1:])]
        constants[key] = {
            "power_fractions": power.tolist(),
            "rate_fractions": rate.tolist(),
            "common_power_fraction": float(power[0]),
            "common_payload_scale": rho,
            "samples": counts[key],
        }
    constants["gate_on_rate"] = counts["on"] / (counts["on"] + counts["off"])
    return constants


def _worker(job: dict[str, Any]):
    cfg, protocol = load_protocol(job["protocol"])
    checkpoint = ROOT / job["checkpoint"]
    base, trained_env, ppo = PPOPolicy.from_checkpoint(checkpoint, device="cpu", deterministic=True)
    if ppo.allocation_control != "joint" or not ppo.common_gate_enabled:
        raise ValueError(f"unexpected action head in {checkpoint}")
    constants = _calibrate(
        base, cfg, job["calibration_seeds"], job["calibration_steps"], job["calibration_burn_in"]
    )
    available = {
        "ppo_original": base,
        "gate_fixed_magnitude": DistilledGatePolicy(base, constants, "fixed_magnitude"),
        "gate_dynamic_rate_only": DistilledGatePolicy(base, constants, "fixed_power_and_rho"),
        "gate_dynamic_private_only": DistilledGatePolicy(base, constants, "fixed_rate_and_magnitude"),
        "gate_private_argmax": DistilledGatePolicy(base, constants, "private_argmax"),
        "gate_fixed_full_action": DistilledGatePolicy(base, constants, "fixed_full_action"),
    }
    methods = [(label, available[label]) for label in job["methods"]]
    episodes, users = [], []
    for label, policy in methods:
        for eval_seed in protocol["eval_seeds"]:
            episode, user_rows = evaluate_trajectory(
                cfg, protocol, label, int(job["training_seed"]), int(eval_seed), policy
            )
            episodes.append(episode)
            users.extend(user_rows)
    return episodes, users, {"training_seed": job["training_seed"], **constants}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/thesis_training_budget_evaluation.yaml")
    parser.add_argument("--out-dir", default="results/thesis_two_mode_audit_v1")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--calibration-seeds", type=int, nargs="+", default=[940000, 940001, 940002])
    parser.add_argument("--calibration-steps", type=int, default=4096)
    parser.add_argument("--calibration-burn-in", type=int, default=1024)
    parser.add_argument(
        "--methods", nargs="+", default=[
            "ppo_original", "gate_fixed_magnitude", "gate_dynamic_rate_only",
            "gate_dynamic_private_only", "gate_private_argmax", "gate_fixed_full_action",
        ],
    )
    args = parser.parse_args(argv)

    _, protocol = load_protocol(ROOT / args.protocol)
    group = protocol["checkpoint_groups"]["rsma_262k"]
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    jobs = [
        {
            "protocol": str(ROOT / args.protocol),
            "checkpoint": item["path"],
            "training_seed": int(item["training_seed"]),
            "calibration_seeds": args.calibration_seeds,
            "calibration_steps": args.calibration_steps,
            "calibration_burn_in": args.calibration_burn_in,
            "methods": args.methods,
        }
        for item in group["checkpoints"]
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_worker, jobs))

    episodes = pd.DataFrame([row for result in results for row in result[0]])
    users = pd.DataFrame([row for result in results for row in result[1]])
    constants = [result[2] for result in results]
    units = build_uncertainty_units(episodes)
    summary = summarise(units, group_keys=["algorithm"], metrics=METRICS, seed=20260910)
    summary["uncertainty_unit"] = "training_seed"

    seed_means = episodes.groupby(["algorithm", "training_seed"]).mean(numeric_only=True)
    reference = seed_means.loc["ppo_original", "mean_aoi"]
    contrasts = []
    for label in [method for method in args.methods if method != "ppo_original"]:
        candidate = seed_means.loc[label, "mean_aoi"]
        delta = candidate - reference
        contrasts.append(
            {
                "candidate": label,
                "reference": "ppo_original",
                "candidate_mean_aoi": candidate.mean(),
                "reference_mean_aoi": reference.mean(),
                "mean_paired_delta": delta.mean(),
                "mean_paired_percent": 100.0 * delta.mean() / reference.mean(),
                "candidate_wins": int((delta < 0).sum()),
                "reference_wins": int((delta > 0).sum()),
                "paired_seeds": len(delta),
            }
        )

    episodes.to_csv(out / "episodes.csv", index=False)
    users.to_csv(out / "users.csv", index=False)
    units.to_csv(out / "uncertainty_units.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "paired_contrasts.csv", index=False)
    (out / "calibration_constants.json").write_text(json.dumps(constants, indent=2), encoding="utf-8")
    manifest = {
        "status": "complete",
        "protocol": args.protocol,
        "calibration_seeds": args.calibration_seeds,
        "calibration_steps": args.calibration_steps,
        "calibration_burn_in": args.calibration_burn_in,
        "calibration_is_disjoint_from_evaluation": not bool(
            set(args.calibration_seeds) & set(protocol["eval_seeds"])
        ),
        "distillation": "per-checkpoint conditional arithmetic means",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary[summary.metric == "mean_aoi"][["algorithm", "mean", "std", "n"]].to_string(index=False))
    print(pd.DataFrame(contrasts).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
