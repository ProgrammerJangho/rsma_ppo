#!/usr/bin/env python3
"""Run the frozen, paired long-horizon thesis evaluation.

The YAML protocol names every checkpoint and held-out channel/ACK seed.  All
methods run in the same RSMA environment; ``sdma_equal`` and ``sdma_power_ppo``
use the exact common-off boundary.  Training-seed uncertainty is kept separate
from evaluation-trajectory uncertainty.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import yaml

from rsma_aoi.agents.ppo import PPOPolicy
from rsma_aoi.baselines.common import equal_split, make_action
from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.evaluation.statistics import summarise
from rsma_aoi.utils.config import EnvConfig, dump_config_snapshot
from rsma_aoi.utils.logging import git_commit_hash


METRICS = (
    "mean_aoi", "weighted_mean_aoi", "aoi_p95", "aoi_p99", "max_aoi",
    "success_rate", "mean_success_probability", "mean_common_power",
    "common_on_rate", "mean_rho", "mean_common_fraction", "mean_common_bler",
    "common_bler_when_on", "mean_private_bler", "mean_peak_aoi_completed",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: str | Path) -> tuple[EnvConfig, dict[str, Any]]:
    """Load and validate the frozen evaluation section."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("evaluation"), dict):
        raise ValueError("protocol YAML must contain an evaluation mapping")
    cfg = EnvConfig.from_dict(raw)
    protocol = raw["evaluation"]
    seeds = protocol.get("eval_seeds")
    steps = protocol.get("episode_steps")
    burn_in = protocol.get("burn_in")
    groups = protocol.get("checkpoint_groups")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("eval_seeds must be a non-empty unique list")
    if not isinstance(steps, int) or not isinstance(burn_in, int) or not 0 <= burn_in < steps:
        raise ValueError("require integer 0 <= burn_in < episode_steps")
    if cfg.episode_length != steps:
        raise ValueError("env.episode_length must equal evaluation.episode_steps")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("checkpoint_groups must be a non-empty mapping")
    if protocol.get("deterministic") is not True:
        raise ValueError("final protocol requires deterministic policy evaluation")
    if protocol.get("paired_reference") not in groups:
        raise ValueError("paired_reference must name a checkpoint group")
    if (cfg.scheduling_enabled or cfg.allocator_mode != "none"
            or cfg.multiple_access != "rsma"
            or cfg.rate_split_mode not in ("simplex_scaled", "per_user_fraction")
            or cfg.link_model != "fbl" or cfg.ack_mode != "sampled"
            or cfg.aoi_cap_enabled or cfg.power_scale_action):
        raise ValueError("final protocol requires direct, uncapped, all-user FBL-RSMA")
    return cfg, protocol


def _environment_difference(expected: EnvConfig, observed: EnvConfig) -> dict[str, tuple[Any, Any]]:
    """Return checkpoint/environment differences, ignoring evaluation horizon."""
    want, got = expected.to_dict(), observed.to_dict()
    want.pop("episode_length", None)
    got.pop("episode_length", None)
    return {key: (want[key], got[key]) for key in want if want[key] != got[key]}


def load_policies(
    protocol: dict[str, Any], cfg: EnvConfig, device: str,
) -> list[tuple[str, int, PPOPolicy, Path, str]]:
    """Load the exact checkpoints and reject mixed environments/action heads."""
    loaded: list[tuple[str, int, PPOPolicy, Path, str]] = []
    seen_paths: set[Path] = set()
    for label, spec in protocol["checkpoint_groups"].items():
        entries = spec.get("checkpoints", [])
        seeds = [item.get("training_seed") for item in entries]
        if not entries or len(set(seeds)) != len(seeds):
            raise ValueError(f"{label}: checkpoints need unique training_seed values")
        for item in entries:
            checkpoint = (ROOT / item["path"]).resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if checkpoint in seen_paths:
                raise ValueError(f"checkpoint appears more than once: {checkpoint}")
            seen_paths.add(checkpoint)
            policy, trained_env, ppo = PPOPolicy.from_checkpoint(
                checkpoint, device=device, deterministic=True
            )
            difference = _environment_difference(cfg, trained_env)
            if difference:
                raise ValueError(f"{label}: checkpoint environment differs: {difference}")
            if ppo.allocation_control != spec["expected_allocation_control"]:
                raise ValueError(
                    f"{label}: expected allocation_control={spec['expected_allocation_control']}, "
                    f"found {ppo.allocation_control}"
                )
            if ppo.common_gate_enabled is not bool(spec["expected_common_gate"]):
                raise ValueError(f"{label}: unexpected common-gate setting")
            loaded.append((label, int(item["training_seed"]), policy, checkpoint, _sha256(checkpoint)))
    return loaded


def _fixed_action(cfg: EnvConfig, label: str, protocol: dict[str, Any]) -> dict[str, np.ndarray]:
    active = np.ones(cfg.num_users, dtype=bool)
    if label == "fixed_rsma":
        fixed = protocol["fixed_rsma"]
        common_fraction = float(fixed["common_power_fraction"])
        rho = float(fixed["common_payload_scale"])
    elif label == "sdma_equal":
        common_fraction, rho = 0.0, 0.0
    else:
        raise ValueError(label)
    power, rate = equal_split(active, common_fraction)
    return make_action(active, power, rate, rho)


def evaluate_trajectory(
    cfg: EnvConfig,
    protocol: dict[str, Any],
    label: str,
    training_seed: int,
    eval_seed: int,
    policy: PPOPolicy | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one policy on one trajectory after the frozen burn-in."""
    env = AoIRSMAEnv(cfg)
    weights = np.asarray(cfg.weight_vector, dtype=np.float64)
    aoi_rows: list[np.ndarray] = []
    ack_rows: list[np.ndarray] = []
    success_rows: list[np.ndarray] = []
    common_power: list[float] = []
    common_on: list[bool] = []
    rho_rows: list[float] = []
    common_bler: list[np.ndarray] = []
    private_bler: list[np.ndarray] = []
    common_rate: list[np.ndarray] = []
    common_fraction: list[np.ndarray] = []
    private_power_share: list[np.ndarray] = []
    burn_in = int(protocol["burn_in"])
    try:
        if policy is not None and hasattr(policy, "reset"):
            policy.reset(seed=eval_seed)
        obs, _ = env.reset(seed=eval_seed)
        for slot in range(cfg.episode_length):
            if policy is None:
                action = _fixed_action(cfg, label, protocol)
            else:
                action = policy.act(obs[None, :], [env.get_state()])[0]
            # Preserve the exact physical allocation for metrics regardless of
            # whether the public action is legacy (combined vectors) or the
            # canonical direct (alpha, q_k, phi_k) schema.
            slot_action, _ = env._parse_action(action)
            obs, _, terminated, truncated, info = env.step(action)
            if info["allocator_calls"] != 0 or info["constraint_violation"] > cfg.feasibility_tol:
                raise AssertionError(f"{label}: allocator override or infeasible action")
            if (
                protocol.get("require_all_users_active", True)
                and not np.all(np.asarray(info["scheduled_users"], dtype=bool))
            ):
                raise AssertionError(f"{label}: final protocol requires all users active")
            if slot < burn_in:
                continue
            aoi_rows.append(np.asarray(info["aoi"], dtype=np.float64))
            ack_rows.append(np.asarray(info["ack"], dtype=np.float64))
            success_rows.append(np.asarray(info["success_probabilities"], dtype=np.float64))
            pf = slot_action.power_fractions
            if cfg.rate_split_mode == "per_user_fraction":
                active_phi = slot_action.rate_fractions[slot_action.active]
                rho = float(active_phi.mean()) if active_phi.size else 0.0
                is_on = bool(pf[0] > 1e-12 and info["common_payload"] > 1e-12)
            else:
                rho = float(
                    info.get(
                        "common_payload_effective_scale",
                        action["common_payload_scale"][0],
                    )
                )
                is_on = bool(pf[0] > 1e-12 and rho > 1e-12)
            common_power.append(float(pf[0]))
            common_on.append(is_on)
            rho_rows.append(rho)
            common_bler.append(np.asarray(info["common_bler"], dtype=np.float64))
            private_bler.append(np.asarray(info["private_bler"], dtype=np.float64))
            common_fraction.append(
                np.asarray(info["common_fraction_per_user"], dtype=np.float64)
            )
            private = (
                np.asarray(action["private_power_fractions"], dtype=np.float64)
                if cfg.rate_split_mode == "per_user_fraction"
                else np.asarray(pf[1:], dtype=np.float64)
            )
            private_power_share.append(
                private / private.sum() if private.sum() > 1e-12 else np.zeros_like(private)
            )
            payload = np.asarray(cfg.payload_vector, dtype=np.float64)
            common_bits = payload - np.asarray(info["private_payload"], dtype=np.float64)
            common_rate.append(common_bits / cfg.blocklength)
            if terminated or truncated:
                break
    finally:
        env.close()

    aoi = np.stack(aoi_rows)
    ack = np.stack(ack_rows)
    success = np.stack(success_rows)
    cbler = np.stack(common_bler)
    pbler = np.stack(private_bler)
    crate = np.stack(common_rate)
    cfrac = np.stack(common_fraction)
    qshare = np.stack(private_power_share)
    on = np.asarray(common_on, dtype=bool)
    completed_intervals: list[np.ndarray] = []
    user_rows: list[dict[str, Any]] = []
    base = {"algorithm": label, "training_seed": training_seed, "eval_seed": eval_seed}
    for user in range(cfg.num_users):
        ack_slots = np.flatnonzero(ack[:, user] > 0.5)
        intervals = np.diff(ack_slots)
        completed_intervals.append(intervals)
        user_rows.append({
            **base, "user": user, "mean_aoi": float(aoi[:, user].mean()),
            "aoi_p95": float(np.quantile(aoi[:, user], .95)),
            "aoi_p99": float(np.quantile(aoi[:, user], .99)),
            "max_aoi": float(aoi[:, user].max()),
            "success_rate": float(ack[:, user].mean()),
            "mean_success_probability": float(success[:, user].mean()),
            "mean_common_rate": float(crate[:, user].mean()),
            "mean_common_fraction": float(cfrac[:, user].mean()),
            "mean_private_power_share": float(qshare[:, user].mean()),
            "mean_common_bler": float(cbler[:, user].mean()),
            "mean_private_bler": float(pbler[:, user].mean()),
            "completed_intervals": int(intervals.size),
            "mean_peak_aoi_completed": float(intervals.mean()) if intervals.size else np.nan,
        })
    peaks = np.concatenate([x for x in completed_intervals if x.size]) if any(
        x.size for x in completed_intervals
    ) else np.asarray([], dtype=np.float64)
    episode = {
        **base,
        "mean_aoi": float(aoi.mean()),
        "weighted_mean_aoi": float((aoi @ weights).mean() / weights.sum()),
        "aoi_p95": float(np.quantile(aoi, .95)),
        "aoi_p99": float(np.quantile(aoi, .99)),
        "max_aoi": float(aoi.max()),
        "success_rate": float(ack.mean()),
        "mean_success_probability": float(success.mean()),
        "mean_common_power": float(np.mean(common_power)),
        "common_on_rate": float(on.mean()),
        # rho belongs only to the legacy simplex-scaled action.  Direct actions
        # report phi_k through mean_common_fraction instead.
        "mean_rho": (
            float(np.mean(rho_rows))
            if cfg.rate_split_mode == "simplex_scaled" else np.nan
        ),
        "mean_common_fraction": float(cfrac.mean()),
        "mean_common_bler": float(cbler.mean()),
        "common_bler_when_on": float(cbler[on].mean()) if on.any() else np.nan,
        "mean_private_bler": float(pbler.mean()),
        "mean_peak_aoi_completed": float(peaks.mean()) if peaks.size else np.nan,
    }
    return episode, user_rows


def _evaluate_method_worker(task):
    """Evaluate one policy generation; process-safe for long protocol audits."""
    cfg, protocol, label, training_seed, checkpoint, device = task
    policy = None
    if checkpoint is not None:
        policy, _, _ = PPOPolicy.from_checkpoint(
            checkpoint, device=device, deterministic=True
        )
    local_episodes: list[dict[str, Any]] = []
    local_users: list[dict[str, Any]] = []
    for eval_seed in protocol["eval_seeds"]:
        episode, users = evaluate_trajectory(
            cfg, protocol, label, training_seed, int(eval_seed), policy
        )
        local_episodes.append(episode)
        local_users.extend(users)
    method_mean = float(np.mean([row["mean_aoi"] for row in local_episodes]))
    return label, training_seed, local_episodes, local_users, method_mean


def build_uncertainty_units(episodes: pd.DataFrame) -> pd.DataFrame:
    """Use training seeds for learned methods and trajectories for fixed ones."""
    learned = episodes[episodes.training_seed >= 0]
    fixed = episodes[episodes.training_seed < 0]
    parts = []
    if not learned.empty:
        part = learned.groupby(["algorithm", "training_seed"], as_index=False)[list(METRICS)].mean()
        part["uncertainty_unit"] = "training_seed"
        parts.append(part)
    if not fixed.empty:
        part = fixed.groupby(["algorithm", "eval_seed"], as_index=False)[list(METRICS)].mean()
        part["training_seed"] = -1
        part["uncertainty_unit"] = "evaluation_trajectory"
        parts.append(part.drop(columns="eval_seed"))
    return pd.concat(parts, ignore_index=True)


def paired_differences(episodes: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Trajectory-paired deltas after averaging each learned policy generation."""
    by_trajectory = episodes.groupby(["algorithm", "eval_seed"], as_index=False)[list(METRICS)].mean()
    reference_rows = by_trajectory[by_trajectory.algorithm == reference].set_index("eval_seed")
    rows = []
    for label, group in by_trajectory.groupby("algorithm"):
        if label == reference:
            continue
        aligned = group.set_index("eval_seed").join(
            reference_rows[list(METRICS)], how="inner", rsuffix="_reference"
        )
        for metric in METRICS:
            delta = aligned[metric] - aligned[f"{metric}_reference"]
            rows.append({
                "algorithm": label, "reference": reference, "metric": metric,
                "mean_difference": float(delta.mean()),
                "std_difference": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                "paired_trajectories": int(len(delta)),
                "wins": int((delta < 0).sum()) if "aoi" in metric else np.nan,
            })
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/thesis_final_evaluation.yaml")
    parser.add_argument("--out-dir", default="results/thesis_final_evaluation_v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1,
                        help="evaluate independent policy generations in parallel")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    torch.set_num_threads(1)
    protocol_path = (ROOT / args.protocol).resolve()
    cfg, protocol = load_protocol(protocol_path)
    policies = load_policies(protocol, cfg, args.device)
    if args.validate_only:
        print(f"PASS: protocol valid; {len(policies)} checkpoints; "
              f"{len(protocol['eval_seeds'])} paired trajectories")
        return 0

    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    methods: list[tuple[str, int, PPOPolicy | None, Path | None, str | None]] = [
        ("fixed_rsma", -1, None, None, None),
        ("sdma_equal", -1, None, None, None),
        *policies,
    ]
    def evaluate_method(method):
        label, training_seed, policy, _, _ = method
        local_episodes: list[dict[str, Any]] = []
        local_users: list[dict[str, Any]] = []
        for eval_seed in protocol["eval_seeds"]:
            episode, users = evaluate_trajectory(
                cfg, protocol, label, training_seed, int(eval_seed), policy
            )
            local_episodes.append(episode)
            local_users.extend(users)
        method_mean = np.mean([row["mean_aoi"] for row in local_episodes])
        return label, training_seed, local_episodes, local_users, method_mean

    episode_rows: list[dict[str, Any]] = []
    user_rows: list[dict[str, Any]] = []
    if args.workers == 1:
        completed = map(evaluate_method, methods)
    else:
        # The environment loop is Python-heavy, so processes provide actual
        # parallelism whereas threads contend on the GIL.  Checkpoints are
        # reloaded in each worker to avoid sharing mutable policy state.
        tasks = [
            (cfg, protocol, label, training_seed,
             None if checkpoint is None else str(checkpoint), args.device)
            for label, training_seed, _, checkpoint, _ in methods
        ]
        executor = ProcessPoolExecutor(max_workers=args.workers)
        completed = executor.map(_evaluate_method_worker, tasks)
    try:
        for label, training_seed, local_episodes, local_users, method_mean in completed:
            episode_rows.extend(local_episodes)
            user_rows.extend(local_users)
            print(
                f"EVALUATED {label} train_seed={training_seed}: mean AoI={method_mean:.5f}",
                flush=True,
            )
    finally:
        if args.workers != 1:
            executor.shutdown(wait=True)

    episodes = pd.DataFrame(episode_rows)
    users = pd.DataFrame(user_rows)
    units = build_uncertainty_units(episodes)
    summary = summarise(units, group_keys=["algorithm"], metrics=METRICS, seed=20260909)
    unit_kind = units.groupby("algorithm").uncertainty_unit.first()
    summary["uncertainty_unit"] = summary.algorithm.map(unit_kind)
    paired = paired_differences(episodes, protocol["paired_reference"])
    episodes.to_csv(out / "episodes.csv", index=False)
    users.to_csv(out / "users.csv", index=False)
    units.to_csv(out / "uncertainty_units.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    paired.to_csv(out / "paired_differences.csv", index=False)
    dump_config_snapshot(out / "config_snapshot.yaml", env=cfg, evaluation=protocol)
    manifest = {
        "status": "complete",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": _sha256(protocol_path),
        "script_sha256": _sha256(Path(__file__)),
        "checkpoint_sha256": {
            str(path.relative_to(ROOT)): digest for _, _, _, path, digest in policies
        },
        "evaluation": "deterministic policies; identical held-out channel/CSIT/ACK seeds; burn-in removed",
        "uncertainty": "learned methods: training-seed means; fixed methods: evaluation trajectories",
        "git_commit": git_commit_hash(ROOT),
        "python": sys.version,
        "torch": torch.__version__,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    table = summary[(summary.metric == "mean_aoi")][
        ["algorithm", "mean", "std", "ci_low", "ci_high", "n", "uncertainty_unit"]
    ].sort_values("mean")
    print(table.to_string(index=False))
    print(f"COMPLETE: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
