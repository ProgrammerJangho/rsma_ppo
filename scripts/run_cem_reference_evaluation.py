#!/usr/bin/env python3
"""Run the bounded-compute online CEM reference beside frozen PPO policies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch

from run_final_evaluation import (  # noqa: E402
    METRICS, evaluate_trajectory, load_policies, load_protocol,
)
from rsma_aoi.baselines.cem_allocation import CEMAllocationPolicy  # noqa: E402
from rsma_aoi.evaluation.statistics import summarise  # noqa: E402
from rsma_aoi.utils.config import dump_config_snapshot  # noqa: E402
from rsma_aoi.utils.logging import git_commit_hash  # noqa: E402


EXTRA_METRICS = (
    "decision_latency_ms", "candidate_evaluations_per_slot",
    "optimizer_solves_per_slot", "boundary_selection_rate",
)
ALL_METRICS = (*METRICS, *EXTRA_METRICS)


class TimedPolicy:
    """Measure complete action-decision wall time without changing the policy."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.elapsed_s = 0.0
        self.calls = 0

    def reset(self, seed: int | None = None) -> None:
        self.elapsed_s = 0.0
        self.calls = 0
        if hasattr(self.policy, "reset"):
            self.policy.reset(seed=seed)

    def act(self, obs, states=None):
        start = time.perf_counter()
        action = self.policy.act(obs, states)
        self.elapsed_s += time.perf_counter() - start
        self.calls += len(obs)
        return action

    @property
    def latency_ms(self) -> float:
        return 1000 * self.elapsed_s / max(self.calls, 1)


def _add_compute_metrics(
    row: dict[str, Any], timed: TimedPolicy, burn_in: int,
) -> None:
    row["decision_latency_ms"] = timed.latency_ms
    base = timed.policy
    if isinstance(base, CEMAllocationPolicy):
        row["candidate_evaluations_per_slot"] = base.candidate_evaluations / max(base.calls, 1)
        row["optimizer_solves_per_slot"] = (
            2.0 if base.mode == "rsma_safeguarded" else 1.0
        )
        post_burn = base.selection_history[burn_in:]
        row["boundary_selection_rate"] = (
            float(np.mean(np.asarray(post_burn) == "sdma"))
            if base.mode == "rsma_safeguarded" and post_burn else np.nan
        )
    else:
        row["candidate_evaluations_per_slot"] = 0.0
        row["optimizer_solves_per_slot"] = 0.0
        row["boundary_selection_rate"] = np.nan


def _uncertainty_units(episodes: pd.DataFrame) -> pd.DataFrame:
    learned = episodes[episodes.training_seed >= 0]
    numerical = episodes[episodes.training_seed < 0]
    parts = []
    if not learned.empty:
        part = learned.groupby(["algorithm", "training_seed"], as_index=False)[
            list(ALL_METRICS)
        ].mean()
        part["uncertainty_unit"] = "training_seed"
        parts.append(part)
    if not numerical.empty:
        part = numerical.groupby(["algorithm", "eval_seed"], as_index=False)[
            list(ALL_METRICS)
        ].mean()
        part["training_seed"] = -1
        part["uncertainty_unit"] = "evaluation_trajectory"
        parts.append(part.drop(columns="eval_seed"))
    return pd.concat(parts, ignore_index=True)


def _paired(episodes: pd.DataFrame, reference: str) -> pd.DataFrame:
    grouped = episodes.groupby(["algorithm", "eval_seed"], as_index=False)[
        list(ALL_METRICS)
    ].mean()
    ref = grouped[grouped.algorithm == reference].set_index("eval_seed")
    rows = []
    for label, frame in grouped.groupby("algorithm"):
        if label == reference:
            continue
        aligned = frame.set_index("eval_seed").join(
            ref[list(ALL_METRICS)], how="inner", rsuffix="_reference"
        )
        for metric in ALL_METRICS:
            delta = aligned[metric] - aligned[f"{metric}_reference"]
            rows.append({
                "algorithm": label, "reference": reference, "metric": metric,
                "mean_difference": float(delta.mean()),
                "std_difference": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                "paired_trajectories": int(len(delta)),
                "wins": int((delta < 0).sum()) if "aoi" in metric else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/thesis_cem_reference_evaluation.yaml")
    parser.add_argument("--out-dir", default="results/thesis_cem_reference_v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol_path = (ROOT / args.protocol).resolve()
    cfg, protocol = load_protocol(protocol_path)
    learned = load_policies(protocol, cfg, args.device)
    cem_spec = protocol.get("cem")
    if not isinstance(cem_spec, dict):
        raise ValueError("protocol requires a cem mapping")
    kwargs = {
        "iterations": int(cem_spec["iterations"]),
        "population": int(cem_spec["population"]),
        "restarts": int(cem_spec["restarts"]),
    }
    expected = kwargs["iterations"] * kwargs["population"] * kwargs["restarts"]
    if expected != int(cem_spec["evaluations_per_solve"]):
        raise ValueError("CEM evaluation budget is inconsistent")
    if args.validate_only:
        print(
            f"PASS: {len(learned)} checkpoints; {len(protocol['eval_seeds'])} trajectories; "
            f"{expected} candidates/solve"
        )
        return 0

    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    methods: list[tuple[str, int, Any]] = [
        (label, seed, policy) for label, seed, policy, _, _ in learned
    ]
    methods.extend([
        ("cem_rsma_raw_960", -1, CEMAllocationPolicy(cfg, "rsma_raw", **kwargs)),
        ("cem_sdma_960", -1, CEMAllocationPolicy(cfg, "sdma", **kwargs)),
        ("cem_rsma_safeguarded_1920", -1,
         CEMAllocationPolicy(cfg, "rsma_safeguarded", **kwargs)),
    ])
    episodes: list[dict[str, Any]] = []
    users: list[dict[str, Any]] = []
    for label, training_seed, policy in methods:
        for eval_seed in protocol["eval_seeds"]:
            timed = TimedPolicy(policy)
            episode, user_rows = evaluate_trajectory(
                cfg, protocol, label, training_seed, int(eval_seed), timed
            )
            _add_compute_metrics(episode, timed, int(protocol["burn_in"]))
            episodes.append(episode)
            users.extend(user_rows)
        values = [
            row["mean_aoi"] for row in episodes
            if row["algorithm"] == label and row["training_seed"] == training_seed
        ]
        print(
            f"EVALUATED {label} train_seed={training_seed}: "
            f"AoI={np.mean(values):.5f}", flush=True,
        )

    episode_frame = pd.DataFrame(episodes)
    unit_frame = _uncertainty_units(episode_frame)
    summary = summarise(
        unit_frame, group_keys=["algorithm"], metrics=ALL_METRICS, seed=20260909
    )
    unit_kind = unit_frame.groupby("algorithm").uncertainty_unit.first()
    summary["uncertainty_unit"] = summary.algorithm.map(unit_kind)
    paired = _paired(episode_frame, protocol["paired_reference"])
    episode_frame.to_csv(out / "episodes.csv", index=False)
    pd.DataFrame(users).to_csv(out / "users.csv", index=False)
    unit_frame.to_csv(out / "uncertainty_units.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    paired.to_csv(out / "paired_differences.csv", index=False)
    dump_config_snapshot(out / "config_snapshot.yaml", env=cfg, evaluation=protocol)
    manifest = {
        "status": "complete",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cem_policy_sha256": hashlib.sha256(
            (ROOT / "src/rsma_aoi/baselines/cem_allocation.py").read_bytes()
        ).hexdigest(),
        "checkpoint_sha256": {
            str(path.relative_to(ROOT)): digest for _, _, _, path, digest in learned
        },
        "selection_information": "AoI and h_hat only; h_true never exposed to policies",
        "compute_accounting": (
            "raw RSMA and SDMA: one 960-candidate solve/slot; safeguarded RSMA: "
            "independent RSMA and SDMA solves, 1920 candidates/slot"
        ),
        "caveat": "myopic finite-budget CEM reference, not a global or long-horizon optimum",
        "git_commit": git_commit_hash(ROOT),
        "python": sys.version,
        "torch": torch.__version__,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    table = summary[summary.metric.isin(("mean_aoi", "decision_latency_ms"))]
    print(table[["algorithm", "metric", "mean", "std", "n", "uncertainty_unit"]].to_string(index=False))
    print(f"COMPLETE: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
