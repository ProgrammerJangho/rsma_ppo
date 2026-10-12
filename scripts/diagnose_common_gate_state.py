#!/usr/bin/env python3
"""Relate a learned common gate to observed channel/AoI state features."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from audit_allocation import audit_policy
from rsma_aoi.agents.ppo import PPOPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, nargs="+")
    parser.add_argument("--eval-seed", type=int, default=910000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--burn-in", type=int, default=512)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or not 0 <= args.burn_in < args.steps:
        parser.error("requires episodes >= 1 and 0 <= burn-in < steps")
    torch.set_num_threads(1)
    seeds = list(range(args.eval_seed, args.eval_seed + args.episodes))
    traces = []
    reference_cfg = None
    for checkpoint in args.checkpoint:
        policy, cfg, ppo = PPOPolicy.from_checkpoint(checkpoint, deterministic=True)
        if not ppo.common_gate_enabled:
            parser.error(f"checkpoint has no common gate: {checkpoint}")
        cfg = replace(cfg, episode_length=args.steps)
        if reference_cfg is not None and cfg.to_dict() != reference_cfg.to_dict():
            parser.error("all checkpoints must use the same environment")
        reference_cfg = cfg
        part = pd.DataFrame(audit_policy(policy, cfg, seeds, "joint_gate"))
        part["training_seed"] = ppo.seed
        traces.append(part)
    trace = pd.concat(traces, ignore_index=True)
    trace = trace[trace.slot > args.burn_in].copy()
    slots = trace.drop_duplicates(["training_seed", "eval_seed", "slot"]).copy()
    slots["common_on"] = slots.common_enabled > 0.5
    features = ["estimated_channel_alignment", "estimated_gain_gap_db", "pre_action_aoi_gap"]

    rows = []
    for on, group in slots.groupby("common_on"):
        row = {"common_on": bool(on), "slots": len(group)}
        for feature in features:
            row[f"mean_{feature}"] = group[feature].mean()
            row[f"p50_{feature}"] = group[feature].median()
        rows.append(row)
    summary = pd.DataFrame(rows)
    correlations = pd.DataFrame([
        {"feature": feature, "correlation_with_common_on": slots[feature].corr(slots.common_on.astype(float))}
        for feature in features
    ])
    try:
        slots["alignment_bin"] = pd.qcut(
            slots.estimated_channel_alignment, 5, duplicates="drop"
        )
        bins = slots.groupby("alignment_bin", observed=True).agg(
            slots=("common_on", "size"),
            mean_alignment=("estimated_channel_alignment", "mean"),
            common_on_rate=("common_on", "mean"),
            mean_aoi_gap=("pre_action_aoi_gap", "mean"),
        ).reset_index()
        bins["alignment_bin"] = bins.alignment_bin.astype(str)
    except ValueError:
        bins = pd.DataFrame()

    args.out_dir.mkdir(parents=True, exist_ok=False)
    trace.to_csv(args.out_dir / "state_trace.csv.gz", index=False)
    summary.to_csv(args.out_dir / "on_off_state_summary.csv", index=False)
    correlations.to_csv(args.out_dir / "state_correlations.csv", index=False)
    bins.to_csv(args.out_dir / "alignment_bins.csv", index=False)
    manifest = {
        "checkpoints": [str(path) for path in args.checkpoint],
        "checkpoint_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.checkpoint
        },
        "eval_seeds": seeds,
        "steps": args.steps,
        "burn_in": args.burn_in,
        "environment": reference_cfg.to_dict() if reference_cfg is not None else None,
        "source_sha256": {
            "scripts/audit_allocation.py": hashlib.sha256(
                (ROOT / "scripts/audit_allocation.py").read_bytes()
            ).hexdigest(),
            "scripts/diagnose_common_gate_state.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(correlations.to_string(index=False))
    if not bins.empty:
        print(bins.to_string(index=False))


if __name__ == "__main__":
    main()
