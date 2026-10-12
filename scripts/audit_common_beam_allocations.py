#!/usr/bin/env python3
"""Paired offline audit of common beams after per-state CEM re-optimisation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from rsma_aoi.agents.ppo import PPOPolicy
from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.evaluation.beam_audit import audit_state_allocations


def collect_and_audit(args: argparse.Namespace) -> pd.DataFrame:
    records = []
    audited = 0
    for checkpoint_index, checkpoint in enumerate(args.checkpoint):
        policy, cfg, _ = PPOPolicy.from_checkpoint(
            checkpoint, device=args.device, deterministic=True
        )
        if cfg.multiple_access != "rsma" or cfg.link_model != "fbl":
            raise ValueError("checkpoints must use FBL RSMA")
        env = AoIRSMAEnv(cfg)
        try:
            for episode in range(args.episodes):
                eval_seed = args.eval_seed + checkpoint_index * 100_000 + episode
                obs, _ = env.reset(seed=eval_seed)
                for slot in range(cfg.episode_length):
                    state = env.get_state()
                    action = policy.act(obs[None, :], states=[state])[0]
                    active = np.asarray(action["schedule"], dtype=bool)
                    if not cfg.scheduling_enabled:
                        active = np.ones(cfg.num_users, dtype=bool)
                    if (
                        slot >= args.burn_in
                        and slot % args.stride == 0
                        and int(active.sum()) >= 2
                        and audited < args.max_states
                    ):
                        paired_seed = args.cem_seed + audited * 1_000_003
                        rows = audit_state_allocations(
                            cfg,
                            np.asarray(state["aoi"], dtype=float),
                            env.true_channel,
                            np.asarray(state["h_hat"]),
                            active,
                            beam_modes=args.beams,
                            csit_mode=args.csit_mode,
                            seed=paired_seed,
                            iterations=args.iterations,
                            population=args.population,
                            restarts=args.restarts,
                        )
                        for row in rows:
                            for key in (
                                "boundary_power_fractions", "boundary_rate_fractions",
                                "raw_power_fractions", "raw_rate_fractions",
                                "raw_success_probabilities", "safe_power_fractions",
                                "safe_rate_fractions", "safe_success_probabilities",
                            ):
                                row.pop(key)
                            row.update({
                                "state_id": audited,
                                "checkpoint": str(Path(checkpoint).resolve()),
                                "checkpoint_index": checkpoint_index,
                                "eval_seed": eval_seed,
                                "slot": slot,
                                "aoi_mean": float(np.mean(state["aoi"])),
                                "aoi_max": float(np.max(state["aoi"])),
                                "urgency_ratio": float(
                                    np.max(np.asarray(state["aoi"])[active] * env.weights[active])
                                    / max(
                                        np.min(np.asarray(state["aoi"])[active] * env.weights[active]),
                                        1e-12,
                                    )
                                ),
                            })
                            records.append(row)
                        audited += 1
                    obs, _, _, truncated, _ = env.step(action)
                    if truncated or audited >= args.max_states:
                        break
                if audited >= args.max_states:
                    break
        finally:
            env.close()
        if audited >= args.max_states:
            break
    if not records:
        raise RuntimeError("no multi-user states were collected")
    return pd.DataFrame(records)


def summarise(trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = trace.groupby("beam", sort=False).agg(
        states=("state_id", "count"),
        mean_raw_gain=("raw_gain_true", "mean"),
        mean_safe_gain=("safe_gain_true", "mean"),
        mean_raw_expected_aoi=("raw_expected_aoi_true", "mean"),
        mean_safe_expected_aoi=("safe_expected_aoi_true", "mean"),
        boundary_selection_rate=("selected_boundary", "mean"),
        mean_common_power_fraction=("common_power_fraction", "mean"),
        mean_rho=("rho", "mean"),
        mean_min_common_gain=("min_common_gain_true", "mean"),
        mean_runtime_ms=("runtime_s", lambda x: 1000.0 * x.mean()),
        evaluations=("evaluations", "mean"),
    ).reset_index()

    wide_raw = trace.pivot(index="state_id", columns="beam", values="raw_gain_true")
    wide_safe = trace.pivot(index="state_id", columns="beam", values="safe_gain_true")
    if "principal" not in wide_raw:
        raise ValueError("principal must be included for paired comparisons")
    comparisons = []
    for beam in wide_raw.columns:
        if beam == "principal":
            continue
        for objective, wide in (("raw", wide_raw), ("sdma_safeguarded", wide_safe)):
            delta = (wide[beam] - wide["principal"]).dropna()
            comparisons.append({
                "beam": beam,
                "objective": objective,
                "states": int(delta.size),
                "mean_delta_gain": float(delta.mean()),
                "median_delta_gain": float(delta.median()),
                "p05_delta_gain": float(delta.quantile(0.05)),
                "p95_delta_gain": float(delta.quantile(0.95)),
                "improve_rate": float((delta > 1e-10).mean()),
                "worsen_rate": float((delta < -1e-10).mean()),
            })
    return summary, pd.DataFrame(comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--beams", nargs="+", default=["principal", "mrt_sum", "maxmin"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--eval-seed", type=int, default=910_000)
    parser.add_argument("--burn-in", type=int, default=64)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-states", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--population", type=int, default=192)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--cem-seed", type=int, default=20260907)
    parser.add_argument("--csit-mode", choices=["estimated", "genie"], default="estimated")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.episodes, args.stride, args.max_states, args.iterations,
           args.population, args.restarts) < 1 or args.burn_in < 0:
        parser.error("counts must be positive and burn-in must be non-negative")
    if "principal" not in args.beams:
        parser.error("--beams must include principal")

    torch.set_num_threads(1)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    trace = collect_and_audit(args)
    summary, comparisons = summarise(trace)
    trace.to_csv(out / "beam_allocation_trace.csv.gz", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    comparisons.to_csv(out / "paired_comparisons.csv", index=False)

    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "purpose": "paired per-state allocation-reoptimised common-beam audit",
        "selection_information": args.csit_mode,
        "true_channel_use": "final scoring only" if args.csit_mode == "estimated" else "genie diagnostic",
        "checkpoints": [str(Path(p).resolve()) for p in args.checkpoint],
        "checkpoint_sha256": {
            str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in args.checkpoint
        },
        "beams": args.beams,
        "cem": {
            "iterations": args.iterations, "population": args.population,
            "restarts": args.restarts,
            "evaluations_per_scheme_state": args.iterations * args.population * args.restarts,
            "paired_seed": args.cem_seed,
        },
        "state_sampling": {
            "episodes": args.episodes, "eval_seed": args.eval_seed,
            "burn_in": args.burn_in, "stride": args.stride,
            "max_states": args.max_states,
        },
        "script_sha256": script_hash,
        "note": "CEM and the SDMA boundary are approximate optimiser references, not global optima.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nPaired against principal:\n" + comparisons.to_string(index=False))
    print(f"\nPASS: {trace.state_id.nunique()} paired states written to {out}")


if __name__ == "__main__":
    main()
