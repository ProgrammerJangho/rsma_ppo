#!/usr/bin/env python3
"""Audit direct PPO allocations against transmitted powers and payloads.

This is a finite-episode integration diagnostic, not a steady-state benchmark.
Both fixed references run inside the SAME RSMA PHY: common-off is SDMA.
No optimiser, true-channel action selection, or scheduling is used.
"""
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
from rsma_aoi.baselines.common import equal_split, make_action
from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.utils.config import EnvConfig, dump_config_snapshot


def audit_policy(policy, cfg, seeds, label):
    records = []
    payload = np.asarray(cfg.payload_vector)
    with_env = AoIRSMAEnv(cfg)
    try:
        for seed in seeds:
            obs, _ = with_env.reset(seed=seed)
            for step in range(cfg.episode_length):
                state = with_env.get_state()
                h_hat = np.asarray(state["h_hat"])
                norms = np.sum(np.abs(h_hat) ** 2, axis=1)
                pairwise_alignment = []
                for i in range(cfg.num_users):
                    for j in range(i + 1, cfg.num_users):
                        denom = max(float(norms[i] * norms[j]), 1e-30)
                        pairwise_alignment.append(float(np.abs(np.vdot(h_hat[i], h_hat[j])) ** 2 / denom))
                mean_alignment = float(np.mean(pairwise_alignment)) if pairwise_alignment else 0.0
                gain_gap_db = float(10.0 * np.log10(max(norms.max(), 1e-30) / max(norms.min(), 1e-30)))
                pre_aoi_gap = float(np.max(state["aoi"]) - np.min(state["aoi"]))
                if policy is None:
                    common = label == "fixed_rsma"
                    pf, rf = equal_split(np.ones(cfg.num_users, bool), 0.25 if common else 0.0)
                    action = make_action(np.ones(cfg.num_users), pf, rf, 0.3 if common else 0.0)
                else:
                    action = policy.act(obs[None, :])[0]
                obs, _, _, _, info = with_env.step(action)
                np.testing.assert_array_equal(action["schedule"], np.ones(cfg.num_users))
                np.testing.assert_array_equal(info["scheduled_users"], np.ones(cfg.num_users))
                if info["allocator_calls"] != 0 or info["constraint_violation"] > cfg.feasibility_tol:
                    raise AssertionError("action overridden by allocator or infeasible")
                # Compare PPO's requested allocation with the actual transmitted allocation.
                pf = action["power_fractions"]
                np.testing.assert_allclose(
                    np.r_[info["common_power"], info["private_power"]],
                    pf * cfg.total_power, rtol=2e-6, atol=1e-7,
                )
                c = action["rate_fractions"]
                c = c / c.sum()
                rho_action = float(action["common_payload_scale"][0])
                rho = float(info.get("common_payload_effective_scale", rho_action))
                common_bits = payload - info["private_payload"]
                total = rho * np.min(payload[c > 1e-12] / c[c > 1e-12])
                np.testing.assert_allclose(common_bits, c * total, rtol=2e-6, atol=1e-6)
                np.testing.assert_allclose(common_bits.sum(), info["common_payload"], atol=1e-6)
                if cfg.aoi_cap_enabled and not np.all(info["aoi"] < cfg.aoi_max):
                    raise AssertionError("AoI cap binds: diagnostic would conceal failures")
                for k in range(cfg.num_users):
                    records.append({
                        "algorithm": label, "eval_seed": seed, "slot": step + 1, "user": k,
                        "estimated_channel_alignment": mean_alignment,
                        "estimated_gain_gap_db": gain_gap_db,
                        "pre_action_aoi_gap": pre_aoi_gap,
                        "aoi": info["aoi"][k], "ack": info["ack"][k],
                        "common_power_fraction": pf[0], "private_power_fraction": pf[k + 1],
                        "common_enabled": float(pf[0] > 1e-12 and rho > 1e-12),
                        "common_payload_share": c[k], "rho": rho,
                        "rho_action": rho_action,
                        "common_bits": common_bits[k], "private_bits": info["private_payload"][k],
                        "common_rate_allocation": common_bits[k] / cfg.blocklength,
                        "common_code_rate": info["common_payload"] / cfg.blocklength,
                        "private_code_rate": info["private_payload"][k] / cfg.blocklength,
                        "common_bler": info["common_bler"][k],
                        "private_bler": info["private_bler"][k],
                        "success_probability": info["success_probabilities"][k],
                    })
    finally:
        with_env.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/thesis_allocation.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--eval-seed", type=int, default=900000)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = EnvConfig.from_yaml(args.config)
    if (cfg.multiple_access != "rsma" or cfg.scheduling_enabled
            or cfg.allocator_mode != "none" or cfg.power_scale_action
            or cfg.rate_split_mode != "simplex_scaled" or cfg.ack_mode != "sampled"
            or cfg.link_model != "fbl"):
        parser.error("requires direct fixed-power, all-user simplex_scaled FBL-RSMA")
    if args.episodes < 1 or cfg.aoi_init != "one" or (cfg.aoi_cap_enabled and cfg.aoi_max <= cfg.episode_length + 1):
        parser.error("requires episodes >= 1, aoi_init=one and a nonbinding/disabled AoI cap")
    policies = [("fixed_rsma", None), ("sdma_equal", None)]
    if args.checkpoint:
        policy, trained_cfg, _ = PPOPolicy.from_checkpoint(args.checkpoint, deterministic=True)
        if trained_cfg.to_dict() != cfg.to_dict():
            parser.error("checkpoint environment differs from audit config")
        policies.append(("ppo_joint", policy))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)  # never silently overwrite a previous audit
    seeds = list(range(args.eval_seed, args.eval_seed + args.episodes))
    rows = []
    for label, policy in policies:
        rows.extend(audit_policy(policy, cfg, seeds, label))
    trace = pd.DataFrame(rows)
    trace.to_csv(out / "allocation_trace.csv", index=False)
    summary = trace.groupby("algorithm").agg(
        mean_aoi=("aoi", "mean"), pooled_aoi_p95=("aoi", lambda x: x.quantile(0.95)),
        max_observed_aoi=("aoi", "max"), success_rate=("ack", "mean"),
        mean_common_power=("common_power_fraction", "mean"),
        mean_common_code_rate=("common_code_rate", "mean"),
    )
    summary.to_csv(out / "summary.csv")
    dump_config_snapshot(out / "config_snapshot.yaml", env=cfg)
    sources = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve(), Path(args.config).resolve()]
    manifest = {
        "purpose": "integration diagnostic; no burn-in, one trained seed, not a performance claim",
        "eval_seeds": seeds, "checkpoint": args.checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None,
        "python": sys.version, "torch": torch.__version__,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "checks": "all users active; no allocator; requested powers/payloads transmitted; no AoI cap binding",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string())
    print(f"PASS: allocation path verified; diagnostic only. Results: {out}")


if __name__ == "__main__":
    main()
