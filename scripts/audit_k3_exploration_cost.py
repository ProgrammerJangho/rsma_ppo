#!/usr/bin/env python3
"""Measure the FBL objective cost of PPO's stochastic action sampling."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsma_aoi.agents.ppo import PPOPolicy  # noqa: E402
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer  # noqa: E402
from rsma_aoi.evaluation.oracle import expected_aoi_of_action  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402


def _score(phy, cfg, z, i, action):
    h_true = z["h_true_real"][i] + 1j * z["h_true_imag"][i]
    h_hat = z["h_hat_real"][i] + 1j * z["h_hat_imag"][i]
    if "episode_total_power" in z:
        phy.total_power = float(z["episode_total_power"][i])
    return expected_aoi_of_action(
        phy, z["aoi"][i], h_true, h_hat,
        np.asarray(cfg.weight_vector, dtype=np.float64),
        float(cfg.aoi_max) if cfg.aoi_cap_enabled else np.inf,
        np.ones(cfg.num_users, dtype=bool), action["power_fractions"],
        action["rate_fractions"], float(np.ravel(action["common_payload_scale"])[0]),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="datasets/rician_k3_nt2_all3_ppo_visited_oracle_v1/test.npz")
    p.add_argument("--config", default="configs/thesis_rician_k3_nt2_all3_ppo_csit02.yaml")
    p.add_argument("--policy", action="append", required=True, metavar="NAME=CHECKPOINT")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--max-states", type=int, default=150)
    p.add_argument("--out-dir", default="results/k3_nt2_all3_exploration_cost_v1")
    args = p.parse_args()
    z = np.load(ROOT / args.dataset, allow_pickle=False)
    cfg = EnvConfig.from_yaml(ROOT / args.config)
    phy = RSMAPhysicalLayer(cfg)
    n = min(args.max_states, len(z["observation"]))
    rows = []
    for spec in args.policy:
        name, raw_path = spec.split("=", 1)
        deterministic, _, _ = PPOPolicy.from_checkpoint(ROOT / raw_path, deterministic=True, device="cpu")
        stochastic, _, _ = PPOPolicy.from_checkpoint(ROOT / raw_path, deterministic=False, device="cpu")
        torch.manual_seed(983000)
        for i in range(n):
            obs = z["observation"][i].astype(np.float32)
            det_action = deterministic.act(obs[None, :], None)[0]
            det_cost = _score(phy, cfg, z, i, det_action)
            sampled = stochastic.act(np.repeat(obs[None, :], args.samples, axis=0), None)
            sampled_costs = np.asarray([_score(phy, cfg, z, i, action) for action in sampled])
            rows.append({
                "policy": name, "state": i, "deterministic_aoi": det_cost,
                "stochastic_mean_aoi": float(sampled_costs.mean()),
                "stochastic_p90_aoi": float(np.quantile(sampled_costs, 0.9)),
                "exploration_cost": float(sampled_costs.mean() - det_cost),
            })
    frame = pd.DataFrame(rows)
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(out / "per_state.csv", index=False)
    summary = frame.groupby("policy", as_index=False).agg(
        states=("state", "size"), deterministic_aoi=("deterministic_aoi", "mean"),
        stochastic_mean_aoi=("stochastic_mean_aoi", "mean"),
        stochastic_p90_aoi=("stochastic_p90_aoi", "mean"),
        exploration_cost=("exploration_cost", "mean"),
        exploration_cost_median=("exploration_cost", "median"),
    )
    summary["exploration_cost_pct"] = 100 * summary.exploration_cost / summary.deterministic_aoi
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nCOMPLETE: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
