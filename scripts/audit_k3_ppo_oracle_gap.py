#!/usr/bin/env python3
"""Decompose the K=3 learned-policy gap to its safeguarded CEM labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsma_aoi.agents.ppo import PPOPolicy  # noqa: E402
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer  # noqa: E402
from rsma_aoi.evaluation.oracle import expected_aoi_of_action  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _common_off(power: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    private = np.clip(np.asarray(power[1:], dtype=np.float64), 0.0, None)
    private = private / private.sum() if private.sum() > 1e-12 else np.full(k, 1.0 / k)
    return np.concatenate(([0.0], private)), np.full(k, 1.0 / k), 0.0


def _score(phy, cfg, aoi, h_true, h_hat, power, rate, rho, estimated=False):
    channel = h_hat if estimated else h_true
    return expected_aoi_of_action(
        phy, aoi, channel, h_hat, np.asarray(cfg.weight_vector, dtype=np.float64),
        float(cfg.aoi_max) if cfg.aoi_cap_enabled else np.inf,
        np.ones(cfg.num_users, dtype=bool), power, rate, rho,
    )


def _load_dataset(directory: Path, splits: list[str]):
    chunks = []
    for split in splits:
        path = directory / f"{split}.npz"
        if path.is_file():
            z = np.load(path, allow_pickle=False)
            chunks.append({key: z[key] for key in z.files if key != "config_hash"})
    if not chunks:
        raise FileNotFoundError(f"no requested splits in {directory}")
    keys = chunks[0].keys()
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--config", default="configs/thesis_rician_k3_nt2_all3_ppo_csit02.yaml")
    parser.add_argument("--policy", action="append", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    data = _load_dataset(dataset, args.splits)
    cfg = EnvConfig.from_yaml(ROOT / args.config)
    phy = RSMAPhysicalLayer(cfg)
    obs = data["observation"].astype(np.float32)
    h_true = data["h_true_real"] + 1j * data["h_true_imag"]
    h_hat = data["h_hat_real"] + 1j * data["h_hat_imag"]
    n, k = obs.shape[0], cfg.num_users
    per_state: list[dict] = []
    checkpoint_hashes = {}

    for spec in args.policy:
        name, raw_path = spec.split("=", 1)
        path = ROOT / raw_path
        actor, _, _ = PPOPolicy.from_checkpoint(path, deterministic=True, device="cpu")
        checkpoint_hashes[name] = _sha256(path)
        actions = actor.act(obs, None)
        for i, action in enumerate(actions):
            if "episode_total_power" in data:
                phy.total_power = float(data["episode_total_power"][i])
            power = np.asarray(action["power_fractions"], dtype=np.float64)
            rate = np.asarray(action["rate_fractions"], dtype=np.float64)
            rho = float(np.ravel(action["common_payload_scale"])[0])
            off_power, off_rate, off_rho = _common_off(power, k)
            raw_true = _score(phy, cfg, data["aoi"][i], h_true[i], h_hat[i], power, rate, rho)
            off_true = _score(phy, cfg, data["aoi"][i], h_true[i], h_hat[i], off_power, off_rate, off_rho)
            raw_est = _score(phy, cfg, data["aoi"][i], h_true[i], h_hat[i], power, rate, rho, True)
            off_est = _score(phy, cfg, data["aoi"][i], h_true[i], h_hat[i], off_power, off_rate, off_rho, True)
            safeguard_on = raw_est <= off_est
            safe_true = raw_true if safeguard_on else off_true
            oracle_power = np.asarray(data["oracle_power_fractions"][i], dtype=np.float64)
            oracle_rate = np.asarray(data["oracle_rate_fractions"][i], dtype=np.float64)
            oracle_rho = float(data["oracle_rho_r"][i])
            actor_components = (power, rate, rho)
            oracle_components = (oracle_power, oracle_rate, oracle_rho)
            hybrid_cost: dict[int, float] = {}
            for mask in range(8):
                components = [
                    oracle_components[j] if mask & (1 << j) else actor_components[j]
                    for j in range(3)
                ]
                hybrid_cost[mask] = _score(
                    phy, cfg, data["aoi"][i], h_true[i], h_hat[i],
                    components[0], components[1], components[2],
                )
            shapley = []
            for component in range(3):
                contribution = 0.0
                others = [j for j in range(3) if j != component]
                for subset_bits in range(1 << len(others)):
                    subset = 0
                    size = 0
                    for bit, other in enumerate(others):
                        if subset_bits & (1 << bit):
                            subset |= 1 << other
                            size += 1
                    weight = (
                        math.factorial(size) * math.factorial(2 - size)
                        / math.factorial(3)
                    )
                    contribution += weight * (
                        hybrid_cost[subset] - hybrid_cost[subset | (1 << component)]
                    )
                shapley.append(contribution)
            actor_on = bool(power[0] > 1e-10 and rho > 1e-10)
            oracle_on = bool(data["oracle_common_gate"][i])
            per_state.append({
                "policy": name,
                "state": i,
                "source": str(data["source"][i]),
                "oracle_expected_aoi": float(data["oracle_expected_aoi"][i]),
                "actor_expected_aoi": raw_true,
                "common_off_expected_aoi": off_true,
                "estimated_safeguard_expected_aoi": safe_true,
                "genie_actor_boundary_expected_aoi": min(raw_true, off_true),
                "actor_regret": raw_true - float(data["oracle_expected_aoi"][i]),
                "estimated_safeguard_regret": safe_true - float(data["oracle_expected_aoi"][i]),
                "genie_actor_boundary_regret": min(raw_true, off_true) - float(data["oracle_expected_aoi"][i]),
                "actor_on": actor_on,
                "oracle_on": oracle_on,
                "estimated_safeguard_on": safeguard_on,
                "actor_gate_correct": actor_on == oracle_on,
                "safeguard_gate_correct": safeguard_on == oracle_on,
                "power_l1": float(np.abs(power - oracle_power).sum()),
                "rate_l1": float(np.abs(rate - oracle_rate).sum()),
                "rho_abs_error": abs(rho - oracle_rho),
                "actor_common_power": float(power[0]),
                "actor_rho": rho,
                "oracle_action_recomputed_aoi": hybrid_cost[7],
                "shapley_power_improvement": shapley[0],
                "shapley_rate_improvement": shapley[1],
                "shapley_rho_improvement": shapley[2],
                **{f"actor_power_{j}": float(power[j]) for j in range(k + 1)},
                **{f"oracle_power_{j}": float(oracle_power[j]) for j in range(k + 1)},
                **{f"actor_rate_{j}": float(rate[j]) for j in range(k)},
                **{f"oracle_rate_{j}": float(oracle_rate[j]) for j in range(k)},
                "oracle_rho": oracle_rho,
            })

    frame = pd.DataFrame(per_state)
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(out / "per_state.csv", index=False)
    summary = frame.groupby("policy", as_index=False).agg(
        states=("state", "size"),
        oracle_expected_aoi=("oracle_expected_aoi", "mean"),
        actor_expected_aoi=("actor_expected_aoi", "mean"),
        actor_regret=("actor_regret", "mean"),
        actor_regret_median=("actor_regret", "median"),
        actor_regret_p90=("actor_regret", lambda x: float(np.quantile(x, 0.9))),
        estimated_safeguard_expected_aoi=("estimated_safeguard_expected_aoi", "mean"),
        estimated_safeguard_regret=("estimated_safeguard_regret", "mean"),
        genie_actor_boundary_regret=("genie_actor_boundary_regret", "mean"),
        gate_accuracy=("actor_gate_correct", "mean"),
        safeguard_gate_accuracy=("safeguard_gate_correct", "mean"),
        actor_on_rate=("actor_on", "mean"),
        oracle_on_rate=("oracle_on", "mean"),
        safeguard_on_rate=("estimated_safeguard_on", "mean"),
        power_l1=("power_l1", "mean"),
        rate_l1=("rate_l1", "mean"),
        rho_mae=("rho_abs_error", "mean"),
        common_power=("actor_common_power", "mean"),
        rho=("actor_rho", "mean"),
        oracle_recompute_error=(
            "oracle_action_recomputed_aoi",
            lambda x: float(np.mean(x.to_numpy() - frame.loc[x.index, "oracle_expected_aoi"].to_numpy())),
        ),
        shapley_power_improvement=("shapley_power_improvement", "mean"),
        shapley_rate_improvement=("shapley_rate_improvement", "mean"),
        shapley_rho_improvement=("shapley_rho_improvement", "mean"),
    )
    summary.to_csv(out / "summary.csv", index=False)
    by_gate = frame.groupby(["policy", "oracle_on"], as_index=False).agg(
        states=("state", "size"), actor_regret=("actor_regret", "mean"),
        gate_accuracy=("actor_gate_correct", "mean"),
        safeguard_regret=("estimated_safeguard_regret", "mean"),
    )
    by_gate.to_csv(out / "by_oracle_gate.csv", index=False)
    component_rows = []
    for policy, group in frame.groupby("policy"):
        for family, width in (("power", k + 1), ("rate", k)):
            for j in range(width):
                actor_values = group[f"actor_{family}_{j}"].to_numpy()
                oracle_values = group[f"oracle_{family}_{j}"].to_numpy()
                correlation = (
                    float(np.corrcoef(actor_values, oracle_values)[0, 1])
                    if actor_values.std() > 1e-12 and oracle_values.std() > 1e-12
                    else float("nan")
                )
                component_rows.append({
                    "policy": policy, "family": family, "component": j,
                    "actor_mean": float(actor_values.mean()),
                    "oracle_mean": float(oracle_values.mean()),
                    "mae": float(np.abs(actor_values - oracle_values).mean()),
                    "correlation": correlation,
                })
        actor_values = group["actor_rho"].to_numpy()
        oracle_values = group["oracle_rho"].to_numpy()
        component_rows.append({
            "policy": policy, "family": "rho", "component": 0,
            "actor_mean": float(actor_values.mean()),
            "oracle_mean": float(oracle_values.mean()),
            "mae": float(np.abs(actor_values - oracle_values).mean()),
            "correlation": float(np.corrcoef(actor_values, oracle_values)[0, 1])
            if actor_values.std() > 1e-12 and oracle_values.std() > 1e-12 else float("nan"),
        })
    pd.DataFrame(component_rows).to_csv(out / "component_tracking.csv", index=False)
    manifest = {
        "dataset": str(dataset), "splits": args.splits, "states": int(n),
        "config": args.config, "checkpoint_sha256": checkpoint_hashes,
        "interpretation": {
            "estimated_safeguard": "choose actor common-on action or its common-off projection using h_hat",
            "genie_actor_boundary": "unattainable diagnostic choosing the lower true-channel objective",
            "oracle": "finite-budget safeguarded CEM label selected using estimated CSIT",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nBY ORACLE GATE")
    print(by_gate.to_string(index=False))
    print(f"\nCOMPLETE: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
