#!/usr/bin/env python3
"""Audit how strongly a trained actor conditions its RSMA action on channel state.

Three diagnostics are produced on held-out oracle-dataset states, so that every
policy is scored on identical inputs alongside the CEM oracle label:

1. ``state_dependence``: Pearson correlations between each action component and
   the observable state features (channel alignment, AoI gap, AoI sum, per-user
   gain gap), reported separately for the gate decision and for the magnitude
   conditional on the gate being on.
2. ``permutation``: an exchange test.  The two users are symmetric by
   construction in the homogeneous configuration, so relabelling them must map
   ``c_1 -> 1 - c_1`` and must leave the gate and the common-power fraction
   unchanged.  The residual measures how far the actor is from equivariance.
3. ``payload``: the realised payload split, computed with the environment's own
   :func:`split_payload`, so the ``simplex_scaled`` cap ``min_k D_k / c_k`` is
   applied rather than assumed.

Usage::

    python scripts/audit_action_state_dependence.py \
        --dataset datasets/rician_20deg_csit02_gate_oracle_v1 \
        --split test \
        --out-dir results/action_state_dependence_v1 \
        --policy 0step=runs/objective_actor_rician_20deg_csit02_single_seed{seed}_v2/checkpoint.pt \
        --policy 262k=results/rician_ppo_20deg_csit02_matched_budget_262k_v1/runs/rsma_joint_gate_seed{seed}/checkpoint.pt \
        --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsma_aoi.agents.ppo import PPOPolicy  # noqa: E402
from rsma_aoi.envs.rsma_phy import split_payload  # noqa: E402

_ON = 1e-3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def state_features(z) -> dict[str, np.ndarray]:
    """Observable state features the transmitter could condition on."""
    h = z["h_hat_real"] + 1j * z["h_hat_imag"]              # (N, K, Nt)
    inner = np.abs(np.sum(np.conj(h[:, 0, :]) * h[:, 1, :], axis=1))
    norms = np.linalg.norm(h[:, 0, :], axis=1) * np.linalg.norm(h[:, 1, :], axis=1)
    alignment = (inner / np.maximum(norms, 1e-12)) ** 2      # squared correlation
    gains = np.sum(np.abs(h) ** 2, axis=2)
    aoi = z["aoi"]
    return {
        "alignment": alignment,
        "aoi_gap": np.abs(aoi[:, 0] - aoi[:, 1]),
        "aoi_sum": aoi.sum(axis=1),
        "gain_gap_db": 10.0 * np.log10(np.maximum(gains[:, 0], 1e-12)
                                       / np.maximum(gains[:, 1], 1e-12)),
    }


def swap_users(obs: np.ndarray, num_users: int, num_antennas: int,
               include_prev_action: bool, channel_mode: str = "complex_flat") -> np.ndarray:
    """Relabel user 0 <-> user 1 inside a flat observation.

    Supports the environment's ``complex_flat``, ``gain_features`` and ``none``
    channel layouts.  Any trailing global feature such as SNR is left intact.
    """
    if num_users != 2:
        raise SystemExit("the exchange test is defined for the two-user configuration")
    k, nt = num_users, num_antennas
    out = obs.copy()
    permutation = np.arange(k)
    permutation[[0, 1]] = permutation[[1, 0]]
    i = 0
    out[:, [i, i + 1]] = obs[:, [i + 1, i]]                       # AoI
    i += k
    if channel_mode == "complex_flat":
        for _ in range(2):                                        # h.real then h.imag
            block = obs[:, i:i + k * nt].reshape(-1, k, nt)[:, permutation, :]
            out[:, i:i + k * nt] = block.reshape(-1, k * nt)
            i += k * nt
    elif channel_mode == "gain_features":
        out[:, i:i + k] = obs[:, i:i + k][:, permutation]         # per-user gains
        i += k
        pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
        pair_to_index = {pair: idx for idx, pair in enumerate(pairs)}
        source = []
        for a, b in pairs:
            mapped = tuple(sorted((int(permutation[a]), int(permutation[b]))))
            source.append(pair_to_index[mapped])
        width = len(pairs)
        out[:, i:i + width] = obs[:, i:i + width][:, source]
        i += width
    elif channel_mode != "none":
        raise ValueError(f"unsupported channel observation mode: {channel_mode}")
    out[:, [i, i + 1]] = obs[:, [i + 1, i]]                       # prev ACK
    i += k
    if include_prev_action:
        out[:, [i + 1, i + 2]] = obs[:, [i + 2, i + 1]]           # prev private power
        i += k + 1
        out[:, [i, i + 1]] = obs[:, [i + 1, i]]                   # prev schedule
    return out


def actor_actions(path: Path, obs: np.ndarray):
    policy, env_cfg, _ = PPOPolicy.from_checkpoint(str(path), deterministic=True)
    acts = policy.act(obs, None)
    return {
        "common_power": np.array([a["power_fractions"][0] for a in acts]),
        "rate_c1": np.array([a["rate_fractions"][0] for a in acts]),
        "rho": np.array([float(np.ravel(a["common_payload_scale"])[0]) for a in acts]),
    }, env_cfg


def payload_rows(name: str, c1: np.ndarray, rho: np.ndarray, on: np.ndarray,
                 payload: np.ndarray) -> dict:
    """Realised payload split under the environment's own mapping."""
    if on.sum() == 0:
        return {"policy": name, "on_rate": 0.0}
    totals, privates = [], []
    for i in np.flatnonzero(on):
        d_c, d_p = split_payload(payload, np.ones(2, dtype=bool),
                                 np.array([c1[i], 1.0 - c1[i]]), float(rho[i]),
                                 "simplex_scaled")
        totals.append(d_c)
        privates.append(d_p)
    privates = np.asarray(privates)
    return {
        "policy": name,
        "on_rate": float(on.mean()),
        "mean_rate_c1_on": float(c1[on].mean()),
        "mean_rho_on": float(rho[on].mean()),
        "mean_common_bits": float(np.mean(totals)),
        "mean_private_bits_user0": float(privates[:, 0].mean()),
        "mean_private_bits_user1": float(privates[:, 1].mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--policy", action="append", required=True,
                   metavar="NAME=PATH_TEMPLATE",
                   help="'{seed}' in the path is substituted with each --seeds value")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = p.parse_args()

    ds = Path(args.dataset)
    z = np.load(ds / f"{args.split}.npz", allow_pickle=False)
    obs = z["observation"].astype(np.float32)
    feats = state_features(z)
    oracle = {
        "common_power": z["oracle_power_fractions"][:, 0],
        "rate_c1": z["oracle_rate_fractions"][:, 0],
        "rho": z["oracle_rho_r"].ravel(),
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dep_rows, perm_rows, pay_rows = [], [], []
    checkpoints: dict[str, str] = {}
    env_cfg = None

    def add_dependence(name: str, act: dict) -> None:
        on = act["common_power"] > _ON
        row = {"policy": name, "on_rate": float(on.mean()),
               "mean_common_power": float(act["common_power"].mean()),
               "mean_common_power_on": float(act["common_power"][on].mean()) if on.any() else float("nan")}
        for fname, fval in feats.items():
            row[f"r_gate__{fname}"] = corr(fval, on.astype(float))
            for aname in ("common_power", "rate_c1", "rho"):
                row[f"r_{aname}_given_on__{fname}"] = (
                    corr(fval[on], act[aname][on]) if on.sum() >= 3 else float("nan"))
        dep_rows.append(row)

    payload_vec = np.array([256.0, 256.0])
    add_dependence("oracle", oracle)
    on_o = oracle["common_power"] > _ON
    r = payload_rows("oracle", oracle["rate_c1"], oracle["rho"], on_o, payload_vec)
    r["mean_common_power_on"] = float(oracle["common_power"][on_o].mean())
    pay_rows.append(r)

    for spec in args.policy:
        if "=" not in spec:
            raise SystemExit(f"--policy needs NAME=PATH, got {spec!r}")
        name, template = spec.split("=", 1)
        for seed in args.seeds:
            path = Path(template.replace("{seed}", str(seed)))
            if not path.is_file():
                raise SystemExit(f"missing checkpoint: {path}")
            tag = f"{name}_seed{seed}"
            checkpoints[tag] = sha256(path)
            act, env_cfg = actor_actions(path, obs)
            add_dependence(tag, act)
            on = act["common_power"] > _ON
            r = payload_rows(tag, act["rate_c1"], act["rho"], on, payload_vec)
            r["mean_common_power_on"] = float(act["common_power"][on].mean()) if on.any() else float("nan")
            pay_rows.append(r)

            swapped = swap_users(
                obs, env_cfg.num_users, env_cfg.num_antennas,
                env_cfg.obs_include_prev_action, env_cfg.obs_channel_mode,
            )
            act_s, _ = actor_actions(path, swapped)
            on_s = act_s["common_power"] > _ON
            perm_rows.append({
                "policy": tag,
                # c_1 must map to 1 - c_1 under relabelling
                "equivariance_residual_rate": float(
                    np.abs(act_s["rate_c1"][on] - (1.0 - act["rate_c1"][on])).mean()) if on.any() else float("nan"),
                # the gate and the common-power magnitude must be invariant
                "gate_agreement": float((on == on_s).mean()),
                "invariance_residual_common_power": float(
                    np.abs(act_s["common_power"][on] - act["common_power"][on]).mean()) if on.any() else float("nan"),
                "mean_rate_c1_original": float(act["rate_c1"][on].mean()) if on.any() else float("nan"),
                "mean_rate_c1_swapped": float(act_s["rate_c1"][on].mean()) if on.any() else float("nan"),
            })

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        keys = list(rows[0].keys())
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write_csv(out / "state_dependence.csv", dep_rows)
    write_csv(out / "permutation_test.csv", perm_rows)
    write_csv(out / "payload_split.csv", pay_rows)
    (out / "manifest.json").write_text(json.dumps({
        "purpose": "action state-dependence, permutation equivariance and payload split of trained RSMA actors",
        "dataset": str(ds),
        "split": args.split,
        "num_states": int(obs.shape[0]),
        "dataset_config_hash": str(z["config_hash"]),
        "checkpoint_sha256": checkpoints,
        "script_sha256": sha256(Path(__file__)),
        "notes": [
            "states come from the oracle dataset's source policies, not from each actor's own trajectory",
            "payload split uses rsma_aoi.envs.rsma_phy.split_payload with mode simplex_scaled",
            "the exchange test assumes the homogeneous two-user configuration is symmetric",
        ],
    }, indent=2))
    print(f"wrote {out}/state_dependence.csv, permutation_test.csv, payload_split.csv")


if __name__ == "__main__":
    main()
