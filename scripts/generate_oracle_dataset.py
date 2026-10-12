#!/usr/bin/env python3
"""Generate a behaviour-cloning dataset from the myopic scheduling oracle.

States are collected from several *source* policies so that the dataset is not
biased towards one policy's state distribution, and each state is labelled with
the oracle's subset choice and its optimised RSMA allocation
(:mod:`rsma_aoi.evaluation.oracle`).

Label CSIT
----------
The **estimated-CSIT** oracle is the default label: it optimises against the
channel estimate the policy itself sees, so it is a target a real policy can
attain.  ``--also-genie`` additionally stores the genie-CSIT label under
``genie_*`` columns for analysis only - never train on it.

Leakage
-------
Splits are made **by trajectory**, not by state: every state of one episode goes
to exactly one split, so temporally adjacent (highly correlated) states cannot
straddle train and test.

Examples:
    python scripts/generate_oracle_dataset.py --train 2000 --val 500 --test 500 --workers 8
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsma_aoi.baselines import build_baseline  # noqa: E402
from rsma_aoi.envs.aoi_env import AoIRSMAEnv  # noqa: E402
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer  # noqa: E402
from rsma_aoi.evaluation.oracle import config_hash, solve_state  # noqa: E402
from rsma_aoi.utils.config import EnvConfig, dump_config_snapshot  # noqa: E402
from rsma_aoi.utils.logging import get_logger  # noqa: E402
from rsma_aoi.utils.action_mapping import (  # noqa: E402
    DIRECT_ACTION_SCHEMA, HARD_SIGMOID_LIMIT,
)

#: Columns every dataset shard must contain (checked by the tests).
REQUIRED_KEYS = (
    "observation", "h_hat_real", "h_hat_imag", "aoi", "oracle_mask", "oracle_count",
    "oracle_power_fractions", "oracle_rate_fractions", "oracle_rho_r",
    "oracle_common_gate",
    "oracle_expected_aoi", "second_mask", "second_expected_aoi", "objective_gap",
    "cem_seed", "source", "trajectory", "split",
)

DIRECT_KEYS = (
    "oracle_common_power_fraction",
    "oracle_private_power_fractions",
    "oracle_common_payload_fractions",
)


def _attach_direct_action_columns(row: dict, active: np.ndarray) -> None:
    """Store canonical ``(alpha, q, phi)`` labels alongside PHY aliases."""
    power = np.asarray(row["oracle_power_fractions"], dtype=np.float64)
    alpha = float(np.clip(power[0], 0.0, 1.0))
    private_mass = float(power[1:].sum())
    if private_mass > 1e-12:
        q = power[1:] / private_mass
    else:
        active_f = np.asarray(active, dtype=np.float64)
        q = active_f / max(float(active_f.sum()), 1.0)
    row["oracle_common_power_fraction"] = np.float32(alpha)
    row["oracle_private_power_fractions"] = q.astype(np.float32)
    row["oracle_common_payload_fractions"] = np.asarray(
        row["oracle_rate_fractions"], dtype=np.float32
    )


def _make_source_policy(name: str, env_cfg: EnvConfig, seed: int):
    """Return a policy for a named state source ('ppo:<checkpoint>' allowed)."""
    if name.startswith("ppo-deterministic:"):
        from rsma_aoi.agents.ppo import PPOPolicy

        policy, _, _ = PPOPolicy.from_checkpoint(
            name.split(":", 1)[1], deterministic=True
        )
        return policy
    if name.startswith("ppo:"):
        from rsma_aoi.agents.ppo import PPOPolicy

        policy, _, _ = PPOPolicy.from_checkpoint(name.split(":", 1)[1], deterministic=False)
        return policy
    policy, overrides = build_baseline(name, env_cfg, seed=seed)
    if overrides:
        raise SystemExit(f"source {name!r} changes the PHY; use an RSMA-compatible policy")
    return policy


def collect_trajectory(
    env: AoIRSMAEnv, policy, seed: int, source: str, max_states: int
) -> list[dict]:
    """Run one episode and return its per-slot raw records."""
    obs, reset_info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(seed=seed)
    records, done = [], False
    while not done and len(records) < max_states:
        state = env.get_state()
        action = policy.act(obs[None, :], [state])[0]
        records.append(
            {
                "observation": obs.astype(np.float32).copy(),
                "aoi": state["aoi"].astype(np.float32).copy(),
                "h_hat": state["h_hat"].copy(),
                "h_true": env.true_channel,      # diagnostic metadata only
                "source": source,
                "trajectory": f"{source}#{seed}",
                "episode_snr_db": np.float32(state["snr_db"]),
                "episode_total_power": np.float32(state["total_power"]),
                "episode_blocklength": np.int16(state["blocklength"]),
                **(
                    {
                        "episode_user_angles_deg": np.asarray(
                            reset_info["user_angles_deg"], dtype=np.float32
                        ),
                        "episode_geometry_index": np.int16(
                            reset_info.get("geometry_index", -1)
                        ),
                    }
                    if "user_angles_deg" in reset_info
                    else {}
                ),
            }
        )
        obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
    return records


def _label_chunk(payload: tuple) -> list[dict]:
    """Worker: attach oracle labels to a chunk of raw records."""
    env_cfg, records, csit_mode, also_genie, base_seed, offset = payload[:6]
    # Keep direct unit-test/legacy calls backward compatible while allowing
    # higher-dimensional overloaded studies to increase the CEM budget.
    cem_iterations, cem_population, cem_restarts = (
        payload[6:9] if len(payload) >= 9 else (5, 192, 1)
    )
    phy = RSMAPhysicalLayer(env_cfg)
    weights = np.asarray(env_cfg.weight_vector, dtype=np.float64)
    aoi_max = float(env_cfg.aoi_max)
    labelled = []
    for i, rec in enumerate(records):
        cem_seed = int(base_seed + offset + i)
        record_cfg = (
            replace(
                env_cfg,
                snr_db=float(rec["episode_snr_db"]),
                snr_db_values=None,
                blocklength=int(rec.get("episode_blocklength", env_cfg.blocklength)),
                blocklength_values=None,
            )
            if "episode_snr_db" in rec else env_cfg
        )
        if not env_cfg.scheduling_enabled:
            # The thesis allocation-only policy always serves every user.  Use
            # the exact common-off SDMA boundary as a candidate and supervise
            # its Bernoulli gate explicitly; a scheduling oracle would produce
            # labels that this policy class cannot execute.
            from rsma_aoi.evaluation.beam_audit import audit_state_allocations

            audit = audit_state_allocations(
                record_cfg,
                rec["aoi"].astype(np.float64),
                rec["h_true"],
                rec["h_hat"],
                np.ones(env_cfg.num_users, dtype=bool),
                beam_modes=(env_cfg.common_beam,),
                csit_mode=csit_mode,
                seed=cem_seed,
                iterations=cem_iterations,
                population=cem_population,
                restarts=cem_restarts,
            )[0]
            selected_boundary = bool(audit["selected_boundary"])
            best_expected = float(audit["safe_expected_aoi_true"])
            raw_expected = float(audit["raw_expected_aoi_true"])
            boundary_expected = float(
                audit["base_expected_aoi"] - audit["sdma_gain_true"]
            )
            second_expected = raw_expected if selected_boundary else boundary_expected
            # Candidate selection is information-feasible: it uses h_hat, so
            # its non-negative margin must also be measured on h_hat.  The
            # rejected action can outperform on this one hidden h_true draw;
            # retain that realised difference under a separate honest name.
            objective_gap = float(
                audit["alternative_expected_aoi_estimated"]
                - audit["safe_expected_aoi_estimated"]
            )
            active = np.ones(env_cfg.num_users, dtype=bool)
            row = {
                "observation": rec["observation"],
                "h_hat_real": rec["h_hat"].real.astype(np.float32),
                "h_hat_imag": rec["h_hat"].imag.astype(np.float32),
                "aoi": rec["aoi"],
                "oracle_mask": active.astype(np.int8),
                "oracle_count": np.int8(env_cfg.num_users),
                "oracle_power_fractions": audit["safe_power_fractions"].astype(np.float32),
                "oracle_rate_fractions": audit["safe_rate_fractions"].astype(np.float32),
                "oracle_rho_r": np.float32(audit["safe_rho"]),
                "oracle_common_gate": np.int8(not selected_boundary),
                "oracle_expected_aoi": np.float32(best_expected),
                "oracle_success": audit["safe_success_probabilities"].astype(np.float32),
                "second_mask": active.astype(np.int8),
                "second_expected_aoi": np.float32(second_expected),
                "objective_gap": np.float32(objective_gap),
                "realized_alternative_gap": np.float32(
                    second_expected - best_expected
                ),
                "cem_seed": np.int64(cem_seed),
                "source": rec["source"],
                "trajectory": rec["trajectory"],
                "h_true_real": rec["h_true"].real.astype(np.float32),
                "h_true_imag": rec["h_true"].imag.astype(np.float32),
            }
            if env_cfg.rate_split_mode == "per_user_fraction":
                _attach_direct_action_columns(row, active)
            if also_genie:
                genie_audit = audit_state_allocations(
                    record_cfg,
                    rec["aoi"].astype(np.float64),
                    rec["h_true"],
                    rec["h_hat"],
                    active,
                    beam_modes=(env_cfg.common_beam,),
                    csit_mode="genie",
                    seed=cem_seed + 10**7,
                    iterations=cem_iterations,
                    population=cem_population,
                    restarts=cem_restarts,
                )[0]
                row["genie_mask"] = active.astype(np.int8)
                row["genie_count"] = np.int8(env_cfg.num_users)
                row["genie_expected_aoi"] = np.float32(
                    genie_audit["safe_expected_aoi_true"]
                )
            for key in (
                "episode_user_angles_deg", "episode_geometry_index",
                "episode_snr_db", "episode_total_power",
                "episode_blocklength",
            ):
                if key in rec:
                    row[key] = rec[key]
            labelled.append(row)
            continue
        phy = RSMAPhysicalLayer(record_cfg)
        rng = np.random.default_rng(cem_seed)
        solutions = solve_state(
            phy, rec["aoi"].astype(np.float64), rec["h_true"], rec["h_hat"],
            weights, aoi_max, csit_mode=csit_mode, rng=rng,
        )
        best, second = solutions[0], solutions[1]
        row = {
            "observation": rec["observation"],
            "h_hat_real": rec["h_hat"].real.astype(np.float32),
            "h_hat_imag": rec["h_hat"].imag.astype(np.float32),
            "aoi": rec["aoi"],
            "oracle_mask": best.active.astype(np.int8),
            "oracle_count": np.int8(best.num_active),
            "oracle_power_fractions": best.power_fractions.astype(np.float32),
            "oracle_rate_fractions": best.rate_fractions.astype(np.float32),
            "oracle_rho_r": np.float32(best.rho_r),
            "oracle_common_gate": np.int8(
                best.power_fractions[0] > 1e-10 and best.rho_r > 1e-10
            ),
            "oracle_expected_aoi": np.float32(best.expected_aoi),
            "oracle_success": best.success_prob.astype(np.float32),
            "second_mask": second.active.astype(np.int8),
            "second_expected_aoi": np.float32(second.expected_aoi),
            "objective_gap": np.float32(second.expected_aoi - best.expected_aoi),
            "cem_seed": np.int64(cem_seed),
            "source": rec["source"],
            "trajectory": rec["trajectory"],
            # metadata only - never an input feature
            "h_true_real": rec["h_true"].real.astype(np.float32),
            "h_true_imag": rec["h_true"].imag.astype(np.float32),
        }
        if env_cfg.rate_split_mode == "per_user_fraction":
            _attach_direct_action_columns(row, best.active)
        if also_genie:
            genie = solve_state(
                phy, rec["aoi"].astype(np.float64), rec["h_true"], rec["h_hat"],
                weights, aoi_max, csit_mode="genie",
                rng=np.random.default_rng(cem_seed + 10**7),
            )[0]
            row["genie_mask"] = genie.active.astype(np.int8)
            row["genie_count"] = np.int8(genie.num_active)
            row["genie_expected_aoi"] = np.float32(genie.expected_aoi)
        for key in (
            "episode_user_angles_deg", "episode_geometry_index",
            "episode_snr_db", "episode_total_power",
            "episode_blocklength",
        ):
            if key in rec:
                row[key] = rec[key]
        labelled.append(row)
    return labelled


def label_records(
    env_cfg: EnvConfig, records: list[dict], csit_mode: str, also_genie: bool,
    seed: int, workers: int, logger, cem_iterations: int = 5,
    cem_population: int = 192, cem_restarts: int = 1,
) -> list[dict]:
    """Label all records, optionally across processes."""
    if workers <= 1:
        return _label_chunk((
            env_cfg, records, csit_mode, also_genie, seed, 0,
            cem_iterations, cem_population, cem_restarts,
        ))
    import multiprocessing as mp

    chunks = np.array_split(np.arange(len(records)), workers)
    payloads = [
        (
            env_cfg, [records[i] for i in chunk], csit_mode, also_genie, seed,
            int(chunk[0]), cem_iterations, cem_population, cem_restarts,
        )
        for chunk in chunks if len(chunk)
    ]
    logger.info("labelling %d states with %d workers", len(records), len(payloads))
    with mp.get_context("spawn").Pool(len(payloads)) as pool:
        return [row for part in pool.map(_label_chunk, payloads) for row in part]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/m2_hybrid.yaml")
    parser.add_argument("--sources", nargs="+",
                        default=["random", "max_aoi_first", "equal_power"],
                        help="state sources; 'ppo:<checkpoint>' is also accepted")
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--val", type=int, default=500)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--csit-mode", choices=["estimated", "genie"], default="estimated")
    parser.add_argument("--also-genie", action="store_true",
                        help="additionally store genie labels (analysis only)")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--states-per-trajectory", type=int, default=25,
                        help="cap per episode so the split has many trajectories to mix")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cem-iterations", type=int, default=5)
    parser.add_argument("--cem-population", type=int, default=192)
    parser.add_argument("--cem-restarts", type=int, default=1)
    parser.add_argument("--out-dir", default="datasets/oracle_m2")
    args = parser.parse_args(argv)

    logger = get_logger()
    env_cfg = EnvConfig.from_yaml(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = args.train + args.val + args.test
    per_source = int(np.ceil(total / len(args.sources)))
    start = time.time()

    # -- collect states, keeping whole trajectories together ----------------
    rng = np.random.default_rng(args.seed)
    raw: list[dict] = []
    env = AoIRSMAEnv(env_cfg)
    used_episode_seeds: set[int] = set()
    for source in args.sources:
        policy = _make_source_policy(source, env_cfg, args.seed)
        collected, episode = 0, 0
        while collected < per_source:
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            while seed in used_episode_seeds:
                seed = int(rng.integers(0, np.iinfo(np.int32).max))
            used_episode_seeds.add(seed)
            budget = min(args.states_per_trajectory, per_source - collected)
            traj = collect_trajectory(env, policy, seed, source, budget)
            raw.extend(traj)
            collected += len(traj)
            episode += 1
        logger.info("collected %d states from %s", collected, source)
    env.close()
    collect_time = time.time() - start

    # -- split BY TRAJECTORY so adjacent states never straddle splits -------
    trajectories = sorted({r["trajectory"] for r in raw})
    rng.shuffle(trajectories)
    by_traj: dict[str, list[dict]] = {t: [] for t in trajectories}
    for record in raw:
        by_traj[record["trajectory"]].append(record)

    assignment: dict[str, str] = {}
    quota = {"train": args.train, "val": args.val, "test": args.test}
    filled = {"train": 0, "val": 0, "test": 0}
    for traj in trajectories:
        # fill the split that is furthest from its quota (proportionally)
        deficits = {s: (quota[s] - filled[s]) / max(quota[s], 1) for s in quota}
        target = max(deficits, key=deficits.get)
        assignment[traj] = target
        filled[target] += len(by_traj[traj])
    for record in raw:
        record["split"] = assignment[record["trajectory"]]

    trimmed: list[dict] = []
    counts = {"train": 0, "val": 0, "test": 0}
    for record in raw:
        split = record["split"]
        if counts[split] < quota[split]:
            counts[split] += 1
            trimmed.append(record)
    logger.info("split sizes: %s (%d trajectories)", counts, len(trajectories))
    for split in ("train", "val", "test"):
        mix = {}
        for record in trimmed:
            if record["split"] == split:
                mix[record["source"]] = mix.get(record["source"], 0) + 1
        logger.info("  %-5s source mix: %s", split, mix)

    # -- label ---------------------------------------------------------------
    label_start = time.time()
    labelled = label_records(
        env_cfg, trimmed, args.csit_mode, args.also_genie, args.seed, args.workers,
        logger, args.cem_iterations, args.cem_population, args.cem_restarts,
    )
    for row, record in zip(labelled, trimmed):
        row["split"] = record["split"]
    label_time = time.time() - label_start

    # -- save ----------------------------------------------------------------
    cfg_hash = config_hash(env_cfg)
    for split in ("train", "val", "test"):
        rows = [r for r in labelled if r["split"] == split]
        if not rows:
            continue
        arrays = {
            key: np.stack([np.asarray(r[key]) for r in rows])
            for key in rows[0]
            if key not in ("source", "trajectory", "split")
        }
        arrays["source"] = np.array([r["source"] for r in rows])
        arrays["trajectory"] = np.array([r["trajectory"] for r in rows])
        arrays["config_hash"] = np.array(cfg_hash)
        np.savez_compressed(out_dir / f"{split}.npz", **arrays)
        logger.info("wrote %s.npz with %d states", split, len(rows))

    meta = {
        "config_hash": cfg_hash,
        "config": args.config,
        "sources": args.sources,
        "csit_mode": args.csit_mode,
        "also_genie": args.also_genie,
        "seed": args.seed,
        "sizes": counts,
        "num_trajectories": len(trajectories),
        "cem": {
            "iterations": args.cem_iterations,
            "population": args.cem_population,
            "restarts": args.cem_restarts,
            "evaluations_per_scheme_state": (
                args.cem_iterations * args.cem_population * args.cem_restarts
            ),
        },
        "trajectory_split": assignment,
        "collect_seconds": round(collect_time, 2),
        "label_seconds": round(label_time, 2),
        "total_seconds": round(time.time() - start, 2),
        "required_keys": list(REQUIRED_KEYS),
    }
    if env_cfg.rate_split_mode == "per_user_fraction":
        meta.update({
            "action_schema": DIRECT_ACTION_SCHEMA,
            "hard_sigmoid_limit": HARD_SIGMOID_LIMIT,
            "canonical_action_columns": list(DIRECT_KEYS),
        })
        meta["required_keys"] += list(DIRECT_KEYS)
    with (out_dir / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    dump_config_snapshot(out_dir / "config_snapshot.yaml", env=env_cfg)

    oracle_counts = np.array([int(r["oracle_count"]) for r in labelled])
    logger.info("oracle |A| distribution: %s",
                {int(c): round(float((oracle_counts == c).mean()), 3)
                 for c in range(1, env_cfg.num_users + 1)})
    logger.info("collect %.1fs | label %.1fs | total %.1fs",
                collect_time, label_time, time.time() - start)
    logger.info("dataset written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
