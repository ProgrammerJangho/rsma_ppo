#!/usr/bin/env python3
"""Myopic scheduling oracle: is a learned schedule size actually suboptimal?

For every collected slot state the script enumerates **all** non-empty user
subsets, approximately optimises the continuous allocation for each subset with
cross-entropy method (CEM) search in the *same latent space the policy uses*
(masked softmax over power / rate logits, sigmoid for ``rho_r``), and scores
each candidate by the **expected** next weighted AoI

    E[Delta_k(t+1)] = p_k * 1 + (1 - p_k) * min(Delta_k(t) + 1, Delta_max)
    objective       = sum_k w_k E[Delta_k(t+1)]        (lower is better)

using success probabilities ``p_k`` produced by the environment's own PHY
(`RSMAPhysicalLayer.evaluate_allocations`, which is proven equivalent to
`transmit` in the test suite).  Expectations are used rather than sampled ACKs so
that subsets are compared without Bernoulli noise.

IMPORTANT - this is a **myopic (one-step) diagnostic, not a long-run oracle.**
It answers "which schedule minimises the *next* slot's weighted AoI", which is
not the same as the optimal stationary policy: it ignores that serving a user now
changes which users are urgent later.  A policy can legitimately differ from this
oracle.  Its value is as a *lower bound check*: if even the one-step oracle
prefers |A| = 1, then a learned |A| = 1 is environment-rational rather than a
collapsed local optimum.

CSIT modes
----------
``genie``     : the allocation is optimised against the true channel (an upper
                bound no real policy can reach).
``estimated`` : the allocation is optimised against ``h_hat`` and then *scored*
                on the true channel - the best a policy with the agent's own
                information could do in one step.  This is the fair comparison.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsma_aoi.baselines import build_baseline  # noqa: E402
from rsma_aoi.envs.aoi_env import AoIRSMAEnv  # noqa: E402
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer  # noqa: E402
from rsma_aoi.evaluation.oracle import optimise_subset  # noqa: E402
from rsma_aoi.utils.config import EnvConfig, dump_config_snapshot  # noqa: E402
from rsma_aoi.utils.logging import get_logger  # noqa: E402


# ---------------------------------------------------------------------------
def collect_states(
    env_cfg: EnvConfig, policy, num_states: int, seed: int, label: str
) -> list[dict]:
    """Roll out ``policy`` and record (AoI, h_true, h_hat, chosen schedule)."""
    env = AoIRSMAEnv(env_cfg)
    states: list[dict] = []
    episode = 0
    while len(states) < num_states:
        obs, _ = env.reset(seed=seed + episode)
        if hasattr(policy, "reset"):
            policy.reset(seed=seed + episode)
        done = False
        while not done and len(states) < num_states:
            structured = env.get_state()
            action = policy.act(obs[None, :], [structured])[0]
            states.append(
                {
                    "aoi": structured["aoi"].copy(),
                    "h_hat": structured["h_hat"].copy(),
                    "h_true": env.true_channel,
                    "policy_schedule": np.asarray(action["schedule"], dtype=bool).copy(),
                    "source": label,
                }
            )
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
        episode += 1
    env.close()
    return states


def _analyse_chunk(payload: tuple) -> list[dict]:
    """Worker entry point: analyse one chunk of states (see :func:`analyse`)."""
    env_cfg, states, csit_mode, seed, offset = payload
    return _analyse_states(env_cfg, states, csit_mode, seed, offset, logger=None)


def analyse(
    env_cfg: EnvConfig, states: list[dict], csit_mode: str, seed: int, logger,
    workers: int = 1,
) -> pd.DataFrame:
    """Run the oracle over every state and subset; one row per state.

    ``workers > 1`` splits the states across processes; each chunk gets its own
    generator seeded from ``seed + offset`` so the result is reproducible and
    independent of the worker count only up to CEM sampling noise.
    """
    if workers <= 1:
        return pd.DataFrame(_analyse_states(env_cfg, states, csit_mode, seed, 0, logger))

    import multiprocessing as mp

    chunks = np.array_split(np.arange(len(states)), workers)
    payloads = [
        (env_cfg, [states[i] for i in chunk], csit_mode, seed, int(chunk[0]))
        for chunk in chunks if len(chunk)
    ]
    logger.info("  dispatching %d chunks to %d workers", len(payloads), workers)
    with mp.get_context("spawn").Pool(len(payloads)) as pool:
        results = pool.map(_analyse_chunk, payloads)
    rows = [row for chunk_rows in results for row in chunk_rows]
    return pd.DataFrame(rows).sort_values("state").reset_index(drop=True)


def _analyse_states(
    env_cfg: EnvConfig, states: list[dict], csit_mode: str, seed: int,
    offset: int, logger,
) -> list[dict]:
    """Core oracle loop over a list of states (returns raw rows)."""
    phy = RSMAPhysicalLayer(env_cfg)
    k = env_cfg.num_users
    weights = np.asarray(env_cfg.weight_vector, dtype=np.float64)
    aoi_max = float(env_cfg.aoi_max)
    subsets = [
        np.array(mask, dtype=bool)
        for mask in itertools.product([False, True], repeat=k)
        if any(mask)
    ]
    rng = np.random.default_rng(seed + offset)
    rows = []
    start = time.time()

    for index, state in enumerate(states):
        aoi = state["aoi"]
        aged = np.minimum(aoi + 1.0, aoi_max)
        urgency = weights * (aged - 1.0)          # objective = const - urgency . p
        base = float(weights @ aged)              # weighted AoI if nothing succeeds
        h_true, h_hat = state["h_true"], state["h_hat"]
        h_opt = h_true if csit_mode == "genie" else h_hat

        per_subset = []
        for active in subsets:
            gain, pf, rf, success, rho = optimise_subset(
                phy, h_true, h_hat, h_opt, active, urgency, rng
            )
            per_subset.append(
                {
                    "active": active,
                    "expected_aoi": base - gain,
                    "success": success,
                    "common_frac": float(pf[0]),
                    "rho_r": rho,
                }
            )

        best = min(per_subset, key=lambda d: d["expected_aoi"])
        policy_mask = state["policy_schedule"]
        policy_entry = next(
            d for d in per_subset if np.array_equal(d["active"], policy_mask)
        )
        rows.append(
            {
                "state": offset + index,
                "source": state["source"],
                "mean_aoi": float(aoi.mean()),
                "max_aoi": float(aoi.max()),
                "oracle_count": int(best["active"].sum()),
                "oracle_expected_aoi": best["expected_aoi"],
                "oracle_mean_success": float(best["success"][best["active"]].mean()),
                "oracle_common_frac": best["common_frac"],
                "oracle_rho_r": best["rho_r"],
                "policy_count": int(policy_mask.sum()),
                "policy_best_expected_aoi": policy_entry["expected_aoi"],
                "subset_match": bool(np.array_equal(best["active"], policy_mask)),
                "count_match": int(best["active"].sum()) == int(policy_mask.sum()),
                "myopic_gap": policy_entry["expected_aoi"] - best["expected_aoi"],
                **{
                    f"best_aoi_count{c}": min(
                        d["expected_aoi"] for d in per_subset if d["active"].sum() == c
                    )
                    for c in range(1, k + 1)
                },
            }
        )
        if logger is not None and (index + 1) % 100 == 0:
            rate = (index + 1) / (time.time() - start)
            logger.info("  %d/%d states (%.1f states/s)", index + 1, len(states), rate)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/m2_hybrid.yaml")
    parser.add_argument("--checkpoint", default=None, help="PPO checkpoint to diagnose")
    parser.add_argument("--states", type=int, default=1000,
                        help="states per source policy")
    parser.add_argument("--sources", nargs="+", default=["ppo", "max_aoi_first"],
                        help="policies whose state distribution is analysed")
    parser.add_argument("--csit-mode", choices=["genie", "estimated"], default="genie")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=1, help="parallel processes")
    parser.add_argument("--out-dir", default="results/oracle")
    args = parser.parse_args(argv)

    logger = get_logger()
    env_cfg = EnvConfig.from_yaml(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    states: list[dict] = []
    for source in args.sources:
        if source == "ppo":
            if not args.checkpoint:
                raise SystemExit("--checkpoint is required to analyse the 'ppo' source")
            from rsma_aoi.agents.ppo import PPOPolicy

            policy, _, _ = PPOPolicy.from_checkpoint(args.checkpoint, deterministic=True)
        else:
            policy, overrides = build_baseline(source, env_cfg, seed=args.seed)
            if overrides:
                raise SystemExit(f"source {source!r} changes the PHY; use an RSMA policy")
        logger.info("collecting %d states from %s", args.states, source)
        states.extend(collect_states(env_cfg, policy, args.states, args.seed, source))

    logger.info("running the myopic oracle over %d states (csit=%s)",
                len(states), args.csit_mode)
    df = analyse(env_cfg, states, args.csit_mode, args.seed, logger, workers=args.workers)
    df.to_csv(out_dir / f"oracle_states_{args.csit_mode}.csv", index=False)
    dump_config_snapshot(
        out_dir / "config_snapshot.yaml", env=env_cfg,
        experiment={"states": len(states), "csit_mode": args.csit_mode,
                    "sources": args.sources, "seed": args.seed,
                    "checkpoint": args.checkpoint},
    )

    print()
    print("=" * 78)
    print(f"MYOPIC SCHEDULING ORACLE  (one-step diagnostic, CSIT={args.csit_mode})")
    print("=" * 78)
    k = env_cfg.num_users
    for source, sub in df.groupby("source"):
        print(f"\n--- source policy: {source}  (n = {len(sub)} states) ---")
        oracle_hist = sub["oracle_count"].value_counts(normalize=True).sort_index()
        policy_hist = sub["policy_count"].value_counts(normalize=True).sort_index()
        print("  active-count distribution (fraction of states):")
        print("    size :  " + "  ".join(f"{c:>6d}" for c in range(1, k + 1)))
        print("    oracle: " + "  ".join(f"{oracle_hist.get(c, 0.0):>6.3f}" for c in range(1, k + 1)))
        print("    policy: " + "  ".join(f"{policy_hist.get(c, 0.0):>6.3f}" for c in range(1, k + 1)))
        print(f"  oracle mean |A|            : {sub['oracle_count'].mean():.3f}")
        print(f"  policy mean |A|            : {sub['policy_count'].mean():.3f}")
        print(f"  exact subset agreement     : {sub['subset_match'].mean():.3f}")
        print(f"  active-count agreement     : {sub['count_match'].mean():.3f}")
        print(f"  oracle expected next AoI   : {sub['oracle_expected_aoi'].mean():.4f}")
        print(f"  policy-subset expected AoI : {sub['policy_best_expected_aoi'].mean():.4f}")
        print(f"  myopic gap (policy-oracle) : {sub['myopic_gap'].mean():.4f}")
        print(f"  oracle mean success prob   : {sub['oracle_mean_success'].mean():.3f}")
        print(f"  oracle common power frac   : {sub['oracle_common_frac'].mean():.3f}")
        print(f"  oracle rho_r               : {sub['oracle_rho_r'].mean():.3f}")
        print("  best expected next AoI when |A| is forced to:")
        for c in range(1, k + 1):
            print(f"    |A| = {c}: {sub[f'best_aoi_count{c}'].mean():.4f}")
    print()
    print("NOTE: this is a MYOPIC (one-step) oracle. It is not the optimal")
    print("long-run policy and a good policy may legitimately deviate from it.")
    logger.info("results written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
