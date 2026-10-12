#!/usr/bin/env python3
"""Compare allocation optimisers and measure how converged the reference is.

The project previously called the CEM result an "oracle".  This script tests
that claim: it runs several optimisers of very different character on the same
states and reports how far apart they land, how the best-so-far value grows with
evaluation budget, and how sensitive each is to its seed.

Nothing here proves global optimality.  The best value found by *any* optimiser
is reported as a **lower reference** on the achievable objective.

Examples:
    python scripts/verify_optimizer.py --config configs/m2_hybrid.yaml --states 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsma_aoi.baselines import build_baseline  # noqa: E402
from rsma_aoi.envs.aoi_env import AoIRSMAEnv  # noqa: E402
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer  # noqa: E402
from rsma_aoi.evaluation.oracle import enumerate_masks  # noqa: E402
from rsma_aoi.optimizers import CEMAllocator, build_allocator, make_problem  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402
from rsma_aoi.utils.logging import get_logger  # noqa: E402

DEFAULT_OPTIMIZERS = ("equal", "grid", "random_shooting", "cem_fast", "cem_full",
                      "cem_heavy", "slsqp")


def collect_states(cfg: EnvConfig, num_states: int, seed: int) -> list[dict]:
    """Roll out max-AoI-first to gather representative states."""
    env = AoIRSMAEnv(cfg)
    policy, _ = build_baseline("max_aoi_first", cfg, seed=seed)
    states, episode = [], 0
    while len(states) < num_states:
        obs, _ = env.reset(seed=seed + episode)
        done = False
        while not done and len(states) < num_states:
            structured = env.get_state()
            states.append({"aoi": structured["aoi"].copy(),
                           "h_hat": structured["h_hat"].copy(),
                           "h_true": env.true_channel})
            action = policy.act(obs[None, :], [structured])[0]
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
        episode += 1
    env.close()
    return states


def compare_optimizers(
    cfg: EnvConfig, states: list[dict], names: tuple[str, ...], seeds: tuple[int, ...],
    sizes: tuple[int, ...] | None, logger,
) -> pd.DataFrame:
    """Run every optimiser on every (state, subset) and record the outcome."""
    phy = RSMAPhysicalLayer(cfg)
    weights = np.asarray(cfg.weight_vector, dtype=np.float64)
    masks = enumerate_masks(cfg.num_users, sizes)
    rows = []
    for index, state in enumerate(states):
        for active in masks:
            problem, base = make_problem(
                phy, state["aoi"], state["h_true"], state["h_hat"], weights,
                float(cfg.aoi_max), active, csit_mode="estimated",
            )
            for name in names:
                for seed in seeds:
                    allocator = build_allocator(name)
                    result = allocator.solve(problem, np.random.default_rng(seed))
                    rows.append({
                        "state": index, "subset": int(np.packbits(active[::-1])[0]),
                        "num_active": int(active.sum()), "optimizer": name, "seed": seed,
                        "gain": result.gain, "expected_aoi": base - result.gain,
                        "evaluations": result.evaluations, "runtime_s": result.runtime_s,
                        "violation": result.violation(active),
                    })
        if (index + 1) % 20 == 0:
            logger.info("  %d/%d states", index + 1, len(states))
    return pd.DataFrame(rows)


def budget_curve(
    cfg: EnvConfig, states: list[dict], sizes: tuple[int, ...] | None, seeds: tuple[int, ...],
) -> pd.DataFrame:
    """Best-so-far objective versus CEM evaluation budget."""
    phy = RSMAPhysicalLayer(cfg)
    weights = np.asarray(cfg.weight_vector, dtype=np.float64)
    masks = enumerate_masks(cfg.num_users, sizes)
    budgets = [(2, 24), (3, 48), (5, 96), (5, 192), (8, 192), (8, 384), (12, 512)]
    rows = []
    for index, state in enumerate(states):
        for active in masks:
            problem, base = make_problem(
                phy, state["aoi"], state["h_true"], state["h_hat"], weights,
                float(cfg.aoi_max), active, csit_mode="estimated",
            )
            for iterations, population in budgets:
                for seed in seeds:
                    allocator = CEMAllocator(iterations=iterations, population=population)
                    result = allocator.solve(problem, np.random.default_rng(seed))
                    rows.append({
                        "state": index, "num_active": int(active.sum()),
                        "evaluations": result.evaluations, "seed": seed,
                        "gain": result.gain, "expected_aoi": base - result.gain,
                    })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/m2_hybrid.yaml")
    parser.add_argument("--states", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--optimizers", nargs="+", default=list(DEFAULT_OPTIMIZERS))
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help="restrict subset sizes (default: all)")
    parser.add_argument("--out-dir", default="results/optimizer_verification")
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args(argv)

    logger = get_logger()
    cfg = EnvConfig.from_yaml(args.config)
    sizes = tuple(args.sizes) if args.sizes else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("collecting %d states from %s", args.states, args.config)
    states = collect_states(cfg, args.states, args.seed)

    logger.info("comparing optimisers: %s", ", ".join(args.optimizers))
    df = compare_optimizers(cfg, states, tuple(args.optimizers), tuple(args.seeds), sizes, logger)
    df.to_csv(out_dir / "optimizer_comparison.csv", index=False)

    # per (state, subset) reference = best value any optimiser found
    reference = df.groupby(["state", "subset"])["gain"].max().rename("reference_gain")
    merged = df.merge(reference, on=["state", "subset"])
    merged["gap"] = merged["reference_gain"] - merged["gain"]

    summary = merged.groupby("optimizer").agg(
        mean_gain=("gain", "mean"),
        mean_gap=("gap", "mean"),
        max_gap=("gap", "max"),
        frac_best=("gap", lambda s: float((s <= 1e-9).mean())),
        seed_spread=("gain", lambda s: float(s.std())),
        evaluations=("evaluations", "mean"),
        runtime_ms=("runtime_s", lambda s: 1000.0 * s.mean()),
        max_violation=("violation", "max"),
    ).sort_values("mean_gap")
    summary.to_csv(out_dir / "optimizer_summary.csv")

    seed_var = (
        merged.groupby(["optimizer", "state", "subset"])["gain"].std()
        .groupby("optimizer").mean().rename("within_problem_seed_std")
    )

    logger.info("computing CEM budget curve")
    curve = budget_curve(cfg, states[: min(len(states), 25)], sizes, tuple(args.seeds))
    curve.to_csv(out_dir / "budget_curve.csv", index=False)
    curve_summary = curve.groupby("evaluations").agg(
        mean_gain=("gain", "mean"), mean_expected_aoi=("expected_aoi", "mean"),
        seed_std=("gain", "std"),
    )
    curve_summary.to_csv(out_dir / "budget_curve_summary.csv")

    pd.set_option("display.width", 200)
    print("\n=== optimizer comparison (reference = best value found by any optimiser) ===")
    print(summary.round(5).to_string())
    print("\n=== seed sensitivity (std of the objective across seeds, same problem) ===")
    print(seed_var.round(6).to_string())
    print("\n=== CEM budget curve (mean over states/subsets/seeds) ===")
    print(curve_summary.round(5).to_string())

    best = summary.index[0]
    print(f"\nLower reference used elsewhere: best-of-all-optimisers "
          f"(closest single optimiser: {best}).")
    print("No optimiser here is proved globally optimal.")
    logger.info("results written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
