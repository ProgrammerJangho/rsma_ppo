#!/usr/bin/env python3
"""Phase map: where does 1-layer RSMA actually beat SDMA and OMA?

Two-stage search over the operating grid (scenario, SNR, CSIT error,
blocklength, payload, channel correlation).  Stage A screens a Latin-hypercube
sample of the grid cheaply; stage B re-runs the most promising cells with more
states, a larger optimiser budget and bootstrap confidence intervals.

METRIC AND ITS LIMIT
--------------------
Every number here is the **expected next-slot weighted AoI** under the
optimiser's allocation - a *myopic, one-step* screening quantity.  It is not a
long-run AoI and must never be reported as one.  Its purpose is to rank
operating points cheaply so that expensive RL is spent only where rate
splitting has room to help.

Gains use the project-wide sign convention (positive = RSMA better):

    gain_vs_sdma = (AoI_SDMA - AoI_RSMA) / AoI_SDMA
    gain_vs_oma  = (AoI_OMA  - AoI_RSMA) / AoI_OMA

Examples:
    python scripts/map_operating_regimes.py --stage coarse --points 120 --workers 12
    python scripts/map_operating_regimes.py --stage refine --top 12 --workers 12
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsma_aoi.baselines import build_baseline  # noqa: E402
from rsma_aoi.envs.aoi_env import AoIRSMAEnv  # noqa: E402
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer  # noqa: E402
from rsma_aoi.evaluation.fairness import audit_resource_parity, build_manifest, rsma_gain  # noqa: E402
from rsma_aoi.evaluation.oracle import urgency_vector  # noqa: E402
from rsma_aoi.evaluation.statistics import bootstrap_ci  # noqa: E402
from rsma_aoi.optimizers import (  # noqa: E402
    AllocationProblem, build_allocator, generate_candidates, normalise_sizes,
)
from rsma_aoi.utils.config import EnvConfig  # noqa: E402
from rsma_aoi.utils.logging import get_logger  # noqa: E402

SCENARIOS = ((2, 2), (4, 4), (4, 2), (6, 4), (8, 4))
SNR_DB = (0.0, 5.0, 10.0, 15.0, 20.0)
CSIT_ERROR = (0.0, 0.1, 0.3, 0.5, 0.7)
BLOCKLENGTH = (64, 128, 256, 512)
PAYLOAD_BITS = (32.0, 64.0, 128.0, 256.0)
CORRELATION = (0.0, 0.5, 0.9)

#: RSMA contains SDMA as the special case ``f_c = 0``, so the RSMA arm is also
#: given the best SDMA allocation as a fallback candidate.  Without this a
#: negative measured gain would only mean the higher-dimensional CEM search got
#: unlucky, not that rate splitting hurts; with it, ``gain_opt_vs_sdma >= 0`` by
#: construction and measures exactly what the common stream *adds*.
#: (arm, multiple_access, allocator preset, use best candidate subset)
ARMS = (
    ("oma_heuristic", "oma", "equal", False),
    ("oma_opt", "oma", None, True),
    ("sdma_equal", "sdma", "equal", False),
    ("sdma_opt", "sdma", None, True),
    ("rsma_equal", "rsma", "equal_rsma", False),
    ("rsma_heuristic", "rsma", "grid", False),
    ("rsma_opt", "rsma", None, True),
)


#: Fixed reference points always included in the screening, even though their
#: payloads are deliberately off-grid: they are the configurations the earlier
#: experiments used, so the map always says where those sit.
ANCHOR_POINTS = (
    {"num_users": 4, "num_antennas": 2, "snr_db": 10.0, "csit_error": 0.3,
     "blocklength": 128, "payload_bits": 192.0, "correlation": 0.8},
    {"num_users": 2, "num_antennas": 2, "snr_db": 10.0, "csit_error": 0.5,
     "blocklength": 128, "payload_bits": 256.0, "correlation": 0.8},
)


def make_config(point: dict) -> EnvConfig:
    """Environment config for one operating point (M2-style defaults)."""
    k, nt = point["num_users"], point["num_antennas"]
    return EnvConfig(
        num_users=k, num_antennas=nt, snr_db=point["snr_db"],
        csit_error_std=point["csit_error"], blocklength=point["blocklength"],
        payload_bits=point["payload_bits"], alpha=point["correlation"],
        multiple_access="rsma", private_beam="rzf", common_beam="principal",
        link_model="fbl", aoi_max=64, episode_length=200,
        scheduling_enabled=True,
    )


def sample_points(num_points: int, seed: int) -> list[dict]:
    """Latin-hypercube sample of the operating grid, plus fixed anchors."""
    from scipy.stats import qmc

    axes = [SCENARIOS, SNR_DB, CSIT_ERROR, BLOCKLENGTH, PAYLOAD_BITS, CORRELATION]
    sampler = qmc.LatinHypercube(d=len(axes), seed=seed)
    raw = sampler.random(num_points)
    points = []
    for row in raw:
        picks = [axis[min(int(u * len(axis)), len(axis) - 1)] for u, axis in zip(row, axes)]
        scenario, snr, csit, n, payload, rho = picks
        points.append({
            "num_users": scenario[0], "num_antennas": scenario[1], "snr_db": snr,
            "csit_error": csit, "blocklength": n, "payload_bits": payload,
            "correlation": rho,
        })
    seen, unique = set(), []
    for point in [dict(a) for a in ANCHOR_POINTS] + points:
        key = tuple(sorted(point.items()))
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return unique


def collect_states(cfg: EnvConfig, num_states: int, seed: int) -> list[dict]:
    """States from a max-AoI-first rollout (a realistic AoI distribution)."""
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


def evaluate_point(payload: tuple) -> dict:
    """Screen one operating point across every arm (worker entry point)."""
    point, num_states, allocator_name, seed, sizes_cap = payload
    started = time.perf_counter()
    cfg = make_config(point)
    weights = np.asarray(cfg.weight_vector, dtype=np.float64)
    aoi_max = float(cfg.aoi_max)
    sizes = normalise_sizes(range(1, min(cfg.num_users, sizes_cap) + 1), cfg.num_users)

    states = collect_states(cfg, num_states, seed)
    phys = {
        scheme: RSMAPhysicalLayer(dataclasses.replace(cfg, multiple_access=scheme))
        for scheme in ("oma", "sdma", "rsma")
    }
    per_arm: dict[str, list[float]] = {arm: [] for arm, *_ in ARMS}
    per_arm_active: dict[str, list[int]] = {arm: [] for arm, *_ in ARMS}
    evaluations = 0

    for index, state in enumerate(states):
        rng = np.random.default_rng(seed * 100_003 + index)
        urgency, base = urgency_vector(state["aoi"], weights, aoi_max)
        candidates = generate_candidates(
            state["aoi"], weights, state["h_hat"], aoi_max, sizes=sizes,
            max_candidates=None if cfg.num_users <= 4 else 10,
        )
        heuristic_mask = candidates[0]
        for mask in candidates:  # max-AoI top-m of the largest allowed size
            if int(mask.sum()) == min(cfg.num_users, cfg.num_antennas):
                heuristic_mask = mask
                break

        sdma_best: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        for arm, scheme, preset, search in ARMS:
            phy = phys[scheme]
            allocator = build_allocator(preset if preset else allocator_name)
            best_gain, best_mask, best_alloc = -np.inf, heuristic_mask, None
            masks = candidates if search else [heuristic_mask]
            for mask in masks:
                problem = AllocationProblem(
                    phy=phy, active=mask, urgency=urgency,
                    h_score=state["h_true"], h_design=state["h_hat"],
                    h_opt=state["h_hat"],
                )
                result = allocator.solve(problem, rng)
                evaluations += result.evaluations
                if result.gain > best_gain:
                    best_gain = result.gain
                    best_mask = mask
                    best_alloc = (mask, result.power_fractions, result.rate_fractions)

            if arm == "sdma_opt":
                sdma_best = best_alloc
            elif arm == "rsma_opt" and sdma_best is not None:
                # RSMA with f_c = 0 reproduces SDMA exactly, so hand it the SDMA
                # optimum as a fallback: the gain then isolates the common stream.
                mask, pf, rf = sdma_best
                fallback_pf = pf.copy()
                fallback_pf[0] = 0.0
                total = fallback_pf.sum()
                if total > 0:
                    fallback_pf = fallback_pf / total
                out = phys["rsma"].evaluate_allocations(
                    state["h_true"], mask, fallback_pf[None, :], rf[None, :],
                    np.asarray([0.0]),
                    beams=phys["rsma"].rsma_beams(state["h_hat"], mask, use_common=True),
                )
                fallback_gain = float(out["success_prob"][0] @ urgency)
                evaluations += 1
                if fallback_gain > best_gain:
                    best_gain, best_mask = fallback_gain, mask

            per_arm[arm].append(base - best_gain)
            per_arm_active[arm].append(int(best_mask.sum()))

    row = dict(point)
    row["num_states"] = len(states)
    row["allocator"] = allocator_name
    row["seed"] = seed
    row["evaluations"] = evaluations
    row["runtime_s"] = time.perf_counter() - started
    row["load_ratio"] = cfg.num_users / cfg.num_antennas
    row["rate_bits_per_use"] = float(cfg.payload_vector[0]) / cfg.blocklength
    for arm in per_arm:
        values = np.asarray(per_arm[arm], dtype=np.float64)
        row[f"aoi_{arm}"] = float(values.mean())
        row[f"active_{arm}"] = float(np.mean(per_arm_active[arm]))
        low, high = bootstrap_ci(values, seed=0, num_resamples=2000)
        row[f"aoi_{arm}_ci_low"], row[f"aoi_{arm}_ci_high"] = low, high

    row["gain_opt_vs_sdma"] = rsma_gain(row["aoi_sdma_opt"], row["aoi_rsma_opt"])
    row["gain_opt_vs_oma"] = rsma_gain(row["aoi_oma_opt"], row["aoi_rsma_opt"])
    row["gain_equal_vs_sdma"] = rsma_gain(row["aoi_sdma_equal"], row["aoi_rsma_equal"])
    row["gain_heur_vs_sdma"] = rsma_gain(row["aoi_sdma_equal"], row["aoi_rsma_heuristic"])
    # Only meaningful when there is a gain to recover; a ratio against a
    # near-zero denominator is noise, not a recovery fraction.
    row["heuristic_gain_recovery"] = (
        row["gain_heur_vs_sdma"] / row["gain_opt_vs_sdma"]
        if row["gain_opt_vs_sdma"] > 0.005 else float("nan")
    )
    parity = audit_resource_parity(cfg, num_states=8)
    row["parity_ok"] = parity.passed
    return row


def run_stage(points: list[dict], args, logger) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(p, args.states, args.allocator, args.seed, args.sizes_cap) for p in points]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(evaluate_point, jobs)):
            rows.append(row)
            if (index + 1) % 10 == 0 or index == 0:
                logger.info(
                    "  %d/%d points (last: K=%d Nt=%d snr=%.0f csit=%.1f n=%d d=%.0f "
                    "gain_vs_sdma=%+.3f)",
                    index + 1, len(jobs), row["num_users"], row["num_antennas"],
                    row["snr_db"], row["csit_error"], row["blocklength"],
                    row["payload_bits"], row["gain_opt_vs_sdma"],
                )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", choices=["coarse", "refine", "explicit"], default="coarse")
    parser.add_argument("--points-json", default=None,
                        help="explicit stage: JSON list of operating points, or a JSON "
                             "object of axis -> values whose product is taken")
    parser.add_argument("--points", type=int, default=120, help="coarse: LHS sample size")
    parser.add_argument("--top", type=int, default=15, help="refine: cells to re-run")
    parser.add_argument("--rank-by", nargs="+", default=["gain_opt_vs_sdma"],
                        help="coarse columns used to pick refinement cells (union of top-N each)")
    parser.add_argument("--states", type=int, default=60)
    parser.add_argument("--allocator", default="cem_fast")
    parser.add_argument("--sizes-cap", type=int, default=4,
                        help="largest subset size considered")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--coarse-csv", default="results/phase_map/coarse.csv")
    parser.add_argument("--out-dir", default="results/phase_map")
    args = parser.parse_args(argv)

    logger = get_logger()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "explicit":
        import itertools
        import json

        spec = json.loads(Path(args.points_json).read_text())
        if isinstance(spec, dict):
            keys = list(spec)
            points = [dict(zip(keys, combo)) for combo in itertools.product(*spec.values())]
        else:
            points = list(spec)
        logger.info("explicit grid: %d operating points, %d states each, allocator=%s",
                    len(points), args.states, args.allocator)
    elif args.stage == "coarse":
        points = sample_points(args.points, args.seed)
        logger.info("coarse screening: %d operating points, %d states each, allocator=%s",
                    len(points), args.states, args.allocator)
    else:
        coarse = pd.read_csv(args.coarse_csv)
        frames = [
            coarse.sort_values(column, ascending=False).head(args.top)
            for column in args.rank_by
        ]
        ranked = pd.concat(frames).drop_duplicates(
            subset=["num_users", "num_antennas", "snr_db", "csit_error",
                    "blocklength", "payload_bits", "correlation"]
        )
        points = [
            {k: (int(r[k]) if k in ("num_users", "num_antennas", "blocklength") else float(r[k]))
             for k in ("num_users", "num_antennas", "snr_db", "csit_error",
                       "blocklength", "payload_bits", "correlation")}
            for _, r in ranked.iterrows()
        ]
        logger.info("refinement: %d cells (ranked by %s), %d states each, allocator=%s",
                    len(points), ", ".join(args.rank_by), args.states, args.allocator)

    started = time.perf_counter()
    df = run_stage(points, args, logger)
    elapsed = time.perf_counter() - started

    name = {"coarse": "coarse", "refine": "refined", "explicit": "focused"}[args.stage]
    df.to_csv(out_dir / f"{name}.csv", index=False)
    manifest, _ = build_manifest(
        f"phase_map_{name}", make_config(points[0]),
        stage=args.stage, points=len(points), states=args.states,
        allocator=args.allocator, seed=args.seed,
        total_phy_evaluations=int(df["evaluations"].sum()),
        wall_clock_s=elapsed,
        metric="expected next-slot weighted AoI (myopic, one-step screening)",
    )
    manifest.save(out_dir / f"{name}_manifest.json")

    pd.set_option("display.width", 240)
    cols = ["num_users", "num_antennas", "snr_db", "csit_error", "blocklength",
            "payload_bits", "correlation", "aoi_rsma_opt", "aoi_sdma_opt", "aoi_oma_opt",
            "gain_opt_vs_sdma", "gain_opt_vs_oma", "heuristic_gain_recovery"]
    top = df.sort_values("gain_opt_vs_sdma", ascending=False)
    print(f"\n=== {name} phase map: top 15 by gain_vs_sdma "
          f"(myopic expected next-slot AoI) ===")
    print(top[cols].head(15).round(4).to_string(index=False))
    negative = float((df["gain_opt_vs_sdma"] < -1e-9).mean())
    print(f"\npositive gain_vs_sdma: {(df['gain_opt_vs_sdma'] > 0).mean():.1%} of cells")
    print(f"negative gain_vs_sdma: {negative:.1%} of cells "
          f"(should be 0 - RSMA contains SDMA; any residual is optimiser noise)")
    print(f"gain_vs_sdma > 5%:     {(df['gain_opt_vs_sdma'] > 0.05).mean():.1%} of cells")
    print(f"parity audit passed:   {df['parity_ok'].all()}")
    print(f"total PHY evaluations: {int(df['evaluations'].sum()):,}   wall clock: {elapsed:.0f}s")
    logger.info("results written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
