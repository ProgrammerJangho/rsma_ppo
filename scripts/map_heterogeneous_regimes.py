#!/usr/bin/env python3
"""Near/far phase map: is there a regime where scheduling AND rate splitting both pay?

The homogeneous study found the two effects living in disjoint regimes: rate
splitting paid at ``K = Nt`` (where there is nothing to schedule) and scheduling
mattered at ``K > Nt`` (where OMA beat RSMA).  Heterogeneous path loss is the
hypothesis for making them coincide: a near/far cell gives the common stream
something to do *and* makes "who to serve" a real decision.

Three conditions are required of a cell (all measured, none assumed):

    A.  optimized RSMA beats optimized SDMA by > 3 %
    B.  optimized RSMA beats OMA
    C.  per-state reference scheduling beats the best *fixed* heuristic by > 3 %
        (i.e. the scheduling decision actually carries value)

Gain decomposition uses a 2x2 factorial over {heuristic, reference} scheduling
x {equal, optimised} allocation::

    scheduling gain = J(heur, equal) - J(ref, equal)
    allocation gain = J(heur, equal) - J(heur, opt)
    total           = J(heur, equal) - J(ref, opt)
    interaction     = total - scheduling - allocation

(J is expected next-slot weighted AoI, lower is better, so a positive "gain"
means the AoI dropped.)  The residual is reported rather than assumed zero.

METRIC LIMIT: like the homogeneous map this ranks cells by the **myopic
one-step** expected AoI.  The homogeneous study showed that screen is
miscalibrated in both directions versus the long run, so surviving cells are
re-measured with full episodes before any conclusion is drawn.
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
from rsma_aoi.envs.geometry import class_labels, near_far_gains_db  # noqa: E402
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer  # noqa: E402
from rsma_aoi.evaluation.fairness import (  # noqa: E402
    audit_geometry_parity, audit_resource_parity, build_manifest, rsma_gain,
)
from rsma_aoi.evaluation.oracle import urgency_vector  # noqa: E402
from rsma_aoi.evaluation.statistics import bootstrap_ci  # noqa: E402
from rsma_aoi.optimizers import (  # noqa: E402
    AllocationProblem, build_allocator, generate_candidates, normalise_sizes,
)
from rsma_aoi.optimizers.candidates import channel_gain_score, top_m_candidates  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402
from rsma_aoi.utils.logging import get_logger  # noqa: E402

SCENARIOS = ((4, 2), (6, 4), (8, 4))
DISPARITY_DB = (0.0, 5.0, 10.0, 15.0, 20.0)
SNR_DB = (5.0, 10.0, 15.0, 20.0)
CSIT_ERROR = (0.1, 0.3, 0.5)
RATE = (1.0, 2.0, 3.0, 4.0)
CORRELATION = (0.0, 0.5, 0.9)
WEIGHTING = ("equal", "inverse_snr")

#: Fixed scheduling heuristics; the best of these is the "no learning" reference.
HEURISTICS = ("max_aoi", "channel_gain", "aoi_x_gain")


def make_config(point: dict, blocklength: int = 128) -> EnvConfig:
    """Environment config for one heterogeneous operating point."""
    k, nt = int(point["num_users"]), int(point["num_antennas"])
    groups = 3 if k >= 6 else 2
    return EnvConfig(
        num_users=k, num_antennas=nt, snr_db=float(point["snr_db"]),
        csit_error_std=float(point["csit_error"]), csit_error_mode="normalized",
        blocklength=blocklength,
        payload_bits=float(point["rate"]) * blocklength,
        alpha=float(point["correlation"]),
        pathloss_model="explicit_near_far",
        path_gains_db=near_far_gains_db(k, float(point["disparity_db"]), groups),
        user_classes=class_labels(k, groups),
        beta_normalization="mean_one",
        aoi_weight_mode=str(point["weighting"]),
        geometry_seed=int(point.get("geometry_seed", 0)),
        multiple_access="rsma", private_beam="rzf", common_beam="principal",
        link_model="fbl", aoi_max=64, episode_length=200, scheduling_enabled=True,
    )


def sample_points(num_points: int, seed: int) -> list[dict]:
    """Stratified Latin-hypercube sample of the heterogeneous grid."""
    from scipy.stats import qmc

    axes = [SCENARIOS, DISPARITY_DB, SNR_DB, CSIT_ERROR, RATE, CORRELATION, WEIGHTING]
    sampler = qmc.LatinHypercube(d=len(axes), seed=seed)
    points = []
    for row in sampler.random(num_points):
        picks = [axis[min(int(u * len(axis)), len(axis) - 1)] for u, axis in zip(row, axes)]
        scenario, disparity, snr, csit, rate, rho, weighting = picks
        points.append({
            "num_users": scenario[0], "num_antennas": scenario[1],
            "disparity_db": disparity, "snr_db": snr, "csit_error": csit,
            "rate": rate, "correlation": rho, "weighting": weighting,
        })
    seen, unique = set(), []
    for point in points:
        key = tuple(sorted(point.items()))
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return unique


def collect_states(cfg: EnvConfig, num_states: int, seed: int) -> list[dict]:
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


def heuristic_masks(state: dict, cfg: EnvConfig, sizes: tuple[int, ...]) -> dict[str, np.ndarray]:
    """One fixed-heuristic subset per rule, all of the same size ``min(K, Nt)``."""
    weights = np.asarray(cfg.weight_vector)
    aged = np.minimum(state["aoi"] + 1.0, float(cfg.aoi_max))
    urgency = weights * (aged - 1.0)
    gains = channel_gain_score(state["h_hat"])
    size = (min(cfg.num_users, cfg.num_antennas),)
    return {
        "max_aoi": top_m_candidates(urgency, size, cfg.num_users)[0],
        "channel_gain": top_m_candidates(gains, size, cfg.num_users)[0],
        "aoi_x_gain": top_m_candidates(urgency * gains, size, cfg.num_users)[0],
    }


def _best_over(phy, state, urgency, masks, allocator, rng):
    """Best (gain, mask, power, rate, evaluations) over a set of candidate subsets."""
    best = (-np.inf, masks[0], None, None)
    evaluations = 0
    for mask in masks:
        problem = AllocationProblem(
            phy=phy, active=mask, urgency=urgency, h_score=state["h_true"],
            h_design=state["h_hat"], h_opt=state["h_hat"],
        )
        result = allocator.solve(problem, rng)
        evaluations += result.evaluations
        if result.gain > best[0]:
            best = (result.gain, mask, result.power_fractions, result.rate_fractions)
    return (*best, evaluations)


def _sdma_fallback_gain(phy_rsma, state, urgency, mask, power, rate) -> float:
    """Score the SDMA optimum inside the RSMA action space (``f_c = 0``).

    RSMA contains SDMA exactly, so handing the RSMA arm this allocation makes
    ``gain_vs_sdma >= 0`` structural: a negative value could only mean the
    higher-dimensional search got unlucky, which would be measuring the
    optimiser rather than the physics.
    """
    if power is None:
        return -np.inf
    pf = np.asarray(power, dtype=np.float64).copy()
    pf[0] = 0.0
    total = pf.sum()
    if total <= 0:
        return -np.inf
    pf = pf / total
    out = phy_rsma.evaluate_allocations(
        state["h_true"], mask, pf[None, :], np.asarray(rate)[None, :], np.asarray([0.0]),
        beams=phy_rsma.rsma_beams(state["h_hat"], mask, use_common=True),
    )
    return float(out["success_prob"][0] @ urgency)


def evaluate_point(payload: tuple) -> dict:
    """Screen one heterogeneous operating point (worker entry point)."""
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
    equal = build_allocator("equal")
    opt = build_allocator(allocator_name)

    acc: dict[str, list[float]] = {
        k: [] for k in (
            "rsma_ref_opt", "rsma_heur_opt", "rsma_ref_equal", "rsma_heur_equal",
            "sdma_ref_opt", "oma_ref_opt", "rsma_heur_best_equal",
        )
    }
    heuristic_scores: dict[str, list[float]] = {h: [] for h in HEURISTICS}
    ref_sizes: list[int] = []
    evaluations = 0

    for index, state in enumerate(states):
        rng = np.random.default_rng(seed * 100_003 + index)
        urgency, base = urgency_vector(state["aoi"], weights, aoi_max)
        candidates = generate_candidates(
            state["aoi"], weights, state["h_hat"], aoi_max, sizes=sizes,
            max_candidates=None if cfg.num_users <= 4 else 12,
        )
        rules = heuristic_masks(state, cfg, sizes)

        # per-heuristic score with the optimised allocator (to pick the best fixed rule)
        for name, mask in rules.items():
            g, _, _, _, ev = _best_over(phys["rsma"], state, urgency, [mask], opt, rng)
            evaluations += ev
            heuristic_scores[name].append(base - g)

        heur_mask = rules["max_aoi"]
        # --- reference arms for the other schemes ---------------------------
        sdma_gain, sdma_mask, sdma_pf, sdma_rf, ev = _best_over(
            phys["sdma"], state, urgency, candidates, opt, rng
        )
        evaluations += ev
        acc["sdma_ref_opt"].append(base - sdma_gain)
        oma_gain, _, _, _, ev = _best_over(phys["oma"], state, urgency, candidates, opt, rng)
        evaluations += ev
        acc["oma_ref_opt"].append(base - oma_gain)

        # --- 2x2 factorial: {heuristic, reference} x {equal, optimised} -----
        g, _, _, _, ev = _best_over(phys["rsma"], state, urgency, [heur_mask], equal, rng)
        evaluations += ev
        acc["rsma_heur_equal"].append(base - g)
        g, _, _, _, ev = _best_over(phys["rsma"], state, urgency, [heur_mask], opt, rng)
        evaluations += ev
        acc["rsma_heur_opt"].append(base - g)
        g, _, _, _, ev = _best_over(phys["rsma"], state, urgency, candidates, equal, rng)
        evaluations += ev
        acc["rsma_ref_equal"].append(base - g)
        g, m, _, _, ev = _best_over(phys["rsma"], state, urgency, candidates, opt, rng)
        evaluations += ev
        fallback = _sdma_fallback_gain(
            phys["rsma"], state, urgency, sdma_mask, sdma_pf, sdma_rf
        )
        evaluations += 1
        if fallback > g:
            g, m = fallback, sdma_mask
        acc["rsma_ref_opt"].append(base - g)
        ref_sizes.append(int(m.sum()))

        best_fixed = min(
            (base - _best_over(phys["rsma"], state, urgency, [mask], equal, rng)[0])
            for mask in rules.values()
        )
        acc["rsma_heur_best_equal"].append(best_fixed)

    row = dict(point)
    row.update({"num_states": len(states), "allocator": allocator_name, "seed": seed,
                "evaluations": evaluations, "runtime_s": time.perf_counter() - started,
                "load_ratio": cfg.num_users / cfg.num_antennas,
                "blocklength": cfg.blocklength,
                "payload_bits": float(cfg.payload_vector[0]),
                "mean_reference_subset_size": float(np.mean(ref_sizes)),
                "reference_subset_size_std": float(np.std(ref_sizes))})
    for key, values in acc.items():
        arr = np.asarray(values)
        row[f"aoi_{key}"] = float(arr.mean())
        low, high = bootstrap_ci(arr, seed=0, num_resamples=2000)
        row[f"aoi_{key}_ci_low"], row[f"aoi_{key}_ci_high"] = low, high

    best_rule = min(HEURISTICS, key=lambda h: float(np.mean(heuristic_scores[h])))
    row["best_fixed_heuristic"] = best_rule
    row["aoi_best_fixed_heuristic_opt"] = float(np.mean(heuristic_scores[best_rule]))

    # --- conditions ------------------------------------------------------
    row["gain_vs_sdma"] = rsma_gain(row["aoi_sdma_ref_opt"], row["aoi_rsma_ref_opt"])
    row["gain_vs_oma"] = rsma_gain(row["aoi_oma_ref_opt"], row["aoi_rsma_ref_opt"])
    row["scheduling_value"] = rsma_gain(
        row["aoi_best_fixed_heuristic_opt"], row["aoi_rsma_ref_opt"]
    )
    row["cond_A_rsma_beats_sdma"] = bool(row["gain_vs_sdma"] > 0.03)
    row["cond_B_rsma_beats_oma"] = bool(row["gain_vs_oma"] > 0.0)
    row["cond_C_scheduling_matters"] = bool(row["scheduling_value"] > 0.03)
    row["all_conditions"] = bool(
        row["cond_A_rsma_beats_sdma"] and row["cond_B_rsma_beats_oma"]
        and row["cond_C_scheduling_matters"]
    )

    # --- 2x2 factorial decomposition (AoI reductions; positive = better) ---
    ref = row["aoi_rsma_heur_equal"]
    row["scheduling_gain"] = ref - row["aoi_rsma_ref_equal"]
    row["allocation_gain"] = ref - row["aoi_rsma_heur_opt"]
    row["total_gain"] = ref - row["aoi_rsma_ref_opt"]
    row["interaction_residual"] = (
        row["total_gain"] - row["scheduling_gain"] - row["allocation_gain"]
    )
    row["interaction_non_negative"] = bool(row["interaction_residual"] >= -1e-9)

    arms = {s: dataclasses.replace(cfg, multiple_access=s) for s in ("rsma", "sdma", "oma")}
    row["parity_ok"] = bool(
        audit_resource_parity(cfg, num_states=6).passed
        and audit_geometry_parity(arms).passed
    )
    row["disparity_actual_db"] = float(cfg.geometry.disparity_db())
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", choices=["coarse", "refine"], default="coarse")
    parser.add_argument("--points", type=int, default=150)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--allocator", default="cem_fast")
    parser.add_argument("--sizes-cap", type=int, default=4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--geometry-seeds", type=int, nargs="+", default=[0],
                        help="refine: repeat each cell over these geometry seeds")
    parser.add_argument("--coarse-csv", default="results/hetero_map/coarse.csv")
    parser.add_argument("--out-dir", default="results/hetero_map")
    args = parser.parse_args(argv)

    logger = get_logger()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "coarse":
        points = sample_points(args.points, args.seed)
    else:
        coarse = pd.read_csv(args.coarse_csv)
        ranked = coarse.sort_values(
            ["all_conditions", "total_gain"], ascending=[False, False]
        ).head(args.top)
        base_points = [
            {k: r[k] for k in ("num_users", "num_antennas", "disparity_db", "snr_db",
                               "csit_error", "rate", "correlation", "weighting")}
            for _, r in ranked.iterrows()
        ]
        points = [
            {**p, "geometry_seed": g} for p in base_points for g in args.geometry_seeds
        ]
    logger.info("%s stage: %d points x %d states, allocator=%s",
                args.stage, len(points), args.states, args.allocator)

    from concurrent.futures import ProcessPoolExecutor

    jobs = [(p, args.states, args.allocator, args.seed, args.sizes_cap) for p in points]
    rows = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(evaluate_point, jobs)):
            rows.append(row)
            if (index + 1) % 10 == 0 or index == 0:
                logger.info("  %d/%d (K=%d disp=%.0fdB rate=%.0f | A=%s B=%s C=%s)",
                            index + 1, len(jobs), row["num_users"], row["disparity_db"],
                            row["rate"], row["cond_A_rsma_beats_sdma"],
                            row["cond_B_rsma_beats_oma"], row["cond_C_scheduling_matters"])
    df = pd.DataFrame(rows)
    name = "coarse" if args.stage == "coarse" else "refined"
    df.to_csv(out_dir / f"{name}.csv", index=False)

    manifest, _ = build_manifest(
        f"hetero_map_{name}", make_config(points[0]), stage=args.stage,
        points=len(points), states=args.states, allocator=args.allocator,
        seed=args.seed, wall_clock_s=time.perf_counter() - started,
        metric="expected next-slot weighted AoI (myopic one-step screening)",
        conditions={"A": "RSMA beats SDMA by >3%", "B": "RSMA beats OMA",
                    "C": "reference scheduling beats best fixed heuristic by >3%"},
    )
    manifest.save(out_dir / f"{name}_manifest.json")

    pd.set_option("display.width", 250)
    cols = ["num_users", "num_antennas", "disparity_db", "snr_db", "csit_error", "rate",
            "correlation", "weighting", "gain_vs_sdma", "gain_vs_oma", "scheduling_value",
            "scheduling_gain", "allocation_gain", "interaction_residual", "all_conditions"]
    print(f"\n=== {name}: cells ranked by total gain (myopic one-step screening) ===")
    print(df.sort_values("total_gain", ascending=False)[cols].head(20).round(4).to_string(index=False))
    print(f"\ncondition A (RSMA > SDMA by 3%)          : {df['cond_A_rsma_beats_sdma'].mean():.1%}")
    print(f"condition B (RSMA > OMA)                 : {df['cond_B_rsma_beats_oma'].mean():.1%}")
    print(f"condition C (scheduling worth >3%)       : {df['cond_C_scheduling_matters'].mean():.1%}")
    print(f"ALL THREE (the intersection)             : {df['all_conditions'].mean():.1%} "
          f"({int(df['all_conditions'].sum())} cells)")
    negative = float((df["gain_vs_sdma"] < -1e-9).mean())
    print(f"negative gain_vs_sdma (must be 0)        : {negative:.1%}")
    print(f"non-negative interaction                 : {df['interaction_non_negative'].mean():.1%}")
    print(f"parity audit passed                      : {df['parity_ok'].all()}")
    logger.info("results written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
