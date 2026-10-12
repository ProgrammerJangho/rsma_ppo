#!/usr/bin/env python3
"""Matched long-run AoI evaluation at the K=3, Nt=2 candidate regime."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsma_aoi.baselines.cem_allocation import CEMAllocationPolicy  # noqa: E402
from rsma_aoi.baselines.cem_scheduling import ScheduledCEMAllocationPolicy  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402
from run_final_evaluation import evaluate_trajectory  # noqa: E402


METHODS = (
    "all3_rsma_raw", "all3_sdma", "all3_safeguarded",
    "scheduled2_rsma_raw", "scheduled2_sdma", "scheduled2_safeguarded",
)


def _policy(cfg: EnvConfig, method: str, iterations: int, population: int):
    mode = {
        "all3_rsma_raw": "rsma_raw",
        "all3_sdma": "sdma",
        "all3_safeguarded": "rsma_safeguarded",
        "scheduled2_rsma_raw": "rsma_raw",
        "scheduled2_sdma": "sdma",
        "scheduled2_safeguarded": "rsma_safeguarded",
    }[method]
    if method.startswith("all3"):
        return CEMAllocationPolicy(
            cfg, mode, iterations=iterations, population=population
        )
    return ScheduledCEMAllocationPolicy(
        cfg, mode, schedule_size=2, iterations=iterations, population=population
    )


def _job(job: dict) -> dict:
    base = EnvConfig.from_yaml(ROOT / job["config"])
    cfg = replace(
        base, user_angles_deg=[-10.0, 0.0, 10.0], user_angle_sets_deg=None,
        snr_db=15.0, snr_db_values=None, episode_length=job["steps"],
        scheduling_enabled=job["method"].startswith("scheduled2"),
    )
    policy = _policy(cfg, job["method"], job["iterations"], job["population"])
    row, _ = evaluate_trajectory(
        cfg, {"burn_in": job["burn_in"], "require_all_users_active": False},
        job["method"], -1,
        job["eval_seed"], policy,
    )
    row.update(
        geometry_span_deg=20, snr_db=15.0, horizon=job["steps"],
        burn_in=job["burn_in"], iterations=job["iterations"],
        population=job["population"],
        candidate_evaluations=getattr(policy, "candidate_evaluations", 0),
        boundary_selection_rate=(
            policy.boundary_selections / max(policy.calls, 1)
        ),
    )
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", default="configs/thesis_rician_k3_nt2_overloaded_csit02.yaml"
    )
    p.add_argument("--out-dir", default="results/k3_nt2_overloaded_longrun_v1")
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[980000, 980001, 980002])
    p.add_argument("--steps", type=int, default=1024)
    p.add_argument("--burn-in", type=int, default=256)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--population", type=int, default=192)
    p.add_argument("--workers", type=int, default=9)
    args = p.parse_args()

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    jobs = [
        dict(
            config=args.config, method=method, eval_seed=seed, steps=args.steps,
            burn_in=args.burn_in, iterations=args.iterations,
            population=args.population,
        )
        for method in METHODS for seed in args.eval_seeds
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        rows = list(pool.map(_job, jobs))
    episodes = pd.DataFrame(rows)
    episodes.to_csv(out / "episodes.csv", index=False)
    summary = episodes.groupby("algorithm", as_index=False).agg(
        mean_aoi=("mean_aoi", "mean"), std_aoi=("mean_aoi", "std"),
        success_rate=("success_rate", "mean"),
        common_on_rate=("common_on_rate", "mean"),
        boundary_selection_rate=("boundary_selection_rate", "mean"),
        candidate_evaluations=("candidate_evaluations", "mean"),
        eval_seeds=("eval_seed", "size"),
    )
    summary.to_csv(out / "summary.csv", index=False)
    wide = episodes.pivot(index="eval_seed", columns="algorithm", values="mean_aoi")
    comparison_rows = []
    for candidate in METHODS:
        if candidate == "scheduled2_sdma":
            continue
        delta = wide[candidate] - wide["scheduled2_sdma"]
        comparison_rows.append({
            "candidate": candidate, "reference": "scheduled2_sdma",
            "mean_delta": float(delta.mean()),
            "candidate_wins": int((delta < 0).sum()),
            "reference_wins": int((delta > 0).sum()),
        })
    pd.DataFrame(comparison_rows).to_csv(out / "paired_comparisons.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "sampled long-run validation of the K=3 Nt=2 one-step candidate regime",
        "config": args.config,
        "geometry_angles_deg": [-10.0, 0.0, 10.0],
        "snr_db": 15.0,
        "eval_seeds": args.eval_seeds,
        "steps": args.steps,
        "burn_in": args.burn_in,
        "cem": {"iterations": args.iterations, "population": args.population},
        "caveat": "finite-budget online CEM; three evaluation seeds",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.sort_values("mean_aoi").to_string(index=False))
    print(f"\nCOMPLETE: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
