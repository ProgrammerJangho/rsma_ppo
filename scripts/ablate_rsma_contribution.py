#!/usr/bin/env python3
"""Separate the RSMA (PHY) gain from the scheduling (MAC) gain.

If a warm-started policy improves the AoI, that improvement could come from
serving more users per slot (a MAC effect that SDMA would also get) or from the
rate splitting itself (a PHY effect).  This script crosses the two axes on a
trained checkpoint, holding the slot duration, power, blocklength and payload
fixed in every arm:

    scheduling in {learned, fixed top-m}  x  allocation in {RSMA, SDMA, equal}

* ``*_sdma``  keeps the policy's schedule but removes the common stream
  (``multiple_access: sdma``): a pure no-rate-splitting ablation.
* ``*_equal`` keeps the schedule but replaces the learned power/rate split with
  an equal one: isolates the learned allocation.

Examples:
    python scripts/ablate_rsma_contribution.py --checkpoint runs/xxx/checkpoint.pt \
        --fixed-checkpoint runs/yyy/checkpoint.pt --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsma_aoi.agents.ppo import PPOPolicy  # noqa: E402
from rsma_aoi.baselines import EqualAllocationPolicy, build_baseline  # noqa: E402
from rsma_aoi.evaluation.evaluator import apply_overrides, evaluate_policy, save_results  # noqa: E402
from rsma_aoi.evaluation.statistics import summarise  # noqa: E402
from rsma_aoi.utils.config import EnvConfig, dump_config_snapshot  # noqa: E402
from rsma_aoi.utils.logging import get_logger  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/m2_hybrid.yaml")
    parser.add_argument("--checkpoint", required=True, help="learned-scheduling policy")
    parser.add_argument("--fixed-checkpoint", default=None,
                        help="fixed-top-m policy (defaults to the heuristic baseline)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out-dir", default="results/rsma_ablation")
    args = parser.parse_args(argv)

    logger = get_logger()
    base_cfg = EnvConfig.from_yaml(args.config)
    sdma_cfg = apply_overrides(base_cfg, {"multiple_access": "sdma"})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    learned, _, _ = PPOPolicy.from_checkpoint(args.checkpoint, deterministic=True)
    if args.fixed_checkpoint:
        fixed, _, _ = PPOPolicy.from_checkpoint(args.fixed_checkpoint, deterministic=True)
    else:
        fixed, _ = build_baseline("max_aoi_first", base_cfg, seed=0)

    arms = [
        ("learned_sched+rsma", learned, base_cfg),
        ("learned_sched+sdma", learned, sdma_cfg),
        ("learned_sched+equal", EqualAllocationPolicy(learned), base_cfg),
        ("fixed_top_m+rsma", fixed, base_cfg),
        ("fixed_top_m+sdma", fixed, sdma_cfg),
        ("fixed_top_m+equal", EqualAllocationPolicy(fixed), base_cfg),
    ]

    frames = []
    for label, policy, cfg in arms:
        for seed in args.seeds:
            df = evaluate_policy(
                policy, cfg, episodes=args.episodes,
                eval_seed=100_000 + 1_000 * seed, algorithm=label,
            )
            df["seed"] = seed
            frames.append(df)
        recent = pd.concat(frames[-len(args.seeds):])
        logger.info("%-22s AoI %.3f | |A| %.2f | common power %.3f",
                    label, recent["weighted_avg_aoi"].mean(),
                    recent["avg_num_scheduled"].mean(),
                    recent["avg_common_power_ratio"].mean())

    raw = pd.concat(frames, ignore_index=True)
    save_results(raw, out_dir, name="episodes")
    summary = summarise(raw, group_keys=["algorithm"], seed=0)
    summary.to_csv(out_dir / "summary.csv", index=False)
    dump_config_snapshot(out_dir / "config_snapshot.yaml", env=base_cfg,
                         experiment={"checkpoint": args.checkpoint,
                                     "fixed_checkpoint": args.fixed_checkpoint,
                                     "seeds": args.seeds, "episodes": args.episodes})

    table = (
        raw.groupby("algorithm")
        .agg(weighted_avg_aoi=("weighted_avg_aoi", "mean"),
             aoi_p95=("aoi_p95", "mean"),
             success_rate=("success_rate", "mean"),
             mean_active=("avg_num_scheduled", "mean"),
             common_power=("avg_common_power_ratio", "mean"),
             common_payload=("avg_common_payload_ratio", "mean"))
        .sort_values("weighted_avg_aoi")
    )
    pd.set_option("display.width", 200)
    print("\n=== RSMA contribution ablation (same power / blocklength / payload) ===")
    print(table.round(4).to_string())

    def get(name: str) -> float:
        return float(table.loc[name, "weighted_avg_aoi"]) if name in table.index else float("nan")

    print("\nDecomposition (lower AoI is better):")
    print(f"  rate-splitting gain, learned schedule : "
          f"{get('learned_sched+sdma') - get('learned_sched+rsma'):+.3f}")
    print(f"  rate-splitting gain, fixed schedule   : "
          f"{get('fixed_top_m+sdma') - get('fixed_top_m+rsma'):+.3f}")
    print(f"  learned-allocation gain (vs equal)    : "
          f"{get('learned_sched+equal') - get('learned_sched+rsma'):+.3f}")
    print(f"  learned-vs-fixed scheduling gain      : "
          f"{get('fixed_top_m+rsma') - get('learned_sched+rsma'):+.3f}")
    logger.info("results written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
