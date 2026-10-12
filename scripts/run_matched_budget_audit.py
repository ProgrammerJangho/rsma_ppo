#!/usr/bin/env python3
"""Train matched objective-warm RSMA and SDMA policies at one PPO budget.

Each training seed uses the objective-pretrained actor with the same seed.  The
script exists to keep the RSMA/SDMA initialization comparison reproducible and
to avoid accidentally reusing a single pretrained actor across all PPO seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsma_aoi.utils.config import load_run_configs  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def _train_one(job: dict) -> dict:
    threads = int(job["threads"])
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)

    import torch

    torch.set_num_threads(threads)
    from rsma_aoi.agents.ppo import PPOTrainer

    env_cfg, ppo_cfg = load_run_configs(job["config"], None)
    seed = int(job["seed"])
    method = str(job["method"])
    ppo_cfg.seed = seed
    ppo_cfg.total_timesteps = int(job["steps"])
    ppo_cfg.pretrained_actor = str(job["pretrained_actor"])
    ppo_cfg.reset_critic = True
    ppo_cfg.checkpoint_interval = 0
    ppo_cfg.eval_interval = 0
    if method == "rsma_joint_gate":
        ppo_cfg.allocation_control = "joint"
        ppo_cfg.common_gate_enabled = True
    elif method == "sdma_power":
        ppo_cfg.allocation_control = "sdma_power"
        ppo_cfg.common_gate_enabled = False
    else:  # pragma: no cover - jobs are constructed below
        raise ValueError(f"unknown method: {method}")
    ppo_cfg.validate()
    env_cfg.validate()

    run_dir = Path(job["run_dir"])
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=run_dir, seed=seed)
    final = trainer.train()
    checkpoint = trainer.save_checkpoint(run_dir / "checkpoint.pt")
    trainer.close()
    return {
        "method": method,
        "seed": seed,
        "steps": int(job["steps"]),
        "pretrained_actor": str(job["pretrained_actor"]),
        "checkpoint": str(checkpoint),
        "final_metrics": {key: float(value) for key, value in final.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/thesis_rician_alignment_20deg_csit02.yaml"
    )
    parser.add_argument("--steps", type=int, default=262_144)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--out-dir", default="results/rician_ppo_20deg_csit02_matched_budget_262k_v1"
    )
    args = parser.parse_args(argv)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config = str(ROOT / args.config)
    templates = {
        "rsma_joint_gate": ROOT / "runs/objective_actor_rician_20deg_csit02_single_seed{seed}_v2/checkpoint.pt",
        "sdma_power": ROOT / "runs/objective_actor_sdma_20deg_csit02_seed{seed}_v1/checkpoint.pt",
    }
    jobs: list[dict] = []
    for method, template in templates.items():
        for seed in args.seeds:
            pretrained = Path(str(template).format(seed=seed))
            if not pretrained.exists():
                raise FileNotFoundError(pretrained)
            jobs.append(
                {
                    "config": config,
                    "method": method,
                    "seed": seed,
                    "steps": args.steps,
                    "threads": args.threads,
                    "pretrained_actor": str(pretrained),
                    "run_dir": str(out_dir / "runs" / f"{method}_seed{seed}"),
                }
            )

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_train_one, jobs))

    manifest_path = out_dir / "manifest.json"
    previous_runs: list[dict] = []
    previous_seeds: list[int] = []
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config") != args.config or int(previous.get("steps", -1)) != args.steps:
            raise ValueError("existing manifest uses a different config or training budget")
        previous_runs = list(previous.get("runs", []))
        previous_seeds = [int(seed) for seed in previous.get("seeds", [])]
    merged = {
        (str(item["method"]), int(item["seed"])): item
        for item in [*previous_runs, *results]
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "matched objective-warm PPO budget audit",
        "config": args.config,
        "steps": args.steps,
        "seeds": sorted(set(previous_seeds) | set(args.seeds)),
        "workers": args.workers,
        "threads_per_worker": args.threads,
        "runs": [merged[key] for key in sorted(merged)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for result in results:
        print(
            f"{result['method']} seed={result['seed']} "
            f"checkpoint={result['checkpoint']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
