#!/usr/bin/env python3
"""Check that every staged 2.10M checkpoint finished and kept its action space.

For each family and training seed the three stages must report cumulative
steps 524,288 / 1,048,576 / 2,097,152, the stage configs must match the target
fine-tuning config except for logging fields, and SDMA checkpoints must carry
``allocation_control=sdma_power``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "rsma": ("runs/matched_objective_rsma_ppo_k3_20deg_from_stage1_seed{s}_524k_v1",
             "runs/matched_objective_rsma_ppo_k3_20deg_seed{s}_continue_524k_v1",
             "runs/matched_objective_rsma_ppo_k3_20deg_seed{s}_continue_to_2097k_v1"),
    "sdma": ("runs/q_only_ppo_k3_20deg_from_stage1_seed{s}_524k_v1",
             "runs/q_only_ppo_k3_20deg_seed{s}_continue_524k_v1",
             "runs/q_only_ppo_k3_20deg_seed{s}_continue_to_2097k_v1"),
}
EXPECTED_STEPS = (524_288, 1_048_576, 2_097_152)
EXPECTED_CONTROL = {"rsma": "joint", "sdma": "sdma_power"}
IGNORED = {"log_interval", "checkpoint_interval", "pretrained_actor", "total_timesteps",
           "recurrent_enabled", "recurrent_hidden_size", "recurrent_seq_len"}


def main() -> int:
    problems = []
    for family, templates in STAGES.items():
        for seed in range(5):
            configs = []
            for template, steps in zip(templates, EXPECTED_STEPS):
                path = ROOT / template.format(s=seed) / "checkpoint.pt"
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                ppo = ckpt["ppo_config"]
                configs.append(ppo)
                status = "ok"
                if int(ckpt["global_step"]) != steps:
                    status = f"STEP {ckpt['global_step']} != {steps}"
                if ppo["allocation_control"] != EXPECTED_CONTROL[family]:
                    status = f"CONTROL {ppo['allocation_control']}"
                if int(ppo["seed"]) != seed:
                    status = f"SEED {ppo['seed']}"
                if status != "ok":
                    problems.append(f"{family} seed {seed} {path.parent.name}: {status}")
                print(f"{family} seed {seed} {path.parent.name}: step {ckpt['global_step']}, "
                      f"control {ppo['allocation_control']}, lr {ppo['actor_lr']}/{ppo['critic_lr']} -> {status}")
            base = {k: v for k, v in configs[0].items() if k not in IGNORED}
            for later in configs[1:]:
                diff = {k: (base.get(k), later.get(k)) for k in base if later.get(k) != base.get(k)}
                if diff:
                    problems.append(f"{family} seed {seed}: config drift {diff}")
    print("\nproblems:", problems or "none")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
