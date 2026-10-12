#!/usr/bin/env bash
set -euo pipefail

# Compare conservative PPO fine-tuning with and without a critic-only warm-up.
for variant in direct criticwarm; do
  for seed in 0 1 2; do
    warm="runs/action_objective_actor_k3_direct_seed${seed}_v1/checkpoint.pt"
    out="runs/ppo_k3_action_warm_conservative_${variant}_seed${seed}_65k_v1"
    if [[ -e "$out/checkpoint.pt" ]]; then
      echo "refusing to overwrite completed run: $out" >&2
      exit 2
    fi
    mkdir -p "$out"
    freeze=0
    if [[ "$variant" == "criticwarm" ]]; then
      freeze=16384
    fi
    (
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
        scripts/train_ppo.py \
        --env-config configs/thesis_rician_k3_nt2_all3_direct.yaml \
        --agent-config configs/thesis_rician_k3_nt2_all3_direct.yaml \
        --seed "$seed" --run-dir "$out" --total-timesteps 65536 \
        --eval-episodes 0 \
        --set "pretrained_actor=$warm" warm_start_log_std=-1.0 \
          entropy_coef=0.0 actor_lr=3e-5 critic_lr=3e-4 \
          "freeze_actor_steps=$freeze" checkpoint_interval=0 eval_interval=0 \
        >"$out/stdout.log" 2>&1
    ) &
  done
done
wait

for variant in direct criticwarm; do
  for seed in 0 1 2; do
    out="runs/ppo_k3_action_warm_conservative_${variant}_seed${seed}_65k_v1"
    echo "variant=$variant seed=$seed"
    tail -3 "$out/stdout.log"
  done
done
