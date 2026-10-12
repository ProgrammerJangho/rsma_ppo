#!/usr/bin/env bash
set -euo pipefail

for variant in critic1e3 critic1e3_entropy0; do
  for seed in 0 1 2; do
    warm="runs/action_objective_actor_k3_direct_seed${seed}_v1/checkpoint.pt"
    out="runs/ppo_k3_${variant}_frozenobs_seed${seed}_262k_v1"
    if [[ -e "$out/checkpoint.pt" ]]; then
      echo "refusing to overwrite completed run: $out" >&2
      exit 2
    fi
    mkdir -p "$out"
    entropy=0.005
    if [[ "$variant" == "critic1e3_entropy0" ]]; then
      entropy=0.0
    fi
    (
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
        scripts/train_ppo.py \
        --env-config configs/thesis_rician_k3_nt2_all3_direct.yaml \
        --agent-config configs/thesis_rician_k3_nt2_all3_direct.yaml \
        --seed "$seed" --run-dir "$out" --total-timesteps 262144 \
        --eval-episodes 0 \
        --set "pretrained_actor=$warm" freeze_observation_stats=true \
          critic_lr=1e-3 "entropy_coef=$entropy" \
          checkpoint_interval=0 eval_interval=0 \
        >"$out/stdout.log" 2>&1
    ) &
  done
done
wait

for variant in critic1e3 critic1e3_entropy0; do
  for seed in 0 1 2; do
    out="runs/ppo_k3_${variant}_frozenobs_seed${seed}_262k_v1"
    echo "variant=$variant seed=$seed"
    tail -3 "$out/stdout.log"
  done
done
