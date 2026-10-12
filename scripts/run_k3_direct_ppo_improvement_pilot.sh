#!/usr/bin/env bash
set -euo pipefail

# PPO-only, one-factor-at-a-time pilot from the same action+objective warm actors.
# Observation statistics stay frozen in every arm.
for variant in critic1e3 entropy0 rollout512; do
  for seed in 0 1 2; do
    warm="runs/action_objective_actor_k3_direct_seed${seed}_v1/checkpoint.pt"
    out="runs/ppo_k3_improve_${variant}_seed${seed}_65k_v1"
    if [[ -e "$out/checkpoint.pt" ]]; then
      echo "refusing to overwrite completed run: $out" >&2
      exit 2
    fi
    mkdir -p "$out"
    extra=()
    case "$variant" in
      critic1e3) extra=(critic_lr=1e-3) ;;
      entropy0) extra=(entropy_coef=0.0) ;;
      rollout512) extra=(rollout_steps=512 minibatch_size=256) ;;
    esac
    (
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
        scripts/train_ppo.py \
        --env-config configs/thesis_rician_k3_nt2_all3_direct.yaml \
        --agent-config configs/thesis_rician_k3_nt2_all3_direct.yaml \
        --seed "$seed" --run-dir "$out" --total-timesteps 65536 \
        --eval-episodes 0 \
        --set "pretrained_actor=$warm" freeze_observation_stats=true \
          checkpoint_interval=0 eval_interval=0 "${extra[@]}" \
        >"$out/stdout.log" 2>&1
    ) &
  done
done
wait

for variant in critic1e3 entropy0 rollout512; do
  for seed in 0 1 2; do
    out="runs/ppo_k3_improve_${variant}_seed${seed}_65k_v1"
    echo "variant=$variant seed=$seed"
    tail -3 "$out/stdout.log"
  done
done
