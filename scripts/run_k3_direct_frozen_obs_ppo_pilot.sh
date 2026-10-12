#!/usr/bin/env bash
set -euo pipefail

# Isolate the effect of preserving the observation scale used by the
# action+objective warm actor.  All other PPO settings remain canonical.
for seed in 0 1 2; do
  warm="runs/action_objective_actor_k3_direct_seed${seed}_v1/checkpoint.pt"
  out="runs/ppo_k3_action_warm_frozenobs_seed${seed}_65k_v1"
  if [[ -e "$out/checkpoint.pt" ]]; then
    echo "refusing to overwrite completed run: $out" >&2
    exit 2
  fi
  mkdir -p "$out"
  (
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
      scripts/train_ppo.py \
      --env-config configs/thesis_rician_k3_nt2_all3_direct.yaml \
      --agent-config configs/thesis_rician_k3_nt2_all3_direct.yaml \
      --seed "$seed" --run-dir "$out" --total-timesteps 65536 \
      --eval-episodes 0 \
      --set "pretrained_actor=$warm" freeze_observation_stats=true \
        checkpoint_interval=0 eval_interval=0 \
      >"$out/stdout.log" 2>&1
  ) &
done
wait

for seed in 0 1 2; do
  out="runs/ppo_k3_action_warm_frozenobs_seed${seed}_65k_v1"
  echo "seed=$seed"
  tail -3 "$out/stdout.log"
done
