#!/usr/bin/env bash
set -euo pipefail

# Paired 65,536-step PPO pilot.  The two reward modes have the same expected
# objective and MDP transitions; only immediate reward sampling noise differs.
for reward_mode in sampled_aoi expected_aoi; do
  for seed in 0 1 2; do
    warm="runs/objective_actor_k3_nt2_all3_direct_seed${seed}_v1/checkpoint.pt"
    out="runs/ppo_k3_direct_${reward_mode}_seed${seed}_65k_v1"
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
        --eval-episodes 0 --set-env "reward_mode=$reward_mode" \
        --set "pretrained_actor=$warm" checkpoint_interval=0 eval_interval=0 \
        >"$out/stdout.log" 2>&1
    ) &
  done
done
wait

for reward_mode in sampled_aoi expected_aoi; do
  for seed in 0 1 2; do
    out="runs/ppo_k3_direct_${reward_mode}_seed${seed}_65k_v1"
    echo "mode=$reward_mode seed=$seed"
    tail -3 "$out/stdout.log"
  done
done
