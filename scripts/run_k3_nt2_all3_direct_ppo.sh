#!/usr/bin/env bash
set -euo pipefail

# Three paired 262,144-step PPO runs, each initialised from the objective actor
# trained with the same seed.  CPU thread limits prevent oversubscription when
# the independent seeds run concurrently.
for seed in 0 1 2; do
  warm="runs/objective_actor_k3_nt2_all3_direct_seed${seed}_v1/checkpoint.pt"
  out="runs/ppo_k3_nt2_all3_direct_seed${seed}_262k_v1"
  if [[ ! -f "$warm" ]]; then
    echo "missing warm-start checkpoint: $warm" >&2
    exit 2
  fi
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
      --seed "$seed" --run-dir "$out" --total-timesteps 262144 \
      --eval-episodes 0 \
      --set "pretrained_actor=$warm" checkpoint_interval=0 eval_interval=0 \
      >"$out/stdout.log" 2>&1
  ) &
done
wait

for seed in 0 1 2; do
  out="runs/ppo_k3_nt2_all3_direct_seed${seed}_262k_v1"
  echo "seed=$seed"
  tail -12 "$out/stdout.log"
done
