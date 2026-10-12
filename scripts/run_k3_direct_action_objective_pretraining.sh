#!/usr/bin/env bash
set -euo pipefail

# Hybrid warm start: retain the differentiable AoI objective while also
# teaching the actor the state-dependent alpha/q/phi structure in existing
# offline allocation labels.
for seed in 0 1 2; do
  out="runs/action_objective_actor_k3_direct_seed${seed}_v1"
  if [[ -e "$out/checkpoint.pt" ]]; then
    echo "refusing to overwrite completed run: $out" >&2
    exit 2
  fi
  mkdir -p "$out"
  (
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
      scripts/pretrain_objective_actor.py \
      --dataset datasets/rician_k3_nt2_all3_direct_csit02_oracle_v1 \
      --env-config configs/thesis_rician_k3_nt2_all3_direct.yaml \
      --agent-config configs/thesis_rician_k3_nt2_all3_direct.yaml \
      --preset action_objective \
      --epochs 60 --batch-size 256 --lr 1e-3 \
      --posterior-saa --posterior-samples 8 --posterior-seed 20260917 \
      --seed "$seed" --out "$out" \
      >"$out/stdout.log" 2>&1
  ) &
done
wait

for seed in 0 1 2; do
  out="runs/action_objective_actor_k3_direct_seed${seed}_v1"
  echo "seed=$seed"
  tail -12 "$out/stdout.log"
done
