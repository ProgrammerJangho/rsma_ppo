#!/usr/bin/env bash
set -euo pipefail

for seed in 0 1 2; do
  out="runs/action_objective_actor_rician_mixed_geometry_snr_symmetric_seed${seed}_v1"
  mkdir -p "$out"
  (
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
      scripts/pretrain_objective_actor.py \
      --dataset datasets/rician_mixed_geometry_snr_symmetric_csit02_oracle_v1 \
      --env-config configs/thesis_rician_mixed_geometry_snr_symmetric_csit02.yaml \
      --preset action_objective \
      --epochs 80 --batch-size 256 --seed "$seed" --out "$out" \
      --set allocation_control=joint common_gate_enabled=true \
      >"$out/stdout.log" 2>&1
  ) &
done
wait

for seed in 0 1 2; do
  out="runs/action_objective_actor_rician_mixed_geometry_snr_symmetric_seed${seed}_v1"
  echo "seed=$seed"
  tail -5 "$out/stdout.log"
done
