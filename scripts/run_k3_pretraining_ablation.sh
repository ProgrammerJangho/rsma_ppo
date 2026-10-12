#!/usr/bin/env bash
set -euo pipefail

for variant in obj128 obj256; do
  for seed in 0 1 2; do
    out="runs/k3_nt2_all3_${variant}_seed${seed}_v1"
    mkdir -p "$out"
    if [[ "$variant" == "obj128" ]]; then
      hidden='hidden_sizes=[128,128]'
      epochs=100
    else
      hidden='hidden_sizes=[256,256]'
      epochs=120
    fi
    (
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      .venv/bin/python scripts/pretrain_objective_actor.py \
        --dataset datasets/rician_k3_nt2_all3_csit02_oracle_v1 \
        --env-config configs/thesis_rician_k3_nt2_all3_ppo_csit02.yaml \
        --preset objective_only --epochs "$epochs" --batch-size 256 \
        --seed "$seed" --out "$out" \
        --set allocation_control=joint common_gate_enabled=true "$hidden" \
        >"$out/stdout.log" 2>&1
    ) &
  done
done
wait

for variant in obj128 obj256; do
  for seed in 0 1 2; do
    echo "== $variant seed$seed =="
    tail -4 "runs/k3_nt2_all3_${variant}_seed${seed}_v1/stdout.log"
  done
done
