#!/usr/bin/env bash
set -euo pipefail

pids=()
for seed in 0 1 2; do
  for mode in rsma sdma; do
    out="runs/objective_actor_rician_mixed_geometry_snr_csit02_${mode}_seed${seed}_v1"
    mkdir -p "$out"
    if [[ "$mode" == "rsma" ]]; then
      opts=(allocation_control=joint common_gate_enabled=true)
    else
      opts=(allocation_control=sdma_power common_gate_enabled=false)
    fi
    (
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
        scripts/pretrain_objective_actor.py \
        --dataset datasets/rician_mixed_geometry_snr_csit02_oracle_v1 \
        --env-config configs/thesis_rician_mixed_geometry_snr_csit02.yaml \
        --epochs 60 --batch-size 256 --seed "$seed" --out "$out" \
        --set "${opts[@]}" >"$out/stdout.log" 2>&1
    ) &
    pids+=("$!")
  done
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

for seed in 0 1 2; do
  for mode in rsma sdma; do
    out="runs/objective_actor_rician_mixed_geometry_snr_csit02_${mode}_seed${seed}_v1"
    echo "$mode seed=$seed"
    tail -4 "$out/stdout.log"
  done
done
exit "$status"
