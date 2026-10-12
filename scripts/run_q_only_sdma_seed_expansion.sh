#!/usr/bin/env bash
# Matched q-only PPO-SDMA: same data, curriculum and budgets as thesis RSMA-PPO.
set -euo pipefail

phase="${1:?usage: $0 pretrain|stage1|target}"
read -r -a seeds <<< "${SEEDS:-0 1 2 3 4}"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
log_dir="runs/q_only_sdma_logs"
mkdir -p "$log_dir"

run_seed() {
  local seed="$1"
  case "$phase" in
    pretrain)
      local out="runs/q_only_objective_actor_k3_mixed_phy_seed${seed}_v1"
      [[ -s "$out/checkpoint.pt" ]] && return 0
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/pretrain_objective_actor.py \
        --dataset datasets/rician_k3_nt2_mixed_phy_sinr_oracle_v2 \
        --env-config configs/thesis_rician_k3_nt2_mixed_phy_sinr.yaml \
        --preset objective_only --epochs 80 --batch-size 256 --lr 1e-3 \
        --seed "$seed" --posterior-samples 8 --posterior-saa \
        --posterior-seed "$((10000 + seed))" --out "$out" \
        --set allocation_control=sdma_power
      ;;
    stage1)
      local warm="runs/q_only_objective_actor_k3_mixed_phy_seed${seed}_v1/checkpoint.pt"
      local out="runs/q_only_ppo_k3_mixed_phy_stage1_seed${seed}_524k_v1"
      [[ -s "$warm" ]]
      [[ -s "$out/checkpoint.pt" ]] && return 0
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/train_ppo.py \
        --env-config configs/thesis_rician_k3_nt2_mixed_phy_sinr_stage1.yaml \
        --seed "$seed" --run-dir "$out" --total-timesteps 524288 \
        --set allocation_control=sdma_power "pretrained_actor=$warm" \
        log_interval=8 checkpoint_interval=0
      ;;
    target)
      local stage1="runs/q_only_ppo_k3_mixed_phy_stage1_seed${seed}_524k_v1/checkpoint.pt"
      local out="runs/q_only_ppo_k3_20deg_from_stage1_seed${seed}_524k_v1"
      [[ -s "$stage1" ]]
      [[ -s "$out/checkpoint.pt" ]] && return 0
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/train_ppo.py \
        --env-config configs/thesis_rician_k3_nt2_20deg_sinr_finetune.yaml \
        --seed "$seed" --run-dir "$out" --total-timesteps 524288 \
        --set allocation_control=sdma_power "pretrained_actor=$stage1" \
        log_interval=8 checkpoint_interval=512
      ;;
    *) echo "unknown phase: $phase" >&2; return 2 ;;
  esac
}

pids=()
for seed in "${seeds[@]}"; do
  run_seed "$seed" > "$log_dir/${phase}_seed${seed}.driver.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "$phase seed ${seeds[$index]} failed; see $log_dir/${phase}_seed${seeds[$index]}.driver.log" >&2
    status=1
  else
    echo "$phase seed ${seeds[$index]} complete"
  fi
done
exit "$status"

