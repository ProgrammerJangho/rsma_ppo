#!/usr/bin/env bash
# Extend target fine-tuning from 524,288 to 2,097,152 cumulative steps in two
# resumed stages, exactly as done for RSMA seed 0:
#   A. 524,288-step checkpoint  -> +524,288 interactions (cumulative 1,048,576)
#   B. 1,048,576-step checkpoint -> +1,048,576 interactions (cumulative 2,097,152)
# Each stage restores actor, critic, optimizer and observation/reward
# normalisation, and restarts learning-rate annealing over its own budget.
#
# Usage: FAMILY=rsma|sdma SEEDS="1 2 3 4" scripts/run_2m_staged_continuation.sh
#
# SDMA must be resumed with allocation_control=sdma_power: load_checkpoint
# restores weights only, and under the default joint control the q-only
# checkpoint loads silently and starts emitting alpha > 0.
set -euo pipefail

family="${FAMILY:?set FAMILY=rsma or FAMILY=sdma}"
read -r -a seeds <<< "${SEEDS:?set SEEDS, e.g. SEEDS=\"1 2 3 4\"}"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
config="configs/thesis_rician_k3_nt2_20deg_sinr_finetune.yaml"
log_dir="runs/continuation_2m_logs"
mkdir -p "$log_dir"

case "$family" in
  rsma)
    base_fmt="runs/matched_objective_rsma_ppo_k3_20deg_from_stage1_seed%s_524k_v1"
    stage_a_fmt="runs/matched_objective_rsma_ppo_k3_20deg_seed%s_continue_524k_v1"
    stage_b_fmt="runs/matched_objective_rsma_ppo_k3_20deg_seed%s_continue_to_2097k_v1"
    extra=()
    ;;
  sdma)
    base_fmt="runs/q_only_ppo_k3_20deg_from_stage1_seed%s_524k_v1"
    stage_a_fmt="runs/q_only_ppo_k3_20deg_seed%s_continue_524k_v1"
    stage_b_fmt="runs/q_only_ppo_k3_20deg_seed%s_continue_to_2097k_v1"
    extra=(allocation_control=sdma_power)
    ;;
  *)
    echo "unknown FAMILY: $family" >&2
    exit 2
    ;;
esac

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# The trainer also writes checkpoint.pt periodically, so completion means the
# checkpoint has reached the stage's final cumulative step.
checkpoint_step() {
  .venv/bin/python -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['global_step'])" "$1" 2>/dev/null
}

run_stage() {
  local seed="$1" resume="$2" out="$3" steps="$4" log_every="$5" ckpt_every="$6"
  local target=$(( $(checkpoint_step "$resume") + steps ))
  if [[ -s "$out/checkpoint.pt" && "$(checkpoint_step "$out/checkpoint.pt")" == "$target" ]]; then
    echo "skip $out (complete at step $target)"
    return 0
  fi
  if [[ -e "$out" ]]; then
    echo "refusing to reuse incomplete $out; remove it first" >&2
    return 1
  fi
  [[ -s "$resume" ]] || { echo "missing $resume" >&2; return 1; }
  .venv/bin/python scripts/train_ppo.py \
    --env-config "$config" \
    --seed "$seed" \
    --run-dir "$out" \
    --total-timesteps "$steps" \
    --resume "$resume" \
    --eval-episodes 0 \
    --set log_interval="$log_every" checkpoint_interval="$ckpt_every" "${extra[@]}"
}

run_seed() {
  local seed="$1"
  local base stage_a stage_b
  base="$(printf "$base_fmt" "$seed")/checkpoint.pt"
  stage_a="$(printf "$stage_a_fmt" "$seed")"
  stage_b="$(printf "$stage_b_fmt" "$seed")"
  run_stage "$seed" "$base" "$stage_a" 524288 16 256
  run_stage "$seed" "$stage_a/checkpoint.pt" "$stage_b" 1048576 32 512
}

pids=()
for seed in "${seeds[@]}"; do
  run_seed "$seed" > "$log_dir/${family}_seed${seed}.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "$family seed ${seeds[$index]} complete"
  else
    echo "$family seed ${seeds[$index]} failed; see $log_dir/${family}_seed${seeds[$index]}.log" >&2
    status=1
  fi
done
exit "$status"
