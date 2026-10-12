# Script guide

## Current thesis workflow

- `validate_channel_model.py`: verify Rician/Gauss–Markov channel statistics.
- `audit_rician_phy_regimes.py`: controlled RSMA/SDMA PHY audit.
- `audit_common_beam_allocations.py`: common-beam comparison.
- `generate_oracle_dataset.py`: create objective-pretraining states and labels.
- `pretrain_objective_actor.py`: pretrain the PPO-compatible allocation actor.
- `run_k3_nt2_all3_direct_pretraining.sh`: reproduce the three canonical
  direct-action posterior-SAA warm starts.
- `run_k3_nt2_all3_direct_ppo.sh`: train the three canonical 262k PPO runs.
- `summarize_k3_direct_matched.py`: build the paired PPO/CEM/SDMA and action
  summaries from the frozen K=3 direct-action evaluation.
- `train_ppo.py`: train one PPO run.
- `run_allocation_ablation.py`: controlled power/split/gate ablation.
- `audit_allocation.py`: trace actions through the actual PHY and AoI update.
- `audit_ppo_convergence.py`: immutable checkpoint and held-out learning curve.
- `plot_ppo_convergence.py`: convergence statistics and thesis figure.
- `summarize_csit_longrun_matched.py`: matched CSIT sweep summary.

## Baselines and diagnostics

`run_baselines.py`, `train_td3.py`, `evaluate.py`,
`evaluate_rician_allocator_longrun.py`, `verify_optimizer.py`, and the remaining
`diagnose_*`, `probe_*`, `summarize_*` scripts are retained as supporting tools.
Scripts whose default configuration is `m1`, `m2`, `op` or `het` belong to the
legacy experiment family unless a current thesis configuration is passed
explicitly.

Every formal run should use a new output directory.  Do not infer validity from
a directory name; inspect its manifest and configuration snapshot.
