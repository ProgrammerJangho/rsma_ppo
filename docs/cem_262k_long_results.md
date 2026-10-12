# 262k PPO and online CEM under the frozen long-horizon protocol

## Purpose and protocol

This is the final numerical-reference comparison for the representative thesis
condition: directional Rician MU-MISO, 20-degree user separation, SNR 10 dB,
relative CSIT error 0.2, blocklength 128 and 256 information bits per user.

The five matched-budget 262,144-interaction RSMA-PPO and SDMA-PPO checkpoints
and all CEM policies use the same held-out seeds 930000--930005, 8,192 slots per
trajectory and a 2,048-slot burn-in.  Policies select actions from current AoI
and estimated channel only.  The true channel is used only by the environment.

- Raw RSMA-CEM: one 5-iteration by 192-population solve, 960 candidates/slot.
- SDMA-CEM: one independent common-off solve, 960 candidates/slot.
- Safeguarded CEM: solve both and select using the estimated-channel objective,
  1,920 candidates/slot.

Raw RSMA-CEM and SDMA-CEM are the matched-compute comparison.  The safeguarded
method has twice their candidate budget.  All CEM variants are finite-budget,
one-step numerical references, not global or long-horizon optima.

## Results

| Method | Mean AoI | Std. | Success | Common on | Candidates/slot | CPU latency |
|---|---:|---:|---:|---:|---:|---:|
| RSMA-PPO, 262k | 1.4574 | 0.0143 | 0.5859 | 0.3679 | 0 | 0.568 ms |
| SDMA-PPO, 262k | 1.5610 | 0.0145 | 0.4927 | 0 | 0 | 0.421 ms |
| Raw RSMA-CEM | **1.2249** | 0.0100 | 0.8074 | 1.0000 | 960 | 2.297 ms |
| SDMA-CEM | 1.3437 | 0.0113 | 0.7151 | 0 | 960 | 1.755 ms |
| Safeguarded CEM | **1.2171** | 0.0106 | 0.8123 | 0.4763 | 1,920 | 3.674 ms |

PPO standard deviations use five training-seed means.  CEM standard deviations
use six evaluation trajectories.  They are different uncertainty units and are
not a single common statistical test.

## Paired interpretation

- RSMA-PPO is 6.64% lower in mean AoI than SDMA-PPO and wins all six paired
  evaluation trajectories.
- At the matched 960-candidate budget, raw RSMA-CEM is 8.85% lower than
  SDMA-CEM.  This is the cleanest allocation-only RSMA-versus-SDMA comparison.
- Raw RSMA-CEM is 15.95% lower than RSMA-PPO; safeguarded CEM is 16.49% lower.
  Both signs hold on all six trajectories.  Equivalently, RSMA-PPO is 19.74%
  higher than safeguarded CEM when the CEM result is used as denominator.
- Safeguarding improves raw RSMA-CEM by 0.64% while doubling its candidate
  budget.  It selects the SDMA boundary in 52.37% of post-burn-in slots.

The final evidence therefore supports two separate statements.  First, RSMA's
allocation class has a real advantage at the representative 20-degree point
under equal numerical-search budgets.  Second, the learned PPO policy captures
only part of that available gain and must not be described as optimal or as
matching the numerical reference.

The measured Python/CPU latency is descriptive only.  It illustrates the
feed-forward-versus-online-search implementation used here, but compiler,
batching, implementation language and hardware can change the ratio.  It is
not a central academic claim.

## Reproducibility

- Protocol: `configs/thesis_cem_262k_long_evaluation.yaml`
- Runner: `scripts/run_cem_reference_evaluation.py`
- CEM policy: `src/rsma_aoi/baselines/cem_allocation.py`
- Raw results: `results/thesis_cem_262k_long_v1/`
- Compact table and figure: `results/thesis_cem_262k_long_summary_v1/`

