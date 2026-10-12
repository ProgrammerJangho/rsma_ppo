# Final online-CEM reference

## Purpose and protocol

This audit measures the remaining gap between the trained PPO policies and a
bounded-compute one-step numerical optimiser at the representative operating
point: directional Rician, 20° separation, SNR 10 dB, relative CSIT error 0.2,
blocklength 128 and 256 bits/user.

Every policy sees only current AoI and `h_hat`. `h_true` is used solely by the
environment after an action has been selected. The audit uses three common
held-out channel/CSIT/ACK seeds, 2,048 slots and a 512-slot burn-in. The five
headline RSMA-PPO and five SDMA-PPO checkpoints are re-evaluated on those same
trajectories.

Three CEM references are separated:

- Raw RSMA-CEM: one `5 iterations × 192 population` solve, 960 candidates/slot.
- SDMA-CEM: one independently optimised common-off solve, 960 candidates/slot.
- Safeguarded RSMA-CEM: solve both and select on `h_hat`, 1,920 candidates/slot.

The safeguarded method includes the exact `P_c=D_c=0` RSMA-to-SDMA boundary.
It has twice the candidate budget of either raw method and is therefore not
labelled a matched-budget comparison against them.

## Results

| Method | Mean AoI | Std. | Success | Candidates/slot | CPU latency |
|---|---:|---:|---:|---:|---:|
| Objective-warm RSMA-PPO | 1.5264 | 0.0096 | 0.5148 | 0 | 0.623 ms |
| SDMA-power PPO | 1.6283 | 0.1241 | 0.4913 | 0 | 0.467 ms |
| Raw RSMA-CEM | **1.2143** | 0.0327 | 0.8145 | 960 | 2.189 ms |
| SDMA-CEM | 1.3303 | 0.0300 | 0.7205 | 960 | 1.736 ms |
| Safeguarded RSMA-CEM | **1.2032** | 0.0251 | 0.8213 | 1,920 | 3.600 ms |

PPO standard deviations use five training-seed means. CEM standard deviations
use three evaluation trajectories. They describe different uncertainty sources
and must not be treated as a common statistical test.

At equal 960-candidate budgets, raw RSMA-CEM lowers mean AoI by 8.72% relative
to SDMA-CEM. Safeguarding lowers raw RSMA-CEM by a further 0.91% and selects the
SDMA boundary in 52.47% of post-burn-in slots. Common-on and common-off states
are both important at this operating point.

Safeguarded CEM lowers AoI by 21.17% relative to objective-warm RSMA-PPO, while
requiring two numerical solves and about 5.8 times the measured CPU decision
latency. Raw CEM is about 3.5 times slower than RSMA-PPO in this Python/CPU
implementation. These latency ratios are descriptive implementation results;
hardware, batching and compiled deployment can change them.

## Interpretation

The PPO policy does not reach the finite-budget numerical reference. This is a
material limitation and should be stated directly. Its benefit is a cheaper
feed-forward decision after offline training, rather than better AoI than
per-slot CEM.

The online one-step CEM result also refines the earlier surrogate concern. At
this representative point, receding one-step optimisation produces lower
long-run AoI than PPO, so the surrogate is not inherently unusable. The old
heterogeneous screening failure mixed different fallback and evaluation
policies and cannot be attributed to the one-step horizon alone. This audit
still does not prove one-step optimality in other geometries, path-gain gaps or
SNRs.

Raw RSMA-CEM uses the common stream in every slot with mean common power 0.482
and mean `rho=0.465`. The safeguarded policy uses the common stream in 47.53% of
slots, with mean common power 0.281 and `rho=0.328` after including common-off
slots. The result supports an adaptive regime switch rather than a fixed RSMA
split.

CEM remains a myopic, random, finite-budget approximation. It is neither a
global optimiser nor a lower bound on achievable long-horizon AoI.

## Reproducibility

- Protocol: `configs/thesis_cem_reference_evaluation.yaml`
- Policy: `src/rsma_aoi/baselines/cem_allocation.py`
- Runner: `scripts/run_cem_reference_evaluation.py`
- Raw results: `results/thesis_cem_reference_v2/`
- Compact table and figure: `results/thesis_cem_reference_summary_v1/`
