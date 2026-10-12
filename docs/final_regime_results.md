# Final spatial and heterogeneous regime results

> **Headline update (2026-09-10).** The 20° table below is the former 65k,
> unmatched-initialization result. Use `training_budget_results.md` for the
> current learned-policy headline: matched objective pretraining, 262,144 PPO
> interactions, RSMA 1.4574 versus SDMA 1.5610 (-6.64%, 5/5 seeds). The
> spatial and near/far sections remain secondary regime evidence at their
> documented budgets.

## Protocol

Every comparison uses deterministic policies, identical held-out
channel/CSIT/ACK seeds, 8,192 slots per trajectory and a 2,048-slot burn-in.
The headline uses five training seeds. Spatial and near/far stress tests use
three training seeds and six paired evaluation trajectories.

## Headline: 20°, relative CSIT error 0.2

| Method | Mean AoI | Training-seed std | 95% bootstrap interval |
|---|---:|---:|---:|
| Objective-warm joint+gate RSMA-PPO | **1.5511** | 0.0089 | [1.5441, 1.5580] |
| SDMA-power PPO | 1.6448 | 0.1280 | [1.5838, 1.7598] |
| Scratch joint+gate RSMA-PPO | 1.6960 | 0.0448 | [1.6607, 1.7297] |

Objective-warm RSMA-PPO lowers mean AoI by 5.70% relative to SDMA-power PPO.
It wins all five same-index training-seed comparisons and all six paired
evaluation trajectories after averaging policies within each method. Scratch
RSMA-PPO is 3.12% worse than SDMA-power PPO. Thus the supported contribution is
the combination of joint RSMA control and objective-aware initialization, not
the larger RSMA action space alone.

## Spatial alignment: perfect CSIT

| Separation | Objective-warm RSMA-PPO | SDMA-power PPO | RSMA relative gain | Seedwise wins |
|---:|---:|---:|---:|---:|
| 5° | **1.5675** | 1.8157 | 13.67% | 3/3 |
| 20° | **1.4866** | 1.6457 | 9.67% | 3/3 |
| 60° | **1.0189** | 1.0216 | 0.26% | 3/3 |

The 60° numerical difference is too small to claim a practically meaningful
RSMA advantage. The policy reduces mean common power from 0.372 at 5° and 0.287
at 20° to 0.019 at 60°. Mean common payload scale falls to 0.00029 and the
common-on fraction to 0.037. The defensible conclusion is that common-stream
use disappears as the channels become separable and the policy approaches the
SDMA boundary.

## Directional-Rician 5 dB near/far stress test

| Method | Mean AoI | Training-seed std | 95% bootstrap interval |
|---|---:|---:|---:|
| Objective-warm joint+gate RSMA-PPO | **2.3720** | 0.0140 | [2.3626, 2.3881] |
| SDMA-power PPO | 2.9601 | 0.1046 | [2.8394, 3.0226] |

RSMA-PPO is 19.87% lower in this one stress condition and wins all three
same-index training-seed comparisons. The result reverses the old archived
near/far experiment, which used the pre-directional channel, different CSIT
normalization and no objective-aware pretraining. It therefore supports an
optimization/model-generation explanation for the old loss. It does not prove
that RSMA wins for arbitrary path-gain gaps; only the normalized 5 dB gap at
20° and relative CSIT error 0.2 was tested.

## Remaining interpretation limits

- Spatial and near/far results have three training seeds; the five-seed claim is
  restricted to the headline operating point.
- SDMA-power PPO has noticeably larger training-seed variance at 5° and 20°.
  Report seedwise values and paired trajectory differences with the mean.
- The fixed RSMA reference uses one deliberately simple allocation and is weak
  in the FBL regime. It is not an upper bound on RSMA.
- CEM is a one-step numerical reference with a much larger inference budget.
  It should be reported separately from learned-policy latency.
- The completed 5/10/15 dB sensitivity is documented in
  `snr_sensitivity_results.md`. It evaluates separately retrained policies and
  therefore supports operating-load sensitivity, not zero-shot robustness.
