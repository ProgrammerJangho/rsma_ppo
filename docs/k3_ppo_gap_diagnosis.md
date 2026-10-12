# K=3, Nt=2 allocation-only PPO gap diagnosis

## 1. Question

The original all-3 PPO converged, used the common stream and improved its own
action+objective warm start, but remained worse than online CEM and even the
all-3 SDMA numerical reference. This audit separates four possible causes:

1. wrong common gate decisions;
2. poor continuous power/rate/`rho` allocation;
3. insufficient actor capacity or unsuitable pretraining loss;
4. destructive PPO exploration/fine-tuning.

All objective comparisons use the implemented finite-blocklength PHY. CEM is a
finite-budget safeguarded label, not a global optimum.

## 2. Same-state gap to CEM

The final PPO actors generated 450 states. Each state was relabelled with the
same 2-restart, 5-iteration, 256-population safeguarded CEM used for K=3
pretraining. Every 0-step and 262k actor was then scored on exactly these states.

| Actor stage | CEM expected AoI | Actor expected AoI | Regret | Gate accuracy |
|---|---:|---:|---:|---:|
| action+objective 0-step | 4.2194 | 4.9532 | 0.7338 | 70.7% |
| action+objective 262k | 4.2194 | 5.1127 | 0.8932 | 65.7% |

The values are weighted next-slot AoI sums, so they are not directly the
per-user long-run mean AoI reported elsewhere. PPO moved farther from the
one-step numerical reference even on states from the final PPO distribution.

Choosing between each actor action and its common-off projection using
estimated CSIT reduced final regret only from 0.8932 to 0.8683. Even an
unattainable true-channel selector reduced it only to 0.8603. Thus at most 3.7%
of the final gap can be removed by correcting this binary gate decision while
retaining the actor's continuous allocations.

## 3. Which continuous head causes the gap?

For every state, all eight hybrids formed from actor/oracle power, common-rate
and `rho` components were evaluated. Shapley attribution gives the following
shares of the actor-to-CEM objective gap.

| Stage | Power | Common-rate allocation | `rho` |
|---|---:|---:|---:|
| 0-step | 29.4% | 33.6% | 37.0% |
| 262k PPO | **63.6%** | 14.8% | 21.6% |

The final actor's mean common-power fraction, 0.448, was close to the oracle
mean 0.434. The failure is state tracking: their per-state common-power
correlation was only about 0.27 and the mean absolute error was 0.344. The PPO
gap is therefore not fixed by choosing a better global common-power constant.
It is mainly a state-dependent power-allocation error.

## 4. Pretraining loss and actor capacity

Three seeds of two new objective-only actors were trained on the unchanged
dataset. They minimise the differentiable FBL one-step objective without an
action-coordinate loss against one arbitrary CEM solution.

| Deterministic policy | Long-run mean AoI | Relative to original 0-step |
|---|---:|---:|
| action+objective, 128x128 | 1.6750 | reference |
| objective-only, 128x128 | 1.5348 | 8.37% lower |
| objective-only, 256x256 | **1.5223** | 9.12% lower |

Increasing capacity from 128x128 to 256x256 gives only another 0.81%. Removing
the action-coordinate loss gives the large improvement. CEM allocations are not
unique, and coordinate averaging among physically similar labels can be poor
near the FBL reliability cliff. Outcome-based training avoids that ambiguity.

Under the exact 1,024-slot CEM protocol, objective-only 256 obtains 1.5258. It
beats all-3 SDMA-CEM 1.5611 by 2.26%, but remains worse than scheduled-2
SDMA-CEM 1.4310 and all-3 safeguarded RSMA-CEM 1.3925. This is a useful
feed-forward RSMA result, not evidence that it reaches the numerical reference.

## 5. PPO exploration and fine-tuning

On 150 held-out PPO-visited states, 32 stochastic samples per state were drawn
from each initial policy distribution.

| Pretraining family | Deterministic objective | Stochastic objective | Sampling cost |
|---|---:|---:|---:|
| action+objective | 4.7551 | 5.5889 | +17.6% |
| objective-only | 4.8149 | 5.4933 | +14.1% |

The broad initial sampling distribution crosses the steep FBL transition even
when its deterministic mean is good. A direct 65,536-step PPO test from the
objective-only 128 actors changed long-run AoI from 1.5348 to 1.6743, a 9.09%
degradation, with the same sign for all three training seeds. Extending this run
to 262k is therefore not justified.

## 6. Conclusion and project decision

The K=3 failure was not caused mainly by lack of common-stream use or a binary
gate error. It came from two optimisation choices:

1. action-coordinate supervision against non-unique CEM labels weakened the
   initial allocator;
2. standard PPO exploration and updates destroyed much of the improved
   objective-only mapping in the FBL-cliff regime.

For the undergraduate thesis, retain K=3 as an overloaded extension and report
the objective-only actor as the strongest learned all-3 allocator. Do not claim
that the K=3 PPO beats SDMA or CEM. A low-noise or KL-constrained PPO design can
be future work; it needs a separate exploration study and is not required for
the current thesis.

## 7. Reproducibility

- PPO-visited labelled states: `datasets/rician_k3_nt2_all3_ppo_visited_oracle_v1/`
- Oracle-gap audit: `results/k3_nt2_all3_ppo_oracle_gap_visited_v3/`
- Exploration audit: `results/k3_nt2_all3_exploration_cost_v1/`
- Pretraining comparison: `results/k3_nt2_all3_pretraining_ablation_v1/`
- Matched CEM protocol: `results/k3_nt2_all3_pretraining_ablation_cem_protocol_v1/`
- 65k PPO control: `results/k3_nt2_all3_pretraining_ppo65k_ablation_v1/`
- Consolidated figure: `results/k3_nt2_all3_gap_diagnosis_v1/k3_ppo_gap_diagnosis.png`
