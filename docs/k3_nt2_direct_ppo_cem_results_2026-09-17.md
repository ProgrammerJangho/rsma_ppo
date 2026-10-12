# K=3, Nt=2 direct-action PPO/CEM matched evaluation

## Protocol

- Channel: directional Rician ULA, 20-degree span with all six user-label
  permutations, 15 dB SNR, relative CSIT error 0.2.
- PHY: all three users active, RZF private beams, principal common beam,
  finite blocklength 128, fixed 256-bit updates.
- Action: gate/rho-free `(alpha, q_k, phi_k)`.
- PPO: three independent 262,144-step runs, each initialised from its own
  objective-only posterior-SAA actor.
- Evaluation: six frozen seeds, 2,048 slots each, first 512 discarded.
- CEM: five iterations and 192 candidates, or 960 candidates per solve.
  Safeguarded CEM performs independent RSMA and SDMA solves (1,920 candidates
  per slot) and selects using transmitter-visible `h_hat` only.
- Reported AoI is the mean across users and post-burn-in slots. Learned-method
  uncertainty units are training seeds; numerical-method uncertainty units are
  evaluation trajectories.

The frozen protocol is `configs/thesis_k3_direct_cem_evaluation.yaml`. The
complete result directory is `results/k3_nt2_all3_direct_cem_matched_v4`.

## Headline result

| Method | Mean AoI | Variation | Gain vs SDMA | Paired wins vs SDMA |
|---|---:|---:|---:|---:|
| SDMA-CEM, 960 | 1.54897 | 0.00802 | reference | — |
| Raw RSMA-CEM, 960 | 1.35211 | 0.00663 | **12.71%** | **6/6** |
| Safeguarded RSMA-CEM, 1,920 | **1.34744** | 0.00665 | **13.01%** | **6/6** |
| PPO seed 0 | 1.61853 | 0.02629 | −4.49% | 0/6 |
| PPO seed 1 | 1.53364 | 0.01693 | +0.99% | 4/6 |
| PPO seed 2 | 1.53942 | 0.01445 | +0.62% | 5/6 |
| PPO, training-seed mean | 1.56386 | 0.04743 across seeds | **−0.96%** | 1/6 after seed averaging |

The current evidence supports an RSMA gain for the numerical allocation
reference in this operating point. It does **not** support a claim that PPO
consistently beats SDMA. Two PPO seeds slightly beat SDMA, one loses clearly,
and their training-seed mean is 0.96% worse.

## What PPO added after objective pretraining

| Seed | Warm actor AoI | 262k PPO AoI | PPO change | Improved trajectories |
|---:|---:|---:|---:|---:|
| 0 | 1.57295 | 1.61853 | **2.90% worse** | 0/6 |
| 1 | 1.53389 | 1.53364 | 0.02% better | 3/6 |
| 2 | 4.53526 | 1.53942 | **66.06% better** | 6/6 |

The seed-average improvement over the warm actors is 38.61%, but this number
is dominated by recovery of the failed seed-2 warm actor. PPO is best described
as reducing sensitivity to a bad objective warm start in this experiment. It
did not reliably improve an already good warm actor.

Warm seed 2 starved user 1: its mean user AoI was 10.245 and success rate was
0.074. After PPO those values became 1.566 and 0.486. This is evidence that the
differentiable one-step pretraining objective can produce a poor long-run
policy even when training itself is numerically stable.

## Allocation behaviour

| Method | alpha | Common-on rate | Mean phi | Common BLER | Private BLER |
|---|---:|---:|---:|---:|---:|
| Raw RSMA-CEM | 0.724 | 1.000 | 0.414 | 0.0249 | 0.3184 |
| Safeguarded RSMA-CEM | 0.467 | 0.633 | 0.297 | 0.0220 | 0.3193 |
| SDMA-CEM | 0 | 0 | 0 | 0 | 0.4562 |
| PPO seed 0 | 0.748 | 0.888 | 0.392 | 0.0119 | 0.5105 |
| PPO seed 1 | 0.821 | 0.999 | 0.416 | 0.0073 | 0.4997 |
| PPO seed 2 | 0.591 | 0.624 | 0.347 | 0.0068 | 0.5015 |

Raw CEM uses nonzero common transmission almost continuously and obtains a
large RSMA gain, so the environment is not simply too easy for common-stream
use. Safeguarded CEM selects the exact SDMA boundary in 36.68% of slots and
still has the best AoI. This supports a state-dependent mixture of an interior
RSMA allocation and the SDMA boundary.

PPO learns substantially higher private BLER than CEM. Its low common BLER is
therefore not sufficient for a good allocation: it devotes more power to the
common stream without matching CEM's balance between common and private
decoding. This points to policy optimisation and generalisation as the current
bottleneck rather than the absence of a useful RSMA operating regime.

The per-user `q_k` and `phi_k` values are stored in
`per_user_action_summary.csv`. The recorded mean `q_k` values sum to one for
every method. For direct actions, legacy `mean_rho` is intentionally `NaN`;
`mean_common_fraction` is the physical `phi_k` statistic.

## Compute measurement

The measured Python decision times were about 0.46 ms for PPO, 2.19 ms for raw
CEM, 1.78 ms for SDMA-CEM, and 3.62 ms for safeguarded CEM. These values are
implementation-specific and are not hardware-independent latency claims. The
defensible distinction is that PPO performs one feed-forward evaluation after
offline training, whereas CEM performs hundreds or thousands of online
candidate evaluations.

## Defensible conclusion

At this `K=3, Nt=2` overloaded operating point, the canonical direct RSMA
action contains a useful allocation that finite-budget CEM finds reliably and
that lowers AoI by about 13% relative to matched SDMA. The current PPO captures
only part of that opportunity and is seed-sensitive. Its clearest demonstrated
benefit is recovery from one pathological warm start, not consistent dominance
over SDMA or CEM.

The next experiment should target the CEM-to-PPO gap. Before increasing model
size or training budget, inspect why PPO private BLER remains near 0.50 while
CEM reaches about 0.32, and test whether a CEM action-distillation term or a
short reference-policy regularisation phase improves all three seeds.
