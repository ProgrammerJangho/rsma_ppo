# Matched PPO training-budget audit

## Question

Does PPO provide a material improvement beyond the differentiable one-step FBL
objective actor, and does the answer differ between the RSMA joint+gate and
SDMA-power action classes?

## Protocol

- Operating point: directional Rician MU-MISO, `K=Nt=2`, 20° separation,
  relative CSIT error 0.2, SNR 10 dB, blocklength 128, payload 256 bits.
- Policies: matched objective-pretraining seed followed by 0, 65,536, or
  262,144 PPO interactions.
- Replication: five independent pretraining/PPO seeds per learned method.
- Evaluation: deterministic actor, six identical held-out channel/CSIT/ACK
  trajectories, 8,192 slots with the first 2,048 removed.
- Uncertainty unit: the mean over held-out trajectories for each training seed.

The 65k and 262k policies are independently trained endpoints.  Both use the
configured linear learning-rate annealing over their full horizon.  Therefore
this is a compute-budget comparison in which the annealing horizon also changes,
not a checkpoint-by-checkpoint continuation of one training run.

## Results

| PPO budget | RSMA mean AoI | SDMA mean AoI | RSMA relative difference | RSMA wins |
|---:|---:|---:|---:|---:|
| 0 | 1.5508 ± 0.0076 | 1.5824 ± 0.0015 | -2.00% | 5/5 |
| 65,536 | 1.5475 ± 0.0061 | 1.5751 ± 0.0130 | -1.75% | 5/5 |
| 262,144 | 1.4574 ± 0.0143 | 1.5610 ± 0.0145 | -6.64% | 5/5 |

The `±` value is the sample standard deviation across five training seeds.

Within each action class:

| Comparison | AoI change | Relative change | Better seeds |
|---|---:|---:|---:|
| RSMA 65k vs 0-step | -0.0033 | -0.21% | 4/5 |
| RSMA 262k vs 65k | -0.0901 | -5.83% | 5/5 |
| RSMA 262k vs 0-step | -0.0934 | -6.02% | 5/5 |
| SDMA 65k vs 0-step | -0.0073 | -0.46% | 3/5 |
| SDMA 262k vs 65k | -0.0141 | -0.90% | 5/5 |
| SDMA 262k vs 0-step | -0.0214 | -1.35% | 5/5 |

For RSMA, 262k PPO interactions also raised the mean predicted packet-success
probability from 0.5010 to 0.5858 and reduced p99 AoI from 3.93 to 3.77.  The
common stream became more selective: common-on rate decreased from 0.436 to
0.368, while common power and common payload scale when represented in the
aggregate action increased from 0.273/0.244 to 0.327/0.344.  This is consistent
with a policy that avoids marginal common transmissions and uses a stronger
common allocation in the states where it enables the stream.

## Interpretation

The earlier 65k-only audit understated the role of PPO.  At that budget PPO was
only a small correction to objective pretraining.  At 262k, RSMA gains 6.02%
over its own pretrained actor and beats the equally trained SDMA action class by
6.64%, with the sign repeated across all five matched seeds.

This supports the claim that PPO can improve the state-dependent common/private
power and common-rate mapping after objective pretraining.  It does not by
itself prove that PPO learned non-myopic planning: the advantage may also come
from more thoroughly optimizing the recurrent state distribution and the
deterministic action mapping.  A causal claim about long-horizon planning would
need a matched multi-step model-based baseline or an intervention on AoI state.

The 262k result should replace the 65k actor in the final learned-policy table.
The zero-step and 65k rows remain necessary because they show that objective
pretraining supplies a strong starting policy and that the PPO benefit is
budget-sensitive.

A subsequent two-mode audit narrows what the 262k actor learned. Its gain needs
the learned gate and continuous user-specific private-power proportions, while
state-dependent common-power magnitude, payload scale, and common-rate
proportions can be replaced by calibrated conditional constants. See
`two_mode_policy_audit.md`; the broad claim that all continuous common
allocations are necessary is no longer supported.

## Artifacts

- Protocol: `configs/thesis_training_budget_evaluation.yaml`
- Training launcher and manifest:
  `scripts/run_matched_budget_audit.py`,
  `results/rician_ppo_20deg_csit02_matched_budget_262k_v1/manifest.json`
- Evaluation rows and summaries: `results/thesis_training_budget_v2/`
- Exact paired contrasts: `results/thesis_training_budget_v2/budget_contrasts.csv`
- Figure: `results/thesis_training_budget_v2/training_budget_aoi.{png,pdf}`
