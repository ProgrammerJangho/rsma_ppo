# Imperfect-CSIT posterior-SAA pretraining

> **Correction (fair-compute audit):** The original `v1` SAA implementation
> flattened posterior samples into the minibatch axis.  At fixed epochs this
> gave `S=8` eight times as many optimiser updates as `S=1`.  All `v1` SAA
> sample-count and downstream PPO comparisons below are retained as audit
> history but are superseded.  The valid results are the `v2` fair-compute
> sections at the end of this document.

## Purpose

The actor observes `h_hat`, while packet success is determined by `h_true`.
Using one realised true channel as the training target can overfit the estimation
error.  Robust pretraining instead minimises a sample-average approximation of

\[
E[J(H,a(\hat H))\mid \hat H].
\]

## Posterior used by the implementation

For mean-preserving relative CSIT, let `m` be the Rician mean, `R` the scatter
covariance, `sigma` the configured error standard deviation, and
`a=sqrt(1-sigma^2)`.  The observation model is

\[
\hat H-m=a(H-m)+\sigma E,\qquad H-m,E\sim CN(0,R).
\]

Therefore

\[
H\mid\hat H\sim CN(m+a(\hat H-m),\sigma^2R).
\]

The beamformers are computed once from `h_hat`.  For every observed state the
same actor action is evaluated over `S` independent posterior channels, and all
success-weighted AoI gains are averaged inside that state before the minibatch
mean.  The posterior axis is retained separately so changing `S` cannot change
the number of optimiser updates per epoch.

Only `csit_error_mode=relative` is supported.  Absolute and normalized modes
need their own posterior derivations and are rejected rather than approximated
silently.

## Smoke validation

Configuration: directional Rician, 20-degree separation, `sigma_e=0.2`, eight
posterior samples, 92/32/32 train/validation/test states, four epochs.  The
posterior-averaged validation AoI decreased from 2.8213 to 2.7809.  This run only
checks the implementation and gradient path; its small dataset and short
training make it unsuitable as a reported performance result.

Artifacts:

- Config: `configs/thesis_rician_alignment_20deg_csit02.yaml`
- Smoke dataset: `datasets/rician_20deg_csit02_saa_smoke_v1/`
- Smoke checkpoint: `archive/runs/objective_actor_rician_20deg_csit02_saa8_smoke_v1/`

## Main experiment design

Full datasets compare posterior sample counts `S in {1,4,8,16}`.  Long-run PPO
uses independently pretrained actor seeds and matched PPO seeds, and compares
single-realisation and SAA-8 initialisation against scratch RSMA and SDMA power
control under identical interaction and evaluation budgets.  The completed
first sweep covers `sigma_e in {0.0,0.1,0.2,0.3}`; estimated-CSIT and
posterior-SAA CEM remain separate online-compute baselines rather than PPO
initialisation controls.

## Superseded v1: first full-size controlled comparison

A 2,400/600/600-state dataset was generated at 20-degree separation and
`sigma_e=0.2`.  With the same actor seed and 60 epochs, the held-out
single-realisation PHY audit was:

| Posterior samples | Test expected AoI | Oracle AoI | Gap | Gate accuracy |
|---:|---:|---:|---:|---:|
| 1 (realised-channel control) | 4.1634 | 3.3418 | 0.8216 | 0.6750 |
| 8 (posterior SAA) | **3.4008** | 3.3418 | **0.0590** | 0.6983 |

The 18.3% lower actor-level test AoI is strong evidence that conditioning the
loss on multiple plausible true channels fixes an estimation-error overfitting
problem.  It is still a one-seed, offline actor comparison.  A thesis claim
requires repeated seeds and long-run PPO fine-tuning; the estimated-CSIT CEM
column is a label/reference, not proof that its action is posterior-optimal.

Full dataset: `datasets/rician_20deg_csit02_gate_oracle_v1/`.
Comparison table: `archive/results/rician_20deg_csit02_saa_pretrain_v1/summary.csv`.

## Superseded v1: long-run PPO fine-tuning

Both seed-0 pretrained actors were fine-tuned for 65,536 interactions with PPO
seeds 0, 1, and 2.  Scratch RSMA and learned SDMA power control used the same
training budget.  Every deterministic policy was evaluated on the same three
4,096-slot trajectories after a 1,024-slot burn-in.

| Method | Average AoI | Pooled p95 | Success rate | Common power | Common payload |
|---|---:|---:|---:|---:|---:|
| Scratch RSMA PPO | 1.7023 ± 0.0403 | 3.33 | 0.4702 | 0.074 | 0.023 |
| Single-realisation warm-start | 1.5386 ± 0.0267 | 2.00 | 0.5118 | 0.369 | 0.358 |
| Posterior-SAA-8 warm-start | **1.4708 ± 0.0203** | **2.00** | **0.5956** | 0.340 | 0.342 |
| SDMA power PPO | 1.6954 ± 0.1653 | 2.89 | 0.4920 | 0 | 0 |

Paired SAA-8 AoI differences were -0.0678 ± 0.0416 versus the single-channel
warm-start, -0.2315 ± 0.0234 versus scratch RSMA, and -0.2247 ± 0.1850 versus
SDMA power.  SAA-8 won all three seed pairs in every comparison.  The SDMA
standard deviation is inflated by its seed-0 AoI of 1.8863; seeds 1 and 2 were
about 1.60, but SAA-8 also beat both of those paired runs.

The deterministic SAA gain is not replicated under sampled execution: sampled
AoI was 1.7374 for SAA-8 and 1.6875 for the single-channel warm-start.  The actor
was pretrained at its modal continuous action, so its distribution tails are
not directly constrained by the physical objective.  Deterministic deployment
is the primary policy, but the sampled reversal must be reported and motivates
a later distribution-aware objective or reduced exploration variance ablation.

Consolidated results:
`archive/results/rician_ppo_20deg_csit02_comparison_v1/`.

## Superseded v1: end-to-end pretraining and PPO seed audit

The first long-run comparison reused pretraining seed 0 for all three PPO
seeds.  To expose pretraining variance, SAA actors were independently trained
with actor/posterior seeds 0/700, 1/701, and 2/702, then paired with PPO seeds
0, 1, and 2.  Offline actor AoI was 3.4008, 3.6678, and 3.6644, confirming that
the unusually strong seed-0 checkpoint was not representative of all starts.

End-to-end deterministic PPO AoI was 1.4480, 1.5964, and 1.5422, giving
1.5288 ± 0.0751.  Paired comparisons were:

| Reference | Mean SAA end-to-end difference | SAA wins |
|---|---:|---:|
| Single-realisation warm-start | -0.0097 ± 0.0790 | 1/3 |
| Scratch RSMA PPO | **-0.1734 ± 0.0362** | 3/3 |
| SDMA power PPO | **-0.1666 ± 0.2371** | 3/3 |

Thus posterior SAA remains a useful and stable warm-start relative to scratch
RSMA in this experiment, but **the evidence does not show that SAA is better
than single-realisation objective pretraining** once pretraining seed variation
is included.  The earlier 4.4% SAA advantage was conditional on the favourable
seed-0 pretrained actor.  Any final comparison of the two objectives must use
matched independent pretraining seeds, and more than three seeds if a small
difference is to be resolved.

## Superseded v1: posterior sample-count ablation

`S=1` was explicitly changed to mean one newly sampled posterior channel, rather
than the dataset's stored realised channel.  The same actor seeds 0/1/2 and
posterior seeds 700/701/702 were used at every sample count.

| Posterior samples | Held-out actor AoI | Objective gap | Gate accuracy |
|---:|---:|---:|---:|
| 1 | 4.1316 ± 0.1260 | 0.7898 ± 0.1260 | 0.6850 ± 0.0101 |
| 4 | 3.7543 ± 0.0583 | 0.4125 ± 0.0583 | 0.6822 ± 0.0067 |
| 8 | **3.5777 ± 0.1532** | **0.2359 ± 0.1532** | 0.6878 ± 0.0092 |
| 16 | 3.5860 ± 0.0691 | 0.2442 ± 0.0691 | 0.6689 ± 0.0125 |

Every seed improved from 1 to 4 and from 4 to 8 samples.  Increasing from 8 to
16 improved two seeds slightly but worsened seed 0; the mean changed by only
+0.0083 AoI while computation and physical-batch size doubled.  Eight posterior
samples are therefore the practical default for the main study.  Sixteen is a
stability ablation, not a better-performing default.  Gate-label accuracy is not
monotone and is secondary because the labels come from a plug-in estimated-CSIT
CEM, whereas the SAA actor optimises a different posterior expectation.

Artifacts: `archive/results/rician_20deg_csit02_posterior_sample_sweep_v1/`.

## Valid v2: fair-compute posterior sample-count ablation

Posterior gains retain shape `(state, sample, user, stream)`.  The actor is
evaluated once per state, PHY outcomes are averaged across the sample axis, and
the optimiser receives exactly the same number of minibatches and updates for
every `S`.

| Posterior samples | Held-out actor AoI | Objective gap | Gate accuracy |
|---:|---:|---:|---:|
| 1 | 4.0959 ± 0.1449 | 0.7541 ± 0.1449 | 0.6789 ± 0.0092 |
| 4 | 3.9711 ± 0.0327 | 0.6293 ± 0.0327 | 0.6750 ± 0.0176 |
| 8 | **3.9354 ± 0.0531** | **0.5936 ± 0.0531** | 0.6767 ± 0.0164 |
| 16 | 3.9598 ± 0.0455 | 0.6180 ± 0.0455 | 0.6711 ± 0.0169 |

Paired changes were -0.1248 for 4 versus 1 (3/3 wins), -0.0357 for 8 versus
4 (3/3 wins), and +0.0244 for 16 versus 8 (1/3 wins).  The posterior benefit
exists but is modest after controlling optimisation steps, and it saturates at
eight samples.  `S=8` remains the efficient ablation setting, but no longer has
the large effect suggested by v1.

Artifacts: `archive/results/rician_20deg_csit02_posterior_sample_sweep_v2/`.

## Valid v2: fair end-to-end PPO result

Each fair SAA-8 actor seed was paired with the same PPO seed.  Deterministic
long-run AoI was 1.5598, 1.5598, and 1.5832, or 1.5676 ± 0.0135.

| Method | Average AoI | Paired SAA wins |
|---|---:|---:|
| Scratch RSMA PPO | 1.7023 ± 0.0403 | 3/3 |
| Single-realisation warm-start | **1.5517 ± 0.0082** | 1/3 |
| Fair posterior-SAA-8 warm-start | 1.5676 ± 0.0135 | — |
| SDMA power PPO | 1.6954 ± 0.1653 | 3/3 |

Fair SAA-8 improves over scratch RSMA by -0.1346 ± 0.0380 and over SDMA by
-0.1278 ± 0.1723, but is worse than seed-matched single-realisation warm-start
by +0.0160 ± 0.0200 and wins only one of three pairs.  The defensible conclusion
is that objective-aware
warm-starting helps PPO under imperfect CSIT; posterior SAA has not demonstrated
an additional long-run advantage at `sigma_e=0.2`.

Artifacts: `archive/results/rician_ppo_20deg_csit02_comparison_v2/`.

## Valid v2: matched high-error long-run check

At `sigma_e=0.3`, independently pretrained actor seed `i` was paired with PPO
seed `i` for both objectives.  Scratch RSMA and SDMA power control used the same
65,536-interaction budget and all methods used the same three 4,096-slot
evaluation trajectories after a 1,024-slot burn-in.

| Method | Average AoI | Mean pooled p95 | Success rate |
|---|---:|---:|---:|
| Scratch RSMA PPO | 1.7369 ± 0.0108 | 3.56 | 0.4632 |
| Single-realisation warm-start | **1.5626 ± 0.0032** | **2.00** | 0.4956 |
| Fair posterior-SAA-8 warm-start | 1.5627 ± 0.0104 | **2.00** | 0.4989 |
| SDMA power PPO | 1.7086 ± 0.1642 | 3.09 | 0.4856 |

The paired SAA-minus-single difference is +0.0001 ± 0.0136, with SAA winning
one of three pairs.  SAA and single-realisation pretraining both beat scratch
RSMA in all three pairs by about 0.174 AoI.  This higher-error experiment again
supports objective-aware warm-starting, but provides no evidence that posterior
sampling itself improves the final long-run policy.

Artifacts: `archive/results/rician_ppo_20deg_csit03_matched_comparison_v2/`.

## Valid v2: complete matched CSIT sweep

The same four-method, three-seed experiment is complete for
`sigma_e in {0.0,0.1,0.2,0.3}`.  Values below are deterministic long-run average
AoI, with the standard deviation computed across training seeds.

| `sigma_e` | Scratch RSMA | Single warm-start | SAA-8 warm-start | SDMA power |
|---:|---:|---:|---:|---:|
| 0.0 | 1.7300 ± 0.1096 | 1.5550 ± 0.0169 | **1.5518 ± 0.0192** | 1.6618 ± 0.1474 |
| 0.1 | 1.7659 ± 0.1623 | 1.5457 ± 0.0190 | **1.5433 ± 0.0182** | 1.6114 ± 0.0428 |
| 0.2 | 1.7023 ± 0.0403 | **1.5517 ± 0.0082** | 1.5676 ± 0.0135 | 1.6954 ± 0.1653 |
| 0.3 | 1.7369 ± 0.0108 | **1.5626 ± 0.0032** | 1.5627 ± 0.0104 | 1.7086 ± 0.1642 |

SAA-minus-single paired AoI differences are -0.0032, -0.0023, +0.0160, and
+0.0001 as `sigma_e` increases.  SAA wins 2/3, 3/3, 1/3, and 1/3 pairs,
respectively.  The differences are small relative to three-seed uncertainty and
do not show a systematic SAA advantage.  In particular, the small difference at
zero posterior variance is a numerical/optimisation sanity residual, not a
robustness gain.

Both objective-aware warm starts beat scratch RSMA in all three seed pairs at
every error level, reducing mean AoI by about 0.135 to 0.223 depending on the
condition.  They also beat the learned SDMA-power control in all recorded pairs,
but SDMA has large seed variance in three conditions, so three seeds are not
enough for a broad RSMA-superiority claim.

The action statistics support a mechanism-level interpretation.  Scratch RSMA
uses only about 0.047--0.102 common-power fraction and 0.014--0.026 common-payload
fraction across the sweep.  The warm-started policies use about 0.254--0.380
common power and 0.224--0.376 common payload.  The defensible result is that
objective-aware initialisation keeps PPO out of a near-SDMA common-stream
collapse.  Posterior sampling is not the observed source of the final gain.

Absolute AoI is not monotone in `sigma_e`.  Training randomness, sampled ACKs,
and the induced state distribution all change across conditions.  Claims should
therefore use within-error matched comparisons rather than interpreting the four
means as a controlled derivative with respect to CSIT error.

Artifacts: `results/rician_ppo_20deg_csit_matched_sweep_v2/`.

## Fair SAA-8 actor sweep over CSIT error

Full 2,400/600/600-state datasets were regenerated at each error level with the
same source-policy mix, trajectory split seed, actor seeds, 60 epochs, and eight
posterior samples.

| `sigma_e` | Actor AoI | Plug-in CEM-label AoI | Actor gap |
|---:|---:|---:|---:|
| 0.0 | 4.1783 ± 0.0392 | 3.4161 | 0.7622 ± 0.0392 |
| 0.1 | **3.8860 ± 0.0381** | 3.1332 | 0.7528 ± 0.0381 |
| 0.2 | 3.9354 ± 0.0531 | 3.3418 | 0.5936 ± 0.0531 |
| 0.3 | 4.2835 ± 0.0495 | 3.7050 | 0.5786 ± 0.0495 |

Absolute offline AoI is not monotone in `sigma_e`, and should not be interpreted
as a pure robustness curve.  Although collection uses the same source types and
seeds, ACK outcomes change the subsequent AoI states, so each error level has a
different state distribution.  The plug-in CEM labels also optimise `h_hat` as
if it were true and are not posterior-optimal.  This sweep establishes that the
fair SAA training path remains numerically stable through `sigma_e=0.3`; a
causal CSIT-degradation claim requires long-run policies evaluated within each
error level against matched scratch, single-realisation, and SDMA controls.

Artifacts: `results/rician_20deg_csit_saa8_actor_sweep_v2/`.
