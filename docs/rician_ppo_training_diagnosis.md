# Controlled-Rician channel study and PPO diagnosis

## Scope and evidence labels

This note records the first controlled channel study after replacing the former
generic fading setup with a mean-power-preserving Rician ULA model.  It is a
diagnostic study, not the final thesis experiment.

- **[Code]** means a fact established by the implementation or saved outputs.
- **[Experiment]** means a result from the controlled runs listed below.
- **[Inference]** means an interpretation that is consistent with those facts
  but is not itself a theorem.

All three geometries use two single-antenna users, two BS antennas, Rician
`K=8 dB`, `SNR=10 dB`, `n=128`, `D=256 bits/user`, perfect CSIT, homogeneous
large-scale gain, Gauss-Markov coefficient `alpha=0.8`, RZF private beams, and
the principal common beam.  Only the users' LoS angle separation changes:
5°, 20°, or 60°.  Thus the experiment isolates spatial separability without
silently changing average channel power.

## What was run

1. A paired, non-learning PHY audit over 1,200 sampled states per geometry.
   CEM optimises common/private power and payload split for the exact FBL
   one-step AoI objective.  The exact common-off SDMA boundary is always an
   explicit candidate.
2. Long-run online CEM allocation for RSMA and SDMA: three paired 2,048-slot
   trajectories per geometry.  CEM is myopic and approximate, not a global or
   dynamic-programming optimum.
3. PPO with three independent training seeds, 65,536 steps each, followed by
   three 4,096-slot evaluation episodes (1,024-slot burn-in) per checkpoint.
   Deterministic evaluation is primary; sampled evaluation is diagnostic.
4. Three PPO initialisations:
   - `low_rho`: common payload mode 0.05, common power mode 0.25;
   - `broad_rho`: common payload mode 0.50 with a broad Beta distribution,
     common power mode 0.25;
   - `joint_init`: common payload mode 0.35 and common power mode 0.50 with a
     concentrated Beta distribution.
5. An SDMA-power PPO whose action only controls the private-power simplex.

The CEM standard deviations below are across evaluation trajectories.  PPO
standard deviations are across training seeds.  They quantify different
sources of variation and must not be presented as interchangeable confidence
intervals.

## Main results

### Long-run mean AoI

| Angle | CEM RSMA | CEM SDMA | PPO low-ρ | PPO broad-ρ | PPO joint-init | PPO SDMA power |
|---:|---:|---:|---:|---:|---:|---:|
| 5°  | 1.3200 | 1.4863 | 1.7092 ± 0.0432 | 2.3313 ± 0.5679 | 1.7841 ± 0.1155 | 1.8140 ± 0.3066 |
| 20° | 1.1771 | 1.2778 | 1.7300 ± 0.1096 | 2.2741 ± 0.2174 | 1.9748 ± 0.0135 | 1.6618 ± 0.1474 |
| 60° | 1.0294 | 1.0074 | 1.0740 ± 0.0103 | 1.3072 ± 0.1668 | 1.2117 ± 0.0443 | 1.0258 ± 0.0053 |

Negative paired AoI differences favour the candidate over its SDMA comparator:

| Angle | PPO low-ρ − PPO SDMA | Seedwise wins | CEM RSMA − CEM SDMA | Trajectory wins |
|---:|---:|---:|---:|---:|
| 5°  | −0.1047 ± 0.3195 | 1/3 | −0.1663 ± 0.0045 | 3/3 |
| 20° | +0.0682 ± 0.0403 | 0/3 | −0.1007 ± 0.0133 | 3/3 |
| 60° | +0.0482 ± 0.0120 | 0/3 | +0.0220 ± 0.0034 | 0/3 |

The apparent 5° mean advantage of low-ρ PPO is not robust: SDMA seed 0 failed
at AoI 2.168, while SDMA seeds 1 and 2 both obtained 1.637 and beat all but one
RSMA seed.  With only three training seeds, these values are descriptive.

### What PPO and CEM did with the common stream

| Angle | Method | Mean common power | Mean common payload fraction ρ | Common BLER when gate is on |
|---:|---|---:|---:|---:|
| 5°  | CEM RSMA | 0.629 | 0.438 | — |
| 5°  | PPO low-ρ | 0.109 | 0.019 | 0.0002 |
| 5°  | PPO joint-init | 0.201 | 0.178 | 0.0622 |
| 20° | CEM RSMA | 0.482 | 0.303 | — |
| 20° | PPO low-ρ | 0.080 | 0.017 | 0.0113 |
| 20° | PPO joint-init | 0.166 | 0.129 | 0.1204 |
| 60° | CEM RSMA | 0.378 | 0.023 | — |
| 60° | PPO low-ρ | 0.084 | 0.017 | 0.1277 |
| 60° | PPO joint-init | 0.213 | 0.140 | 0.4438 |

`broad_rho` was worse in every geometry.  Its common BLER when enabled reached
about 0.33 on average at 5°, 0.69 at 20°, and 0.96 at 60°.  Raising common
payload without simultaneously finding adequate common power pushes the FBL
decoder through a steep error-probability cliff.

## Diagnosis

### 5° and 20°: primarily an optimisation/action-representation problem

**[Experiment]** Both the offline exact-FBL audit and repeated online CEM choose
substantial common power and payload and beat the corresponding SDMA allocator.
The result therefore does not support the claim that the standard 1-layer RSMA
PHY is intrinsically inferior in these aligned geometries.

**[Code]** PPO samples power, common-rate simplex, total common-payload scale,
and a Bernoulli common gate as separate policy factors.  Their joint effect on
BLER is strongly coupled.  A useful action requires all factors to enter a
decodable region together, while many partially coordinated samples give almost
zero success and weak/noisy policy-gradient credit.

**[Inference]** The low-ρ policy found a safe near-SDMA basin; broad exploration
crossed into high common BLER; a hand-picked joint initial point helped only
some seeds and did not approach the CEM basin.  Initialisation alone is
therefore insufficient.  A representation that encodes decodability, or an
explicit information-feasible SDMA boundary selector, is the next justified
algorithmic change.

### 60°: a natural SDMA-favouring regime

**[Experiment]** SDMA wins even when RSMA and SDMA receive the same online CEM
budget.  CEM drives common payload close to zero (ρ ≈ 0.023), which is the RSMA
boundary rather than a meaningful common-stream operating point.

**[Inference]** With well-separated user directions, RZF suppresses private
interference and the common stream adds a worst-user decoding constraint and
uses power that private streams exploit more efficiently.  This is a useful
negative result and should remain in the thesis.  It must be phrased as a result
for the specified finite-blocklength geometry and beam/allocator class, not a
universal theorem about RSMA.

### One-step expected AoI surrogate

**[Experiment]** In this controlled generate-at-will experiment, applying the
one-step CEM decision at every slot preserves the sign of the offline audit:
RSMA wins at 5°/20° and loses at 60°.

**[Inference]** This shows that the earlier heterogeneous screening reversal
cannot be attributed to horizon length alone.  That earlier pipeline also
changed fallback, information, scheduling, and allocation conditions between
screening and validation.  The controlled result does not prove that a
one-step surrogate is generally optimal; queueing, stochastic arrivals,
retransmissions, weights, or capped/tail AoI can restore genuine long-horizon
effects.

## Behaviour cloning result

Gate-aware oracle datasets were generated for all three geometries.  The
60° dataset correctly contains only common-off labels and behaviour cloning
reproduces its oracle closely.  At 5°/20°, deterministic regression of
multi-modal CEM allocations averages incompatible modes and gives test
one-step-AoI gaps of about 3.70 and 2.00.  This BC model should not be used as
the main PPO warm start in its current form.

## Decision and next implementation priority

The common beam remains `principal`; max-min stays a diagnostic ablation.  Beam
choice is not the dominant failure in these runs.

The next action design should satisfy four conditions:

1. PPO still decides the common/private power split and the per-user common-rate
   allocation, as required by the thesis direction.
2. `P_c=0, D_c=0` is an exact executable boundary, not a limit that softmax and
   sigmoid can only approximate.
3. Common payload is parameterised relative to a channel-dependent FBL-decodable
   budget computed from transmitter-side CSIT, so random exploration does not
   spend most samples beyond the common decoder's cliff.
4. The mapping and any SDMA fallback use `h_hat`; `h_true` remains evaluation-only.

A safeguarded or FBL-normalised action is a research hypothesis.  It must be
compared with the existing direct action, SDMA-power PPO, and matched-budget CEM
across all three geometries and imperfect-CSIT levels.  A one-step safety check
must not be described as a guarantee on long-run AoI.

## Follow-up: FBL-normalised action pilot

An optional `common_payload_mapping=fbl_normalized` mode was implemented.  If
the policy emits utilization `u`, the environment computes the common SINR on
`h_hat` under the policy's selected power, inverts the same normal approximation
used by the PHY at `common_target_bler`, and transmits

`D_c = u min(D_split_cap, D_FBL(h_hat, P, epsilon_target))`.

It leaves the direct mode unchanged and logs both the policy scalar and the
effective payload scale.  Unit tests verify the direct identity, monotonicity,
configuration restrictions, and the nominal worst-user BLER bound.

A 32,768-step, one-training-seed pilot with target BLER 0.1 obtained
deterministic AoI 2.096 (5°), 2.362 (20°), and 1.046 (60°).  The learned raw
utilizations were only 0.142, 0.257, and 0.126; effective payload fractions were
0.042, 0.096, and 0.017.  Catastrophic common BLER disappeared, but the actor
again collapsed toward little common payload.  The pilot is therefore a useful
negative ablation and is not promoted to a three-seed main experiment.

## Follow-up: objective-aware allocator pretraining audit

The existing differentiable FBL mirror was applied to the three oracle datasets
with an identical learned allocator.  The table reports one-step expected AoI
on the held-out test states.  Oracle CEM is the reference column.

| Angle | Action-only BC | Best objective-aware variant | Oracle reference | Best regret |
|---:|---:|---:|---:|---:|
| 5°  | 9.258 | 5.149 (`action_objective`) | 4.464 | 0.685 |
| 20° | 7.801 | 3.384 (`objective_only`) | 2.908 | 0.475 |
| 60° | 10.415 | 2.042 (`objective_only`) | 2.014 | 0.028 |

This validates the diagnosis that action-space distance is a poor loss under a
steep FBL threshold.  Directly loading this standalone allocator into PPO would
mix incompatible network architectures and is not valid.

## Objective-aware PPO actor warm-start

The PPO actor itself was subsequently pretrained.  Its modal power, common-rate
simplex, and common-payload heads are evaluated by the differentiable FBL PHY.
For the Bernoulli common gate, training uses the exact expectation

`p_gate J_common + (1-p_gate) J_SDMA-boundary`,

which supplies a gradient while deterministic deployment retains the exact
0/1 common-off decision.  The checkpoint contains the PPO actor's own weights
and observation normaliser, so PPO loads it without architecture conversion.

The offline objective uses each dataset sample's realised `h_true` to compute
the training loss, but `h_true` is never an actor input.  In the current
perfect-CSIT study `h_true=h_hat`.  For a later imperfect-CSIT study, the
dataset must contain enough independent error realisations that this empirical
loss approximates the conditional expectation given `h_hat`; otherwise the
pretraining objective will overfit individual channel errors.

After one fixed pretraining run per geometry, PPO was fine-tuned for 65,536
steps with three independent PPO seeds and evaluated using the same long-run
protocol as the original PPO.

| Angle | Original low-ρ PPO | Objective-warm-start PPO | Paired change | Improved seeds |
|---:|---:|---:|---:|---:|
| 5°  | 1.7092 ± 0.0432 | **1.5730 ± 0.0035** | −0.1362 | 3/3 |
| 20° | 1.7300 ± 0.1096 | **1.5010 ± 0.0172** | −0.2290 | 3/3 |
| 60° | 1.0740 ± 0.0103 | **1.0232 ± 0.0038** | −0.0508 | 3/3 |

At both 5° and 20°, the warm-start policy also beats the independently trained
SDMA-power PPO in all three paired seeds (mean differences −0.2409 and
−0.1608).  The unusually poor SDMA seed 0 still inflates the 5° mean difference,
but the other two warm-start seeds also beat their SDMA counterparts.
At 60°, it places only 1.7% mean power and 0.03% payload on the common stream,
so its AoI 1.0232 should be interpreted as successfully learning the SDMA
boundary.  Its tiny −0.0026 difference from SDMA-power PPO is not evidence of an
RSMA gain, particularly because matched-budget CEM favours SDMA there.

The uniform objective-only policies use mean common power/payload fractions of
0.372/0.348 at 5° and 0.288/0.258 at 20°, with common BLER 0.007 and 0.019 when
enabled.  These are meaningful common-stream operating points rather than
near-SDMA solutions.  An earlier 5° diagnostic used a geometry-specific mixed
loss and fell back toward low payload; it is retained only as a sensitivity run.
The controlled table above fixes `objective_only` at every geometry and uses the
validation metric, rather than the final test number, for checkpoint selection.

The learned-policy comparison does not establish global RSMA superiority.
Matched-budget CEM still attains 1.320/1.177 at 5°/20°, while CEM-SDMA attains
1.486/1.278; both are below the corresponding warm-start PPO values.  The
supported algorithmic claim is improved PPO training and a correct regime
switch, not superiority over numerical allocation optimisation.

### Current research conclusion

The supported contribution is no longer “PPO finds RSMA allocations from
scratch.”  The evidence supports the more specific statement that a steep FBL
success surface makes action imitation and uninformed PPO exploration
unreliable, while **objective-aware physical pretraining improves subsequent
PPO control** and learns when to retain or disable the common stream.

Before the final thesis experiment, repeat the pretraining itself with at least 
three seeds, then cross each selected pretraining checkpoint with multiple PPO  
fine-tuning seeds.  Add imperfect-CSIT levels only after that two-level seed    
design is fixed.  The current three-seed table measures PPO fine-tuning
variation conditional on one fixed pretrained actor.

As a first seed audit, the 20° objective-only pretraining was repeated with
seeds 0, 1, and 2.  Held-out hard-gate expected AoI was respectively 3.7147,
3.7172, and 3.6741 (mean 3.7020, standard deviation 0.0242); gate accuracy was
0.7250, 0.7050, and 0.7067.  This rules out large pretraining-seed instability
at the representative 20° geometry, but it does not yet measure the crossed
pretraining-by-PPO seed interaction.

## Follow-up: held-out PPO convergence and critic audit

The original curves were training rollouts under sampled actions.  They showed
finite losses and falling AoI, but could not establish that the deployed
deterministic policy improved.  A new audit therefore saves immutable
checkpoints and evaluates every checkpoint on identical held-out channel,
CSIT-error, and ACK seeds.  The evaluation uses three PPO seeds, three held-out
4,096-slot trajectories per seed, and the 20°/`sigma_e=0.2` objective-aware
warm start.  The pretraining checkpoint is held fixed so this experiment
isolates PPO fine-tuning variation.

| PPO setting | Initial AoI/user | Final AoI/user | Initial-to-final | Last-16-update critic EV |
|---|---:|---:|---:|---:|
| actor/critic LR `3e-4/3e-4` | 1.5463 | 1.5374 | -0.58% | -1.217 |
| actor/critic LR `3e-4/1e-3` | 1.5463 | **1.5149** | **-2.03%** | **-0.377** |

Both settings remain numerically stable: no non-finite core metrics, PPO KL
explosion, constraint violation, or early stop occurred.  With the larger
critic LR, held-out AoI decreases monotonically after the early transient in
the sampled checkpoints (Spearman step/AoI correlation `-0.95`) and all three
paired training seeds beat the reference at the final checkpoint.  The mean
paired improvement is 0.0225 AoI/user.  Three training seeds are too few to
claim a universally optimal learning rate or a strong significance result.

The critic is improved, not fully solved: explained variance remains negative.
Removing reward normalization produced value losses around hundreds and no
held-out gain; removing value clipping did not improve the critic.  Those
variants are rejected.  The thesis CSIT configurations therefore freeze
`actor_lr=3e-4`, `critic_lr=1e-3`, retain reward normalization and value
clipping, and use deterministic held-out checkpoint curves as the primary
convergence evidence.  A positive explained variance is not claimed as a
requirement for policy validity in this stochastic ACK process.

Artifacts are under `results/ppo_convergence_critic_3seed_v1/`; the reusable
runner is `scripts/audit_ppo_convergence.py` and the plotting/diagnostic script
is `scripts/plot_ppo_convergence.py`.

## Reproducible artifacts

- Channel definition: `docs/channel_model_recommendation.md`
- PHY audit: `docs/rician_phy_regime_audit.md`
- PPO/CEM consolidated data: `results/rician_ppo_study_v1/`
- Plot: `results/rician_ppo_study_v1/rician_ppo_study.png`
- Initial, broad-rho and joint-initialisation diagnostics:
  `archive/results/rician_ppo_*_3seed_v1/`
- Online allocator runs: `results/rician_allocator_longrun_cem_full_v1/`
- FBL-normalised pilots: `archive/results/rician_ppo_*deg_fblnorm_pilot_v1/`
- Objective-aware allocator audit: `results/rician_objective_allocator_study_v1/`
- PPO-compatible objective pretraining: current selections in `runs/`, older
  variants in `archive/runs/`
- Objective-aware PPO actor audit: `results/rician_objective_actor_study_v1/`
- Objective-warm-start PPO: `results/rician_ppo_{20,60}deg_objective_warmstart_3seed_v1/`
  and the uniform-preset 5° run
  `results/rician_ppo_5deg_objective_warmstart_uniform_3seed_v1/`
