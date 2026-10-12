# Mixed-PHY SINR observation and curriculum audit (updated 2026-09-22)

## Question and scope

Can one K=3, Nt=2 direct-action PPO policy use the RSMA common-power fraction
`alpha` differently as geometry, SNR and finite blocklength change, instead of
converging to one fixed-environment value? The selected pipeline is repeated
with five independent training seeds. Confidence intervals use the five
training-seed means as the statistical units.

## Causal PHY-aware observation

`obs_channel_mode: beam_sinr_features` uses only the current channel estimate
`h_hat`: channel gains/correlations, principal-common-beam gains, the full RZF
private gain matrix, and nominal common/private SINRs obtained by applying the
previous allocation to the current estimated channel. At reset, the nominal
SINR uses alpha=0.25 and equal private power. The observation never uses
`h_true` and does not depend on the action PPO has yet to choose.

## Curriculum

- Geometry span: 5, 20 and 60 degrees, with all six K=3 user permutations.
- SNR: 5, 10, 15 and 20 dB.
- Blocklength: 64, 128 and 256 channel uses.
- Relative CSIT error standard deviation: 0.2.
- Fixed 256-bit generate-at-will updates; direct `(alpha, q_k, phi_k)` action;
  all three users active.
- Independent uniform episode-level sampling gives 36 PHY regimes.
- Pretraining dataset: 3,600/900/900 states, three-state trajectories to keep
  regime frequencies approximately balanced.
- Action+model-based FBL pretraining: 80 epochs and 8-sample posterior SAA,
  independently initialized for training seeds 0--4.
- Initial mixed PPO: seed 0, 524,288 steps.
- Stage 1 PPO: geometry 5/20/60 degrees, SNR 10/15/20 dB and
  n=128/256, 524,288 steps.
- Stage 2 stress exposure: the complete 36-regime grid, 1,048,576 steps.
- Final target fine-tuning: each Stage 1 checkpoint at 20 degrees, 15 dB and
  n=128 for 524,288 steps. The complete pipeline was repeated for seeds 0--4.
  The Stage 2 checkpoint was not used because it sacrificed the feasible
  regimes without resolving the hard-regime tail.

The implementation also randomizes blocklength per episode, exposes it in the
observation, updates both PHY implementations, and stores per-state
blocklength/geometry metadata for objective pretraining.

## Alpha adaptation

| policy | alpha at 5/20/60 deg | alpha at SNR 5/10/15/20 dB | alpha at n=64/128/256 |
|---|---:|---:|---:|
| mixed PPO 524k | 0.605 / 0.560 / 0.256 | 0.393 / 0.415 / 0.492 / 0.594 | 0.573 / 0.451 / 0.397 |
| initial mixed PPO plus 131k target fine-tuning | 0.629 / 0.586 / 0.277 | 0.366 / 0.416 / 0.536 / 0.672 | 0.566 / 0.477 / 0.450 |
| staged PPO plus 524k target fine-tuning, seed 0 | 0.694 / 0.657 / 0.405 | 0.329 / 0.525 / 0.692 / 0.795 | 0.616 / 0.571 / 0.568 |
| **five-seed mean** | **0.619 / 0.589 / 0.397** | **0.250 / 0.453 / 0.659 / 0.778** | **0.500 / 0.542 / 0.562** |
| five-seed standard deviation | 0.055 / 0.051 / 0.049 | 0.084 / 0.053 / 0.028 / 0.021 | 0.106 / 0.038 / 0.026 |

The geometry response is physically interpretable: the policy assigns less
common power when channels separate. Within-trajectory alpha standard
deviation is about 0.13--0.17 in representative regimes, so this is not merely
one constant per environment. The geometry and SNR directions survive all five
training seeds. The blocklength response is weaker, especially at n=64 where
the seed standard deviation is 0.106. These are empirical responses rather
than universal laws; several n=64, low-SNR regimes are close to infeasible for
three simultaneous 256-bit updates.

## Performance

Each grid value averages six paired 768-slot trajectories, one per user
permutation, with 256 burn-in slots.

| 20 deg, 15 dB, n=128 | five-seed warm | five-seed final PPO | SDMA |
|---|---:|---:|---:|
| mean AoI | 2.481 | **1.463** | 2.822 |
| training-seed standard deviation | 0.270 | **0.022** | — |
| 95% CI half-width | 0.336 | **0.027** | — |
| mean p95 AoI | 7.508 | **2.000** | 7.500 |
| mean alpha | 0.737 | **0.762** | 0 |

At the target point, the final PPO improves over its own independently trained
warm actor by 40.51% +/- 7.88 percentage points and over SDMA by
48.17% +/- 0.95 percentage points. These are 95% Student-t intervals over five
training-seed means.

In the 18 feasible regimes defined prospectively as SNR >= 10 dB and
n >= 128, the five final policies achieve mean AoI 1.701 versus SDMA 2.798.
The mean paired gain is 26.29% +/- 1.13 percentage points (95% Student-t
interval over five training seeds).

The complete-grid number is not a defensible headline. In the n=64 and/or
5 dB stress regimes, all methods can approach zero packet success and the AoI
cap scale. A few denominator-sensitive percentage gains dominate the average:
the five-seed stress-stratum gain over SDMA is -5.77% with a 95% half-width of
88.94 percentage points. Its sign is unresolved. Absolute AoI and
feasible/stress strata must therefore be reported separately.

The final staged policy also improves on the earlier fixed-environment
gain-feature PPO under the same six permutations, seeds, 768 slots and
256-slot burn-in:

| policy | target mean AoI |
|---|---:|
| previous fixed-regime canonical PPO | 1.533 |
| final staged mixed-to-target SINR PPO | **1.456** |
| mixed warm actor | 2.333 |

For seed 0, the paired improvement over the old fixed policy is 5.05% with a 95% t
interval of +/-1.82 percentage points across the six trajectories. This is
evidence that the PHY-aware observation and staged curriculum can improve the
target point as well as produce regime-dependent alpha. The old fixed policy
was not retrained for five seeds, so this particular architecture comparison
remains a seed-0 diagnostic rather than the headline.

## Training diagnosis and decision

All training phases finished without NaN or constraint violations. Stage 1 was
the useful general policy: over the feasible stratum it improved on warm by
18.04% and SDMA by 25.29%. Extending the complete-grid Stage 2 to 1,048,576
steps reduced those gains to -4.02% and +7.72%; the hard regimes still had
catastrophic outliers. More raw-AoI mixed training is therefore not a solution
to regime imbalance.

All five final target runs improved their rollout return. Four finished around
-790 to -802 and seed 3 around -831, yet their frozen target AoI is tightly
clustered between 1.444 and 1.500. Held-out paired AoI rather than rollout
return or critic quality therefore supports the policy choice. All 616 tests
passed after the implementation changes.

Use the Stage 1 plus target-fine-tuned checkpoint as the current primary PPO.
The defensible finding is that causal PHY-aware state makes alpha respond to
geometry, SNR and blocklength, while staged training improves the target AoI.
Do not claim universal robustness over the 36-cell grid. The target and
feasible-stratum claims now have five-training-seed support; the stress-stratum
claim does not. Any later attempt to solve the stress regimes should use a
predeclared stratified objective or per-regime regret; that is a new objective
and should not silently replace mean AoI.

## Artifacts

- Configs: `configs/thesis_rician_k3_nt2_mixed_phy_sinr.yaml` and
  `configs/thesis_rician_k3_nt2_mixed_phy_sinr_stage1.yaml` and
  `configs/thesis_rician_k3_nt2_20deg_sinr_finetune.yaml`
- Dataset: `datasets/rician_k3_nt2_mixed_phy_sinr_oracle_v2/`
- Runs: `runs/action_objective_actor_k3_mixed_phy_sinr_seed{0..4}_v1/`,
  `runs/ppo_k3_mixed_phy_sinr_stage1_seed{0..4}_524k_v1/`,
  `runs/ppo_k3_mixed_phy_sinr_stage2_seed0_1m_v1/`, and
  `runs/ppo_k3_20deg_sinr_from_stage1_seed{0..4}_524k_v1/`
- Results: `results/ppo_k3_mixed_phy_sinr_524k_eval_v1/` and
  `results/ppo_k3_mixed_phy_sinr_stage1_524k_eval_v1/`,
  `results/ppo_k3_mixed_phy_sinr_stage2_1m_eval_v1/`, and
  `results/ppo_k3_20deg_sinr_from_stage1_524k_eval_v1/`
- Five-seed results:
  `results/ppo_k3_mixed_phy_seed_expansion_target_v1/` and
  `results/ppo_k3_mixed_phy_seed_expansion_grid_v1/`
