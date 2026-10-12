# Configuration guide

The active thesis configurations begin with `thesis_`.

- `thesis_rician_mixed_geometry_snr_direct.yaml`: canonical ``K=2,N_t=2``
  gate-free action over mixed geometry and SNR.
- `thesis_rician_k3_nt2_all3_direct.yaml`: canonical overloaded ``K=3,N_t=2``
  all-user allocation-only PPO configuration.
- `thesis_rician_k3_nt2_mixed_phy_sinr.yaml`: experimental K=3, Nt=2
  generalisation policy over geometry, SNR and blocklength using causal
  estimated-CSIT beam/SINR features.
- `thesis_rician_k3_nt2_20deg_sinr_finetune.yaml`: matched-width 20-degree,
  15-dB, n=128 target fine-tuning configuration for that mixed policy.
- `thesis_direct_fraction.yaml`: compact direct-action development fixture.

- `thesis_rician_alignment_{5,20,60}deg.yaml`: perfect-CSIT spatial controls.
- `thesis_rician_alignment_20deg_csit{00,01,02,03}.yaml`: primary CSIT sweep.
- `thesis_rician_20deg_csit02_snr{05,15}.yaml`: low/high-SNR sensitivity
  points around the 10 dB primary configuration.
- `thesis_final_evaluation.yaml`: former frozen 65k held-out protocol, retained
  for traceability; its unmatched-initialization headline is superseded.
- `thesis_rician_nearfar5_csit02.yaml`: representative directional-Rician
  near/far stress test with a 5 dB gain gap.
- `thesis_spatial_{5,20,60}deg_evaluation.yaml` and
  `thesis_nearfar5_evaluation.yaml`: frozen long-run regime protocols.
- `thesis_snr{05,10,15}_evaluation.yaml`: matched three-seed SNR sensitivity
  protocols with a shared held-out trajectory set.
- `thesis_cem_reference_evaluation.yaml`: short matched-trajectory comparison
  of the five-seed headline policies and bounded-compute online CEM.
- `thesis_matched_initialization_evaluation.yaml`: frozen six-trajectory audit
  separating RSMA/SDMA action class, objective pretraining and PPO effects.
- `thesis_training_budget_evaluation.yaml`: matched five-seed 0/65k/262k PPO
  budget audit and current learned-policy headline at the representative
  20-degree imperfect-CSIT point.
- `thesis_allocation.yaml` and `thesis_ablation.yaml`: compact action-path and
  ablation configurations.
- `env_default.yaml` and `ppo_default.yaml`: documented library defaults.
- `td3_default.yaml`: retained off-policy comparator defaults.

The `m0`, `m1`, `m2`, `het`, `op` and `sweep_*` files describe earlier project
phases.  They remain here because integration tests and general-purpose scripts
load them directly.  They are compatibility fixtures, not the current thesis
experiment family.

Configurations containing `rate_split_mode: simplex_scaled` or
`common_gate_enabled: true` reproduce the pre-redesign action and must not be
used to train new headline policies.

Legacy spatial, action-ablation, and CSIT experiments use actor/critic learning
rates `3e-4/3e-4` and must retain that label.  The active gate-free K=3 direct
action configuration uses `3e-4/1e-3`: after fixing warm-start observation
normalisation, a matched 3-seed audit reduced 262k held-out AoI by 0.53% versus
`3e-4/3e-4` and won all six paired evaluation trajectories.  Do not mix these
two training generations in an unlabeled aggregate.

The mixed-PHY SINR configurations are an experimental generalisation branch.
They do not replace `thesis_rician_k3_nt2_all3_direct.yaml` as the best
single-operating-point reference without a future multi-seed comparison.
