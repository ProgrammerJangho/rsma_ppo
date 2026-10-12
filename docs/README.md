# Current research documents

Read these documents in order.  The canonical action document overrides every
older gate, simplex and ``rho`` description.  For other topics, later result
documents override earlier diagnostic assumptions when they disagree.

0. [`canonical_action_definition.md`](canonical_action_definition.md) — current
   gate-free allocation action.  It supersedes older gate, simplex and ``rho``
   action descriptions below.
   - [`action_redesign_implementation_2026-09-17.md`](action_redesign_implementation_2026-09-17.md)
     records changed code paths, compatibility boundaries and verification.

**Latest PPO result:** [`mixed_phy_sinr_curriculum_2026-09-20.md`](mixed_phy_sinr_curriculum_2026-09-20.md)
records the causal PHY-aware observation, staged geometry/SNR/blocklength
curriculum, failed full-grid stress extension, and the selected Stage-1-to-target
checkpoint. It is the current source for the primary K=3, Nt=2 PPO result.

**Thesis-ready result chapter:** [`thesis_results_chapter_2026-09-22.md`](thesis_results_chapter_2026-09-22.md)
contains the frozen five-training-seed tables, statistical interpretation,
allowed claims, limitations and figure captions. Its artifact manifest is
`results/thesis_mixed_phy_final_v1/manifest.json`.

**Integrated thesis draft:** [`thesis_manuscript_draft_2026-09-22.md`](thesis_manuscript_draft_2026-09-22.md)
connects the current system model, related work, PPO method, experimental
protocol, frozen results and limitations. Its machine-readable bibliography is
[`thesis_references.bib`](thesis_references.bib).
Use [`thesis_submission_checklist_2026-09-22.md`](thesis_submission_checklist_2026-09-22.md)
for the remaining school-template and claim-review work.

**Matched pretraining audit:** [`matched_pretraining_results_2026-09-24.md`](matched_pretraining_results_2026-09-24.md)
repeats the target RSMA/SDMA comparison with the same objective-only pretraining
loss, data, epochs, PPO curriculum and budgets. This document supersedes the
pretraining-confounded interpretation in the earlier q-only result note.

1. [`project_midterm_status.md`](project_midterm_status.md) — historical
   gate/`rho`-generation midterm record; its 2026-09-22 notice points to the
   current action and result documents.
2. [`final_evaluation_protocol.md`](final_evaluation_protocol.md) — frozen
   long-horizon main comparison, uncertainty unit and supported interpretation.
3. [`final_experiment_matrix.md`](final_experiment_matrix.md) — missing training
   runs and the statistical scope of headline, spatial and near/far studies.
4. [`final_regime_results.md`](final_regime_results.md) — completed five-seed
   headline, matched spatial controls and corrected near/far stress test.
5. [`snr_sensitivity_results.md`](snr_sensitivity_results.md) — matched
   5/10/15 dB retraining and long-run evaluation.
6. [`cem_262k_long_results.md`](cem_262k_long_results.md) — final frozen-protocol
   comparison of 262k PPO and bounded-compute CEM.
7. [`cem_reference_results.md`](cem_reference_results.md) — superseded short-horizon
   CEM audit retained for traceability and implementation-cost diagnosis.
8. [`matched_initialization_results.md`](matched_initialization_results.md) —
   matched RSMA/SDMA objective pretraining and zero-step versus PPO effects.
9. [`training_budget_results.md`](training_budget_results.md) — matched
   0/65k/262k PPO audit and the final five-seed learned-policy result.
10. [`two_mode_policy_audit.md`](two_mode_policy_audit.md) — nested controls that
   identify which parts of the converged continuous action remain state dependent.
11. [`mixed_geometry_policy_audit.md`](mixed_geometry_policy_audit.md) — one-policy
   5/20/60-degree test of whether boundary concentration is a fixed-geometry artefact.
12. [`mixed_geometry_snr_results.md`](mixed_geometry_snr_results.md) — one-policy
   mixed geometry/SNR generalization, CEM comparison and common-off projection.
13. [`k3_nt2_overloaded_results.md`](k3_nt2_overloaded_results.md) — K=3, Nt=2
   all-user/scheduled-user CEM study plus all-3 allocation-only PPO and its
   remaining numerical-reference gap.
14. [`symmetric_rsma_followup_results.md`](symmetric_rsma_followup_results.md) —
   symmetry-aware observation/action pretraining, paired common contribution and
   matched CEM comparison.
15. [`k3_ppo_gap_diagnosis.md`](k3_ppo_gap_diagnosis.md) — same-state CEM gap,
   action-head attribution, objective-only pretraining and destructive PPO
   exploration in the overloaded extension.
16. [`rsma_characteristic_diagnosis.md`](rsma_characteristic_diagnosis.md) — why
   the current learned policy underuses RSMA and which fixes target the evidence.
17. [`thesis_allocation.md`](thesis_allocation.md) — thesis scope, PPO action and
   the precise meaning of common-rate allocation.
18. [`channel_model_recommendation.md`](channel_model_recommendation.md) — selected
   directional Rician/Gauss–Markov channel and its controls.
19. [`rician_phy_regime_audit.md`](rician_phy_regime_audit.md) — fixed-beam FBL
   RSMA/SDMA behaviour across spatial regimes.
20. [`common_beam_allocation_audit.md`](common_beam_allocation_audit.md) — why the
   principal beam remains the default and max-min stays an ablation.
21. [`common_gate_followup.md`](common_gate_followup.md) — exact common-off boundary
   and the gate mechanism.
22. [`objective_action_ablation.md`](objective_action_ablation.md) — contribution
   of power, rate split and common gate.
23. [`imperfect_csit_saa_pretraining.md`](imperfect_csit_saa_pretraining.md) —
   posterior-SAA negative result and matched CSIT comparison.
24. [`rician_ppo_training_diagnosis.md`](rician_ppo_training_diagnosis.md) — current
   consolidated PPO conclusion and convergence audit.

Superseded proposals, the original allocation pilot, and the comparison with
the imported older code are under `archive/docs/`.  They are useful for tracing
decisions but should not be cited as the current implementation specification.
