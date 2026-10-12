# Gate/rho-free action redesign implementation report

## Implemented action

For fixed update sizes `D_k`, every current `*_direct.yaml` policy uses

\[
a_t=(\alpha,\mathbf q,\boldsymbol\phi),
\]

with

\[
P_c=\alpha P_{max},\qquad P_k=(1-\alpha)q_kP_{max},
\quad \sum_{k\in A}q_k=1,
\]

\[
D_{c,k}=\phi_kD_k,\qquad D_{p,k}=(1-\phi_k)D_k,
\qquad C_k=D_{c,k}/n.
\]

The environment action dictionary is now explicit:

- `schedule`
- `common_power_fraction` (`alpha`)
- `private_power_fractions` (`q_k`)
- `common_payload_fractions` (`phi_k`)

There is no gate key and no `rho` key.  A direct actor has neither a
`common_gate_head` nor a `rho_r_head` in its checkpoint.

## Exact SDMA boundary

The logistic sigmoid pilot could approach but never equal zero.  Its replacement

\[
h(z)=\operatorname{clip}(1/2+z/6,0,1)
\]

lets `alpha` and every `phi_k` equal zero exactly.  This represents the SDMA
boundary without a separate Bernoulli decision.  PPO continues to store and
score the Gaussian latent `z`; the clipped map is only a deterministic action
transform.

The PHY itself was not changed to treat an arbitrary near-zero payload as zero.
The finite-blocklength near-zero effect therefore remains visible rather than
being hidden behind an unphysical tolerance.

## Compatibility and provenance

The old `simplex_scaled` implementation, gate and `rho` fields remain readable
for archived checkpoints and result reproduction.  They are not emitted by a
new direct policy.  The environment accepts the former combined direct-action
keys through a compatibility adapter, but its declared Gym action space and all
new PPO/CEM policies use the explicit schema above.

Existing configs tied to published result folders were not overwritten.  New
training uses:

- `configs/thesis_rician_mixed_geometry_snr_direct.yaml`
- `configs/thesis_rician_k3_nt2_all3_direct.yaml`
- `configs/thesis_direct_fraction.yaml`

The pre-change source/config/test snapshot is under
`archive/action_redesign_before_20260916/`.

## Verification

- Full suite after dataset/pretraining integration: **601 passed**.
- Python compile check: passed for `src`, `scripts` and `tests`.
- End-to-end unit test: a PPO actor outputs exact `alpha=phi=0`; RSMA and SDMA
  success probabilities agree to `1e-12` and common power/payload are zero.
- `K=2,N_t=2` mixed geometry/SNR: 1,024-step PPO smoke, checkpoint save/load
  and finite update metrics passed with zero feasibility violations.
- `K=3,N_t=2` all-user: the same 1,024-step smoke passed.
- Restored checkpoints expose only the four public action keys above; their
  state dictionaries contain no gate or `rho_r` parameters.
- `run_direct_fraction_study.py --steps 512`: completed all paired CEM states
  and three legacy/direct training seeds with the new schema.

The short runs verify execution only.  Their AoI values are not evidence that
the redesigned action improves the final policy.

## Direct CEM dataset and objective warm start

The first two items of the experiment-generation plan are complete for the
`K=3, N_t=2`, all-three-user setting.

- Dataset: `datasets/rician_k3_nt2_all3_direct_csit02_oracle_v1`
- Splits: 3,600 train / 900 validation / 900 test, with 216 disjoint
  trajectories
- Label budget: 5 CEM iterations × 256 candidates × 2 restarts = 2,560
  candidates per state
- Schema: `direct_alpha_q_phi_hard_sigmoid_v1`
- Stored canonical labels: `alpha`, private-power simplex `q_k`, and payload
  fractions `phi_k`
- Posterior objective training: 8 samples from `p(h|h_hat)` per state
- Actor seeds: 0, 1, 2; 60 epochs each

The objective-pretraining path previously used a logistic sigmoid even though
PPO and CEM used the clipped linear transform.  It now calls the same
`hard_sigmoid_torch` implementation, and a regression test compares its modal
allocation directly with the deployed PPO distribution.

Imperfect-CSIT candidate selection is performed on `h_hat`.  Consequently, a
rejected candidate may be better on one hidden `h_true` draw.  The dataset now
records two distinct quantities:

- `objective_gap`: non-negative selection margin on the information available
  to the policy;
- `realized_alternative_gap`: diagnostic difference on the hidden realised
  channel, which may be negative.

The held-out single-realisation expected AoI of the three warm starts was
13.0212, 12.5976, and 13.1213.  The corresponding finite-budget CEM labels
average 10.3405.  Thus the warm start is useful but does not reproduce the CEM
reference; the difference must remain visible in later comparisons.

A 2,048-step PPO smoke run loaded the seed-1 actor with no missing or unexpected
parameters, completed four updates with finite losses, produced zero action
constraint violations, and saved/restored a direct-action checkpoint.

## Remaining experiment generation

The 262,144-step three-seed PPO run and matched six-trajectory PPO/CEM/SDMA
evaluation are complete. See
`docs/k3_nt2_direct_ppo_cem_results_2026-09-17.md`.

The next experiment should diagnose and reduce the remaining CEM-to-PPO gap,
especially PPO's private BLER and its training-seed sensitivity.
