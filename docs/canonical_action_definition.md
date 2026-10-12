# Canonical RSMA-PPO action after the gate/rho redesign

## Decision

The thesis model uses fixed per-user update sizes `D_k` and the continuous
allocation action

\[
a_t=(\alpha,\mathbf q,\boldsymbol\phi).
\]

- `alpha in [0,1]`: fraction of total power assigned to the common stream.
- `q_k >= 0`, `sum q_k = 1` over active users: private-power simplex.
- `phi_k in [0,1]`: fraction of user `k`'s fixed payload placed in the common
  codeword.  The `phi_k` values are independent and do not sum to one.

The transmitted powers are

\[
P_c=\alpha P_{max},\qquad P_k=(1-\alpha)q_kP_{max}.
\]

The payload split and its correspondence to standard RSMA common-rate notation
are

\[
D_{c,k}=\phi_kD_k,\qquad D_{p,k}=(1-\phi_k)D_k,
\]
\[
D_c=\sum_kD_{c,k},\qquad C_k=D_{c,k}/n.
\]

Every active user decodes the complete `D_c`-bit common codeword for SIC.  The
user-specific `C_k` is its contribution to that codeword, not a separate stream.

## Removed decisions

The current policy has no Bernoulli common gate and no common-payload magnitude
`rho`.  The former simplex `c_k` plus `rho` coordinates are retained in source
only for loading and reproducing archived experiments.  Current environment
actions use the explicit keys `common_power_fraction`,
`private_power_fractions` and `common_payload_fractions`; misleading legacy
keys are accepted only by an input compatibility adapter.

## Exact boundaries

The direct heads use an endpoint-capable hard sigmoid

\[
h(z)=\operatorname{clip}(1/2+z/6,0,1).
\]

Consequently the continuous policy can produce exact zero and one.  Exact
`alpha=0` and `phi_k=0` for all active users represent SDMA without adding a
discrete gate.  PPO log probabilities are evaluated on the stored Gaussian
latent variables, so clipping the deterministic action transform does not
invalidate the PPO probability ratio.

## Experiment provenance

New policies must use a `*_direct.yaml` configuration.  Existing checkpoints,
datasets and result tables generated with `simplex_scaled`, `rho` or a common
gate keep their original interpretation and cannot be compared as though they
used this action.  They require new objective data, pretraining and PPO runs.
