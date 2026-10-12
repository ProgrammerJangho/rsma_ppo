# Mixed-geometry policy and boundary-concentration audit

## Question and controlled design

This experiment tests whether the common/private power concentration observed
at fixed 20-degree geometry is an artefact of training on one geometry. One PPO
policy sees a uniform episode-level choice among symmetric 5, 20 and 60 degree
ULA angle separations. Geometry is fixed within an episode, preserving the
Gauss--Markov process around one LoS mean. The channel observation contains the
complex estimated channel, so the actor can infer the current spatial regime.

Five independently objective-pretrained actors were fine-tuned for 262,144 PPO
interactions. Evaluation uses six new paired seeds per fixed geometry, 8,192
slots and 2,048 burn-in. Conditional constants are estimated once from a
separate mixed-geometry calibration set and reused unchanged at all three test
geometries. They are not retuned per geometry.

## What the full policy does

| Separation | Common-on | Common power `alpha` given on | `rho` given on | Common share to user 0 given on |
|---:|---:|---:|---:|---:|
| 5 degrees | 0.937 | 0.874 | 0.954 | 0.513 |
| 20 degrees | 0.571 | 0.770 | 0.827 | 0.553 |
| 60 degrees | 0.015 | 0.339 | 0.235 | 0.722 |

The actor clearly identifies geometry through its gate: it nearly always uses
the common stream at 5 degrees, uses both modes at 20 degrees, and almost always
selects the exact SDMA boundary at 60 degrees. Conditional magnitude also
changes, but the 60-degree conditional means are based on only 1.5% common-on
states and have little effect on its long-run performance.

The per-checkpoint global calibration averages are `alpha=0.824` and
`rho=0.892` when the common stream is on.

## Controlled compression tests

| Control | 5 degrees | 20 degrees | 60 degrees |
|---|---:|---:|---:|
| Full PPO mean AoI | 1.5150 | 1.4490 | 1.0402 |
| One global fixed `alpha,rho` | +0.31% | -0.27% | +0.54% |
| Per-slot common-recipient argmax | -0.26% | -0.10% | -0.003% |
| Both fixed `alpha,rho` and recipient argmax | -0.05% | -0.56% | +0.39% |

Replacing the continuous common-rate simplex by its per-slot argmax allocates
the common codeword to one user's split message. It preserves or slightly
improves performance at every geometry. Combining this discrete recipient
choice with the single global common-power/payload magnitudes remains within
0.56% of the full actor everywhere.

Fixing the common-rate vector at its arithmetic mean is not a valid substitute:
it raises AoI by 468% at 5 degrees and 177% at 20 degrees. Under the current
`simplex_scaled` mapping, an equal vector and high `rho` approximately doubles
the total common payload relative to a one-hot vector. The resulting failure is
a finite-blocklength payload cliff, not evidence that fractional common-rate
values are required. The argmax control isolates this nonlinearity.

## Supported conclusion

The fixed-geometry-artifact explanation is rejected for the tested model. A
single policy trained across 5/20/60 degrees still obtains its performance from:

1. a state-dependent exact common-off gate;
2. a discrete choice of which user's split update occupies the common stream;
3. continuous state-dependent private-power allocation; and
4. approximately fixed high common power and payload when the gate is on.

Thus the effective controller switches between the exact SDMA boundary and an
asymmetric NOMA-like RSMA corner. The full action space can represent interior
RSMA allocations, and its raw outputs vary continuously, but those continuous
common magnitudes and fractional common-rate values are not needed for the
measured long-run AoI.

This is a structural result of the current `K=Nt=2`, fixed-beam, equal-payload,
equal-weight, generate-at-will FBL model over the tested geometry set. It is not
a theorem about general RSMA. The `simplex_scaled` payload parameterisation,
short-packet decoding cliff and fixed precoders are plausible causes and must
be named as scope limitations.

The mixed policy is still not permutation equivariant on the offline exchange
test. Geometry randomisation reduces common-power residuals but gate agreement
is only 0.53--0.62. This remains a separate robustness issue; it does not alter
the paired compression result.

## Artifacts

- Mixed channel config: `configs/thesis_rician_mixed_geometry_csit02.yaml`
- Training: `results/rician_ppo_mixed_geometry_csit02_262k_v1/`
- Main audit: `results/thesis_mixed_geometry_audit_v1/`
- Rate-argmax audit: `results/thesis_mixed_rate_argmax_v1/`
- Offline state dependence: `results/action_state_dependence_mixed_geometry_v1/`
- Reproduction scripts: `scripts/run_mixed_geometry_study.py`,
  `scripts/audit_mixed_geometry_policy.py`, `scripts/audit_mixed_rate_argmax.py`
