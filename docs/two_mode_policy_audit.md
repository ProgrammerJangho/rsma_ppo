# Two-mode simplification audit of the 262k RSMA policy

## Motivation and protocol

The 262k actor frequently uses high common power and payload when its exact
common gate is on. This raised the possibility that its continuous action heads
had collapsed to a switch between a fixed SDMA action and a fixed
common-dominant action.

For each of five checkpoints, conditional action means were estimated on
calibration seeds 940000--940002 (4,096 slots, 1,024 burn-in). Performance was
then measured on the disjoint frozen seeds 930000--930005 (8,192 slots, 2,048
burn-in). Every control retains the original actor's gate decision.

| Policy | State-dependent quantities retained |
|---|---|
| PPO original | gate and all continuous allocations |
| Fixed magnitude | gate, private-power proportions, common-rate proportions |
| Dynamic private only | gate, private-power proportions |
| Dynamic rate only | gate, common-rate proportions |
| Private argmax | gate, discrete private-user argmax |
| Fixed full action | gate only |

## Long-run result

| Policy | Mean AoI | Change from PPO | PPO wins |
|---|---:|---:|---:|
| Fixed magnitude | **1.4539** | -0.24% | 0/5 |
| PPO original | 1.4574 | reference | -- |
| Dynamic private only | 1.4625 | +0.36% | 4/5 |
| Private argmax | 1.5910 | +9.17% | 5/5 |
| Dynamic rate only | 1.7605 | +20.80% | 5/5 |
| Fixed full action | 1.7730 | +21.66% | 5/5 |

Common power and total common payload can be fixed without loss. Fixing the
common-rate proportions as well costs only 0.36% if continuous private-power
proportions are retained. Conversely, retaining dynamic common-rate proportions
while fixing power loses 20.8%. Replacing continuous private power by only its
argmax loses 9.17%.

The supported conclusion at this operating point is:

> The learned gate and continuous user-specific private-power allocation are
> material. State-by-state variation of common-power magnitude, total common
> payload, and user common-rate proportions is not material.

This invalidates the broad interpretation that continuous joint power and
common-rate control is necessary. The old power-only/split-only ablation used
particular fixed defaults; it did not use a calibrated fixed common split.

## Payload semantics and the “multicast” claim

The implemented `simplex_scaled` mapping is

\[
D_c=\rho\min_k D_k/c_k,\qquad D_{c,k}=c_kD_c,\qquad
D_{p,k}=D_k-D_{c,k}.
\]

It is not `rho` times the sum of all users' payloads. On the separate
calibration trajectories, the actors allocate roughly 94.5--97.5% of the
common rate to user 0. Their conditional mean actions produce approximately
219--255 total common bits, private payloads of 9--42 bits for user 0, and
242--250 bits for user 1. The claimed “448 common bits and 32 private bits per
user” does not follow the implemented mapping.

All users decode the common codeword for SIC, but its contents are predominantly
a split part of one user's unicast update. This is not multicast traffic. It is
an asymmetric, common-dominant 1-layer RSMA solution close to the two-user NOMA
boundary: almost all of one user's message is decoded by both users while the
other user's message remains private.

## Thesis implication

The PPO action can remain common/private power plus common-rate allocation. The
result must be reported honestly: PPO learns a useful asymmetric common split,
but its slot-to-slot common-rate variation is redundant in the representative
fixed-geometry regime. The contribution should emphasize objective pretraining,
exact mode selection, and continuous private-power adaptation.

If state-dependent common-rate allocation must be central, a regime where the
useful common recipient changes over time is needed, such as time-varying user
geometry, heterogeneous AoI weights, or asymmetric payloads. That is a new
experiment and cannot be inferred from the present result.

The exact alignment correlations quoted in the external diagnosis had no
repository script or raw table, so those numbers are not yet independently
reproducible. CEM and the 262k actor should also be evaluated on one identical
protocol before comparing their numerical gap or conditional actions.

## Artifacts

- Reproducible audit: `scripts/audit_two_mode_policy.py`
- Consolidated result: `results/thesis_two_mode_audit_v4/`
- Paired effects: `results/thesis_two_mode_audit_v4/paired_contrasts.csv`
- Calibration constants: `results/thesis_two_mode_audit_v4/calibration_constants.json`
