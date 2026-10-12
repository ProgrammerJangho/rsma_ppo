# Frozen main evaluation protocol

> **Superseded learned-policy headline (2026-09-10).** This document records the
> former 65k protocol. Its RSMA/SDMA comparison used unmatched initialization
> and must not be cited as the action-class effect. The current matched
> five-seed 0/65k/262k result is in `training_budget_results.md`; the final
> 262k values are RSMA 1.4574 and SDMA 1.5610 (-6.64%, 5/5 matched seeds).

## Purpose

This evaluation answers the central action-design question at the representative
20° separation and imperfect-CSIT point. It reuses existing checkpoints and
changes only the held-out evaluation horizon. It is therefore an evaluation
audit rather than another training generation.

## Frozen conditions

- Directional Rician MU-MISO: `K=2`, `Nt=2`, K-factor 8 dB
- User angles: -10° and +10°
- Gauss–Markov coefficient: 0.8
- Relative CSIT error: `sigma_e=0.2`
- SNR: 10 dB
- FBL: blocklength 128, payload 256 bits/user
- RZF private beam and principal common beam
- Five training seeds and six paired held-out seeds: 920000–920005
- 8,192 slots per trajectory; first 2,048 slots removed as burn-in
- Deterministic checkpoint action
- Actor/critic training generation: `3e-4/3e-4`

The exact environment, paths and training seeds are in
`configs/thesis_final_evaluation.yaml`. The runner rejects missing checkpoints,
duplicate training seeds, different PHY environments and unexpected PPO action
heads. Every checkpoint hash is stored in the result manifest.

## Main result

| Method | Mean AoI | Standard deviation | 95% bootstrap interval | Unit |
|---|---:|---:|---:|---|
| Objective-warm joint+gate RSMA-PPO | 1.5511 | 0.0089 | [1.5441, 1.5580] | 5 training seeds |
| SDMA-power PPO | 1.6448 | 0.1280 | [1.5838, 1.7598] | 5 training seeds |
| Scratch joint+gate RSMA-PPO | 1.6960 | 0.0448 | [1.6607, 1.7297] | 5 training seeds |
| Equal SDMA | 2.7950 | 0.1901 | [2.6554, 2.9364] | 6 evaluation trajectories |
| Fixed RSMA (`f_c=0.25`, `rho=0.30`) | 16.9793 | 0.7665 | [16.4161, 17.5225] | 6 evaluation trajectories |

Learned-policy intervals use training-seed means as the uncertainty unit. Fixed
methods have no training seed, so their intervals describe held-out trajectory
variation. These intervals must not be presented as identical sources of
uncertainty.

After averaging the five trained policies within each method on every held-out
trajectory, objective-warm RSMA-PPO beats SDMA-power PPO on all six trajectories.
Its paired mean AoI difference is -0.0937, or about 5.7% relative to SDMA-power
PPO. Scratch RSMA-PPO is 0.0513 worse than SDMA-power PPO on average.

## Supported interpretation

The comparison supports the joint claim that the common/private power and
common-rate action is useful **when paired with objective-aware initialization**.
It does not show that adding an RSMA action space automatically improves PPO:
the scratch joint policy fails to beat the learned SDMA boundary.

The fixed RSMA result is deliberately retained. Its common BLER is about 0.872,
showing that the fixed payload split operates beyond the useful FBL region. It
is a weak heuristic baseline and evidence of the FBL cliff, not evidence that
RSMA itself is structurally inferior.

The headline result is one operating point. Spatial, near/far and SNR stress
tests are documented separately. The bounded-compute CEM reference is reported
in `cem_reference_results.md` because its shorter trajectory and per-slot
numerical optimisation cost differ from the headline protocol.
