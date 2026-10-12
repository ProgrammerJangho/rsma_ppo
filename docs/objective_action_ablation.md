# Objective-aware PPO action ablation

> **Interpretation update (2026-09-10).** The original power-only and split-only
> controls use particular fixed default allocations. They show that those
> defaults are poor, but do not prove that every common allocation must vary by
> state. The 262k audit in `two_mode_policy_audit.md` shows that fixing common
> power, total common payload, and user common-rate proportions costs only 0.36%
> when state-dependent private-power proportions and the gate are retained.
> Claims of necessary continuous joint common-rate control are superseded.

## Question

This experiment tests whether PPO should control the RSMA power allocation,
common-rate/payload allocation, or both.  The representative operating point is
the two-user 20-degree directional-Rician channel with relative CSIT error
`sigma_e=0.2`.  Every learned RSMA arm uses its own three-seed FBL-AoI objective
pretraining followed by 65,536 PPO interactions.  Evaluation uses the same three
4,096-slot trajectories after a 1,024-slot burn-in.

The action arms are:

| Arm | Learned by PPO | Fixed or discrete part |
|---|---|---|
| `power_only` | common/private power simplex | equal common-rate allocation and common-payload fraction `rho=0.3` |
| `split_only` | common-rate allocation and `rho` | common-power fraction 0.25 and equal private power |
| `joint` | power simplex, common-rate allocation, and `rho` | common always on |
| `joint_gate` | joint continuous action | exact common on/off Bernoulli gate |
| `sdma_power` | private-power allocation | exact common-off policy class |

The first attempted run inherited `common_gate_enabled=true` from the environment
configuration in nominally non-gated arms.  `power_only` and `split_only` were
rejected by the trainer, while the nominal `joint` arm ran with a gate.  Those
`v1` directories are invalid and excluded.  The runner now explicitly forces
the gate off for every mode except `joint_gate`; the valid results are `v2`.

## Long-run result

| Action arm | Average AoI | Mean pooled p95 | Success rate | Common power | `rho` | Common-on rate |
|---|---:|---:|---:|---:|---:|---:|
| `power_only` | 3.0426 ± 0.0446 | 8.00 | 0.2794 | 0.3982 | 0.3000 fixed | 1.000 |
| `split_only` | 2.6853 ± 0.1795 | 8.78 | 0.5438 | 0.2500 fixed | 0.1407 | 1.000 |
| `joint` | 1.5881 ± 0.0617 | 2.67 | 0.5173 | 0.3512 | 0.2525 | 1.000 |
| `joint_gate` | **1.5517 ± 0.0082** | **2.00** | 0.5067 | 0.3550 | 0.3426 | 0.466 |
| `sdma_power` | 1.6954 ± 0.1653 | 2.89 | 0.4920 | 0 | 0 | 0 |

Paired AoI differences are:

| Candidate minus reference | Mean ± seed SD | Candidate wins |
|---|---:|---:|
| `joint - power_only` | -1.4545 ± 0.0934 | 3/3 |
| `joint - split_only` | -1.0972 ± 0.1198 | 3/3 |
| `joint_gate - joint` | -0.0364 ± 0.0608 | 2/3 |
| `joint_gate - sdma_power` | -0.1438 ± 0.1680 | 3/3 |

The direct conclusion from these original arms is that the particular fixed
power and common-rate/payload controls leave a large AoI loss. They justify the
joint action space but do not establish that every component must remain
state-dependent after convergence. The exact gate
has a small and seed-variable mean effect, but reduces mean p95 from 2.67 to 2.0
and greatly reduces training-seed variation.  More seeds are required before
claiming a mean-AoI gate gain.

## Why success rate and AoI disagree

`split_only` has the highest aggregate success rate but poor AoI.  User-level
results show that its user 1 has empirical success rate about 0.505 and mean
inter-delivery time about 1.98 slots, yet its mean maximum observed
inter-delivery interval is 34.4 slots and p99 AoI is 19.4.  Successful updates
are temporally clustered and long failure bursts dominate the age process.
This is direct evidence that a one-step expected-success objective cannot by
itself predict long-run AoI.

`power_only` retains `rho=0.3` and equal common-rate allocation.  Its mean common
BLER when enabled is 0.364, compared with 0.018 for joint and 0.013 for
joint-plus-gate.  The fixed payload split frequently places the common codeword
on the steep, unreliable side of the finite-blocklength transition.

## Limits on the claim

The first table used one fixed setting in each restricted arm.  A follow-up
sensitivity sweep below tests whether that choice alone caused the gap.

The SDMA arm is a separately trained learned-power policy, rather than a
state-by-state counterfactual projection of the joint RSMA policy.  Its large
three-seed variance also prevents a general RSMA-over-SDMA claim.  The result is
evidence for this channel and training budget only.

Artifacts are in
`results/rician_ppo_20deg_csit02_objective_action_ablation_v2/`.

## Fixed-parameter sensitivity

Power-only PPO was retrained at fixed
`rho in {0.05,0.10,0.20,0.30}`.  Split-only PPO was retrained at fixed common
power in `{0.10,0.25,0.40,0.60}`.  Every new point uses independent
objective-pretraining seeds 0--2 followed by the same 65,536-step PPO and
long-run evaluation protocol.  The full curves are reported; the best point is
not used as an unbiased held-out hyperparameter selection.

| Restricted action | Fixed value | Average AoI | p95 | Common BLER when on |
|---|---:|---:|---:|---:|
| power only | `rho=0.05` | **1.8680 ± 0.0710** | 4.11 | 0.029 |
| power only | `rho=0.10` | 2.2276 ± 0.0474 | 5.78 | 0.078 |
| power only | `rho=0.20` | 2.9334 ± 0.0391 | 7.78 | 0.275 |
| power only | `rho=0.30` | 3.0426 ± 0.0446 | 8.00 | 0.364 |
| split only | common power 0.10 | 2.8065 ± 0.0198 | 10.00 | 0.020 |
| split only | common power 0.25 | 2.6853 ± 0.1795 | 8.78 | 0.033 |
| split only | common power 0.40 | **2.5849 ± 0.1600** | 7.67 | 0.061 |
| split only | common power 0.60 | 2.6328 ± 0.1286 | 6.89 | 0.109 |

The lowest-AoI power-only point remains 0.2799 ± 0.0741 worse than joint and
loses all three seed pairs.  The lowest-AoI split-only point remains
0.9969 ± 0.0989 worse than joint and also loses all pairs.  Thus the large joint
gain is not explained by the original `rho=0.3` or common-power 0.25 choices
within the tested grid.

The power-only curve also quantifies the finite-blocklength sensitivity.  As
fixed `rho` increases from 0.05 to 0.30, common BLER rises from about 0.029 to
0.364 and AoI rises from 1.868 to 3.043.  A fixed payload fraction cannot adapt
well under these 65k policies and tested constants. For split-only,
changing fixed common power improves the result only modestly and no tested
value approaches joint control.

The later 262k nested audit uses constants calibrated from each converged actor
on disjoint trajectories. It finds that fixed common power, `rho`, and
common-rate proportions match the full actor when the gate and continuous
private-power proportions are retained. Therefore this grid must not be cited
as proof that state-dependent common-rate/payload decisions are necessary.

These conclusions remain local to the tested grid, channel geometry, two-user
model, and training budget.  A continuous proof that no fixed parameter can
match joint control is neither claimed nor needed for the action-design
ablation.

Sensitivity artifacts are in
`results/rician_ppo_20deg_csit02_fixed_action_sensitivity_v1/`.
