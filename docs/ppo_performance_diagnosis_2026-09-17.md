# PPO performance diagnosis (2026-09-17)

This note records the PPO-only audit for the canonical gate-free K=3, Nt=2
allocation problem.  No new CEM search or CEM comparison was used.

## Reward definition

With the canonical settings (`reward_scale=1/3`, `lambda_power=0`, and
`lambda_tail=0`), the realised reward is exactly

\[
r_t=-\frac{1}{3}\sum_{k=1}^{3}\Delta_k(t+1),
\]

so maximising discounted return directly minimises next-slot mean AoI over the
trajectory.  An optional conditional expected-AoI reward was implemented and
tested, but it worsened held-out AoI by 1.01%, 3.85%, and 1.05% for the three
training seeds.  The canonical experiment therefore retains sampled AoI.

## Failure found in the original warm-start PPO

The action+objective pretrained actor restores the observation running mean and
variance saved during pretraining.  The original PPO rollout immediately
updated those statistics.  As a result, the same physical state was mapped to
a changing normalised input even when actor parameters were frozen.  On six
held-out trajectories, 65k PPO with this behaviour degraded mean AoI from
1.5484 to 1.8403 (18.85%).

Narrowing the initial action standard deviation or adding a critic-only warm-up
did not fix the regression.  They worsened mean AoI by 53.40% and 48.28%,
respectively.  These negative controls rule out action noise and a randomly
initialised critic as the primary cause.

The PPO configuration now supports `freeze_observation_stats`.  When enabled,
the saved pretraining input scale remains fixed while PPO updates the actor and
critic.

## Corrected PPO result

All values below use the same six held-out environment seeds, 2048 steps per
seed, and a 512-step burn-in.

| Policy | Seed-averaged mean AoI | Gain vs warm actor | Paired wins |
|---|---:|---:|---:|
| Action+objective warm actor | 1.5484 | -- | -- |
| PPO 65k, frozen observation stats | 1.5365 | 0.77% | 6/6 |
| PPO 262k, frozen observation stats | **1.4755** | **4.71%** | **6/6** |
| PPO 262k, frozen stats, critic LR 1e-3 | **1.4677** | **5.21%** | **6/6** |
| PPO 65k, updating observation stats | 1.8403 | -18.85% | 0/6 |

The 262k seed results are 1.5331, 1.4586, and 1.4349.  The corresponding warm
actor results are 1.5853, 1.4964, and 1.5636.  Thus the average improvement is
not caused by one lucky training seed.

A one-factor PPO audit then compared critic LR `1e-3`, zero entropy bonus, and
a 512-step rollout.  At 65k, their gains over the corrected canonical run were
0.60%, 0.32%, and 0.06%, respectively.  At 262k, critic LR `1e-3` reduced AoI
from 1.4755 to 1.4677 and won 6/6 paired trajectories.  Combining it with zero
entropy worsened AoI to 1.4829.  The selected configuration therefore keeps
`entropy_coef=0.005` and `rollout_steps=128`, and changes only `critic_lr` to
`1e-3`.

## Reward convergence

For the corrected 262k runs, the initial and final 10% moving AoI values were
2.0313 to 1.6706, 1.8190 to 1.5488, and 1.9769 to 1.5434.  The final 32-update
moving averages were within 0.06%, 0.48%, and 0.33% of their best values.  Tail
slopes were -0.022, -0.032, and -0.065 AoI per 100k steps.  This supports a
plateau/slow-improvement interpretation rather than divergence.

With critic LR `1e-3`, tail explained variance is 0.072, 0.139, and 0.124.  It
remains modest, so the critic should not be described as an accurate AoI
predictor, but it is positive and substantially more useful than the original
critic.

## State-dependent actions

The corrected policy changes allocation strongly with AoI:

- correlation of a user's private-power share with its own AoI ranges from
  -0.029 to 0.569 across seeds;
- correlation of its common-payload fraction with its own AoI ranges from
  0.615 to 0.776;
- common-power fraction also varies substantially (`std=0.062` to `0.180`).

Direct response to instantaneous channel gain remains weak: private-power/gain
correlations are 0.012 to 0.069, while common-payload/gain correlations are
-0.139 to -0.042.  The policy therefore learns a clear freshness response, but
the evidence does not support claiming rich fast-fading adaptation.  Under the
current fixed large-scale geometry and temporally correlated Rician channel,
most of its useful state dependence comes from AoI.

## Reproducible artifacts

- `results/ppo_k3_frozen_obs_budget_comparison_v1/`: matched held-out policy comparison
- `results/ppo_k3_frozen_obs_262k_diagnostics_v1/`: convergence and action audit
- `results/ppo_k3_critic_262k_comparison_v1/`: final critic/entropy comparison
- `results/ppo_k3_critic1e3_262k_diagnostics_v1/`: selected-policy diagnostics
- `results/ppo_k3_improvement_pilot_v1/`: one-factor PPO improvement pilot
- `results/ppo_k3_reward_pilot_v1/`: sampled versus expected reward
- `results/ppo_k3_low_exploration_comparison_v1/`: narrow-exploration negative control
- `results/ppo_k3_conservative_comparison_v1/`: conservative update and critic-warm negative controls
- `scripts/run_k3_direct_frozen_obs_ppo_262k.sh`: corrected three-seed training
- `scripts/audit_direct_ppo_learning.py`: reward/action diagnostics

The canonical thesis result should use sampled-AoI reward, the action+objective
warm start, fixed observation-normalisation statistics, actor/critic learning
rates `3e-4/1e-3`, entropy coefficient `0.005`, 128-step rollouts, and 262144
PPO steps.
