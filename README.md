# RSMA–AoI PPO thesis project

This repository studies finite-blocklength 1-layer RSMA for downlink AoI
minimisation.  The current thesis scope is deliberately narrow: a two-antenna
BS serves fixed-size status updates over a directional Rician channel, and PPO
chooses a common-power fraction ``alpha``, a private-power simplex ``q_k`` and
per-user common-payload fractions ``phi_k``.  RZF private beams and the
principal common beam are fixed so the experiment isolates resource allocation.

The supported research claim is that objective-aware pretraining, causal
PHY-aware observations and staged PPO training learn a state-dependent RSMA
allocation in the overloaded `K=3, N_t=2` system.  Five independent training
seeds support the target and feasible-regime results.  The project does not
assume that RSMA always beats SDMA; near-infeasible short-blocklength and
low-SNR conditions are retained as unresolved stress regimes.

## Current experiment

- System: `K=3`, `N_t=2`, overloaded MU-MISO downlink, 1-layer RSMA.
- Channel: mean-power-preserving Rician ULA with Gauss–Markov time evolution.
- PHY: normal-approximation FBL BLER, perfect CSIR, configurable imperfect CSIT.
- Traffic: generate-at-will; successful delivery resets AoI to one slot.
- PPO action: ``alpha``, private-power simplex ``q_k`` and independent payload
  fractions ``phi_k``.  There is no learned gate and no ``rho`` action.
- PPO optimiser: Stage 1 actor/critic LR `3e-4/1e-3`, target fine-tuning LR
  `1e-4/5e-4`, reward normalisation and clipped value loss.
- Primary evaluation: deterministic policies on held-out common-random-number
  trajectories; sampled rollouts are training diagnostics.

The canonical configurations are
`configs/thesis_rician_k3_nt2_mixed_phy_sinr.yaml`,
`configs/thesis_rician_k3_nt2_mixed_phy_sinr_stage1.yaml` and
`configs/thesis_rician_k3_nt2_20deg_sinr_finetune.yaml`.  The frozen tables,
figures and checkpoint hashes are under
`results/thesis_mixed_phy_final_v1/`. Older `simplex_scaled`, gate and `rho`
configurations are retained only to reproduce results generated before the
action redesign.

## Repository map

| Path | Purpose |
|---|---|
| `src/rsma_aoi/` | Environment, channel, PHY, PPO/TD3 and evaluation library |
| `configs/` | Executable experiment configurations; `thesis_*` is the current family |
| `scripts/` | Training, evaluation and reproducible audit entry points |
| `tests/` | Unit and integration tests |
| `docs/` | Current model decisions and experiment conclusions |
| `datasets/` | Current oracle datasets used for objective-aware pretraining |
| `runs/` | Current reusable pretraining checkpoints |
| `results/` | Current thesis evidence and compact study indexes |
| `references/` | Original project documents supplied by the researcher |
| `archive/` | Preserved legacy project, superseded documents and old experiments |

Start with [`docs/README.md`](docs/README.md).  Files under `archive/` are
historical evidence and are not the source of truth for the current model.

## Verify the project

From WSL:

```bash
cd /home/alexalex2/rsma_ppo
.venv/bin/python -m pytest -q
```

The full implementation suite passed 616 tests after the action redesign. The
latest focused environment/configuration verification passed 159 tests after
the five-seed artifact build.

## Reproduce the frozen training pipeline

```bash
SEEDS="0 1 2 3 4" scripts/run_mixed_phy_seed_expansion.sh pretrain
SEEDS="0 1 2 3 4" scripts/run_mixed_phy_seed_expansion.sh stage1
SEEDS="0 1 2 3 4" scripts/run_mixed_phy_seed_expansion.sh target

.venv/bin/python scripts/evaluate_mixed_phy_seed_expansion.py \
  --out-dir results/<new_target_result_directory>
.venv/bin/python scripts/evaluate_mixed_phy_multiseed_grid.py \
  --out-dir results/<new_grid_result_directory>
```

The runner skips completed checkpoints. Evaluation output directories must be
new. Every formal comparison uses independent training seeds and identical
held-out channel/CSIT/ACK seeds.

## Scope boundaries

Scheduling, stochastic arrivals, queues, HARQ, blocklength control, learned
precoders and explicit Digital Twin state mismatch are outside the current core
experiment.  OMA/NOMA, TD3 and older heterogeneous studies remain available as
secondary or archived baselines; they do not define the current thesis model.
