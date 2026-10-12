# Matched objective-pretraining audit

## Question and protocol

This experiment removes the main initialization confound in the thesis
comparison.  Both fixed-beam RSMA and exact common-off SDMA use independently
trained, model-based objective-pretraining seeds before the same 65,536-step PPO
fine-tuning budget.

The frozen protocol is
`configs/thesis_matched_initialization_evaluation.yaml`: directional Rician,
20-degree separation, SNR 10 dB, relative CSIT error 0.2, blocklength 128 and
256 bits/user.  It uses five matched training/pretraining seeds and six new
paired channel/CSIT/ACK trajectories (930000--930005), each 8,192 slots with a
2,048-slot burn-in.  These evaluation seeds were selected before inspecting
the results and must not be used for further hyperparameter tuning.

SDMA pretraining optimises only the private-power simplex.  Common power and
common payload are exactly zero in both pretraining and deployment.  Its loss
uses the same differentiable FBL expected next-AoI objective as RSMA, but does
not imitate the RSMA CEM reference action.

## Results

| Method | Mean AoI | Training-seed SD | p95 AoI | Success rate |
|---|---:|---:|---:|---:|
| RSMA objective pretraining, PPO 0 step | 1.5508 | 0.0076 | 2.0 | 0.5010 |
| RSMA objective pretraining + PPO | **1.5475** | 0.0061 | 2.0 | 0.5077 |
| SDMA objective pretraining, PPO 0 step | 1.5824 | 0.0015 | 2.0 | 0.4849 |
| SDMA objective pretraining + PPO | 1.5751 | 0.0130 | 2.0 | 0.4890 |
| Scratch RSMA + PPO | 1.6928 | 0.0467 | 3.13 | 0.4705 |
| Scratch SDMA + PPO | 1.6437 | 0.1267 | 2.53 | 0.4892 |

The paired five-seed contrasts are:

| Contrast (candidate versus reference) | Mean AoI change | Mean percent | Candidate wins |
|---|---:|---:|---:|
| RSMA versus SDMA at PPO step 0 | -0.0316 | **-2.00%** | 5/5 |
| RSMA versus SDMA after PPO | -0.0276 | **-1.75%** | 5/5 |
| PPO versus step 0 for RSMA | -0.0033 | -0.21% | 4/5 |
| PPO versus step 0 for SDMA | -0.0073 | -0.46% | 3/5 |
| Objective-warm versus scratch RSMA final | -0.1453 | -8.54% | 5/5 |
| Objective-warm versus scratch SDMA final | -0.0685 | -3.76% | 4/5 |

## Interpretation

The previous 5.7% headline difference was inflated by comparing objective-warm
RSMA with scratch SDMA.  After matching initialization, RSMA retains a smaller
1.75--2.00% AoI advantage and wins all five paired seeds.  The result supports
a conditional fixed-beam RSMA action-class benefit at this operating point; it
does not establish general RSMA superiority.

Objective pretraining is the dominant learning component.  It improves the
final RSMA result by 8.54% relative to scratch PPO and substantially reduces
the unstable scratch-SDMA outcome.  PPO adds only 0.21% for RSMA and 0.46% for
SDMA on average.  PPO therefore cannot be presented as the primary source of
the performance gain under the current 65k budget.

The clean thesis account is:

1. the constrained RSMA action class yields a modest matched advantage over
   SDMA at the representative point;
2. differentiable objective pretraining makes both action classes trainable and
   stable across seeds;
3. PPO provides only a small subsequent correction in the current model; and
4. online CEM remains substantially better in AoI, while learned actors avoid
   iterative online search.

Five training seeds are enough for a bounded bachelor-thesis comparison but
not for a broad asymptotic or journal-level statistical claim.  The 5/5 signs
and paired design should be reported alongside the effect size rather than
using the bootstrap interval as definitive proof.

## Artifacts

- SDMA pretraining checkpoints: `runs/objective_actor_sdma_20deg_csit02_seed*_v1/`
- Warm SDMA PPO checkpoints:
  `results/rician_ppo_20deg_csit02_sdma_objective_warm_5seed_v2/`
- Frozen protocol: `configs/thesis_matched_initialization_evaluation.yaml`
- Long-run results: `results/thesis_matched_initialization_v4/`
- Reproducible contrasts: `scripts/summarize_matched_initialization.py`

