# Final experiment matrix

> **역사 기록 (2026-09-22):** 이 matrix는 gate/`rho` action 세대의 최종
> 실험 계획이다. 현재 gate-free K=3, Nt=2 논문의 동결 결과는
> `thesis_results_chapter_2026-09-22.md`와
> `results/thesis_mixed_phy_final_v1/manifest.json`을 기준으로 한다.

## Decision

Three additions are retained, with different statistical scope.

1. Spatial regime comparison is mandatory. Reuse the existing three-seed
   objective-warm RSMA policies and add matched SDMA-power PPO at 5° and 60°.
2. Near/far is mandatory as one representative stress test because earlier
   heterogeneous results used mismatched screening and deployment policies.
   Use 20°, a 5 dB gain gap and relative CSIT error 0.2.
3. Expand only the 20° homogeneous headline methods to five full training
   seeds. Ten seeds and a full crossed pretraining/PPO seed design are outside
   the undergraduate-thesis scope.

## Runs to reuse

| Condition | Method | Existing training seeds |
|---|---|---|
| 5°, perfect CSIT | objective-warm joint+gate PPO | 0, 1, 2 |
| 20°, CSIT 0.2 | objective-warm joint+gate PPO | 0, 1, 2 |
| 20°, CSIT 0.2 | scratch joint+gate PPO | 0, 1, 2 |
| 20°, CSIT 0.2 | SDMA-power PPO | 0, 1, 2 |
| 60°, perfect CSIT | objective-warm joint+gate PPO | 0, 1, 2 |

## Completed additions

| Purpose | Condition | Method | Added seeds |
|---|---|---|---|
| Spatial matched control | 5° | SDMA-power PPO | 0, 1, 2 |
| Headline five-seed result | 20°, CSIT 0.2 | objective-warm, scratch, SDMA-power PPO | 3, 4 |
| Spatial matched control | 60° | SDMA-power PPO | 0, 1, 2 |
| Heterogeneous stress test | 20°, CSIT 0.2, gain gap 5 dB | objective-warm joint+gate and SDMA-power PPO | 0, 1, 2 |
| Numerical reference | 20°, CSIT 0.2 | 262k RSMA/SDMA PPO, raw RSMA/SDMA CEM and safeguarded CEM | 5 PPO seeds, 6 evaluation trajectories |

The spatial and heterogeneous studies use three training seeds because they are
secondary regime tests. The headline uses five. Every final comparison uses the
same deterministic held-out channel/CSIT/ACK trajectories, an 8,192-slot
horizon and a 2,048-slot burn-in.

The near/far experiment uses `beta_normalization=mean_one`. It changes relative
user strength while preserving the average large-scale channel power, so it
does not silently turn into an overall-SNR comparison.

All rows above were completed on 2026-09-09. Consolidated numerical results and
their interpretation are in `docs/final_regime_results.md` and
`results/thesis_regime_summary_v1/`.

The final numerical-reference row was completed on 2026-09-15 using the same
930000--930005 seeds, 8,192-slot horizon and 2,048-slot burn-in as the frozen
headline protocol.  Its interpretation is in `docs/cem_262k_long_results.md`.
