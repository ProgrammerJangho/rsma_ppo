# 대칭 관측·CEM action 사전학습 후속 실험 — 2026-09-16

## 1. 질문과 통제

기존 mixed geometry/SNR PPO는 별도 SDMA actor보다 평균 AoI가 낮았지만, 같은 actor를
common-off boundary로 투영하면 오히려 더 좋아졌다. 따라서 기존 차이는 RSMA common
stream의 이득이 아니라 private-power 학습 품질 차이였다.

이번 실험은 그 진단에서 직접 확인된 두 원인만 수정했다.

1. raw complex channel 대신 global-phase 불변 `gain_features`를 사용했다.
2. `[-theta/2,+theta/2]`와 반대 사용자 순서를 모두 학습 분포에 넣고, safeguarded CEM의
   gate/allocation label과 differentiable FBL objective를 함께 쓰는
   `action_objective` 사전학습을 적용했다.

PHY, RZF private beam, principal common beam, `alpha/c/rho` action, 채널 조건과 PPO
budget은 유지했다. 따라서 결과 변화는 beam이나 PHY 변경으로 설명할 수 없다.

## 2. 사전학습은 의도한 문제를 고쳤다

Held-out test set에서 CEM label의 common-on 비율은 26.9%였다. 새 0-step actor는
32.2--33.2%를 출력했고, gate accuracy는 89.1--89.2%였다. 기존 objective-only actor의
gate accuracy 69--75%, seed별 common-on 4.1--27.6%보다 안정적이다.

사용자 교환 감사도 개선됐다.

| 정책 | Gate agreement | Common-rate 교환 오차 |
|---|---:|---:|
| 기존 0-step, 3 seeds | 71.9--98.8% | 0.500--0.973 |
| 새 action-objective 0-step | 93.6--94.4% | 0.116--0.129 |

따라서 phase-invariant observation, 사용자 순서 augmentation, action supervision은
실제로 index shortcut과 seed별 gate 붕괴를 줄였다.

## 3. PPO 후 일부 대칭성은 다시 약해졌다

262k PPO는 수치 오류 없이 완료됐고 마지막 critic explained variance는
0.836/0.989/0.951이었다. 그러나 common-on 비율은 seed별 17.1%, 6.1%, 3.9%로 다시
벌어졌다. Common-rate 교환 오차도 0.228/0.196/0.597로 증가했다.

따라서 representation/pretraining 수정은 충분조건이 아니다. PPO의 sampled long-run
objective가 CEM label과 다른 action으로 이동시키며, seed 2에서는 사용자 대칭성도
상당 부분 잃었다. 다만 아래 장기 결과상 이 이동 전체를 실패라고 볼 수는 없다.

## 4. 4096-slot 장기 결과

각 actor를 그대로 평가한 결과와, 같은 actor의 private-power 비율은 보존하면서
common power와 payload만 0으로 만든 paired boundary를 비교했다. 학습 seed 3개,
평가 seed 6개를 사용했다.

| Split | 정책 | 평균 AoI | Common-on | 같은 actor의 common-off | Common 기여 |
|---|---|---:|---:|---:|---:|
| Seen 9조건 | 0-step | 3.3767 | 34.2% | 3.0429 | **-10.97%** |
| Seen 9조건 | 262k PPO | **1.6965** | 9.3% | 1.7700 | **+4.15%** |
| Interpolation 4조건 | 0-step | **1.6178** | 32.3% | 1.9784 | **+18.23%** |
| Interpolation 4조건 | 262k PPO | **1.4791** | 8.1% | 1.5326 | **+3.49%** |

`Common 기여 = 100(AoI_common-off - AoI_actor)/AoI_common-off`이다. 양수면 common
stream이 AoI를 낮춘다. 비율은 trajectory별 비율의 평균이 아니라 먼저 AoI를 집계한
뒤 계산했다.

262k 정책은 세 학습 seed 모두에서 common 기여의 부호가 양수였다.

| Split | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Seen | +10.03% | +1.48% | +0.37% |
| Interpolation | +8.29% | +1.82% | +0.04% |

부호는 일관되지만 효과 크기는 seed 0에 크게 의존한다. 세 seed만으로 정밀한 효과
크기를 일반화해서는 안 된다.

## 5. RSMA operating regime이 나타났는가

262k 정책의 조건별 paired common 기여는 다음과 같다.

| Geometry | 5 dB | 10 dB | 15 dB |
|---|---:|---:|---:|
| 5° | +9.28% | +6.17% | +8.52% |
| 20° | +3.18% | +2.02% | +1.65% |
| 60° | -0.05% | +0.02% | -0.01% |

Interpolation에서는 10°에서 +6.97/+5.06%, 40°에서는 +0.11/-0.28%였다.

이는 물리적으로 해석 가능한 패턴이다. 정렬된 5°/10°에서는 common stream이 private
interference를 완화하고, 20°에서는 이득이 작아지며, 분리된 40°/60°에서는 gate가
거의 꺼져 SDMA boundary로 수렴한다. 기존 actor의 common contribution이 seen/unseen에서
-0.36/-0.24%였던 것과 달리, 이번 actor는 제한적이지만 실제 RSMA 동작을 학습했다.

## 6. 동일 프로토콜 CEM 비교

기존 CEM과 정확히 같은 seen 9조건, evaluation seed 3개, 2048 slots, burn-in 512로
새 actor를 다시 평가했다.

| 방식 | 평균 AoI | Common-on |
|---|---:|---:|
| Safeguarded RSMA-CEM, 1920 candidates | **1.3658** | 24.9% |
| SDMA-CEM, 960 candidates | 1.4496 | 0% |
| Raw RSMA-CEM, 960 candidates | 1.4937 | 100% |
| 새 symmetric 262k PPO | 1.6986 | 9.0% |
| 같은 PPO의 common-off | 1.7661 | 0% |

새 PPO에서 common stream은 3.82% 이득을 주지만, PPO AoI는 raw RSMA-CEM보다
13.72%, safeguarded CEM보다 24.37%, SDMA-CEM보다 17.18% 높다. 반대로 표현하면
safeguarded CEM은 PPO보다 19.59% 낮은 AoI를 기록한다.

Raw RSMA-CEM은 SDMA-CEM보다 3.04% 나쁘다. RSMA가 SDMA를 포함한다는 이론적 경계는
raw CEM의 유한 탐색만으로 자동 보장되지 않으며, exact SDMA candidate를 포함한
safeguard가 있어야 평균 5.78% 이득이 난다.

## 7. 냉정한 결론

이번 수정은 실패가 아니다. 기존 연구의 가장 큰 해석 오류였던 “별도 SDMA actor보다
좋으므로 common stream이 유용하다”를 paired control로 바로잡았고, 수정 후에는 실제로
common stream이 aligned geometry에서 AoI를 낮춘다. 하나의 정책이 geometry/SNR에 따라
RSMA와 SDMA boundary를 선택하는 특성도 나타났다.

그러나 “PPO가 RSMA allocation을 잘 최적화한다”는 주장에는 아직 부족하다. PPO가 얻은
common 이득은 작고 seed별 편차가 크며, 전체 AoI는 CEM 계열보다 높다. 현 상태에서
가능한 정확한 주장은 다음이다.

> Phase-invariant observation, user-order augmentation, FBL-aware action
> pretraining을 적용하면 mixed geometry/SNR PPO가 aligned regime에서 common stream을
> 선택하고, 동일 actor의 SDMA boundary보다 낮은 장기 AoI를 달성한다. 다만 bounded
> numerical optimization과의 격차는 남는다.

학부논문에서는 이 결과를 **RSMA 우월성**보다 **유한 블록길이에서 RSMA/SDMA mode를
학습하는 데 필요한 representation과 initialization**의 실증으로 쓰는 편이 안전하다.

## 8. 근거 파일

- 설정: `configs/thesis_rician_mixed_geometry_snr_symmetric_csit02.yaml`
- 사전학습: `scripts/run_symmetric_mixed_pretraining.sh`
- PPO: `scripts/run_symmetric_mixed_ppo.py`
- paired 평가: `scripts/evaluate_symmetric_mixed_followup.py`
- 0-step 감사: `results/action_state_dependence_symmetric_pretrain_v1/`
- 262k 감사: `results/action_state_dependence_symmetric_262k_v1/`
- 장기 평가: `results/thesis_symmetric_mixed_geometry_snr_v1/`
- CEM protocol 평가: `results/thesis_symmetric_mixed_geometry_snr_cem_protocol_v1/`

