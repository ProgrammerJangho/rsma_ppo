# Mixed geometry + SNR 확장 실험 — 2026-09-15

## 1. 질문과 결론

하나의 PPO가 각도 분리도와 SNR이 함께 바뀌는 분포를 학습하면 고정 조건별 PPO보다
일반적인 정책이 되는지, 그리고 그 정책이 SDMA 및 online CEM보다 낮은 장기 AoI를
달성하는지 확인했다.

결과는 두 층으로 나뉜다.

1. 별도로 학습한 `SDMA-power PPO`와 비교하면 mixed `RSMA joint+gate PPO`는 학습
   조건 평균에서 3.32%, 보간 조건 평균에서 4.18% 낮은 AoI를 기록했다. 세 training
   seed에서 모두 같은 부호였다.
2. 그러나 RSMA actor의 common stream만 끄고 private-power 비율을 그대로 유지한
   정책이 원 RSMA actor보다 학습 조건에서 0.36%, 보간 조건에서 0.24% 더 좋았다.
   따라서 위 3--4%를 common/private RSMA 분할의 이득으로 해석할 수 없다. 이는
   RSMA actor가 별도 SDMA actor보다 더 좋은 private-power mapping을 학습한 효과다.

한편 동일한 상태 정보로 매 slot 탐색한 safeguarded CEM은 SDMA-CEM보다 5.78%
낮은 AoI를 보였다. 현재 action class 안에는 유효한 RSMA operating point가 있지만,
262k mixed PPO는 이를 안정적으로 포착하지 못했다는 결론이 가장 정확하다.

## 2. 구현과 protocol

환경은 episode마다 다음 slow regime을 독립적으로 뽑고 512 slot 동안 유지한다.

- 사용자 각도 분리도: 5°, 20°, 60°
- SNR: 5, 10, 15 dB
- directional Rician `K=8`, Gauss--Markov `alpha=0.8`
- relative imperfect CSIT `sigma_e=0.2`
- `K=Nt=2`, blocklength 128, 사용자별 payload 256 bits
- fixed RZF private beam과 principal common beam

Small-scale fading과 CSIT error는 slot마다 변한다. SNR은 observation의 마지막 좌표에
`snr_db / 20`으로 포함한다. 환경, differentiable pretraining objective 및 CEM이 모두
episode의 실제 total power를 사용하도록 연결했다.

RSMA joint+gate와 SDMA power actor를 각각 3 seeds, 262,144 interactions로 학습했다.
두 방법 모두 같은 mixed oracle dataset으로 objective pretraining한 뒤 PPO를 적용했다.

장기 PPO 평가는 다음과 같다.

- seen: 3 geometry × 3 SNR의 9조건
- unseen interpolation: `(10°, 7.5 dB)`, `(10°, 12.5 dB)`,
  `(40°, 7.5 dB)`, `(40°, 12.5 dB)`
- 3 training seeds × 6 공통 evaluation trajectories
- trajectory당 4,096 slots, burn-in 1,024 slots
- 조건을 같은 가중치로 평균하고 training seed를 불확실성 단위로 사용

CEM 비교는 계산량 때문에 seen 9조건에서 3 trajectories, 2,048 slots, burn-in 512를
사용했다. 이 표 안의 PPO도 같은 짧은 trajectory로 다시 평가했다. CEM은 global optimum이
아닌 finite-budget one-step reference다.

## 3. PPO 장기 평가

| 분포 | RSMA joint+gate | 별도 SDMA-power | RSMA 상대 개선 | 조건별 승리 |
|---|---:|---:|---:|---:|
| seen 9조건 | 1.7024 ± 0.0279 | 1.7608 ± 0.0300 | 3.32% | 6/9 |
| unseen 4조건 | 1.5119 ± 0.0137 | 1.5777 ± 0.0263 | 4.18% | 3/4 |

`±`는 조건과 evaluation trajectory를 먼저 평균한 뒤 얻은 3개 training-seed 평균의
표준편차다. Seen에서 same-index seed별 RSMA 개선은 3.97%, 2.92%, 3.06%였고,
unseen에서는 5.79%, 3.38%, 3.32%였다.

| 각도 | SNR | RSMA PPO | SDMA PPO | RSMA 상대 개선 |
|---:|---:|---:|---:|---:|
| 5° | 5 dB | 2.7304 | 2.7024 | -1.03% |
| 5° | 10 dB | 1.6694 | 1.6467 | -1.38% |
| 5° | 15 dB | 1.5239 | 1.5422 | 1.19% |
| 20° | 5 dB | 2.5655 | 2.6169 | 1.96% |
| 20° | 10 dB | 1.5628 | 1.5656 | 0.18% |
| 20° | 15 dB | 1.2590 | 1.4717 | 14.45% |
| 60° | 5 dB | 1.8031 | 1.7968 | -0.35% |
| 60° | 10 dB | 1.1999 | 1.4205 | 15.53% |
| 60° | 15 dB | 1.0071 | 1.0842 | 7.12% |

60°에서 common gate를 거의 사용하지 않으면서 큰 차이가 난다는 사실이 common-off
projection 검사의 출발점이었다.

## 4. Common-off projection

각 RSMA checkpoint의 출력에서 common power와 common payload를 0으로 만들고,
그 checkpoint가 선택한 두 private-power 비율만 합이 1이 되도록 재정규화했다. 새로운
정책을 학습하거나 상수를 calibration하지 않았으며, 원 평가와 같은 trajectories를 썼다.

| 분포 | 원 RSMA | common-off projection | common stream 순개선 | projection vs 별도 SDMA |
|---|---:|---:|---:|---:|
| seen | 1.7024 | **1.6961** | **-0.36%** | +3.67% |
| unseen | 1.5119 | **1.5082** | **-0.24%** | +4.40% |

세 training seed 모두에서 common-off projection이 원 RSMA보다 좋았다. 조건별로는
seen 9조건 중 1개, unseen 4조건 중 1개에서만 common stream이 AoI를 낮췄다.

이 결과가 뜻하는 바는 다음과 같다.

- mixed RSMA actor는 채널 상태에 따라 좋은 private-power action을 학습했다.
- 별도로 학습한 SDMA actor는 같은 수준의 private-power mapping에 도달하지 못했다.
- 따라서 `RSMA PPO 1.7024 vs SDMA PPO 1.7608`은 학습된 정책 두 개의 비교로는
  유효하지만, RSMA PHY 구조가 SDMA보다 3.32% 좋다는 증거는 아니다.
- common-off projection은 현재 mixed 실험의 더 강한 learned-SDMA baseline이다.

## 5. Geometry와 SNR에 대한 action 적응

장기 on-policy trajectory의 common gate 사용률은 다음과 같다.

| 각도 \ SNR | 5 dB | 10 dB | 15 dB |
|---:|---:|---:|---:|
| 5° | 3.26% | 10.06% | 25.12% |
| 20° | 0.40% | 3.10% | 13.80% |
| 60° | 0.00% | 0.05% | 0.50% |

정책은 정렬된 채널과 높은 SNR에서 common을 더 자주 켰다. Gate가 켜진 슬롯의
`alpha`는 대략 5°에서 0.52, 20°에서 0.47--0.49였고, `rho`는 SNR과 함께 증가했다.
고정 20° 정책에서 관측된 `alpha≈0.83`, `rho≈0.87` 포화는 이번 mixed geometry+SNR
정책에서 유지되지 않았다. 따라서 그 포화는 현재 PHY의 필연적인 구조라기보다 고정
operating point와 해당 학습 결과에 의존한 현상으로 보는 것이 맞다.

다만 action이 환경 변수에 반응했다는 사실과 그 action이 장기 AoI에 도움이 됐다는
사실은 다르다. Common-off projection은 현재 학습된 gate 및 common allocation이
평균적으로 불필요하거나 약간 해로웠음을 보인다.

## 6. 동일 protocol의 CEM subset

| 방식 | 후보 예산/slot | 평균 AoI | SDMA-CEM 대비 |
|---|---:|---:|---:|
| safeguarded RSMA-CEM | 1,920 | **1.3658** | +5.78% |
| SDMA-CEM | 960 | 1.4496 | 기준 |
| raw RSMA-CEM | 960 | 1.4937 | -3.04% |
| RSMA joint+gate PPO | feed-forward | 1.6859 | — |
| SDMA-power PPO | feed-forward | 1.7515 | — |

Raw RSMA-CEM은 9조건 중 4조건에서만 SDMA-CEM을 이겼다. 특히 5 dB와 60°에서
common을 항상 켜는 raw 탐색이 불리했다. Exact SDMA 후보를 함께 비교하는 safeguarded
CEM은 60°에서 사실상 SDMA와 같고 나머지 대부분에서 낮은 AoI를 기록했다. 이는
RSMA action space가 SDMA 경계를 포함해야 한다는 기존 결론을 다시 지지한다.

같은 짧은 subset에서 raw CEM은 RSMA-PPO보다 11.40%, safeguarded CEM은 18.99%
낮은 AoI였다. 따라서 환경 변동성만으로 PPO가 현재 CEM을 이긴다는 가설은 기각한다.

## 7. 논문에서 허용되는 주장

이번 결과로 주장할 수 있는 내용은 다음과 같다.

1. Episode-randomized geometry와 SNR을 하나의 관측 기반 PPO가 처리할 수 있고,
   보간 조건에서도 동작한다.
2. 정책의 common gate는 spatial alignment와 SNR에 체계적으로 반응한다.
3. 고정 geometry에서 관측한 높은 `alpha/rho` 포화는 보편적인 구조적 결과가 아니다.
4. 현재 mixed PPO의 실제 성능 이점은 common stream보다 private-power adaptation에서
   나왔다.
5. One-step safeguarded CEM과 PPO 사이에는 여전히 큰 optimization gap이 있다.

다음 표현은 피해야 한다.

- “Mixed 환경에서 PPO-RSMA가 SDMA보다 3.32% 우수하다”를 RSMA PHY의 순이득으로
  제시하는 것
- 환경 변동 때문에 PPO가 CEM보다 유리하다고 주장하는 것
- 4개의 보간점 결과를 연속 SNR/geometry 전 범위 일반화로 확대하는 것
- 현재 episode-randomized 실험을 slot마다 이동성과 path loss가 변하는 현실 채널과
  동일시하는 것

## 8. 한계와 다음 판단

- Geometry와 SNR은 episode 단위 slow variable이며 episode 내부에서 전환되지 않는다.
- Training seeds가 3개이므로 이 실험은 headline보다 secondary ablation에 적합하다.
- Unseen 조건은 범위 내부 보간 4개뿐이며 외삽을 시험하지 않았다.
- Oracle pretraining은 one-step expected AoI 목적이므로 장기 common 사용을 잘못
  유도했을 가능성이 있다.
- CEM은 짧은 subset과 finite population을 사용했으며 long-horizon optimum이 아니다.

학부논문 마감 관점에서 within-episode SNR/geometry transition을 추가로 구현할 필요는
없다. 이번 negative ablation을 통해 “상태가 다양하면 PPO-RSMA가 CEM을 이길 것”이라는
가설을 이미 검증했고, 답은 현재 구현에서 `아니다`였다. 본문의 중심은 고정 headline
조건의 matched comparison과 CEM gap에 두고, 이 실험은 일반화 및 한계 절에 배치하는
것이 가장 안전하다.

## 9. 재현 파일

- Config: `configs/thesis_rician_mixed_geometry_snr_csit02.yaml`
- Oracle dataset: `datasets/rician_mixed_geometry_snr_csit02_oracle_v1/`
- Objective pretraining: `runs/objective_actor_rician_mixed_geometry_snr_csit02_*_v1/`
- PPO checkpoints: `results/rician_ppo_mixed_geometry_snr_csit02_262k_v1/`
- 평가 원자료와 그림: `results/thesis_mixed_geometry_snr_v1/`
- 평가: `scripts/evaluate_mixed_geometry_snr_study.py`
- Common-off projection: `scripts/evaluate_mixed_geometry_snr_projection.py`
- 파생표와 그림: `scripts/analyze_mixed_geometry_snr_study.py`
