# RSMA-PPO 기반 AoI 최소화: 상세 실험 결과

## 1. 연구 질문과 평가 범위

본 실험은 사용자 3명, 송신 안테나 2개의 overloaded downlink에서 PPO가 common 전력 alpha, private 전력 simplex q_k, common payload fraction phi_k를 상태에 따라 조절해 장기 평균 AoI를 낮출 수 있는지 평가한다. 채널은 directional Rician ULA와 Gauss-Markov 시간 상관을 사용하며 CSIT 상대 오차는 0.2다. Private beam은 RZF, common beam은 principal singular direction으로 고정한다.

최종 수치는 독립적으로 학습한 5개 seed를 동일한 channel, CSIT error 및 ACK 난수에서 평가한 결과다. 각 checkpoint는 사용자 순열 6개에서 768 slots 평가하고 초기 256 slots를 제외했다.

Geometry span g는 양끝 사용자 각도의 차이이며 배치는 (-g/2, 0, g/2)다. 5/20/60도는 각각 (-2.5, 0, 2.5), (-10, 0, 10), (-30, 0, 30)도와 그 사용자 순열을 뜻한다.

사전학습 데이터는 geometry span 5/20/60도, SNR 5/10/15/20 dB, n=64/128/256을 포함한다. 이후 PPO Stage 1은 SNR 10/15/20 dB와 n=128/256에서 524,288 steps, 목표 조건 미세조정은 20도·15 dB·n=128에서 524,288 steps 수행했다. 따라서 stress는 사전학습에는 포함됐으나 최종 PPO curriculum에서는 제외됐다. 격자 평가도 목표 조건 미세조정을 마친 최종 checkpoint를 사용한다.

## 2. 목표 조건 결과

목표 조건은 geometry 20도, SNR 15 dB, blocklength 128, 사용자당 payload 256 bit다.

| 방법 | 평균 AoI | seed SD | 95% CI half-width | p95 AoI | 성공확률 |
|---|---:|---:|---:|---:|---:|
| Objective-pretrained actor | 2.481 | 0.270 | 0.336 | 7.508 | 0.463 |
| RSMA-PPO, objective-only matched | **1.463** | 0.012 | 0.015 | 2.033 | **0.577** |
| RSMA-PPO, action+objective 진단 | 1.463 | 0.022 | 0.027 | **2.000** | 0.568 |
| 고정 규칙 RSMA, alpha=0.70 | 1.656 | 0.046* | - | 3.000 | 0.502 |
| q-only PPO-SDMA | 1.744 | 0.060 | 0.074 | 3.167 | 0.447 |
| Max-AoI 단일 사용자 MRT | 2.000 | 0.000* | - | 3.000 | 0.333 |
| Common-off 투영 | 2.650 | 0.125 | 0.155 | - | - |
| Max-AoI top-2 SDMA | 2.822 | - | - | 7.500 | 0.330 |

주 SDMA는 매 slot 현재 AoI가 가장 큰 두 사용자를 선택하고 private 전력을 균등 분배한다. PPO와 SDMA의 48.17% 차이에는 common stream뿐 아니라 all-user schedule과 학습된 q_k의 효과가 포함된다.

q-only PPO-SDMA는 RSMA-PPO와 동일한 all-user schedule, RZF beam, 데이터 상태 집합, objective-only 사전학습 80 epochs, PPO curriculum·학습량과 5개 seed를 사용하며 alpha=phi=0으로 고정하고 q_k만 학습한다. 따라서 matched 비교에서 남는 체계적 차이는 RSMA의 common 전력·payload 제어 자유도다. 다만 행동 차원과 최적화 난도까지 같지는 않으므로 common stream 하나의 인과 효과를 완전히 분리한 것으로 해석하지 않는다.

Posterior channel 8개에 대한 SAA를 두 actor의 differentiable FBL-AoI 목적 손실에 사용한다. RSMA는 alpha-q-phi 공간에서, q-only는 common-off 공간에서 같은 목적을 직접 최적화한다. 기존 action+objective RSMA는 추정 채널 CEM label의 행동 모방 손실을 추가했지만, 목표 조건의 최종 AoI는 objective-only RSMA와 구분되지 않았다.

q-only 대비 개선율은 각 training seed에서 방법별 6개 순열 평균 AoI를 먼저 구하고, 그 두 평균으로 개선율을 계산한 뒤 5개 seed에 걸쳐 평균한다. 이 값이 16.01%이며 95% 신뢰구간 half-width는 3.40 percentage point다. 평균 AoI 절대 차이는 0.280 ± 0.072다. RSMA-PPO는 30개 paired 조합 모두에서 낮은 AoI를 보였다. Objective-only RSMA와 기존 action+objective RSMA의 차이는 -0.08% ± 2.26%p로 구분되지 않았다. Common-off 투영의 43.76%는 q_k를 재학습하지 않은 제거 실험이므로 순기여 추정치로 사용하지 않는다.

별표 표준편차는 학습 seed가 없는 규칙 정책의 6개 test permutation 간 변동이다. Max-AoI 단일 사용자 정책은 매 slot 가장 오래된 사용자 한 명에게 전체 전력을 주며 단일 사용자 RZF는 MRT와 같은 방향이다. 고정 규칙 RSMA는 가장 오래된 사용자의 packet 전체를 common에 두고 나머지 두 packet을 private으로 보내며, test와 분리된 validation 궤적에서 alpha=0.70을 선택했다. PPO의 paired gain은 고정 규칙 대비 11.63% ± 1.63%p, 단일 사용자 대비 26.87% ± 1.34%p였고 두 비교 모두 30개 궤적에서 이겼다.

고정 규칙 RSMA는 PPO와 같은 principal common beam과 RZF private beam을 사용한다. 최대 AoI 사용자는 phi=1과 private 전력 0, 나머지 두 사용자는 phi=0과 private 전력 1/2씩으로 설정했다. 표의 성공확률은 실제 ACK 비율이 아니라 PHY가 계산한 사용자-slot별 packet 성공확률의 평균이다. p95 AoI는 각 궤적의 burn-in 이후 사용자-slot AoI 표본의 경험적 95백분위수를 구한 뒤 궤적 간 평균한 값이다.

![목표 조건의 matched 및 AoI-aware 기준선 비교](../results/thesis_rule_baselines_target_v1/figure_target_rule_baselines.png)

**그림 1.** 붉은 점선 2.0은 K=3에서 slot당 최대 한 명을 완벽하게 갱신할 때의 하한이다. 이보다 낮은 AoI는 동시 update의 이점과 양립한다.

![동일 objective-only 사전학습 조건에서 RSMA-PPO와 q-only PPO-SDMA의 paired 비교](../results/matched_pretraining_target_v1/figure_matched_pretraining.png)

**그림 2.** 첫 두 열은 동일 seed의 objective-only RSMA/SDMA paired 비교이며, 세 번째 열은 기존 action+objective RSMA 진단 결과다.

## 3. Mixed-regime 결과

| 구간 | PPO 평균 AoI | top-2 SDMA 평균 AoI | 단일 사용자 평균 AoI | PPO의 단일 사용자 대비 paired gain |
|---|---:|---:|---:|---:|
| Feasible: SNR >= 10 dB, n >= 128 | **1.701** | 2.798 | 2.000 | **14.97% ± 1.65%p** |
| Stress: SNR=5 dB 또는 n=64 | 151.388 | 198.967 | **67.933** | 백분율 평균으로 요약하지 않음 |

이 절의 단일 사용자 및 top-2 기준선 대비 paired gain은 각 궤적 s에서 다음과 같이 계산한 뒤 seed 내 평균, seed 간 평균 순으로 집계한다. ref는 해당 기준선이며, 2절의 q-only 대비 16.01%와는 집계 순서가 다르다.

\[
G_s=100\,\frac{\bar\Delta_{\mathrm{ref},s}-\bar\Delta_{\mathrm{PPO},s}}{\bar\Delta_{\mathrm{ref},s}}
\]

따라서 표의 평균 AoI 두 개로 계산한 비율과 일치하지 않는다. Feasible 구간에서는 PPO가 단일 사용자 전송보다 5개 학습 seed 모두에서 우수했다. Stress 구간은 일부 셀의 AoI가 수백까지 폭증해 백분율 gain 평균이 불안정하지만, 절대 AoI 평균에서는 단일 사용자 전송이 현재 all-user PPO보다 낮았다. 이 조건은 사전학습에는 포함됐으나 최종 PPO 단계에서는 제외됐다. 활성 집합을 줄이면 beam과 간섭 구조도 함께 바뀌므로 학습 분포 변화와 scheduling의 영향을 분리할 수 없다. 또한 수치는 512-slot 측정 창에 의존한다. 18개 stress regime 중 평균 성공확률이 10^-3 미만인 regime은 두 방식 모두 3개였고, 평균 AoI가 100을 넘은 regime은 PPO 6개, 단일 사용자 3개였다.

## 4. 상태에 따른 action 변화

| 변화 축 | 조건 | 평균 alpha | 95% CI half-width |
|---|---:|---:|---:|
| Geometry | 5도 | 0.619 | 0.068 |
| Geometry | 20도 | 0.589 | 0.063 |
| Geometry | 60도 | 0.397 | 0.061 |
| SNR | 5 dB | 0.250 | 0.104 |
| SNR | 10 dB | 0.453 | 0.066 |
| SNR | 15 dB | 0.659 | 0.035 |
| SNR | 20 dB | 0.778 | 0.026 |
| Blocklength | 64 | 0.500 | 0.132 |
| Blocklength | 128 | 0.542 | 0.047 |
| Blocklength | 256 | 0.562 | 0.032 |

Alpha는 채널이 더 분리되는 60도에서 감소하고 SNR이 증가할수록 증가했다. 각 값은 나머지 두 축을 평균한 marginal response다.

| 현재 AoI 순위 | 평균 q_k | 평균 phi_k |
|---|---:|---:|
| 가장 낮은 사용자 | 0.271 | 0.215 |
| 중간 사용자 | 0.291 | 0.263 |
| 가장 높은 사용자 | 0.438 | 0.713 |

정책은 오래된 사용자에게 private 전력과 common payload를 더 많이 할당했다. 최대 AoI가 2에서 3 이상으로 증가할 때 alpha는 0.766에서 0.699로 감소했다. 이는 긴급한 사용자의 private 전달을 보호하는 행동과 양립하지만 인과적 최적성을 뜻하지 않는다.

Target fine-tuning 로그의 on-policy mean AoI 8-point moving average는 다섯 seed 모두 약 1.91–2.00에서 1.54–1.63으로 감소했다. Logged episodic return도 약 -1027~-980에서 -828~-789로 증가했다. 환경은 사용자 AoI 합의 음수에 reward_scale=1/3을 곱해 평균 보상을 만들고 그 episode 합을 기록한다. 이 logged return은 PPO 내부 running reward normalization 이전의 값이며, PPO 갱신에는 이후 정규화된 보상이 사용된다. 두 곡선은 학습 중 stochastic action과 방문 state에서 계산한 수렴 진단값이며 frozen deterministic 평가 AoI와 같은 통계량은 아니다.

## 5. Stress 영역 해석

Blocklength 64에서 256 bit를 전송하려면 4 bit/channel-use가 필요하다. 본 채널 정규화에서는 사용자당 평균 채널 norm 제곱이 N_t=2이므로 모든 전력을 한 사용자에게 주고 이상적인 MRT를 적용한 상계도 log2(1+2*10^(5/10))=2.87 bit/channel-use로 4보다 작다. 실제 시스템에는 다중사용자 간섭, 불완전 CSIT 및 FBL penalty가 추가된다. 고정 payload, all-user schedule과 principal common beam도 영향을 줄 수 있으나 각 원인은 아직 분리하지 않았다.

## 6. 현재 결론

q-only PPO-SDMA가 휴리스틱 SDMA의 2.822를 1.744로 낮춘 결과에는 학습된 q_k와 all-user 전송의 차이가 함께 반영된다. 고정 규칙 RSMA는 1.656, RSMA-PPO는 1.463을 달성했다. 사전학습 손실을 objective-only로 맞춘 q-only 대비 seed별 평균 개선율은 16.01%였다. 기존 16.05%와 사실상 같아 사전학습 손실 불일치가 격차의 원인이라는 가설은 지지되지 않는다. 이 값과 고정 규칙 대비 11.63%를 목표 조건 밖으로 일반화하지 않는다. Stress에서는 단일 사용자 기준선이 더 낮은 AoI를 보였으나, 사전학습 이후의 학습 분포 변화·PHY 난도·행동 공간 제약의 영향은 분리되지 않았다.

## 7. 평가 범위

CSIT 오차는 sigma_e=0.2 한 점에서만 평가했다. 평가 방법들은 paired 비교를 위해 동일한 channel, CSIT error 및 ACK seed 한 세트를 공유하므로, 보고한 신뢰구간은 독립 training seed의 변동만 반영하고 새로운 평가 궤적에 대한 불확실성은 포함하지 않는다. q-only와의 비교는 objective-only 사전학습 손실·epoch와 PPO 예산을 맞췄다. 별도의 full-grid Stage 2 진단 checkpoint는 최종 정책에 사용하지 않았다.
