# Feed-forward PPO 상태 적응성 감사

## 질문과 절차

524k, 1.05M, 2.10M seed-0 checkpoint를 동일한 20도·15 dB·n=128의
6개 held-out 사용자 순열에서 비교했다. 각 정책의 상태별 `alpha`, `q_k`,
`phi_k`를 기록하고, 별도 calibration seed에서 구한 상수 또는 AoI-rank
규칙으로 action 일부를 대체했다.

두 종류의 counterfactual을 사용했다.

1. **One-step:** 동일한 실제 채널과 AoI 상태에서 action만 바꿔 expected
   next-slot AoI를 비교한다.
2. **Closed-loop:** 바뀐 action으로 512-slot 평가를 계속해 미래 AoI와 정책
   방문 상태까지 반영한다.

Closed-loop의 큰 손실은 즉시 action 손실뿐 아니라 FBL cliff와 off-policy
상태분포 이동이 누적된 값이다. 따라서 action 성분의 직접 민감도는 one-step,
실제 제어 결과는 closed-loop를 함께 봐야 한다.

## 2.10M 정책의 핵심 결과

### alpha: 포화 경향은 있지만 상수가 아니다

- 전체 평균: 0.8220
- 5--95 percentile: 0.7505--0.8846
- 채널 정렬도 Q1/Q2/Q3/Q4: 0.7866/0.8307/0.8339/0.8367
- max AoI가 2일 때 0.8270, 3 이상일 때 0.7517

즉 정책은 정렬도가 낮거나 긴급 사용자가 심하게 누적된 상태에서 common
power를 줄인다. Q2 이후에는 alpha가 포화에 가까워 정렬도에 대한 연속 조절은
주로 낮은 정렬도 영역에서 일어난다.

alpha를 별도 calibration 평균으로 고정하면 one-step AoI가 1.97%, closed-loop
AoI가 5.44% 증가했다. alpha의 상태별 조절은 작지만 실제 성능에 기여한다.

### q_k: 사실상 AoI-aware private scheduling이다

| AoI 순위 | 평균 q_k |
|---|---:|
| youngest | 0.0044 |
| middle | 0.0295 |
| oldest | **0.9661** |

`q_k`와 relative AoI의 Pearson 상관은 0.939다. unique oldest 상태에서
최대-AoI 사용자 집합에 들어간 private power는 평균 96.54%이고, q가 가장 큰
사용자가 실제 최대-AoI 사용자일 확률은 97.85%다. 최대 AoI가 동률인 상태에는
그 집합에 거의 100%를 주며, 두 명 동률일 때 그중 채널 이득이 더 큰 사용자를
선택한 비율은 65.1%였다.

단순 AoI-rank q 규칙으로 바꾸면 one-step AoI는 0.88%, closed-loop AoI는
6.46% 증가했다. 대부분의 구조는 “oldest에 private power 집중”으로 설명되지만,
동률 처리와 연속 상태에 따른 미세 조절도 장기 성능에 필요하다.

### phi_k: 채널 정렬도에 가장 강하게 적응한다

정렬도 Q1/Q2/Q3/Q4에서 사용자 평균 phi는
0.310/0.402/0.459/0.520으로 증가했다. phi 평균과 정렬도의 Pearson/Spearman
상관은 각각 0.758/0.827이다. alpha보다 phi가 채널 geometry를 더 세밀하게
추적한다.

AoI 순위별 평균은 youngest/middle/oldest에서 0.909/0.057/0.302였다. 정책은
oldest 사용자에게 private power를 집중하면서 다른 사용자의 payload를 common
stream으로 옮기는 역할 분담을 학습했다. 이 순위 통계에는 AoI 동률의 stable
tie-breaking이 포함되므로 각 숫자를 보편적 규칙으로 해석하면 안 된다.

actor의 사용자별 phi 비율은 유지하되 전체 phi 수준만 calibration 평균으로
고정하면 one-step AoI가 13.22%, closed-loop AoI가 37.65% 증가했다. 완전 고정
phi의 장기 손실은 1,000%를 넘었지만, 이는 FBL cliff 이후의 발산이 누적된
수치이므로 contribution 숫자로 사용하는 것은 부적절하다.

## 학습량에 따른 변화

| 항목 | 524k | 1.05M | 2.10M |
|---|---:|---:|---:|
| 평균 alpha | 0.756 | 0.782 | 0.822 |
| alpha 표준편차 | 0.104 | 0.112 | 0.091 |
| 평균 phi | 0.364 | 0.396 | 0.423 |
| q entropy | 0.092 | 0.097 | 0.056 |
| q-relative AoI 상관 | 0.875 | 0.918 | 0.939 |

학습량 증가로 정책은 높은 common 사용점으로 이동하는 동시에 private power를
최대-AoI 사용자에게 더 선명하게 집중했다. alpha의 변동 폭은 다소 줄었지만
상태 적응성이 사라지지는 않았다. 특히 phi의 정렬도 적응은 더 강해졌다.

## 결론

2.10M 정책은 alpha, q_k, phi_k를 모두 상태에 따라 사용한다. 세 action의
역할은 서로 다르다.

- `q_k`: AoI urgency에 따른 private-power scheduling
- `phi_k`: 채널 정렬도에 따른 common payload 제어
- `alpha`: 낮은 정렬도와 큰 AoI 누적 상태에서 common power를 줄이는 보조 제어

따라서 “학습량을 늘리면서 alpha가 단순 상수로 붕괴했다”는 결론은 맞지 않는다.
더 정확한 해석은 **높은 alpha operating point 주변에서 q_k와 phi_k가 핵심
상태 적응을 담당하고, alpha가 제한된 범위의 추가 보정을 수행한다**는 것이다.

이 감사에는 학습 seed가 하나이고 geometry/SNR도 고정되어 있다. 다른 seed와
mixed-regime에서 동일한 역할 분담이 재현되어야 일반적인 정책 특성으로 주장할
수 있다.

## 산출물

- 감사 데이터: `results/feedforward_state_adaptation_audit_v4/`
- 핵심 그림: `state_adaptation_audit.png`
- closed-loop 결과: `counterfactual_summary.csv`
- one-step 결과: `one_step_counterfactual_summary.csv`
- 상태 상관: `global_state_correlations.csv`, `user_state_correlations.csv`
