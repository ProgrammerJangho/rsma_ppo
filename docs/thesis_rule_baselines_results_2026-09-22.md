# Max-AoI 단일 사용자 및 고정 규칙 RSMA 기준선 결과

## 평가 설계

- **Max-AoI 단일 사용자:** 매 slot weighted AoI가 가장 큰 사용자 한 명을 선택하고 전체 전력을 사용한다. 한 사용자 RZF 방향은 MRT와 동일한 방향이다.
- **고정 규칙 RSMA:** 세 사용자를 모두 활성화하고, 최대 AoI 사용자의 256-bit update 전체를 common stream에 배치한다. 나머지 두 사용자는 private stream으로 전송하며 private 전력을 균등 분배한다.
- 고정 규칙의 common 전력 비율 alpha는 test와 분리된 validation 궤적에서 0.05 간격 grid search로 선택했다. 3개 validation repeat와 6개 사용자 순열을 사용했고 alpha=0.70이 선택됐다.
- Test는 최종 PPO와 동일한 20도, 15 dB, blocklength 128 조건과 동일한 6개 frozen 평가 seed를 사용했다.
- 성공확률은 실제 ACK 비율이 아니라 PHY가 계산한 사용자-slot별 packet 성공확률의 평균이다. p95 AoI는 각 궤적의 burn-in 이후 사용자-slot AoI 표본의 경험적 95백분위수다.

## 목표 조건 결과

| 방법 | 평균 AoI | 변동성 | p95 AoI | 성공확률 |
|---|---:|---:|---:|---:|
| RSMA-PPO | **1.463** | training-seed SD 0.022 | **2.000** | **0.568** |
| 고정 규칙 RSMA, alpha=0.70 | 1.656 | test-permutation SD 0.046 | 3.000 | 0.502 |
| q-only PPO-SDMA | 1.744 | training-seed SD 0.060 | 3.167 | 0.447 |
| Max-AoI 단일 사용자 MRT | 2.000 | test-permutation SD 0.000 | 3.000 | 0.333 |

RSMA-PPO는 고정 규칙 RSMA보다 seed별 paired gain 기준 11.63% 낮은 AoI를 보였고 95% Student-t 신뢰구간 half-width는 1.63 percentage point였다. Max-AoI 단일 사용자 대비 gain은 26.87% ± 1.34 percentage point였다. 두 비교 모두 5개 training seed와 6개 사용자 순열의 30개 paired 궤적 전부에서 PPO가 이겼다.

고정 규칙이 q-only PPO-SDMA보다 낮은 AoI를 보인 것은 최대 AoI 사용자의 packet을 common stream으로 보내는 간단한 RSMA 구조도 유용하다는 뜻이다. 최종 PPO가 이 규칙보다 다시 낮은 AoI를 보인 결과는 validation으로 조정한 단일 alpha와 고정 payload 규칙만으로 상태 적응 행동을 모두 설명할 수 없음을 보여준다.

Max-AoI 단일 사용자 방식의 AoI 2.0은 K=3에서 매 slot 최대 한 명만 완벽하게 갱신할 때 round-robin이 달성하는 (K+1)/2와 같다. PPO가 이를 넘은 것은 한 slot의 동시 update가 장기 AoI를 낮춘다는 증거다. 이 비교만으로 common stream의 순효과를 주장할 수는 없으며, 그 효과는 matched q-only PPO-SDMA 비교에서 분리한다.

## 전체 regime 결과

| 구간 | PPO 평균 AoI | 단일 사용자 평균 AoI | PPO paired gain | 95% CI half-width |
|---|---:|---:|---:|---:|
| Feasible: SNR >= 10 dB, n >= 128 | **1.701** | 2.000 | **14.97%** | 1.65%p |
| Stress: SNR=5 dB 또는 n=64 | 151.388 | **67.933** | 우열을 단일 평균 gain으로 요약하지 않음 | - |

Feasible 영역에서는 단일 사용자 전송이 거의 항상 성공해 AoI 2에 붙지만 PPO가 5개 학습 seed 모두에서 더 낮다. Stress 영역에서는 방법별 AoI가 수백까지 폭증하는 셀이 있어 궤적별 백분율 평균이 불안정하다. 절대 AoI 평균에서는 Max-AoI 단일 사용자 전송이 PPO보다 강하다. 다만 이 값은 512-slot 측정 창에 의존하며 성공확률이 거의 0인 경우 정상상태 평균이 아니다. 18개 stress regime 중 평균 성공확률이 10^-3 미만인 regime은 두 방식 모두 3개였고, 평균 AoI가 100을 넘은 regime은 PPO 6개와 단일 사용자 3개였다. 활성 사용자 집합을 줄이면 beam과 간섭 구조도 함께 바뀌므로 scheduling만의 인과 효과로 분리하지 않는다.
