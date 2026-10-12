# 최종 PPO의 common-stream 기여 및 AoI 조건부 행동 감사

## 목적

최종 결과의 SDMA 기준선은 최대 AoI 사용자 두 명을 선택해 private 전력을 균등
분배하는 휴리스틱이다. 따라서 최종 PPO와 이 기준선의 차이에는 common stream,
사용자별 private 전력, 활성 사용자 수 및 AoI 기반 자원 배분의 효과가 함께
포함된다. 본 감사는 최종 PPO 행동에서 common 부분만 제거해 common stream의
직접 기여와 AoI에 따른 행동 변화를 별도로 확인한다.

## 평가 방법

- 대상: 독립적으로 학습한 최종 PPO checkpoint 5개
- 조건: geometry 20도, SNR 15 dB, blocklength 128, 사용자당 payload 256 bit
- 궤적: 사용자 순열 6개, 순열당 768 slot, burn-in 256 slot
- 공통 난수: 원 정책과 common-off 투영이 channel, CSIT error 및 ACK 난수를 공유
- common-off 투영: actor가 출력한 private 전력 simplex q는 유지하고,
  common 전력 비율 alpha와 common payload fraction phi를 0으로 고정

이 투영은 별도로 재학습한 PPO-SDMA가 아니다. RSMA에서 학습된 q를 그대로 쓰는
반사실적 분석이므로, common stream을 제거했을 때 동일 actor의 성능이 얼마나
변하는지를 측정한다.

## 결과

| 정책 | 평균 AoI |
|---|---:|
| 최종 PPO | 1.463 |
| common-off 투영 | 2.650 |

training-seed별 paired gain의 평균은 43.76%이고 95% Student-t 신뢰구간
half-width는 6.21 percentage point이다. 최종 PPO는 5개 seed와 6개 사용자
순열의 30개 paired 궤적 모두에서 common-off 투영보다 낮은 AoI를 보였다.

AoI 순위별 행동 평균은 다음과 같다.

| 현재 AoI 순위 | private 전력 비율 q | common payload fraction phi |
|---|---:|---:|
| 가장 낮은 사용자 | 0.271 | 0.215 |
| 중간 사용자 | 0.291 | 0.263 |
| 가장 높은 사용자 | 0.438 | 0.713 |

정책은 오래된 사용자에게 private 전력과 common payload를 더 많이 할당했다.
최대 AoI가 2에서 3 이상으로 증가할 때 평균 alpha는 0.766에서 0.699로 감소했다.
이는 매우 오래된 사용자의 private 전달을 보호하는 행동과 양립하지만, 관측
상관만으로 인과적 최적성을 주장할 수는 없다.

## 해석 범위

본 결과는 최종 actor에서 common stream이 실질적으로 사용되며 성능에 기여한다는
근거다. 그러나 주 SDMA 기준선과 PPO의 48.17% 차이를 common stream만의 효과로
분해하지는 못한다. 최종 비교를 더 엄격하게 하려면 동일한 all-user 상태와 행동
공간에서 alpha=phi=0으로 제한한 q-only PPO를 별도로 학습해야 한다.

