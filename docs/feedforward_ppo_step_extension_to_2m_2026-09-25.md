# Feed-forward PPO 2.10M-step 확대 실험

## 설정

seed-0 objective-only RSMA feed-forward PPO의 누적 1,048,576 target-step
checkpoint에서 추가 1,048,576 interactions를 학습해 총 2,097,152 target
steps까지 확대했다. 환경, 행동 `(alpha, q_k, phi_k)`, reward 및 PPO update
설정은 기존 실험과 같다.

actor, critic, optimizer, observation/reward normalization 상태를 복원했으며,
추가 구간에서 learning-rate annealing을 다시 시작했다. 따라서 이 결과는
처음부터 단일 schedule로 수행한 2.10M 학습이 아니라 524k 단위의 첫 연장과
1.05M 단위의 두 번째 연장으로 구성된 staged fine-tuning 결과다.

평가는 동일한 6개 사용자 각도 순열, 궤적당 768 slot, 256-slot burn-in 및
동일 channel/CSIT/ACK seed로 수행했다.

## 학습량별 결과

| 총 target steps | 평균 AoI | 최초 524k 대비 개선 |
|---:|---:|---:|
| 524,288 | 1.4667 | 0.00% |
| 655,360 | 1.4641 | 0.18% |
| 786,432 | 1.4409 | 1.76% |
| 917,504 | 1.4346 | 2.19% |
| 1,048,576 | 1.4285 | 2.60% |
| 1,310,720 | 1.4233 | 2.96% |
| 1,572,864 | 1.4084 | 3.97% |
| 1,835,008 | 1.4052 | 4.19% |
| 2,097,152 | **1.4009** | **4.48%** |

2.10M 정책은 1.05M 정책보다 `1.93 ± 0.44%p`, 최초 524k 정책보다
`4.48 ± 0.61%p` 개선됐으며 두 비교 모두 6개 궤적 전부에서 승리했다.
이 신뢰구간의 표본 단위는 학습 seed가 아니라 고정 사용자 순열이다.

## 최종 정책 변화

| 정책 | 평균 AoI | p95 AoI | 성공확률 | 평균 alpha | alpha slot 표준편차 | common payload |
|---|---:|---:|---:|---:|---:|---:|
| 524k | 1.4667 | 2.0 | 0.5855 | 0.7561 | 0.1006 | 279.69 |
| 1.05M | 1.4285 | 2.0 | 0.6240 | 0.7817 | 0.1066 | 304.33 |
| 2.10M | **1.4009** | 2.0 | **0.6358** | 0.8220 | 0.0872 | 324.63 |

학습량 증가에 따라 common power와 common payload가 계속 증가했고 성공확률과
AoI도 함께 개선됐다. alpha의 slot별 표준편차는 1.05M 이후 감소했으므로,
정책이 더 높은 common-power operating point로 수렴하는 경향은 있다. 완전히
상수화되지는 않았지만, 최종 논문에서는 상태 적응성을 다시 감사한 뒤
연속 alpha 제어에 관한 주장을 정해야 한다.

같은 평가의 seed-0 SDMA-PPO AoI 1.6882와 비교하면 2.10M RSMA-PPO는
17.01% 낮다. 이 수치 역시 한 학습 seed의 결과이므로 기존 5-seed headline을
대체할 수 없다.

## 판단

seed 0에서는 학습량 확대가 아직 유효했으며 2.10M checkpoint가 전체 sweep의
best다. 다만 개선 폭은 후반으로 갈수록 줄었다. 1.57M에서 2.10M까지의
추가 개선은 약 0.53%이므로 plateau에 접근하고 있다. 같은 seed의 step을 다시
늘리는 것보다 2.10M protocol을 다른 seed에서 재현하는 편이 연구적으로 더
중요하다.

## 산출물

- best checkpoint:
  `runs/matched_objective_rsma_ppo_k3_20deg_seed0_continue_to_2097k_v1/checkpoint.pt`
- 전체 checkpoint 표:
  `results/feedforward_ppo_seed0_step_extension_to_2m_v1/checkpoint_aoi.csv`
- 최종 비교표:
  `results/feedforward_ppo_seed0_step_extension_to_2m_v1/final_summary.csv`
- 수렴 그림:
  `results/feedforward_ppo_seed0_step_extension_to_2m_v1/feedforward_step_extension_to_2m.png`
