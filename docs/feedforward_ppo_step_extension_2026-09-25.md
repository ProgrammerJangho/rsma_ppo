# Feed-forward PPO 학습량 확대 실험

## 실험 설계

기존 objective-only RSMA feed-forward PPO seed-0의 524,288-step target
checkpoint에서 학습을 재개해 target interaction을 총 1,048,576까지 늘렸다.
시스템, 행동 `(alpha, q_k, phi_k)`, reward, PPO update 설정과 평가 궤적은
기존 실험과 동일하다.

checkpoint에는 actor, critic, optimizer, observation/reward normalization
통계를 모두 복원했다. 추가 524,288-step 구간에서는 기존 learning-rate
설정을 사용해 annealing schedule을 한 번 더 시작했다. 따라서 이 결과는
처음부터 하나의 1M schedule로 학습한 결과가 아니라 **524k checkpoint에서
재개한 두 번째 fine-tuning stage**의 결과다.

평가는 사용자 각도 6개 순열, 각 768 slot, 앞 256 slot burn-in 및 동일한
channel/CSIT/ACK seed를 사용했다.

## 결과

| 총 target interaction | 평균 AoI | 524k 대비 개선율 |
|---:|---:|---:|
| 524,288 | 1.4667 | 0.00% |
| 655,360 | 1.4641 | 0.18% |
| 786,432 | 1.4409 | 1.76% |
| 917,504 | 1.4346 | 2.19% |
| 1,048,576 | **1.4285** | **2.60%** |

최종 1.05M 정책은 524k 정책보다 6개 평가 궤적 모두에서 낮은 AoI를
기록했다. 궤적별 paired 개선율은 `2.60 ± 0.88%p`(95% CI)였다. 이 CI의
단위는 학습 seed가 아니라 고정된 6개 사용자 순열이므로, 알고리즘 전체의
통계적 일반화를 의미하지 않는다.

| 정책 | 평균 AoI | p95 AoI | 성공확률 | 평균 alpha | alpha slot 표준편차 | common payload |
|---|---:|---:|---:|---:|---:|---:|
| 524k feed-forward PPO | 1.4667 | 2.0 | 0.5855 | 0.7561 | 0.1006 | 279.69 |
| 1.05M feed-forward PPO | **1.4285** | 2.0 | **0.6240** | 0.7817 | 0.1066 | 304.33 |

학습량을 늘리면서 common power와 common payload가 증가했지만, 성공확률도
0.5855에서 0.6240으로 올랐고 alpha의 slot별 변화도 유지됐다. 따라서 현재
결과는 단순한 action saturation이나 common-stream 붕괴로 보이지 않는다.
학습 core metric에는 non-finite 값이 없었고 sampled-policy AoI도 후반에
계속 감소했다.

같은 seed-0 SDMA-PPO의 평균 AoI 1.6882와 비교하면 최종 RSMA-PPO의 개선율은
15.38%다. 다만 기존 논문의 5-seed RSMA 평균 1.4635를 seed-0 1.4285로
교체해서는 안 된다. 최종 논문 숫자로 사용하려면 선택한 1.05M 설정을 다른
학습 seed에도 사전에 고정해 반복해야 한다.

## 판단

이번 seed-0 결과에서는 feed-forward PPO의 기존 524k target 학습이 완전히
수렴한 상태가 아니었다. GRU-PPO보다 단순한 feed-forward 정책에 계산 예산을
더 주는 편이 효과적이었다. 원래 동일한 524k target 예산에서도 feed-forward
PPO가 GRU-PPO보다 1.01% 좋았고, 예산을 1.05M으로 늘린 feed-forward PPO는
524k GRU-PPO보다 3.58% 낮은 AoI를 기록했다. 후자의 비교는 학습 예산이
다르므로 architecture 우열의 공정 비교로 사용하지 않는다.

현재 best checkpoint는 누적 1,048,576-step 정책이다. 다음 우선순위는
학습량을 무한히 더 늘리는 것이 아니라, 이 설정을 최소 3개 추가 seed에서
재현해 개선이 seed-0 우연인지 확인하는 것이다.

## 산출물

- 최종 checkpoint:
  `runs/matched_objective_rsma_ppo_k3_20deg_seed0_continue_524k_v1/checkpoint.pt`
- 학습량별 결과:
  `results/feedforward_ppo_seed0_step_extension_v1/checkpoint_aoi.csv`
- paired 결과:
  `results/feedforward_ppo_seed0_step_extension_v1/paired_comparison.csv`
- 수렴 그림:
  `results/feedforward_ppo_seed0_step_extension_v1/feedforward_step_extension.png`
