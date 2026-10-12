# KICS 2.10M 결과: 학습 seed 5개 × 평가 seed 5개

작성일: 2026-09-25

## 무엇을 했나

- RSMA-PPO와 SDMA-PPO(q-only) 모두 seed 0~4를 목표 조건에서 누적 2,097,152 step까지 학습했다.
  동일한 objective-only 사전학습과 혼합 조건 Stage 1(524,288 step) 이후, 목표 조건 524,288 step
  체크포인트에서 (A) 524,288 step, (B) 1,048,576 step을 이어 학습했다. 각 단계는 actor, critic,
  optimizer, 관측·보상 정규화 상태를 복원하고 learning-rate annealing은 단계마다 새로 시작했다.
  실행: `scripts/run_2m_staged_continuation.sh`, 검증: `scripts/verify_2m_checkpoints.py`.
- SDMA 연장은 `allocation_control=sdma_power`를 명시했다. `load_checkpoint`는 가중치만 복원하므로
  이 설정 없이 이어 학습하면 q-only 체크포인트가 오류 없이 joint(α>0) 정책으로 바뀐다.
- 평가: 20°, 15 dB, n=128, CSIT 오차 0.2. 학습 seed마다 평가 seed 5개(eval_seed = 990278 +
  10000·repeat + 100·permutation) × 사용자 순열 6개 = 30개 궤적, 768 slot, 처음 256 slot 제외.
  모든 방법이 같은 채널·CSIT·ACK 난수를 쓴다. 스크립트: `scripts/evaluate_kics_2m_5train_5eval.py`.
- 집계: 순열 평균 → 평가 seed별 값 → 평가 seed 5개 평균 = 학습 seed별 성능. 학습된 정책의 95% CI는
  학습 seed 5개 평균에 대한 Student-t 구간(df=4)이다. 150개 궤적을 독립 표본으로 쓰지 않았다.
  규칙 기준선은 학습 seed가 없으므로 평가 seed 5개에 대한 구간이다.

## 목표 조건 성능

| 방법 | 평균 AoI | 95% CI | CI 단위 | p95 | 성공확률 | 평균 α |
|---|---|---|---|---|---|---|
| RSMA-PPO 2.10M | 1.4009 | ±0.0071 | 학습 seed 5개 | 2.00 | 0.630 | 0.808 |
| SDMA-PPO 2.10M | 1.6551 | ±0.0327 | 학습 seed 5개 | 3.00 | 0.472 | 0.000 |
| RSMA-PPO 0.52M | 1.4700 | ±0.0142 | 학습 seed 5개 | 2.06 | 0.573 | 0.760 |
| SDMA-PPO 0.52M | 1.7449 | ±0.0734 | 학습 seed 5개 | 3.12 | 0.446 | 0.000 |
| 고정 규칙 RSMA (α=0.70) | 1.6447 | ±0.0148 | 평가 seed 5개 | 3.00 | 0.509 | 0.700 |
| Max-AoI 단일 사용자 MRT | 2.0000 | ±0.0000 | 평가 seed 5개 | 3.00 | 0.333 | 0.000 |

## 학습 seed별 평균 AoI (평가 seed 5개 × 순열 6개 평균)

| training_seed | RSMA-PPO 2.10M | RSMA-PPO 0.52M | SDMA-PPO 2.10M | SDMA-PPO 0.52M |
|---|---|---|---|---|
| 0 | 1.4082 | 1.4743 | 1.6316 | 1.6922 |
| 1 | 1.4038 | 1.4688 | 1.6365 | 1.7105 |
| 2 | 1.3932 | 1.478 | 1.6915 | 1.84 |
| 3 | 1.4012 | 1.4781 | 1.6416 | 1.719 |
| 4 | 1.398 | 1.4507 | 1.6742 | 1.7626 |

## 부록: 평가 seed별 평균 AoI (학습 seed 5개 × 순열 6개 평균)

학습 seed 변동과 별개로, 같은 정책 집합이 평가 난수에 따라 얼마나 달라지는지 보여 준다.

| eval_repeat | 고정 규칙 RSMA (α=0.70) | Max-AoI 단일 사용자 MRT | RSMA-PPO 2.10M | RSMA-PPO 0.52M | SDMA-PPO 2.10M | SDMA-PPO 0.52M |
|---|---|---|---|---|---|---|
| 0 | 1.6565 | 2.0 | 1.3958 | 1.4635 | 1.6533 | 1.7439 |
| 1 | 1.6556 | 2.0 | 1.4054 | 1.4782 | 1.6633 | 1.7516 |
| 2 | 1.6431 | 2.0 | 1.4076 | 1.4725 | 1.6581 | 1.7467 |
| 3 | 1.6276 | 2.0 | 1.3957 | 1.4674 | 1.6438 | 1.732 |
| 4 | 1.6407 | 2.0 | 1.3998 | 1.4684 | 1.6569 | 1.7501 |

## 쌍별 비교 (같은 학습 seed·같은 평가 seed끼리 대응)

gain = 100 × (기준 − 방법) / 기준. 학습 seed별로 평가 seed 5개 평균을 낸 뒤 학습 seed 5개로 CI를 구했다.
wins는 학습 seed 5개, 학습·평가 seed 쌍 25개, 궤적 150개 중 방법이 더 낮은 수다.

| comparison | gain_pct | gain_ci95_training_seed | abs_diff | abs_diff_ci95_training_seed | training_seed_wins | seed_eval_pair_wins | trajectory_wins |
|---|---|---|---|---|---|---|---|
| rsma_ppo_2m_vs_sdma_ppo_2m | 15.338 | 2.067 | 0.254 | 0.039 | 5 | 25 | 150 |
| rsma_ppo_2m_vs_fixed_rule_rsma | 14.823 | 0.432 | 0.244 | 0.007 | 5 | 25 | 150 |
| rsma_ppo_2m_vs_max_aoi_single_user | 29.957 | 0.355 | 0.599 | 0.007 | 5 | 25 | 150 |
| rsma_ppo_2m_vs_rsma_ppo_524k | 4.696 | 0.999 | 0.069 | 0.015 | 5 | 25 | 149 |
| sdma_ppo_2m_vs_sdma_ppo_524k | 5.098 | 2.16 | 0.09 | 0.043 | 5 | 25 | 150 |
| rsma_ppo_524k_vs_sdma_ppo_524k | 15.676 | 3.568 | 0.275 | 0.075 | 5 | 25 | 150 |
| sdma_ppo_2m_vs_fixed_rule_rsma | -0.633 | 1.988 | -0.01 | 0.033 | 3 | 12 | 67 |

## 행동 역할 통계 (RSMA-PPO 2.10M, 학습 seed별)

| training_seed | alpha_mean | alpha_sd | alpha_p05 | alpha_p95 | q_relative_aoi_pearson | q_argmax_is_oldest_unique_pct | unique_oldest_state_pct | phi_mean | phi_alignment_spearman | phi_alignment_pearson | alpha_alignment_spearman | q_youngest | phi_youngest | q_middle | phi_middle | q_oldest | phi_oldest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.8211 | 0.0887 | 0.7295 | 0.8843 | 0.9367 | 97.7249 | 95.0065 | 0.4205 | 0.819 | 0.7439 | 0.0716 | 0.0046 | 0.9022 | 0.0295 | 0.0606 | 0.9659 | 0.2987 |
| 1 | 0.794 | 0.2445 | 0.2933 | 1.0 | 0.717 | 81.4716 | 83.3464 | 0.4568 | 0.2255 | 0.194 | 0.0821 | 0.1017 | 0.5705 | 0.1875 | 0.237 | 0.7108 | 0.563 |
| 2 | 0.8151 | 0.0473 | 0.7516 | 0.8724 | 0.9435 | 97.8407 | 96.1784 | 0.4143 | 0.8082 | 0.734 | -0.3752 | 0.0026 | 0.937 | 0.0501 | 0.0445 | 0.9473 | 0.2615 |
| 3 | 0.7736 | 0.0536 | 0.7058 | 0.8451 | 0.9562 | 98.9802 | 95.1172 | 0.3719 | 0.6802 | 0.5861 | -0.471 | 0.0018 | 0.8839 | 0.0189 | 0.0767 | 0.9793 | 0.1551 |
| 4 | 0.8384 | 0.091 | 0.758 | 0.9144 | 0.9542 | 97.9864 | 97.9688 | 0.4539 | 0.8105 | 0.7615 | 0.11 | 0.0085 | 0.4693 | 0.0367 | 0.4908 | 0.9547 | 0.4015 |

요약 (학습 seed 5개 평균, 95% CI):

| statistic | mean | ci95_halfwidth | min | max |
|---|---|---|---|---|
| alpha_mean | 0.8085 | 0.0312 | 0.7736 | 0.8384 |
| alpha_sd | 0.105 | 0.0999 | 0.0473 | 0.2445 |
| alpha_p05 | 0.6476 | 0.2473 | 0.2933 | 0.758 |
| alpha_p95 | 0.9032 | 0.0739 | 0.8451 | 1.0 |
| q_relative_aoi_pearson | 0.9015 | 0.1285 | 0.717 | 0.9562 |
| q_argmax_is_oldest_unique_pct | 94.8008 | 9.2725 | 81.4716 | 98.9802 |
| unique_oldest_state_pct | 93.5234 | 7.2167 | 83.3464 | 97.9688 |
| phi_mean | 0.4235 | 0.043 | 0.3719 | 0.4568 |
| phi_alignment_spearman | 0.6687 | 0.3158 | 0.2255 | 0.819 |
| phi_alignment_pearson | 0.6039 | 0.2976 | 0.194 | 0.7615 |
| alpha_alignment_spearman | -0.1165 | 0.3505 | -0.471 | 0.11 |
| q_youngest | 0.0238 | 0.0541 | 0.0018 | 0.1017 |
| phi_youngest | 0.7526 | 0.2685 | 0.4693 | 0.937 |
| q_middle | 0.0646 | 0.0865 | 0.0189 | 0.1875 |
| phi_middle | 0.1819 | 0.2349 | 0.0445 | 0.4908 |
| q_oldest | 0.9116 | 0.1402 | 0.7108 | 0.9793 |
| phi_oldest | 0.336 | 0.1919 | 0.1551 | 0.563 |

## 행동 대체 실험 (closed-loop, RSMA-PPO 2.10M)

대체 상수와 AoI 순위 규칙은 별도 보정 궤적(eval_seed = 980278 + 100·permutation)에서 정책별로 구했다.
평가 궤적은 위 5×6과 같다.

| mode | aoi_increase_pct | ci95_training_seed | min_seed | max_seed | training_seeds_worse |
|---|---|---|---|---|---|
| fixed_alpha | 38.372 | 82.34 | 5.236 | 156.713 | 5 |
| rank_q | 13.483 | 26.042 | 0.774 | 50.776 | 5 |
| fixed_phi_level | 65.309 | 87.448 | 31.538 | 191.256 | 5 |

## 36개 조건 격자 (2.10M RSMA-PPO, 조건당 평가 seed 1개 × 순열 6개)

| stratum | training_seeds | ppo_mean_aoi | single_user_mean_aoi | gain_mean_pct | gain_seed_std_pct | gain_ci95_halfwidth_pct |
|---|---|---|---|---|---|---|
| feasible | 5 | 1.764 | 2.0 | 11.808 | 3.993 | 4.959 |
| stress | 5 | 170.822 | 67.933 | -2877.712 | 2348.767 | 2916.38 |

## 해석상 주의

- 학습량은 두 방법이 같다(누적 2.10M). 다만 행동 차원과 최적화 난도가 달라 RSMA-PPO와 SDMA-PPO의
  차이를 common stream만의 인과효과로 단정하지 않는다.
- 목표 조건 학습은 세 단계로 나뉘어 learning-rate annealing이 두 번 재시작됐다. 하나의 연속 스케줄로
  학습한 것과 같지 않다.
- 격자 평가는 목표 조건으로 미세조정한 최종 정책을 적용한 것이며, stress 조건은 PPO 학습에 포함되지
  않았다(사전학습에만 포함). feasible 평균 AoI는 0.52M 정책보다 약간 높아졌다(목표 조건 과적합 가능성).
- 행동 역할 분리는 seed에 따라 다르다. seed 0, 2, 3, 4는 q_k–상대 AoI 상관 0.94~0.96, φ_k–정렬도
  Spearman 0.68~0.82, α의 slot 간 SD 0.05~0.09로 역할이 분리됐지만, seed 1은 α를 넓게(SD 0.24,
  5~95%: 0.29~1.00) 조절하는 다른 해를 찾았고 AoI는 비슷했다(1.404). 그래서 논문 표 2는 평균±CI 대신
  학습 seed 중앙값 [최소–최대]를 쓴다. 대체 실험의 AoI 증가도 seed 1에서 크게 튄다(+51~191%).
- seed-0 전용 감사(`docs/feedforward_policy_state_adaptation_audit_2026-09-25.md`)의 수치(α 고정 +5.44%
  등)는 seed 0 진단값이다. 5개 seed 결과는 위 표를 따른다.
- 논문 v2(2026-09-28)부터 stress 조건(SNR 5 dB 또는 n=64)의 수치와 "단일 사용자 전송이 우세했다"는
  결론을 뺐다. PPO가 학습하지 않은 조건이라 분포 밖 일반화 실패, 전체 사용자 활성화, RSMA 구조, 물리적
  난도 중 원인을 구분할 수 없기 때문이다. 논문에는 "학습에 포함된 SNR·blocklength 범위에 한정"이라는
  범위 한계로만 남겼다. 위 격자 표의 stress 행은 내부 기록이다.
