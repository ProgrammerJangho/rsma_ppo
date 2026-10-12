# 졸업논문 결과 장 초안 — Mixed-PHY RSMA-PPO (2026-09-22)

## 1. 최종 실험 범위

본 결과는 `thesis_mixed_phy_final_v1`로 동결한 gate-free RSMA-PPO를
대상으로 한다. 시스템은 단일 기지국, `K=3`, `N_t=2`인 overloaded
downlink MU-MISO이며, 고정 RZF private beam과 principal common beam을
사용한다. 채널은 directional Rician-ULA와 Gauss--Markov 시간 상관을
따르고 상대 CSIT 오차 표준편차는 0.2이다. PHY는 256-bit update와
finite-blocklength normal approximation을 사용한다.

PPO의 연속 행동은

\[
a_t=(\alpha,\mathbf q,\boldsymbol\phi)
\]

이다. `alpha`는 common-stream 전력 비율, `q_k`는 private-power simplex,
`phi_k`는 사용자 `k`의 고정 payload 중 common codeword에 포함되는
비율이다. 별도의 common gate와 `rho`는 사용하지 않는다.

관측에는 AoI와 현재 추정 채널 `h_hat`으로 계산한 channel gain,
correlation, principal-common gain, RZF gain matrix 및 이전 allocation을
적용한 nominal common/private SINR가 포함된다. 실제 채널 `h_true`는
정책 입력에 사용하지 않는다.

## 2. 학습 및 통계 절차

각 training seed는 다음 전체 절차를 독립적으로 수행했다.

1. 동일하게 고정한 train/validation/test dataset에서 서로 다른 초기화,
   minibatch 순서 및 posterior-SAA 난수를 사용해 80 epoch objective
   pretraining을 수행했다.
2. Geometry 5/20/60도, SNR 10/15/20 dB, blocklength 128/256인 Stage 1
   mixed distribution에서 PPO를 524,288 step 학습했다.
3. 20도, 15 dB, blocklength 128인 target 조건에서 524,288 step
   fine-tuning했다.
4. Training seed 0--4의 최종 정책을 동일한 여섯 user permutation,
   channel/CSIT/ACK seed에서 각각 768 slot 평가하고 처음 256 slot을
   burn-in으로 제외했다.

신뢰구간의 독립 표본 단위는 evaluation slot이나 user가 아니라 **다섯
training-seed 평균**이다. 아래 95% 신뢰구간은 자유도 4인 Student-t
구간이다. SDMA는 학습하지 않는 고정 baseline이므로 training-seed
표준편차를 부여하지 않는다.

## 3. 목표 operating point 성능

20도, 15 dB, `n=128`에서 최종 결과는 다음과 같다.

| 방법 | Mean AoI | Seed SD | 95% CI half-width | Mean p95 AoI | 평균 성공확률 |
|---|---:|---:|---:|---:|---:|
| Objective-pretrained actor | 2.481 | 0.270 | 0.336 | 7.508 | 0.463 |
| **Final PPO** | **1.463** | **0.022** | **0.027** | **2.000** | **0.568** |
| Objective-only matched RSMA-PPO | **1.463** | 0.012 | 0.015 | 2.033 | **0.577** |
| Objective-only q-only PPO-SDMA | 1.744 | 0.060 | 0.074 | 3.167 | 0.447 |
| SDMA | 2.822 | — | — | 7.500 | 0.330 |

최종 PPO의 seed별 mean AoI는 1.444--1.500으로 좁게 분포했다. PPO는
각 seed의 pretraining actor보다 평균 40.51% 낮은 AoI를 보였으며 95%
신뢰구간 half-width는 7.88 percentage point이다. SDMA 대비 평균 개선은
48.17%이고 신뢰구간 half-width는 0.95 percentage point이다.

고정 SDMA와의 48.17%에는 사용자 활성화와 학습된 private 전력의 차이가
포함된다. 이를 줄이기 위해 RSMA와 q-only SDMA를 동일한 objective-only
사전학습 loss·epoch, Stage 1, target fine-tuning 예산 및 training seed로
다시 학습했다. 이 matched 비교에서 RSMA의 q-only 대비 seed별 paired
개선율은 **16.01% ± 3.40%p**였고 30개 평가 궤적 모두에서 RSMA의 AoI가
낮았다. 기존 action+objective RSMA와 objective-only RSMA의 최종 차이는
-0.08% ± 2.26%p로 구분되지 않았다. 따라서 기존 격차가 사전학습 loss
차이 때문에 생겼다는 가설은 지지되지 않는다.

이 결과로부터 다음 두 주장은 가능하다.

- Objective pretraining 출력만 사용한 경우보다 장기 sampled-AoI PPO
  fine-tuning이 목표 조건의 AoI를 일관되게 낮춘다.
- 동일한 beam, 총전력, channel/ACK realization을 사용한 SDMA보다 최종
  RSMA-PPO의 AoI가 낮다.
- 동일한 objective-only 사전학습과 PPO 예산에서 common 전력·payload
  제어가 허용된 RSMA policy가 q-only SDMA policy보다 목표 조건의 AoI가 낮다.

반면 p95 AoI가 모든 PPO seed에서 2.0으로 포화되므로 이 조건에서는
tail 개선의 크기를 세밀하게 비교하는 지표로 사용하기 어렵다.

## 4. Common 전력 비율의 상태 적응성

최종 정책의 `alpha`를 전체 36개 regime에서 평가한 결과는 다음과 같다.
괄호는 training-seed 표준편차이다.

| 변화 변수 | 조건별 alpha 평균 (seed SD) |
|---|---|
| Geometry | 5도: 0.619 (0.055), 20도: 0.589 (0.051), 60도: 0.397 (0.049) |
| SNR | 5 dB: 0.250 (0.084), 10 dB: 0.453 (0.053), 15 dB: 0.659 (0.028), 20 dB: 0.778 (0.021) |
| Blocklength | 64: 0.500 (0.106), 128: 0.542 (0.038), 256: 0.562 (0.026) |

Geometry가 5도에서 60도로 증가하면 사용자의 공간 채널이 더 잘
분리되고 평균 common 전력은 감소한다. 반대로 SNR이 증가할수록 common
stream을 신뢰성 있게 decoding할 수 있어 `alpha`가 증가한다. 두 방향은
다섯 training seed에서 모두 유지되며, PPO가 하나의 고정 `alpha`만
학습했다는 설명과 일치하지 않는다.

Blocklength에 대한 변화는 더 작고 `n=64`의 seed 편차도 0.106으로 크다.
따라서 blocklength에 관한 단조 법칙을 주장하지 않고, 정책이
blocklength 정보를 사용하되 해당 효과는 geometry와 SNR보다 약하다고
해석한다.

## 5. 적용 가능한 regime과 실패 영역

학습 전에 구분한 feasible stratum은 SNR 10/15/20 dB와 blocklength
128/256의 18개 regime이다.

| Stratum | PPO mean AoI | SDMA mean AoI | Paired gain vs SDMA | 95% CI half-width |
|---|---:|---:|---:|---:|
| **Feasible** | **1.701** | 2.798 | **26.29%** | **1.13%p** |
| Stress (`5 dB` 또는 `n=64`) | 151.388 | 198.967 | -5.77% | 88.94%p |

Stress stratum은 절대 AoI 평균만 보면 PPO가 낮지만, trajectory별
percentage gain은 작은 SDMA denominator와 거의 실패하는 trajectory에
매우 민감하다. Training-seed 신뢰구간도 0을 크게 포함하므로 우열의
부호를 주장할 수 없다. 이 영역은 `K=3` 사용자가 각각 256 bit를 짧은
blocklength로 동시에 전송하는 조건 자체가 FBL decoding cliff에 가까운
stress test로 해석한다.

따라서 논문의 성능 주장은 target operating point와 feasible stratum에
한정한다. 전체 36개 regime의 평균 하나로 보편적 우수성을 주장하지
않는다.

## 6. 학습 곡선 해석

다섯 target fine-tuning run 모두 초기보다 높은 episode return으로
종료했고 NaN, constraint violation, KL 폭주는 발생하지 않았다. Seed
0/1/2/4의 후반 return은 약 -790~-802, seed 3은 약 -831이었다. 그러나
held-out target AoI는 모든 seed에서 1.5 이하로 수렴했다.

Rollout return의 차이는 유한 episode 표본과 학습 중 stochastic action의
영향을 함께 받는다. 따라서 checkpoint 선택과 최종 주장은 training
return 순위가 아니라 deterministic frozen evaluation을 기준으로 한다.

## 7. 논문에서 사용할 수 있는 결론

### 주장 가능한 내용

1. Causal PHY-aware observation과 staged curriculum을 사용한 PPO가
   geometry와 SNR에 따라 common 전력 비율을 일관되게 조절한다.
2. 20도, 15 dB, `n=128` 조건에서 최종 RSMA-PPO는 다섯 training seed에
   걸쳐 SDMA보다 낮은 AoI를 보인다.
3. Feasible 18-regime 범위에서도 SDMA 대비 이점이 유지된다.
4. Full-grid raw-AoI 학습량을 단순히 늘리면 stress regime이 학습을
   지배하여 feasible 성능을 악화시킬 수 있다. Stage 1 후 target
   fine-tuning이 더 안정적이다.

### 주장하면 안 되는 내용

1. RSMA가 모든 geometry, SNR, blocklength에서 SDMA보다 우수하다.
2. PPO가 전역 최적 allocation을 찾았다.
3. Blocklength에 대한 `alpha` 변화가 일반적인 RSMA 법칙이다.
4. p95=2 결과만으로 tail AoI가 충분히 최적화됐다고 결론낸다.
5. 본 결과를 stochastic arrival, queue, HARQ 또는 learned precoder
   시스템으로 일반화한다.

## 8. 권장 그림과 캡션

### Figure 1. Target performance

> Mean AoI at the target operating point (20-degree geometry, 15 dB SNR,
> blocklength 128). Dots represent independently trained policies; black
> squares show the training-seed mean and standard deviation. The SDMA policy
> has no training randomness.

파일: `results/thesis_mixed_phy_final_v1/figure_1_target_performance.pdf`

### Figure 2. Alpha adaptation

> Common-stream power fraction selected by one PPO policy across geometry,
> SNR, and blocklength regimes. Markers and error bars denote the mean and 95%
> Student-t interval over five independently trained policies.

파일: `results/thesis_mixed_phy_final_v1/figure_2_alpha_response.pdf`

### Figure 3. Training curves

> Target fine-tuning curves for five independent training seeds. Each line is
> an eight-point moving average of episodic return. Final claims use frozen
> deterministic evaluation rather than the training-return ranking.

파일: `results/thesis_mixed_phy_final_v1/figure_3_training_curves.pdf`

## 9. 재현성과 동결 상태

동결된 checkpoint, config 및 평가 script의 SHA-256은
`results/thesis_mixed_phy_final_v1/manifest.json`에 기록했다. 본문 표는
같은 폴더의 `table_*.csv` 또는 `table_*.tex`를 직접 사용한다. 새로운
학습 결과가 생겨도 이 폴더를 덮어쓰지 않고 새 version을 만들어야 한다.
