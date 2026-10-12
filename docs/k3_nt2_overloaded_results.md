# K=3, Nt=2 overloaded RSMA 진단 — 2026-09-16

## 1. 목적과 설정

기존 `K=Nt=2`에서 RZF-SDMA가 너무 강해 common stream의 역할이 작다는 가설을
검증하기 위해 `K=3, Nt=2` extension을 추가했다. RZF는 overloaded 행렬에서도
regularized inverse를 계산하지만 세 private stream의 간섭을 모두 제거할 수 없다.

주요 조건은 다음과 같다.

- Directional Rician, K-factor 8 dB, Gauss--Markov `alpha=0.8`
- Relative CSIT error 0.2
- FBL `n=128`, 사용자당 payload 256 bits
- Principal common beam, RZF private beams
- 사용자별 동일 path gain과 AoI weight
- Geometry span 5/20/60도:
  `[-2.5,0,2.5]`, `[-10,0,10]`, `[-30,0,30]`
- SNR 5/10/15 dB

비교 방식은 다음과 같다.

| 방식 | 설명 |
|---|---|
| all-3 RSMA | 세 사용자를 동시에 서비스하는 1-layer RSMA |
| all-3 SDMA | 세 private stream을 동시에 전송하는 overloaded SDMA |
| all-3 safeguarded | 추정 채널에서 RSMA와 exact SDMA boundary 중 선택 |
| scheduled-2 RSMA | 세 가지 2-user subset을 모두 평가해 최선 선택 |
| scheduled-2 SDMA | 동일 subset 탐색을 사용하는 SDMA |
| scheduled-2 safeguarded | subset과 RSMA/SDMA mode를 함께 선택 |

모든 CEM 선택에는 현재 AoI와 `h_hat`만 사용했고 최종 score는 `h_true`에서 계산했다.

## 2. 전체 9조건 one-step 감사

조건당 16개 독립 채널과 네 AoI profile, 총 64개 paired sample을 사용했다. 각 CEM
solve는 `2 restarts x 5 iterations x 256 population = 2,560` candidates다.

All-3 safeguarded RSMA의 scheduled-2 SDMA 대비 변화는 다음과 같다. 양수는 낮은
expected next AoI를 뜻한다.

| Geometry | 5 dB | 10 dB | 15 dB |
|---|---:|---:|---:|
| 5° | -1.66% | +1.54% | +10.72% |
| 20° | -0.38% | -2.10% | +11.63% |
| 60° | -2.20% | -9.09% | -3.55% |

낮은 SNR과 분리된 geometry에서는 scheduled SDMA가 강하다. 정렬된 5°·15 dB와
중간 정렬 20°·15 dB에서는 세 사용자 동시 RSMA가 scheduled-2 SDMA보다 약 11%
좋았다. 그러나 대부분의 aligned/intermediate 조건에서 scheduled-2 RSMA가 all-3
RSMA보다 더 낮은 one-step AoI를 냈다.

## 3. CEM budget sensitivity

K=3 action 차원 증가 때문에 raw RSMA 탐색이 불리할 수 있어 동일 채널 일부에서
candidate budget을 2,560에서 9,216으로 높였다. 높은 budget에서도 다음 패턴이
유지됐다.

- 5°·10/15 dB: scheduled-2 RSMA가 all-3 RSMA보다 좋음
- 20°·10 dB: scheduled-2 RSMA가 좋음
- 20°·15 dB: all-3 RSMA가 scheduled-2 RSMA와 비슷하거나 조금 좋음
- 60°: scheduled-2 SDMA boundary가 우세

따라서 all-3 결과가 작은 CEM budget 때문에 생긴 단순 artefact라고 보기는 어렵다.

## 4. 20°·15 dB 확대 감사

가장 유망한 20°·15 dB를 32개 채널 x 4 AoI profile, 총 128 paired sample과
9,216-candidate CEM으로 확대했다.

| 방식 | Expected next AoI |
|---|---:|
| all-3 RSMA | **3.9708** |
| all-3 SDMA | 4.4352 |
| scheduled-2 RSMA | 4.0046 |
| scheduled-2 SDMA | 4.4338 |

Paired 차이는 다음과 같다.

| 비교 | 평균 차이 | 95% 구간 | 후보 승리 |
|---|---:|---:|---:|
| all-3 RSMA - all-3 SDMA | -0.4644 | [-0.6533, -0.2755] | 91/128 |
| all-3 RSMA - scheduled-2 SDMA | -0.4630 | [-0.7421, -0.1839] | 74/128 |
| all-3 RSMA - scheduled-2 RSMA | -0.0338 | [-0.1381, +0.0706] | 52/128 |

RSMA와 SDMA의 차이는 명확하지만 all-3와 scheduled-2 RSMA의 차이는 통계적으로
분리되지 않았다. Equal-AoI profile에서는 all-3가 scheduled-2 RSMA보다 4.88%
좋았지만 한 사용자가 긴급한 profile에서는 scheduling이 대체로 강했다.

## 5. Sampled long-run AoI

20°·15 dB를 1,024 slots, burn-in 256, 공통 evaluation seed 3개로 검증했다. 온라인
CEM은 매 slot 현재 AoI와 `h_hat`으로 allocation/subset을 다시 선택했다.

| 방식 | 평균 AoI | Common-on | Scheduled-2 SDMA 대비 |
|---|---:|---:|---:|
| scheduled-2 raw RSMA | **1.3367** | 100%* | +6.59% |
| scheduled-2 safeguarded | 1.3390 | 29.2% | +6.43% |
| all-3 safeguarded | 1.3925 | 53.4% | +2.69% |
| all-3 raw RSMA | 1.3973 | 100%* | +2.36% |
| scheduled-2 SDMA | 1.4310 | 0% | 기준 |
| all-3 SDMA | 1.5611 | 0% | -9.09% |

`*` Raw CEM은 softmax/sigmoid 내부점만 생성하므로 exact zero common을 만들지 못한다.
Raw의 100%는 모든 slot에서 common이 실질적으로 중요했다는 증거로 사용하지 않는다.
Safeguarded 정책의 mode 선택이 해석 가능한 지표다.

세 evaluation seed 모두에서:

- all-3 RSMA는 all-3 SDMA보다 10.49% 낮았다.
- all-3 RSMA는 scheduled-2 SDMA보다 2.36% 낮았다.
- scheduled-2 RSMA는 scheduled-2 SDMA보다 6.59% 낮았다.
- scheduled-2 RSMA는 all-3 RSMA보다 4.34% 낮았다.

## 6. 판정

`K=3, Nt=2`는 현재 시스템에서 RSMA 특성을 확인할 수 있는 유효한 extension이다.
동일한 RZF와 FBL model에서 RSMA는 overloaded all-3 SDMA의 residual interference를
완화했고, scheduled SDMA보다도 낮은 장기 AoI를 기록했다. 따라서 기존 K=2 환경이
RSMA 필요성을 약하게 만든다는 가설에는 근거가 생겼다.

하지만 최선의 방식은 all-3 transmission이 아니라 scheduled-2 RSMA였다. 현재 결과는
다음 연구 질문을 지지한다.

> Overloaded user pool에서 AoI-aware subset selection과 RSMA allocation을 함께
> 사용하면 SDMA scheduling보다 freshness를 개선할 수 있는가?

이는 “항상 K>Nt stream을 동시에 전송해야 RSMA답다”는 주장보다 정확하다. 전체 등록
사용자는 송신 안테나보다 많고, 정책은 현재 AoI와 채널에 따라 동시 서비스 집합과
common/private allocation을 결정한다.

## 7. 한계와 다음 단계

- 장기 평가는 유망한 한 조건과 evaluation seed 3개뿐이다.
- 사용자 수 증가와 함께 total offered payload도 512에서 768 bits로 증가했다.
- CEM은 finite-budget one-step optimizer이며 global/long-horizon optimum이 아니다.
- Scheduled CEM은 세 subset을 모두 계산하므로 all-3 policy보다 계산량이 크다.
- K=3 all-user allocation-only PPO는 별도 extension으로 학습·평가했다. 아래 8절처럼
  수렴은 했지만 online CEM 및 scheduled SDMA에는 도달하지 못했다.
- Generalized/group-common RS는 구현하지 않았으며 현재 결과는 1-layer RSMA에 한정된다.

다음 PPO를 실행한다면 all-3 allocation-only PPO 하나만 학습하기보다 schedule-size 2의
AoI-aware subset action과 RSMA allocation을 함께 평가해야 한다. 다만 학부논문 범위를
유지하려면 기존 K=2를 main study로 두고 K=3 결과는 overloaded extension으로 사용하는
것이 안전하다.

## 8. 근거

- 설정: `configs/thesis_rician_k3_nt2_overloaded_csit02.yaml`
- One-step audit: `scripts/audit_k3_nt2_overloaded.py`
- Scheduled CEM: `src/rsma_aoi/baselines/cem_scheduling.py`
- Long-run evaluation: `scripts/evaluate_k3_nt2_overloaded_longrun.py`
- 전체 9조건: `results/k3_nt2_overloaded_cem_audit_v1/`
- Budget sensitivity: `results/k3_nt2_overloaded_cem_budget9216_v1/`
- 20°·15 dB 확대: `results/k3_nt2_overloaded_20deg15db_heavy_v1/`
- Long-run: `results/k3_nt2_overloaded_longrun_v1/`

## 8. All-3 allocation-only PPO

### 8.1 설정

PPO의 역할을 scheduling과 beam 설계에서 분리했다. 세 사용자는 항상 활성화하며,
정책은 `[P_c,P_1,P_2,P_3]`, 세 사용자의 common-rate simplex, `rho`, exact common
gate만 결정한다. Private/common beam은 각각 RZF/principal beam으로 고정했다.

학습 조건은 20° span, 15 dB, relative CSIT error 0.2, Rician K-factor 8 dB,
blocklength 128, 사용자별 payload 256 bits다. 고정 user index를 외우는 것을 줄이기
위해 `[-10,0,10]`의 여섯 순열을 episode마다 무작위화했다. CEM oracle dataset은
train/validation/test 3600/900/900 상태이고, 세 objective-aware actor를 각각
262,144 PPO step으로 fine-tuning했다.

### 8.2 학습 전후와 common-off paired 결과

고정 물리 조건 `[-10,0,10]`, 4,096 slots, burn-in 1,024, 여섯 evaluation seed에서
세 training seed를 평가했다. Common-off는 같은 actor의 private power 비율을 유지한
채 common power와 payload만 0으로 projection한다.

| 방식 | 평균 AoI | success rate | common-on | common power | rho |
|---|---:|---:|---:|---:|---:|
| 0-step actor | 1.6750 | 0.4952 | 0.8024 | 0.5703 | 0.4789 |
| 262k PPO | **1.6326** | 0.4745 | 0.4926 | 0.3956 | 0.4183 |
| 0-step common-off | 2.4680 | 0.3890 | 0 | 0 | 0 |
| 262k common-off | 2.1121 | 0.3976 | 0 | 0 | 0 |

PPO fine-tuning은 전체 평균을 2.53% 개선했지만 training seed별 부호는 일관되지
않았다. Seed 0은 1.57% 악화, seed 1은 7.26% 개선, seed 2는 1.48% 개선됐다.
반면 common-on 정책은 common-off projection을 0-step과 262k 모두 18/18 paired
trajectory에서 이겼다. Aggregate common gain은 각각 32.13%, 22.70%다. 이는 이
overloaded 조건에서 common stream이 실제로 필요하다는 근거지만, projection은
독립적으로 최적화한 SDMA 정책이 아니므로 RSMA-SDMA 최적 정책 비교로 해석하지 않는다.

### 8.3 동일 프로토콜 CEM/SDMA 비교

기존 online-CEM 평가와 같은 evaluation seed 3개, 1,024 slots, burn-in 256으로
맞추면 262k PPO의 평균 AoI는 1.6325다.

| 방식 | 평균 AoI | 262k PPO 대비 |
|---|---:|---:|
| scheduled-2 raw RSMA-CEM | **1.3367** | PPO가 22.13% 높음 |
| all-3 safeguarded CEM | 1.3925 | PPO가 17.23% 높음 |
| scheduled-2 SDMA-CEM | 1.4310 | PPO가 14.08% 높음 |
| all-3 SDMA-CEM | 1.5611 | PPO가 4.58% 높음 |
| all-3 262k PPO | 1.6325 | 기준 |

따라서 학습은 정상 완료됐지만, 현재 actor/pretraining/PPO 조합은 CEM이 확인한
allocation operating point를 충분히 amortize하지 못했다. 특히 all-3 SDMA numerical
baseline에도 미달하므로 “PPO가 overloaded 환경에서 SDMA를 이겼다”는 주장은 할 수
없다. K=3 결과의 긍정적 핵심은 numerical RSMA reference의 이점이며, learned-policy
결과는 남아 있는 optimization gap이다.

### 8.4 User-order 감사

최종 PPO를 동일 geometry의 여섯 user 순열에서 평가한 평균 AoI 범위는
1.6225--1.6650이고 max-minus-min spread는 0.0425였다. 모든 순열을 학습에 포함한
효과로 catastrophic index shortcut은 보이지 않지만, 완전한 permutation equivariance도
아니다. 향후에는 permutation-equivariant actor가 더 직접적인 해결책이다.

### 8.5 재현 경로

- PPO 설정: `configs/thesis_rician_k3_nt2_all3_ppo_csit02.yaml`
- Oracle dataset: `datasets/rician_k3_nt2_all3_csit02_oracle_v1/`
- 0-step actors: `runs/action_objective_actor_k3_nt2_all3_seed*_v1/`
- 262k checkpoints: `results/rician_ppo_k3_nt2_all3_262k_v1/`
- 학습 곡선: `results/rician_ppo_k3_nt2_all3_262k_v1/training_curves.png`
- 장기 paired 평가: `results/thesis_k3_nt2_all3_ppo_long_v1/`
- CEM-protocol 평가: `results/thesis_k3_nt2_all3_ppo_cem_protocol_v1/`
- 순열 감사: `results/k3_nt2_all3_ppo_permutation_audit_v1/`

## 9. PPO gap 후속 진단

후속 same-state 감사에서 262k actor와 CEM의 차이는 common gate보다 continuous
allocation에서 발생했다. Actor action과 그 common-off projection 중 더 좋은 것을
true channel로 고르는 비현실적인 selector도 regret의 3.7%만 제거했다. 세 action
component를 Shapley 방식으로 분해하면 최종 gap의 63.6%가 power, 21.6%가 `rho`,
14.8%가 common-rate allocation에 귀속됐다.

더 중요한 수정은 pretraining loss였다. CEM action coordinate loss를 제거하고 FBL
objective만 직접 최소화하자 128x128 actor의 장기 AoI가 1.6750에서 1.5348로,
256x256 actor는 1.5223으로 개선됐다. 네트워크 크기의 추가 효과보다 action loss 제거
효과가 훨씬 컸다. 동일한 짧은 CEM protocol에서 objective-only 256은 all-3 SDMA
1.5611보다 2.26% 낮았지만 scheduled-2 SDMA 1.4310 및 safeguarded CEM 1.3925에는
미달했다.

Objective-only 128 actor에서 65k PPO를 수행하면 1.5348에서 1.6743으로 9.09%
악화됐고 세 seed 모두 같은 부호였다. 초기 stochastic action sampling은 deterministic
actor보다 one-step objective를 평균 14.1% 악화시켰다. 따라서 K=3에서는 standard
PPO exploration이 FBL cliff 부근의 좋은 allocator를 훼손한다는 진단이 가장 근거가
강하다. 상세 근거는 `docs/k3_ppo_gap_diagnosis.md`에 있다.
