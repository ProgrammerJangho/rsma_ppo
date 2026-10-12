# 프로젝트 중간 점검 — 2026-09-16

> **상태 주의 (2026-09-22):** 이 문서는 gate/`rho` action을 사용하던 이전 결과
> 세대의 중간 점검 기록이다. 현재 gate-free `(alpha, q_k, phi_k)` 정의는
> `canonical_action_definition.md`, 최신 5-seed PPO 결과와 남은 범위는
> `mixed_phy_sinr_curriculum_2026-09-20.md`를 기준으로 한다. 아래 수치는 역사적
> 비교와 결정 추적을 위해 보존하며 현재 headline으로 사용하지 않는다.

## 1. 현재 판단

프로젝트는 구현 검증 단계를 지나 논문 가설을 시험할 수 있는 상태다. 채널, FBL
PHY, 1-layer RSMA, exact common-off 경계, PPO action, objective-aware warm start와
평가 도구가 동작하며 전체 테스트 596개가 통과한다. 핵심 결과도 일관된 연구
이야기를 지지한다.

2026-09-09 후속 점검에서 공식 CSIT 설정을 기존 결과 세대와 같은 actor/critic LR
`3e-4/3e-4`로 고정했다. `critic_lr=1e-3` 결과는 optimizer sensitivity audit으로만
보존한다. 또한 최종 평가 protocol과 checkpoint inventory를 별도 YAML로 고정하고,
기존 checkpoint의 장기 paired held-out 평가와 RSMA/SDMA의 초기화 조건을 맞춘
후속 감사를 완료했다. 후속 감사가 현재 headline 결과이며, 서로 다른 초기화를 쓴
이전 5.7% 차이는 pretraining 효과가 섞인 진단 결과로만 보존한다.

## 2. 현재 논문 질문

> Directional Rician MU-MISO의 finite-blocklength AoI 제어에서 PPO가
> common/private 전력과 사용자별 common-rate action을 안정적으로 학습할 수 있는가?
> Objective-aware pretraining과 exact common-off gate가 FBL decoding cliff에서의
> 학습 실패를 줄이고, 채널 분리도와 CSIT 품질에 따라 common stream을 유지하거나
> SDMA 경계로 전환하게 하는가? 수렴 정책에서 실제로 상태 의존성이 필요한 action
> 성분은 무엇인가?

이 질문은 단순한 “RSMA + PPO 결합”보다 현재 코드와 결과에 잘 맞는다.

## 3. 완료된 구현

| 항목 | 상태 | 근거 |
|---|---|---|
| 1-layer RSMA common/private PHY | 완료 | common 선복호, SIC 후 private SINR |
| Finite-blocklength BLER | 완료 | normal approximation과 결합 성공확률 |
| Directional Rician ULA channel | 완료 | 평균 전력 보존, angle/K-factor 설정 |
| Gauss–Markov temporal channel | 완료 | 시간 상관 및 CSIT error 분리 |
| PPO power allocation | 완료 | common/private power simplex |
| Common-rate allocation | 완료 | common payload의 사용자별 simplex |
| Common payload scale | 완료 | 총 common bits 결정 |
| Exact common-off gate | 완료 | 실행 가능한 SDMA boundary |
| Objective-aware actor pretraining | 완료 | 실제 FBL objective 기반 warm start |
| Fixed RZF/principal beams | 완료 | beam은 action에서 제외 |
| CEM allocator/reference | 완료 | exact common-off 후보 포함 |
| Deterministic held-out evaluation | 완료 | checkpoint별 고정 channel/ACK seed |
| Tail/peak/user metrics | 구현 완료 | p95, p99, peak, 사용자별 AoI/서비스 간격 |
| Reproducibility metadata | 대부분 완료 | config snapshot, seed, source/checkpoint hash |

## 4. 완료된 실험과 해석

### PHY regime

- 5°와 20°에서는 matched-budget CEM-RSMA가 CEM-SDMA보다 낮은 장기 AoI를 보였다.
- 60°에서는 CEM-SDMA가 더 좋았고 RSMA allocator가 common payload를 거의 0으로
  줄였다.
- 이는 RSMA가 항상 우수하다는 결론보다 spatial separability에 따른 regime
  transition을 지지한다.

### PPO 학습 실패와 warm start

- 낮은 `rho` 초기화는 안전한 near-SDMA basin에 머물렀다.
- 넓은 `rho` 탐색은 FBL BLER cliff를 자주 넘어 성능이 악화됐다.
- Objective-aware actor warm start는 기존 low-rho PPO 대비 AoI를 5°에서
  1.7092→1.5730, 20°에서 1.7300→1.5010, 60°에서 1.0740→1.0232로 낮췄다.
- 60° 결과의 common power/payload는 거의 0이므로 RSMA gain보다 SDMA boundary를
  학습한 결과로 해석해야 한다.

### Action ablation

20°, `sigma_e=0.2`의 기존 3-seed 결과는 power-only 3.0426, split-only 2.6853,
joint 1.5881, joint+gate 1.5517, learned SDMA-power 1.6954 AoI/user다. 이 결과는
기존에 선택한 고정 power/rate 설정이 나쁘고 gate가 작은 추가 개선을 준다는 것을
보인다. 그러나 고정 설정을 충분히 보정한 대조군이 아니므로, 연속적인 common-rate
제어가 필요하다는 증거로 사용하지 않는다. 아래 262k two-mode 감사가 이 해석을
대체한다.

### Imperfect CSIT

20°에서 single-realisation objective warm start는 `sigma_e=0.0,0.1,0.2,0.3`에
대해 약 1.5550, 1.5457, 1.5517, 1.5626의 AoI를 기록했다. Posterior SAA-8은
일관된 장기 이득을 보이지 않았다. 따라서 SAA는 핵심 기여가 아니라 negative
ablation이다.

### PPO convergence

Held-out checkpoint audit에서 critic LR `1e-3`은 초기 1.5463에서 최종 1.5149로
AoI를 2.03% 낮췄다. 세 PPO seed 모두 기존 critic LR보다 좋았고 NaN, KL 폭주,
constraint violation 또는 early stop은 없었다. Critic explained variance는
-1.217에서 -0.377로 개선됐지만 여전히 음수다. 정책 학습은 정상으로 판단하되
critic의 완전 수렴은 주장하지 않는다.

## 5. 현재 강점

1. 결과가 RSMA 우월성을 전제로 하지 않고 60° negative regime을 보존한다.
2. FBL cliff, action coupling, warm-start 효과가 코드와 수치로 연결된다.
3. RSMA가 SDMA를 포함하도록 exact boundary를 구현해 잘못된 비교를 줄였다.
4. 동일 channel/ACK seeds와 deterministic evaluation으로 비교 조건을 통제한다.
5. 학부논문 범위에 맞게 beam/precoder/blocklength/scheduling을 고정했다.

## 6. 해결해야 할 문제

### 해결됨 — 실험 세대 불일치

기존 action ablation과 CSIT sweep에 맞춰 공식 config를 `3e-4/3e-4`로 복구했다.
따라서 optimizer 세대 불일치는 해소됐다. `1e-3` audit은 정책 안정성 확인과 optimizer
sensitivity 근거이며 main-table 정책과 혼합하지 않는다.

### 완료 — 초기 평가와 초기화 confound 확인

최종 protocol은 20°, `sigma_e=0.2`, SNR 10 dB, 8,192 slot, burn-in 2,048 slot,
6개 공통 trajectory로 고정했다. Headline 정책을 5 training seeds로 확장한 결과:

| 방식 | 평균 AoI | 불확실성 단위 |
|---|---:|---|
| objective-warm joint+gate PPO | 1.5511 | 5 training seeds |
| SDMA-power PPO | 1.6448 | 5 training seeds |
| scratch joint+gate PPO | 1.6960 | 5 training seeds |
| equal SDMA | 2.7950 | 6 trajectories |
| fixed RSMA (`f_c=0.25`, `rho=0.30`) | 16.9793 | 6 trajectories |

Objective-warm 정책은 trajectory별로 training-seed 평균을 낸 paired 비교에서
SDMA-power PPO보다 6/6 trajectory 및 5/5 same-index training seed에서 낮은 AoI를
보였고 평균 차이는 -0.0937(약 5.7%)였다. Scratch joint policy는 SDMA-power보다
평균 0.0513 높았다. 따라서
RSMA action만으로 개선된 것이 아니라 objective-aware initialization이 핵심이라는
해석이 필요하다. Fixed RSMA의 큰 손실은 부적절한 고정 `rho`가 FBL cliff를 넘는
현상을 다시 보여준다.

이 비교에서는 RSMA만 objective-aware initialization을 사용하고 SDMA는 scratch에서
학습했다. 따라서 5.7%를 RSMA 구조의 이득으로 인용하지 않는다.

### 완료 — matched-initialization 감사

SDMA-power actor에도 common-off FBL one-step objective를 직접 최소화하는 사전학습을
구현했다. RSMA와 SDMA 각각 사전학습 직후(`0-step`)와 65,536-step PPO 이후를
5 training seeds, 새로운 6개 held-out trajectory에서 평가했다.

| 비교 | candidate AoI | reference AoI | paired 차이 | 승리 seed |
|---|---:|---:|---:|---:|
| RSMA vs SDMA, 0-step | 1.5508 | 1.5824 | -2.00% | 5/5 |
| RSMA vs SDMA, PPO 이후 | 1.5475 | 1.5751 | -1.75% | 5/5 |
| PPO 증가분, RSMA | 1.5475 | 1.5508 | -0.21% | 4/5 |
| PPO 증가분, SDMA | 1.5751 | 1.5824 | -0.46% | 3/5 |

따라서 초기화 confound를 제거한 뒤에도 65,536-step 조건에서 고정 beam RSMA action
class에는 약 1.75--2.00%의 작은 이점이 남는다. 이 시점의 PPO 추가 개선은 두 방식
모두 0.5% 미만이었다. 그러나 아래 262k budget audit 결과 때문에 PPO를 항상 작은
보정으로만 기술하지 않는다.

### 완료 — 0/65k/262k matched budget 감사

동일 번호의 objective-pretrained actor에서 시작해 RSMA joint+gate와 SDMA-power를
262,144 interactions까지 각각 5 seeds로 학습했다. 0/65k/262k 정책을 같은 6개
held-out trajectory에서 평가한 결과는 다음과 같다.

| PPO budget | RSMA AoI | SDMA AoI | RSMA 상대 차이 | RSMA 승리 seed |
|---:|---:|---:|---:|---:|
| 0 | 1.5508 | 1.5824 | -2.00% | 5/5 |
| 65,536 | 1.5475 | 1.5751 | -1.75% | 5/5 |
| 262,144 | 1.4574 | 1.5610 | -6.64% | 5/5 |

262k RSMA는 자기 0-step actor보다 6.02% 개선됐고 5/5 seeds에서 같은 부호를
보였다. SDMA의 대응 개선은 1.35%였다. 따라서 PPO의 효과가 없다는 결론은 기각한다.
다만 65k와 262k는 각 horizon 전체에 learning-rate annealing을 적용한 독립 run이다.
이 결과는 계산 예산과 annealing horizon이 함께 증가한 endpoint 비교이며, PPO가
비근시안적 계획을 학습했다는 직접 증거는 아니다.

### 완료 — 262k action 단순화 감사

각 262k checkpoint의 gate-on 조건부 action 평균을 별도 calibration trajectory에서
구하고, 평가 trajectory와 분리한 뒤 중첩 대조군을 비교했다.

| 정책 | 평균 AoI | 원 PPO 대비 |
|---|---:|---:|
| common magnitude만 고정 | 1.4539 | -0.24% |
| 원 PPO | 1.4574 | 기준 |
| private power만 연속 제어 | 1.4625 | +0.36% |
| private-user argmax만 유지 | 1.5910 | +9.17% |
| common-rate만 연속 제어 | 1.7605 | +20.80% |
| gate 외 모든 allocation 고정 | 1.7730 | +21.66% |

대표 20° 조건에서는 common power 크기, `rho`, 사용자별 common-rate 비율의
slot별 변화가 필요하지 않았다. 반면 gate와 사용자별 private power의 연속적인
상태 의존 제어는 중요했다. 따라서 논문에서는 “모든 power/rate action의 연속 공동
제어가 필수”라고 주장하지 않는다. 구현 action은 그대로 유지하되, 수렴 정책이
학습한 유효 구조를 exact mode selection과 private-power adaptation으로 기술한다.

또한 높은 `rho`는 전체 사용자 payload의 합에 곱해지는 값이 아니다. 현재 mapping은
`D_c = rho * min_k(D_k/c_k)`이고, common codeword에는 주로 한 사용자의 split
unicast payload가 들어간다. 따라서 이 정책을 multicast라고 부르지 않고, 비대칭적인
common-dominant 1-layer RSMA 또는 two-user NOMA 경계에 가까운 해로 해석한다.

### 완료 — 하나의 정책을 이용한 mixed-geometry 재검증

5°·20°·60° 중 하나를 episode마다 균등 선택해 하나의 objective-warm PPO를 5 seeds,
262,144 interactions로 학습했다. 정책은 5°/20°/60°에서 common gate를 각각
0.937/0.571/0.015 비율로 켰다. 따라서 채널 정렬도에 따른 mode 선택은 명확하다.

혼합 분포에서 checkpoint별로 한 번만 구한 `alpha≈0.824`, `rho≈0.892`를 세
geometry에 공통 적용해도 원 PPO 대비 AoI 변화는 +0.31%/-0.27%/+0.54%였다.
common-rate simplex를 매 slot argmax 사용자 선택으로 바꾸면 -0.26%/-0.10%/-0.003%,
두 축약을 함께 적용해도 -0.05%/-0.56%/+0.39%였다.

따라서 fixed-geometry 인공물이라는 가설은 현재 모델 범위에서 기각한다. 유효 정책은
exact SDMA gate, common recipient의 이산 선택, 연속 private-power 배분 및 gate-on
상태의 거의 고정된 높은 common power/payload로 축약된다. 이는 일반 RSMA의 정리가
아니며 `K=Nt=2`, fixed beam, equal payload/weight, `simplex_scaled` FBL 모델에 대한
결과로 한정한다.

### 완료 — mixed geometry + SNR 확장과 common-off 원인 분해

5°/20°/60°와 5/10/15 dB의 9조건을 episode마다 무작위로 선택하는 하나의 정책을
3 seeds, 262,144 interactions로 학습했다. 별도로 학습한 SDMA-power PPO보다 seen
평균 AoI는 3.32%, 4개 보간 조건 평균은 4.18% 낮았다. 그러나 RSMA actor의 common
stream만 끄고 private-power 비율을 유지한 projection은 원 RSMA보다 seen 0.36%,
unseen 0.24% 더 좋았다. 따라서 이 3--4% 차이는 RSMA common stream의 이득이 아니라
private-power 학습 품질 차이다.

정책의 common-on 비율은 5°에서 SNR 5/10/15 dB에 따라 3.26/10.06/25.12%로
증가했고, 60°에서는 0.00/0.05/0.50%였다. 환경 변수에는 반응했지만 그 common action이
장기 AoI를 개선하지는 못했다. 고정 20° 정책의 높은 `alpha/rho` 포화도 유지되지
않았으므로 이를 시스템의 구조적 필연으로 보지 않는다.

동일 짧은 protocol에서 raw RSMA-CEM은 SDMA-CEM보다 3.04% 나빴지만, exact SDMA
경계를 포함한 safeguarded CEM은 5.78% 좋았고 RSMA-PPO보다 18.99% 낮은 AoI를
기록했다. 환경 변동성만으로 PPO가 CEM을 이긴다는 가설은 현재 결과에서 기각한다.
세부 protocol과 조건별 표는 `docs/mixed_geometry_snr_results.md`에 있다.

### 완료 — 대칭 관측·action-objective 사전학습 후속 실험

위 mixed 실험에서 확인한 index shortcut과 common-action 학습 실패를 직접 겨냥해
phase-invariant `gain_features`, 양방향 사용자 순서 augmentation, CEM action label과
FBL objective를 결합한 사전학습을 적용했다. PHY, beam, action과 262k budget은
바꾸지 않았다.

새 262k actor와 같은 actor의 exact common-off projection을 4096-slot paired
protocol로 비교한 결과, common stream은 seen 9조건에서 평균 AoI를 4.15%, 보간
4조건에서 3.49% 낮췄다. 5°에서는 6.17--9.28%, 20°에서는 1.65--3.18% 이득이었고,
40°/60°에서는 약 0%로 SDMA boundary에 수렴했다. 세 training seed 모두 평균 이득의
부호는 양수였지만 크기는 +0.37--10.03%로 편차가 크다.

동일 CEM protocol에서는 새 PPO 1.6986, raw RSMA-CEM 1.4937, SDMA-CEM 1.4496,
safeguarded CEM 1.3658이었다. 따라서 **common stream의 실제 기여는 회복했지만
allocator 품질 격차는 남았다**. 현재 가장 정확한 결론과 표는
`docs/symmetric_rsma_followup_results.md`에 있다.

### 완료 — K=3, Nt=2 overloaded extension

세 사용자를 두 안테나로 동시에 서비스하는 1-layer RSMA를 all-3 SDMA 및 모든
2-user subset을 탐색하는 scheduled RSMA/SDMA와 비교했다. 가장 유망한 20°·15 dB의
128-sample high-budget one-step 감사에서 all-3 RSMA는 all-3 SDMA보다 expected AoI를
10.47% 낮췄다. Scheduled-2 RSMA와의 0.84% 차이는 95% 구간이 0을 포함했다.

동일 조건의 sampled long-run에서는 all-3 RSMA 1.3973, all-3 SDMA 1.5611,
scheduled-2 RSMA 1.3367, scheduled-2 SDMA 1.4310이었다. 세 평가 seed 모두에서
RSMA가 대응 SDMA보다 좋았지만 전체 최선은 scheduled-2 RSMA였다. 따라서 K=3은
RSMA 유효성을 강화하는 extension이지만, 주된 후속 질문은 강제 all-user overload보다
AoI-aware subset selection과 RSMA allocation의 결합이다. 세부 결과는
`docs/k3_nt2_overloaded_results.md`에 있다.

All-3 allocation-only PPO도 세 seed, 262,144 step으로 완료했다. 4,096-slot 장기
평가에서 0-step actor의 평균 AoI 1.6750을 1.6326으로 2.53% 개선했고, paired
common-off 2.1121보다 22.70% 낮아 common stream 자체의 실효성은 확인했다. 그러나
동일한 짧은 CEM protocol에서는 all-3 SDMA 1.5611, scheduled-2 SDMA 1.4310,
all-3 safeguarded RSMA-CEM 1.3925보다 모두 나빴다. 따라서 K=3 PPO는 정상 학습된
negative optimization-gap 결과이며 headline 우월성 근거로 쓰지 않는다. 여섯 user
순열의 평균 AoI spread는 0.0425로 catastrophic index shortcut은 없었지만 완전한
permutation equivariance도 확인되지 않았다.

후속 진단으로 이 negative optimization gap의 원인을 분해했다. CEM과 같은 상태에서
비교하면 gate를 완전히 고쳐도 제거 가능한 final regret은 최대 3.7%였고, Shapley
분해상 63.6%가 state-dependent power allocation에서 발생했다. 비유일한 CEM action
coordinate를 따라가는 loss를 제거한 objective-only actor는 장기 AoI 1.5223을 기록해
기존 0-step 1.6750과 262k PPO 1.6326을 모두 개선했다. 동일 short protocol에서는
all-3 SDMA-CEM보다 2.26% 낮지만 scheduled-2 SDMA/RSMA와 all-3 safeguarded CEM에는
미달한다. Objective-only actor에 65k PPO를 추가하면 9.09% 악화됐으므로 K=3 PPO를
더 연장하지 않는다. 이는 FBL cliff에서 stochastic exploration 비용이 14.1%였던
감사와 일치한다. `docs/k3_ppo_gap_diagnosis.md`에 상세 결과를 기록했다.

### 완료 — 공간 정렬도와 near/far 재검증

- Perfect-CSIT 5°/20°/60°에서 objective-warm RSMA-PPO와 learned SDMA-power PPO를
  동일 조건으로 비교했다. RSMA 상대 gain은 각각 13.67%, 9.67%, 0.26%였다.
- 60°에서는 common power 0.019, `rho=0.00029`, common-on 0.037로 SDMA 경계에
  접근했다. 실질적 RSMA gain을 주장하지 않는다.
- Directional-Rician 20°, relative CSIT error 0.2, normalized 5 dB near/far를 다시
  학습·평가했다. RSMA-PPO 2.3720, SDMA-power PPO 2.9601로 과거 성능 역전이
  재현되지 않았다.
- Near/far 결과는 한 gain gap의 stress test이며 보편적인 heterogeneous 우월성으로
  확대하지 않는다.

### 완료 — 최종 통계 프로토콜

- Headline은 5 training seeds, secondary regime은 3 training seeds다.
- 모든 최종 평가는 6개 paired trajectories, 8,192 slots, burn-in 2,048를 사용한다.
- mean/tail/peak, BLER, common-on, power와 `rho`를 동일 표본에서 저장한다.
- training-seed uncertainty와 evaluation-trajectory uncertainty를 분리한다.
- SDMA seed variance 때문에 paired difference와 seedwise wins를 함께 저장한다.

### 완료 — CEM과 262k PPO의 protocol 일치

최종 262k RSMA/SDMA PPO와 CEM을 동일 seed 930000--930005, 8,192 slots 및 2,048-slot
burn-in에서 재평가했다. 평균 AoI는 RSMA-PPO 1.4574, SDMA-PPO 1.5610, raw
RSMA-CEM 1.2249, SDMA-CEM 1.3437, safeguarded CEM 1.2171이다. 동일 960-candidate
예산에서 raw RSMA-CEM은 SDMA-CEM보다 8.85% 낮다. Safeguarded CEM은 RSMA-PPO보다
16.49% 낮고 6/6 trajectory에서 같은 부호이며, post-burn-in 슬롯의 52.37%에서
SDMA 경계를 선택한다.

따라서 RSMA action class의 대표 operating-point 이점과 PPO의 numerical-reference
gap이 동시에 확인됐다. CEM은 finite-budget one-step reference이지 global 또는
long-horizon optimum이 아니다. 최종 근거는 `docs/cem_262k_long_results.md`와
`results/thesis_cem_262k_long_v1/`이다.

### Major — baseline 표의 통일

최종 핵심 baseline은 fixed RSMA, CEM-RSMA, equal/optimised SDMA, SDMA-power PPO,
scratch PPO와 objective-warm PPO면 충분하다. OMA는 conventional reference로 넣을
수 있지만 자원 정수화를 다시 감사해야 한다. NOMA, WMMSE, SAC/TD3는 현재 fixed-beam
AoI 질문의 필수 baseline이 아니다.

### 완료 — SNR operating-load sensitivity

SNR 5/10/15 dB에서 조건별 oracle dataset과 3개의 objective-aware actor를 새로
생성하고, RSMA-PPO와 SDMA-power PPO를 조건별 3 seed로 재학습했다. 동일한 6개
trajectory의 장기 평가에서 평균 AoI는 각각 RSMA 2.4803/1.5440/1.0342,
SDMA 3.2489/1.6709/1.2164였다. 모든 조건에서 3/3 same-index seed와 6/6 paired
trajectory를 이겼다.

다만 5 dB의 `rho=0.00051`로 common payload가 사실상 꺼져 있고 SDMA seed 분산이
크다. 따라서 5 dB gain은 RSMA common stream의 이득보다 objective-aware 학습과
전력 정책의 안정성으로 해석한다. 각 SNR에서 재학습했으므로 unseen SNR 일반화도
주장하지 않는다.

### Minor — pretraining variation

Headline seed 0~4는 같은-index objective-pretraining seed와 PPO seed를 사용한다.
Spatial 5°/20°/60°의 기존 3-seed PPO는 한 고정 pretrained actor에 조건부다. 이
제한을 논문에 명시하면 secondary ablation으로 허용 가능하다. 완전한 crossed seed
실험은 필수로 보지 않는다.

### Engineering — version control

현재 폴더에는 Git metadata가 없다. 다음 변경 전에 repository를 초기화하고 현재
정리 상태를 baseline commit으로 남기는 것이 필요하다.

## 7. 남은 필수 작업

1. 확정된 표와 그림을 논문 본문의 실험 설정, 결과, negative regime 및 한계 절에
   반영한다.
2. KICS 1--2페이지 형식에 맞춰 핵심 결과만 압축한 투고 원고를 작성한다.

동일 protocol의 최종 CEM reference가 완료됐다. Raw RSMA-CEM과 SDMA-CEM은 각각
960 candidates/slot, safeguarded CEM은 두 후보를 모두 풀어 1,920 candidates/slot을
사용한다. PPO가 numerical reference에 도달했다는 주장은 하지 않는다. PPO actor는
offline 학습 후 고정 횟수의 feed-forward 계산으로 결정한다는 구조적 차이만 주장한다.
현재 Python 구현에서 측정한 latency 비율은 구현과 하드웨어에 민감하므로 핵심 학술
결과로 사용하지 않는다. 기존 3개 2,048-slot short audit은 traceability 용도이며
최종 성능 표에는 사용하지 않는다.

## 8. 하지 않는 것이 좋은 확장

- stochastic queue/HARQ
- blocklength를 RL action에 추가
- full precoder learning
- Digital Twin/semantic state 추가
- max-min common beam 전체 PPO 재학습
- SAC/DDPG/TD3 대규모 알고리즘 경쟁
- “RSMA는 항상 SDMA보다 우수하다”는 결론

이들은 현재 기여를 흐리고 실험 행렬을 크게 늘린다. 향후 연구 항목으로 남기는 것이
적절하다.

## 9. 현실적인 진행률

| 영역 | 진행률 판단 |
|---|---:|
| 핵심 시스템/PHY 구현 | 90% |
| PPO action과 학습 안정성 | 95% |
| 핵심 가설 검증 | 95% |
| 최종 baseline/통계 실험 | 100% |
| 논문용 표·그림 | 95% |
| 본문 작성 | 20% |

전체 연구 구현·실험은 완료 단계다. 필수 실험은 모두 끝났으며, 논문 본문 작성과
표·그림 배치가 남은 작업의 중심이다.
