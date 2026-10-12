# RSMA–PPO–AoI 프로젝트 심사위원 관점 평가 (2026-09-09)

이 문서는 `docs/`, `configs/`, `src/`, `results/`의 현재 상태와 관련 문헌을 대조해 작성한 외부 심사 형식의 평가다. 수치는 모두 저장소의 문서·CSV·학습 로그에서 가져왔다. 저장소에 근거가 없는 항목은 **확인 필요**로 표시했다.

---

> **갱신 고지 (2026-09-10).** 이 평가의 1순위 지적이었던 초기화 confound와 PPO 기여량 문제가 matched-budget 감사로 해소됐다. 아래 절들은 **9절의 갱신 내용이 우선**한다.
>
> | 절 | 원래 주장 | 현재 상태 |
> |---|---|---|
> | 0, 3.5, 4-1 | PPO 기여 0.6~2% | **폐기.** 65k 예산 한정 결과였음. 262k에서 RSMA 6.02% |
> | 3.8, 4-2 | SDMA 대조군만 스크래치 (초기화 confound) | **해소.** 양쪽 모두 seed별 objective 사전학습 |
> | 3.9 | headline 1.5511 vs 1.6448 (5.7%) | **대체.** 1.4574 vs 1.5610 (6.64%), matched |
> | 7 | headline CEM 없음, 지연 0.04 ms 이상 | **해소.** 다만 CEM과 headline의 평가 protocol 불일치가 새로 발견됨 |
>
> 여전히 유효한 지적: 문제 규모(`K=N_t=2`, 빔 고정), MDP framing 미정당화, CEM과의 격차, 누락 baseline, 하이퍼파라미터 선택 경로 오염, near/far 역전의 원인 미분해.

## 0. 한 줄 판정 (2026-09-10 개정)

> 구현·평가 규율은 학부논문 수준을 크게 넘고, 초기화 매칭 이후 **RSMA 행동 클래스와 PPO의 기여가 각각 분리되어 확인됐다.** 262k 예산에서 PPO는 RSMA에 6.02%, SDMA에 1.35%를 기여하며(각 5/5 seed), 이 4.5배 비대칭이 "왜 PPO인가"에 대한 이 프로젝트의 가장 강한 증거다. 남은 약점은 방법이 아니라 **범위와 위치 설정**이다. 문제 규모가 `K=N_t=2`에 빔 고정이고, myopic CEM이 여전히 학습 정책보다 낮은 AoI를 내며, 장기 계획이 필요한 이유가 실험으로 증명되지 않았다. 저널을 노린다면 규모나 결정 변수 중 하나를 열어야 하고, 학부논문이라면 현재 증거로 충분히 강하다.

---

## 1. 프로젝트가 실제로 무엇인지 (코드 기준 요약)

| 항목 | 현재 thesis 설정 (`configs/thesis_rician_alignment_20deg_csit02.yaml`) |
|---|---|
| 시스템 | 단일 BS, `K=2`, `N_t=2`, 1-layer RSMA, 모든 사용자 항상 활성 (scheduling off) |
| 채널 | Rician-ULA (K-factor 8 dB), 사용자 각도 ±10°, Gauss–Markov `alpha=0.8`, 평균전력 보존 |
| CSIT | relative error `sigma_e ∈ {0, .1, .2, .3}`, mean-preserving; CSIR perfect |
| PHY | 정규근사 FBL BLER, `n=128`, `D=256 bit/user` (사용자당 2 bit/use), 완벽 SIC, 파일럿 오버헤드 없음 |
| 빔 | private: RZF (고정), common: principal eigenvector (고정) — **PPO가 빔을 결정하지 않음** |
| 트래픽/AoI | generate-at-will, 성공 시 AoI=1, 실패 시 +1, cap 없음, ACK Bernoulli 샘플링 |
| PPO action | 전력 simplex (K+1), common-rate simplex (K), `rho∈[0,1]` (Beta), Bernoulli common gate |
| 관측 | AoI/1024, `h_hat` 실수·허수 8차원, 이전 ACK, 이전 전력비, 이전 schedule (총 17차원) |
| 보상 | `-0.5 × Σ_k Δ_k(t+1)`; tail/power 페널티 계수 0 |
| PPO | 65,536 steps (4 env × 128), minibatch 128, 4 epoch, `[128,128]` tanh, LR 3e-4 anneal, γ=0.99, λ=0.95 |
| 사전학습 | CEM 라벨 2,400/600/600 states → 미분가능 FBL PHY로 actor 직접 최적화 (60 epoch) |
| 평가 | deterministic policy, 6 paired trajectory × 8,192 slot, burn-in 2,048, headline 5 training seeds |

학습 로그 기준 처리량은 약 610 step/s이므로 PPO 한 run은 CPU에서 약 2분이다.

---

## 2. 관련 문헌과 유사도

### 2.1 가장 가까운 선행연구

| 논문 | 문제 | 최적화 변수 | 방법 | FBL | CSIT | AoI | 우리와의 관계 |
|---|---|---|---|---|---|---|---|
| **Kaya, METU M.S. thesis, Aug 2025** | MU-MISO `K=M=2` 1-layer RSMA, AoI 최소화 | **정규화 precoder 전체(9) + 전력(2) + rate-split ratio(2)** = 13차원 | TD3, γ=0.01, 9,800 ep × 1,000 step | 정규근사 | perfect 학습 → outdated/양자화 CSIT 테스트 (α=0.98) | generate-at-will, 성공 시 AoI=n | **동일 문제·동일 규모.** 우리는 빔을 고정하고 gate·사전학습을 추가 |
| Yan et al., VTC-Fall 2023 | 위성 RSMA short-packet 다중 UE | 전력 배분 | PSO(정적) + DRL(동적, 알고리즘 **확인 필요**) | closed-form BLER/AAoI | shadowed-Rician | 해석적 AAoI | RSMA+FBL+AoI+DRL 조합의 최초 사례 |
| Ali, Abidrabbu, Furqan, Arslan, IEEE IoT J. Sep 2025 | HARQ-RSMA, 우선순위 다른 common/private 패킷 | common/private 전력 + 재전송 | MDP + Lyapunov drift (DRL 아님) | closed-form BLER 근사 | — | AoI + feedback 상태 | AoI를 상태로 쓰는 RSMA 전력제어 |
| arXiv 2604.13691, Apr 2026 (자율주행 short-packet RSMA) | `K=4`, `N_t=5`, 200 km/h, outdated CSIT | 전력 배분 + rate-splitting ratio | multi-start two-step SCA | 구간선형 BLER | Gauss–Markov(Jakes) outdated | closed-form AAoI | ZF private + random common beam으로 **빔 고정** 후 전력/분할 최적화 — 우리와 가장 유사한 문제 구조를 해석적으로 품 |
| Dizdar & Wang, arXiv 2312.07159 | RSMA AoII | scheduling + precoding + power | Big-M + SCA | 무한 | perfect | AoII | RSMA < SDMA 신선도 |
| Hieu et al., IEEE WCL 2021 (arXiv 2107.00238) | RSMA sum-rate, `K=M=4` | 전력 simplex + common-rate 분배 (softmax) | **PPO** | 없음 | imperfect (P^-0.6) | 없음 | 우리 action 구조와 유사(전력 simplex + common-rate 분배); 세부 파라미터화 동일성은 확인 필요 |
| Huang et al., IEEE WCL 2022 / TCOM 2024 | LEO RSMA 전력 배분 | 전력 | PPO / DQN+PPO | 없음 | — | 없음 | PPO-RSMA 선례 |
| Xu, Mao, Clerckx et al., IEEE TVT 2022 (arXiv 2105.06198) | FBL RSMA sum-rate / max-min | precoder + common rate | SCA | 정규근사 | perfect/imperfect | 없음 | FBL-RSMA precoder/common-rate 최적화의 기준 연구; RSMA가 같은 rate를 더 짧은 blocklength로 달성 (common stream on/off regime에 대한 구체적 정리는 원문 확인 필요) |
| Irkiçatal et al., arXiv 2403.05974 | RSMA precoder+전력 | precoder | MADDPG | 없음 | imperfect | 없음 | DRL이 precoder까지 학습 |
| arXiv 2507.08653 (Safe DRL, PAoI, FBL) | WNCS 전력 최소화 | 전력/스케줄 | teacher–student safe DRL | FBL | — | peak-AoI 제약 | "제약을 action 구조로 보장"하는 접근의 최신 예 |
| arXiv 2512.21412 (Freshness-aware RL survey, Dec 2025) | — | — | — | — | — | — | RSMA+AoI+RL 항목이 **없음** → 공백은 맞지만 아직 커뮤니티가 주목하지 않는 조합 |

### 2.2 판정: "RSMA + PPO + AoI 조합" 자체는 새롭지 않다

- RSMA + FBL + AoI + DRL 조합은 **Yan 2023(위성)** 과 **Kaya 2025(MU-MISO, 같은 `K=M=2`)** 가 이미 했다. Kaya는 심지어 precoder까지 학습했고, imperfect CSIT까지 다뤘다.
- PPO + RSMA(전력 simplex + common-rate 분배)는 Hieu 2021, Huang 2022가 했다.
- "빔 고정 후 전력·분할만 최적화하는 short-packet RSMA AoI"는 2604.13691이 SCA로 더 큰 규모(`K=4`)에서 해석적 AAoI까지 도출했다.

따라서 novelty는 조합이 아니라 **문제 정의·최적화 방식의 차별성**에서만 찾을 수 있다. 그 관점에서 실제로 남는 것은 다음 넷이다.

1. policy class 안에 `P_c=D_c=0`를 **정확히** 포함하는 gate (Kaya·Hieu의 softmax는 경계를 근사만 함).
2. FBL decoding cliff 때문에 action-distance 모방(BC)이 실패하고, 미분가능 PHY로 outcome을 직접 최적화한 **objective-aware actor 사전학습**이 필요하다는 발견과 구현.
3. 각도 분리도(5°/20°/60°) × CSIT × SNR로 통제된 **regime 지도**: common stream 사용률이 0.72 → 0.49 → 0.02로 줄고 60°에서는 SDMA 경계로 수렴.
4. paired CRN 평가, 불확실성 단위 분리, negative result 보존 등 **평가 규율**.

이 중 2번이 방법론적 기여에 가장 가깝지만, 학습 커뮤니티에서 "differentiable simulator로 policy를 사전학습한 뒤 RL fine-tune"은 잘 알려진 패턴이다. 무선 DRL 문헌에서의 novelty는 "incremental"로 평가될 가능성이 높다. 3번은 RSMA 이론(Clerckx/Mao의 정렬·직교 채널 논의, Xu 2022의 FBL 결과)을 AoI 지표로 재확인한 것이다.

---

## 3. 항목별 평가

### 3.1 전체 시스템 설계 및 문제 정의

**잘된 점**
- 채널을 alignment / strength / temporal correlation / CSIT / load 다섯 축으로 분리하고, 각도 20°·5°·60°가 실제로 realized correlation 0.86/0.67/0.14로 분리됨을 검증했다(`docs/rician_phy_regime_audit.md`).
- 평균전력 보존형 Rician-ULA와 mean-preserving CSIT 모델을 명시적으로 유도했다.
- 하나의 PHY 구현으로 RSMA/SDMA/OMA/NOMA를 설정만 바꿔 실행하고 자원 parity를 감사한다.

**공격 지점**
- **문제 규모가 trivial하다.** `K=N_t=2`, 스케줄링 없음, 빔 고정, 연속 action 4개 + 이진 1개. Kaya는 같은 규모에서 precoder까지 학습했고, 2604.13691은 `K=4`다. 심사위원은 "이 규모라면 격자 탐색이나 CEM으로 매 슬롯 풀면 된다"고 말할 것이고, 실제로 저장소의 CEM이 그것을 증명한다(3.9절).
- **long-horizon MDP framing의 근거가 약하다.** generate-at-will + 무큐 + 무HARQ에서는 AoI 상태가 urgency 가중치로만 작용한다. Kaya는 γ=0.01(사실상 myopic)을 채택했고, 우리 문서도 "one-step CEM이 부호를 보존하고 PPO보다 낫다"고 기록했다. 즉 현재 문제는 MDP를 요구하지 않는다. 문서 8절이 queue/HARQ/blocklength 확장을 "하지 않는 것이 좋다"고 못박은 것은 실험 행렬 관점에서는 옳지만, 학술적으로는 RL을 정당화할 유일한 구조를 스스로 제거한 셈이다.
- 가정 목록(완벽 SIC, 완벽 CSIR, Gaussian interference에 정규근사 적용, 파일럿 오버헤드 무시, 연속 bit) 은 표준이지만 RSMA 커뮤니티 심사위원은 imperfect SIC와 common beam 설계 부재를 반드시 지적한다.

### 3.2 RSMA 구조와 common/private stream 설계

**잘된 점**
- 성공 규칙 `P_succ,k=(1-ε_c,k)(1-ε_p,k)`, common stream 전체를 복호해야 SIC가 가능하다는 결합, `D_c,tot = rho·min_k(D_k/c_k)`로 feasibility를 구조적으로 보장하는 파라미터화는 정확하고 잘 문서화되어 있다.
- "common stream이 성공을 두 사용자에게 동시에 묶어 burstiness를 키운다"는 메커니즘 분석(inter-delivery CV 0.425 vs 0.806)은 이 프로젝트에서 가장 통찰력 있는 관찰이다.

**공격 지점**
- **RSMA의 이득은 precoder에서 나온다.** common beam을 principal eigenvector로 고정하면 1-layer RSMA의 핵심 설계 자유도를 버린 것이다. 저장소 감사에서도 maxmin beam이 allocation 재최적화 후 모든 regime에서 raw 이득을 냈다(+0.04~+0.16). "beam은 주된 실패 원인이 아니다"는 진단은 PPO 학습 실패에 대한 진단이지 RSMA 성능에 대한 진단이 아니다.
- gate가 켜지는 조건이 **채널 정렬도와 무관(상관 −0.05)** 하고 두 사용자 AoI 유사도와 상관(−0.285)한다는 진단은 정직하지만, 심사위원에게는 "정책이 RSMA가 유리한 물리적 조건을 학습한 것이 아니라 동시 업데이트 타이밍을 학습했다"는 뜻으로 읽힌다. common stream의 물리적 역할이 증명되지 않았다.
- 5 dB에서 `rho=0.00051`인데 23.7% 이득, 60°에서 `rho=0.0003`인데 0.26% 이득: 문서 스스로 인정하듯 이득의 상당 부분은 RSMA가 아니라 초기화·전력정책 안정성이다.

### 3.3 전력 분할 및 사용자별 전력 할당

- 전력 simplex `[P_c, P_1, P_2]`와 common-rate simplex `c`를 masked logistic-normal latent로 파라미터화하고, PPO ratio를 latent에서 정확히 계산하는 설계는 올바르며 테스트로 고정되어 있다(ratio 재채점 오차 <1e-3). 이 부분은 흠잡을 데 없다.
- 다만 `rho_P`(총전력 스케일)는 꺼져 있어 항상 최대전력을 쏜다. 에너지 지표가 없으므로 문제없지만 "전력 할당"이라는 표현은 정확히 "전력 비율 분할"이다.
- 사용자별 전력 정책이 만드는 결과(ACK 음의 상관 −0.554로 업데이트를 교대시키는 SDMA 정책)는 흥미롭지만 이는 RSMA가 아니라 **학습형 SDMA**의 성질이다.

### 3.4 PPO 구조와 State / Action / Reward

**State**: AoI, `h_hat`, 이전 ACK/action. 적절하다. 다만 AoI를 1024로 나눠 0.001 수준의 값이 들어가고 running normalizer에 의존한다. 동작하지만 심사위원이 물어보면 설명이 필요하다. `P_max`·`n`을 상태에 넣어 한 agent로 SNR·blocklength를 일반화한 Kaya와 달리, 우리는 SNR마다 재학습했다(zero-shot 일반화 주장 불가).

**Action**: 4연속 + 1이진, 제약 보존. 우수.

**Reward**: `-Σ w_k Δ_k(t+1)` 하나. tail/power 항 계수 0. 결과표에서 가장 뚜렷한 차이가 p95 AoI(2.0 vs 2.5~3.2)인데 이를 목적함수에 반영하지 않은 것은 아쉽다. γ=0.99와 512-slot truncation + bootstrap은 표준.

**PPO 구현**: clipped surrogate, GAE, 값 clipping, obs/reward 정규화, 분리된 critic trunk, target-KL. 표준적이고 테스트됨.

### 3.5 PPO가 해당 문제에 적절하게 적용되었는가 — **2026-09-10 개정: 해소됨**

> 이 절의 원래 판정("PPO 기여 0.6~2%, 가장 큰 취약점")은 **폐기한다.** 그 수치는 65k 예산에 갇힌 결과였다. matched-budget 감사(`docs/training_budget_results.md`)가 이를 직접 반증했다.

`results/thesis_training_budget_v2/budget_contrasts.csv` 기준, 5개 독립 사전학습/PPO seed의 paired 결과다.

| 행동 클래스 | 0-step | 65k | 262k | PPO 기여 (262k vs 0-step) | 승 |
|---|---:|---:|---:|---:|---:|
| RSMA joint+gate | 1.5508 | 1.5475 | **1.4574** | **−6.02%** | 5/5 |
| SDMA power | 1.5824 | 1.5751 | 1.5610 | −1.35% | 5/5 |

세 가지가 확인된다.

1. **65k에서 PPO 기여는 실제로 −0.21%였다.** 원래 측정은 맞았고 해석이 틀렸다. 예산 부족이었지 방법의 성질이 아니었다.
2. **262k에서 PPO는 RSMA에 6.02%, SDMA에 1.35%를 기여한다.** 4.5배 비대칭이다. 더 큰 행동 공간에 PPO가 파낼 여지가 실제로 존재하고, 충분한 예산에서만 파진다. 논문의 "왜 PPO인가"는 이 비대칭으로 답해야 한다.
3. **행동 클래스 자체의 이득도 분리됐다.** PPO 0 step에서 RSMA가 SDMA보다 2.00% 낮고 5/5 seed에서 이긴다. RL 없이 행동 클래스와 목적 기반 사전학습만으로 얻는 몫이다.

critic도 예산과 함께 개선된다. RSMA seed 0의 explained variance는 −2.151(첫 10%) → −0.377(중간) → **+0.012**(마지막 10%)로 끝에서 0을 넘고, update의 19.5%가 EV > 0 구간에 있다. 반면 SDMA는 −5.218 → −0.800 → −0.158로 끝까지 음수이고 EV > 0인 update가 9.6%뿐이다. **RSMA 쪽 critic이 더 잘 학습된다는 사실이 PPO 기여 비대칭의 부분적 설명이 된다.**

남는 질문은 좁아졌지만 사라지지 않았다.

1. 두 arm 모두 학습의 80~90%를 EV ≤ 0 구간에서 보낸다. EVPO 정리 1(`docs/ppo_trends_2026.md` 참조)에 따르면 그 구간의 critic은 분산을 키운다. **EV 게이팅을 넣으면 SDMA가 더 개선되어 격차가 줄어들 위험**이 있다. 반대로 줄지 않으면 강한 방어가 된다. 어느 쪽이든 실험할 가치가 있다.
2. 262k 이득이 장기 계획에서 오는지, 단순히 상태 분포와 결정론적 행동 사상을 더 철저히 최적화한 결과인지는 구분되지 않았다. 문서도 이를 명시한다. 인과 주장을 하려면 multi-step model-based 대조군이나 AoI 상태 개입이 필요하다.
3. 왜 on-policy PPO인가는 여전히 답이 없다. 선행연구(Kaya 2025)가 TD3이고 저장소에도 TD3가 구현돼 있는데 최종 비교에 없다. 다만 무선 PPO 문헌고찰(Future Internet 18(7):340, 2026)이 "TD3는 후반에 불안정해지고 PPO는 단조 상승한다"고 보고하므로 인용 방어는 가능하다.

### 3.6 AoI 정의 및 최적화 방식

- 정의는 표준(슬롯 단위, 성공 시 1로 리셋, 무한 cap). Kaya는 리셋값을 `n`으로 두어 blocklength를 AoI 단위에 반영했으므로, blocklength 비교를 할 계획이 있다면 우리 정의는 그 축을 잃는다.
- 평균 AoI만 최적화하고 p95/p99/peak는 보고만 한다. p95는 결과 차이가 가장 선명한 지표이므로 tail-aware 목적함수 ablation이 없는 것은 손해다.
- "one-step 성공확률 최대화 ≠ 장기 AoI 최소화"를 데이터로 보인 점(split-only: 성공률 최고, AoI 나쁨; max inter-delivery 34.4 slot)은 좋은 기여다. 그러나 같은 저장소의 CEM 결과(myopic이 PPO보다 21% 좋음)와 합치면 "장기 효과가 존재하지만 현재 PPO는 그것을 잡지 못한다"가 된다.

### 3.7 RSMA + PPO + AoI 결합 구조의 논리적 타당성

논리 사슬을 그대로 쓰면 다음과 같다.

1. FBL cliff 때문에 무작위 탐색이 실패한다 → **증명됨** (broad-rho 정책 common BLER 0.33~0.96; scratch PPO의 정렬도 상관 0.103).
2. 그래서 objective-aware 사전학습이 필요하다 → **증명됨** (action-only BC gap 3.70/2.00 vs objective-aware 0.685/0.475; 사전학습 actor의 정렬도 상관 0.585 ≈ 오라클 0.589).
3. 사전학습 후 PPO가 정책을 실질적으로 개선한다 → **2026-09-10 증명됨** (262k에서 RSMA 6.02%, 5/5 seed). 단 65k에서는 −0.21%였으므로 **예산 조건부**다.
4. 그 개선이 행동 클래스에 따라 다르다 → **증명됨** (RSMA 6.02% 대 SDMA 1.35%). 이 비대칭이 결합 구조를 정당화하는 핵심 고리다.
5. 그 결과 RSMA가 SDMA보다 AoI가 낮다 → **매칭 조건에서 증명됨** (0-step 2.00%, 262k 6.64%, 각 5/5). 단 20°/σ=0.2/10 dB 한 지점이며, 다른 regime 표는 아직 매칭되지 않았다.
6. 그 정책이 myopic 수치 최적화보다 낫다 → **아직 아님** (safeguarded CEM 1.2032). 결정 지연 절충으로 서술해야 한다.

즉 1~5는 성립하고 6이 남는다. 09-09 평가에서 3이 약점이었던 것은 예산 문제였다. 결합 구조의 논리적 타당성 자체는 이제 방어 가능하며, 남은 과제는 "학습형 allocator가 슬롯별 수치 최적화 대비 무엇을 주는가"라는 위치 설정이다.

### 3.8 Baseline 및 실험 설계

**잘된 점**
- paired common-random-number 평가, training-seed와 trajectory 불확실성 분리, seedwise win 보고, negative regime 보존, checkpoint hash 기록. 무선 DRL 논문 평균보다 훨씬 엄격하다.
- 학습형 SDMA-power PPO를 같은 encoder·optimizer·예산으로 둔 것은 올바른 대조군이다.
- power-only / split-only / joint / joint+gate ablation과 고정값 감도 sweep은 action 설계 주장을 지지한다.

**공격 지점**
- ~~**초기화가 매칭되지 않았다.**~~ → **2026-09-10 해소.** matched-budget 감사에서 RSMA와 SDMA 모두 같은 index의 objective 사전학습 actor를 받고, 262k 예산과 `reset_critic: true`가 동일하다. 10개 run의 `config_snapshot.yaml`로 확인했다. 다만 **이 매칭은 20°/σ=0.2/10 dB 한 지점에만 적용됐다.** 5°/20°/60° 공간 regime, near/far, SNR 5/15 dB 결과는 여전히 스크래치 SDMA를 대조군으로 쓰므로, 그 표들의 13.7%·9.7%·19.9%·23.7%·15.0% 이득은 아직 confound 상태다. 논문에 넣을 때 매칭 여부를 표에 명시하거나 해당 조건도 재실행해야 한다.
- **CEM이 main table에 없고, protocol이 다르다.** headline 조건의 CEM은 이제 존재한다(safeguarded RSMA-CEM 1.2032). 그러나 CEM 감사는 seed 960000–960002, 2,048 slot, burn-in 512로 평가했고 새 headline은 seed 930000–930005, 8,192 slot, burn-in 2,048이다. **두 값을 같은 표에 넣을 수 없다.** 게다가 CEM 감사가 비교한 대상은 옛 65k 정책(1.5264)이지 262k 정책이 아니다. 동결된 주 프로토콜(6×8,192)에서 CEM을 재실행하는 것이 논문 표 통합의 선결 조건이다. **격차의 크기는 그 재실행 전까지 어떤 형태로도 인용하지 않는다.** (2026-09-10 정정: 이전 판에 있던 "약 21%에서 17%로 좁혔다"는 환산은 horizon과 protocol이 다른 두 값을 섞은 것이므로 삭제했다.)
- ~~CEM 추론 지연 0.04 ms~~ → **해소.** 재측정값은 raw RSMA-CEM 2.189 ms, safeguarded 3.600 ms, RSMA-PPO 0.623 ms다. 다만 Python/CPU 구현값이므로 복잡도 우위로 일반화하면 안 된다.
- fixed RSMA(16.98)와 equal SDMA(2.80)는 straw man이다. 문서도 인정하지만 표에 넣는 순간 심사위원은 "약한 baseline으로 표를 채웠다"고 본다. 설계안이 요구한 TD3(선행연구 알고리즘), WMMSE/SCA precoder 참조, max-AoI-first가 모두 빠졌다.
- 3~5 seed. 5개 값으로 bootstrap 95% CI를 만드는 것은 통계적으로 취약하다. SDMA-PPO 표준편차 0.128(5 seed), 0.699(5 dB, 3 seed)는 이득의 상당 부분이 SDMA 학습 실패 seed에서 온다는 뜻이다.
- pilot 단계에서 초기 `f_c=0.2, rho=0.05`, gate 확률, Beta concentration이 **test seed를 관찰하며** 선택됐다(문서가 명시). 최종 seed 세트(920000~)는 분리됐지만 하이퍼파라미터 선택 경로가 오염됐다는 지적은 피하기 어렵다.
- 2차 실험(5°/60°)은 단일 사전학습 actor에 조건부다. crossed seed 설계가 없다.

### 3.9 성능 결과와 결과 해석

| 결과 | 해석의 타당성 |
|---|---|
| **[현행] matched 262k: 1.4574 vs 1.5610 (6.64%, 5/5)** | **초기화·예산 매칭 상태의 유일한 headline.** 0-step 행(2.00%, 5/5)이 행동 클래스 몫을, 262k 행이 PPO 몫을 분리해 준다. 논문 주 표는 이것이어야 함 |
| ~~[폐기] headline 1.5511 vs 1.6448 (5.7%)~~ | 초기화 confound로 superseded. archive 처리됨 |
| 5° 13.7%, 20° 9.7%, 60° 0.26% | regime transition 해석은 정직하고 좋음. 다만 60°는 "RSMA가 SDMA와 같다"가 아니라 "학습이 SDMA로 수렴했다"이므로 이득 0은 당연 |
| near/far 19.9% | 한 조건. SDMA seed 분산 0.105. 과거 역전 결과(−45.5%, archive)와 상반되므로 심사위원이 반드시 원인을 묻는다. 문서는 "model generation 차이"로 설명하지만 정량적 분해가 없음 |
| SNR 5/10/15 dB | 5 dB 이득은 `rho≈0`이라 RSMA 이득이 아님(문서 인정). 15 dB에서 성공확률 0.84→0.97, AoI 1.03은 하한 1에 근접 |
| CSIT sweep 1.555/1.546/1.552/1.563 | 평평함. "robust"라기보다 CSIT 오차가 이 operating point에서 중요하지 않다는 뜻. 2604.13691처럼 CSIT 열화에 따라 RSMA 이득이 커지는 스토리가 없음 |
| SAA negative | 정직한 negative result. 공정 계산예산으로 재검증한 점은 칭찬할 만함 |
| tail 지표 | **matched 감사에서 p95가 모든 학습 정책에서 2.0으로 포화**되어 구분력을 잃었다. tail 주장은 p99(RSMA 262k에서 3.93 → 3.77)로 옮겨야 한다 |
| equal SDMA 성공률 0.585 > sdma_262k 0.493인데 AoI는 2.78 vs 1.56 | "성공률과 AoI가 어긋난다"는 메커니즘 주장의 가장 깨끗한 사례. 본문에서 활용할 것 |

전반적으로 해석은 과장이 없고 오히려 보수적이다. 개정 이전의 핵심 문제("이득은 초기화에서 온다")는 해소됐다. 남은 해석상의 과제는 262k 이득이 장기 계획에서 오는지 최적화 철저성에서 오는지 구분되지 않는다는 점이며, 문서도 이를 명시하고 있다.

### 3.10 기존 연구 대비 novelty와 차별성

| 축 | Kaya 2025 | 2604.13691 (2026) | 본 프로젝트 | 차별성 판정 |
|---|---|---|---|---|
| 규모 | `K=M=2` | `K=4, N_t=5` | `K=N_t=2` | 열세 |
| 결정 변수 | precoder+전력+split | 전력+split (빔 고정) | 전력+split+gate (빔 고정) | gate만 추가 |
| 최적화 | TD3 (γ=0.01) | SCA (해석적 AAoI) | 지도 사전학습 + PPO | 사전학습이 실질 기여 |
| CSIT | perfect 학습→imperfect 테스트 | outdated (속도 축) | 조건별 재학습 | 일반화 주장 불가 |
| 채널 통제 | i.i.d. Rayleigh / GM | GM, 거리 | **각도·강도·CSIT 분리** | 우위 |
| 대조군 | SCA 기반 RSMA 2종 | SDMA(격자), NOMA | 학습형 SDMA, CEM(부분), 고정 | 학습형 SDMA는 우위, CEM 미완 |
| 통계 | 1,000 채널 평균, seed 미보고 | 10^5 realization | 3~5 seed paired | 우위 |
| negative regime | 없음 | 없음 | 60°, 5 dB, SAA | 우위 |
| 해석적 결과 | 없음 | closed-form AAoI | 없음 | 열세 |

결론: **평가 규율과 regime 통제는 선행연구보다 낫고, 문제 규모·결정 변수·이론적 결과·RL의 실질 기여는 선행연구보다 못하다.** 저널 심사위원은 후자를 본다.

### 3.11 계산 복잡도 및 실제 적용 가능성

- 추론: `[128,128]` MLP 한 번. 슬롯당 마이크로초 수준. 문제없음.
- 학습 파이프라인: oracle 데이터셋(states × subset × CEM 2,880 evals) → 미분가능 PHY 사전학습(60 epoch) → PPO. CEM 한 번 돌리는 것보다 파이프라인이 무겁다. `K` 증가 시 oracle이 `2^K` subset을 열거해야 하고, 빔이 고정이라 `K>N_t`에서는 scheduling이 필요한데 이 프로젝트가 이미 Bernoulli scheduling head 붕괴를 기록했다.
- 배포 관점: 슬롯마다 `h_hat`·ACK가 필요하고 blocklength 128이면 슬롯이 수십 μs~ms 수준. PPO 추론은 가능하지만 `K=2` CEM도 가능하다(지연 **확인 필요**). "학습이 필요한 이유"를 계산량으로 정당화하려면 `K≥4`가 필요하다.
- 재현성: 실행 manifest·hash는 좋으나 **git 저장소가 없다**(`git_commit: null`). README 테스트 수 569 vs 실제 수집 571(사소).

---

## 4. Reviewer가 공격할 가능성이 높은 지점 (2026-09-10 재정렬)

1. **myopic CEM이 학습 정책보다 낫다** — safeguarded CEM 1.2032. 262k가 격차를 약 17%로 좁혔지만 여전히 진다. 왜 RL인가에 답해야 한다. 게다가 CEM과 headline의 평가 protocol이 달라 아직 같은 표에 넣지 못한다. (3.8, 3.9절)
2. **toy 규모 + 빔 고정** — RSMA의 핵심 자유도인 precoder를 제거한 채 RSMA를 평가. Kaya 2025는 같은 `K=M=2`에서 precoder까지 학습했다. (3.1, 3.2절)
3. **MDP framing 미정당화** — generate-at-will·무큐. 262k 이득이 장기 계획인지 최적화 철저성인지 구분 안 됨. Kaya는 γ=0.01을 채택했다. (3.1, 3.6절)
4. **매칭이 한 지점에만 적용됐다** — 5°/20°/60°, near/far, SNR 5/15 dB 표는 여전히 스크래치 SDMA 대조군이다. 그 조건들의 이득은 아직 confound 상태. (3.8절)
5. **critic이 대부분의 학습에서 EV ≤ 0** — RSMA 80.5%, SDMA 90.4%의 update가 해당 구간. EVPO 정리 1 기준으로 그 구간 critic은 분산을 키운다. EV 게이팅이 격차를 줄일 위험도 함께 존재. (3.5절)
6. **5 seed, test seed로 pilot 튜닝, crossed seed 없음** — 초기 `f_c`, gate 확률, Beta concentration이 test seed를 관찰하며 선택됐다. (3.8절)
7. **common stream의 물리적 역할** — 정렬도 의존성은 확인됐다(사전학습 상관 0.585 ≈ 오라클 0.589). 그러나 5 dB·60°의 이득은 `rho≈0`이므로 RSMA 이득이 아니다. (3.2절)
8. **누락 baseline** — TD3(구현되어 있음), SCA/WMMSE precoder, max-AoI-first.
9. **near/far 역전** — 과거 −45.5%와 현재 +19.9%의 정량적 원인 분해가 없다.
10. **가정** — 완벽 SIC, common beam principal 고정, 파일럿 오버헤드 없음, SNR 독립 CSIT 오차.

**해소된 항목:** 초기화 confound(구 2위), "PPO가 한 일이 무엇인가"(구 1위), 65k 예산 의문(구 6위).

---

## 5. 종합 평가

| 항목 | 점수(5) | 09-09 | 근거 |
|---|:-:|:-:|---|
| Novelty | 2.5 | 2 | 조합은 기존. 다만 "행동 클래스별 PPO 기여의 비대칭"은 문헌에 없는 관측 |
| Technical soundness | 4 | 4 | PHY·action 파라미터화·PPO 구현 정확, 테스트로 고정 |
| Experimental rigor | 4 | 3.5 | matched 초기화·예산·5 seed·paired 대조로 크게 개선. 매칭이 한 지점에만 적용된 점이 감점 |
| Significance | 2.5 | 2 | `K=2`·빔 고정은 그대로지만 RL 기여가 6.02%로 실증됨 |
| Clarity / honesty | 4.5 | 4 | 자기 결과를 superseded로 명시하고 archive 처리한 점은 드물게 좋은 규율 |

**예상 판정 (개정)**
- 학부 졸업논문: **통과, 최상위권.** 자기 headline을 스스로 반증하고 교체한 과정 자체가 방법론적 성숙도의 증거다.
- IEEE 컨퍼런스(VTC/ICC/GLOBECOM 워크숍): **수정 후 Accept 가능성이 실질적으로 높아졌다.** 남은 필수 작업은 CEM을 동일 protocol에서 재실행해 주 표에 넣는 것, 그리고 기여를 "행동 클래스별 PPO 이득의 비대칭"으로 서술하는 것이다.
- IEEE 저널(TWC/TCOM/IoTJ): **여전히 Major revision.** 규모(`K=N_t=2`), 빔 고정, 이론적 결과 부재, CEM 대비 열세가 남아 있다. 규모나 결정 변수 중 하나를 열지 않으면 어렵다.

---

## 6. 연구 수준을 높이기 위한 우선 개선 과제 (2026-09-10 개정)

### 1) ~~초기화를 매칭하고 "기여의 주체"를 분리하라~~ → **완료 (2026-09-10)**
- objective-aware SDMA 사전학습 구현, 양쪽 arm 5 seed 262k 매칭 학습, 0/65k/262k 3점 예산 대조 완료.
- 결과: 행동 클래스 몫 2.00%(0-step), PPO 몫 RSMA 6.02% 대 SDMA 1.35%.
- **남은 부분:** 이 매칭이 20°/σ=0.2/10 dB 한 지점에만 적용됐다. 공간 regime·near/far·SNR 표는 여전히 스크래치 SDMA 대조군이므로 논문에 매칭 여부를 명시하거나 재실행할 것.

### 2) CEM을 동일 protocol에서 재실행해 주 표에 넣어라 (현재 1순위)
- 현재 CEM(1.2032)은 seed 960000–960002·2,048 slot·burn-in 512이고, headline(1.4574)은 seed 930000–930005·8,192 slot·burn-in 2,048이다. **같은 표에 넣을 수 없다.** CEM 감사가 비교한 대상도 옛 65k 정책이다.
- 동결된 주 protocol(6×8,192, burn-in 2,048)에서 CEM-RSMA·CEM-SDMA를 재실행하고 262k 정책과 나란히 놓는다.
- CEM이 여전히 이기면 두 선택지뿐이다. (a) 기여를 "CEM의 학습형 amortization"으로 재정의하고 결정 지연(0.62 ms 대 3.60 ms)과 함께 성능–비용 절충으로 서술하거나, (b) 장기 구조(아래 3번)를 넣어 myopic CEM이 실패하는 조건을 만든다.

### 3) MDP가 필요한 이유를 실험으로 만들어라
- 문서가 "하지 말라"고 한 항목 중 **하나만** 고른다. 비용 대비 효과는 (i) 사용자별 Bernoulli 도착 + 1-packet 버퍼, 또는 (ii) 에너지 예산/평균전력 제약(`rho_P` action은 이미 구현됨), 또는 (iii) HARQ-CC 1회 재전송. 어느 것이든 myopic CEM과 장기 정책의 차이가 생기며 그때 비로소 γ=0.99 PPO가 의미를 갖는다.
- 저비용 대안: 262k 이득이 장기 계획에서 오는지 검정하기 위해 **γ를 낮춘 대조군**(예: γ=0.5 또는 0.9)을 같은 예산으로 학습한다. γ가 낮아도 6% 이득이 유지되면 장기 계획이 원인이 아니라는 뜻이고, 사라지면 원인이라는 직접 증거가 된다. Kaya가 γ=0.01을 채택한 것과 대비할 수 있어 서술 가치도 크다.
- critic 쪽은 `docs/ppo_trends_2026.md`의 권고(EV 게이팅, WSRL식 critic warmup, `reference_kl_coef` 활성화)를 따른다. 262k에서 RSMA critic이 EV +0.012까지 올라왔으므로 여지가 남아 있다.

### 4) 규모 또는 결정 변수 중 하나를 확장하라
- `N_t=2, K=4` 또는 `N_t=4, K=6` 한 쌍에서 subset-categorical scheduling head와 함께 재실험한다. RSMA는 overloaded·정렬 채널에서 이득이 크다는 것이 이론이므로, 이득이 커지면 스토리가 생기고 작아지면 negative regime이 하나 더 생긴다.
- 또는 `K=2`를 유지하되 common beam을 maxmin/학습형으로 열고 Kaya의 RSMA-LowLatency(SCA) 참조를 재현해 넣는다. 빔이 고정된 RSMA 논문은 RSMA 심사위원을 설득하지 못한다.

### 5) 통계·프로토콜·서사를 정리하라
- 2차 실험(공간 regime·near/far·SNR)의 대조군 초기화를 매칭하거나, 매칭되지 않았음을 표에 명시한다.
- TD3(이미 구현됨)를 선행연구 대응 baseline으로 넣고 동일 interaction 예산에서 비교한다. Future Internet 18(7):340(2026)의 "TD3는 후반에 불안정, PPO는 단조 상승" 서술을 함께 인용한다.
- **tail 지표를 p95에서 p99로 바꾼다.** matched 감사에서 p95가 모든 학습 정책에서 2.0으로 포화되어 구분력을 잃었다.
- 서사를 "RSMA-PPO가 SDMA를 이긴다"에서 **"FBL decoding cliff 위에서 RSMA 자원배분 정책을 학습 가능하게 만드는 조건(exact SDMA 경계 + outcome-aware 초기화)과, 충분한 예산에서 PPO가 두 행동 클래스에 비대칭적으로 기여한다는 관측"** 으로 바꾼다.
- 하이퍼파라미터 선택은 validation seed로만. 기존 pilot 튜닝 경로의 오염은 한계로 명시한다.
- git 초기화, README 테스트 수 정정(현재 573).

---

## 7. 확인 필요 항목

- ~~CEM 슬롯당 지연 0.04 ms~~ → **해결됨.** `docs/cem_reference_results.md`(2026-09-09 19:43)에서 allocator 전체 시간을 재측정: raw RSMA-CEM 2.189 ms, safeguarded 3.600 ms, RSMA-PPO 0.623 ms (Python/CPU). 기존 0.04 ms는 allocator 계산을 제외한 policy timing이었다.
- ~~headline operating point(σ_e=0.2)에서의 CEM 값 없음~~ → **해결됨.** 같은 문서: raw RSMA-CEM 1.2143, SDMA-CEM 1.3303, safeguarded RSMA-CEM 1.2032, RSMA-PPO 1.5264, SDMA-PPO 1.6283 (3 trajectory × 2,048 slot). Safeguarded CEM이 RSMA-PPO보다 21.17% 낮다. 동일 960-candidate 예산에서 RSMA-CEM이 SDMA-CEM보다 8.72% 낮다는 결과는 **학습 confound가 없는 유일한 RSMA-vs-SDMA 증거**이므로 본문에서 강조할 가치가 있다.
- **[신규] CEM(1.2032)과 headline(1.4574)의 평가 protocol 불일치.** 전자는 seed 960000–960002·2,048 slot·burn-in 512, 후자는 seed 930000–930005·8,192 slot·burn-in 2,048이다. 동일 protocol 재실행 전에는 두 값을 나란히 제시할 수 없다.
- **[신규] 2차 실험(5°/20°/60°, near/far, SNR 5/15 dB)의 대조군 초기화가 매칭되지 않았다.** matched 감사는 20°/σ=0.2/10 dB 한 지점에만 적용됐다.
- **[신규] 262k 이득의 원인.** 장기 계획인지, 상태 분포와 결정론적 행동 사상의 더 철저한 최적화인지 구분되지 않았다. γ 감소 대조군으로 저비용 검정이 가능하다.
- Yan et al. 2023이 사용한 DRL 알고리즘 2종의 정확한 명칭 (Codex 검토는 TD3라고 보고했으나 본 검토에서는 원문 확인 불가 → 여전히 확인 필요).
- Kaya 2025의 imperfect CSIT 양자화 정확한 정의(“소수점 첫째 자리만”)와 SNR 의존성.
- archive의 near/far −45.5% 결과와 현재 +19.9% 결과 사이의 정량적 원인 분해(문서는 정성적 설명만 제공).
- objective-aware 사전학습이 imperfect CSIT에서 단일 `h_true` realization으로 학습됨: 오차 realization 수가 조건부 기대를 근사하기에 충분한지 여부.

---

## 8. Codex 교차검토 반영 (2026-09-09 추가)

Codex가 이 보고서를 검토한 뒤 제시한 수정 의견 중 저장소로 검증한 결과:

| Codex 의견 | 검증 | 반영 |
|---|---|---|
| CEM 결과·지연이 이미 재측정됨 | `docs/cem_reference_results.md`, `results/thesis_cem_reference_v2/summary.csv`로 확인 | 7절 갱신 |
| "지도 사전학습"은 부정확, objective_only는 action loss 0 | `src/rsma_aoi/agents/allocator_net.py:205` `LossWeights(0.0, 1.0, 0.0)` 확인 | 0절 표현 수정 |
| gate 효과는 2.3%로 작고 action-space 정확성 기여로 써야 함 | joint 1.5881 → joint+gate 1.5517 확인 | 이미 3.8절과 일치 |
| PPO 기여 0.6~2%는 고정 actor 감사 조건부 | 감사가 단일 사전학습 checkpoint 사용 확인 | 0절에 단서 추가 |
| Hieu "원형", Xu "원조" 표현 과함 | 원문 세부 미확인 | 2.1절 완화 |
| Yan 2023은 TD3 | 본 검토에서 원문 확인 불가 | 확인 필요 유지 |
| queue/HARQ, K≥4, precoder 학습, TD3, 10 seed는 학부논문 범위 초과 | 판단 문제 | 6절은 "연구 수준을 저널급으로 높이는" 과제이며, 학부논문 최소 요건은 6-1(warm SDMA + zero-step 비교)과 CEM 표 포함으로 축소 가능 |

Codex가 언급하지 않았지만 유지되는 지적: near/far −45.5%→+19.9% 역전의 정량적 원인 분해 부재, pilot 단계에서 test seed를 보며 선택한 초기화 하이퍼파라미터, 2차 실험의 단일 사전학습 actor 의존.

새로 강조할 점: 사전학습 objective가 **one-step expected AoI**이므로 사전학습 actor는 사실상 CEM의 amortized 근사이고, PPO는 그 위의 미세조정이다. 따라서 "PPO = CEM의 amortization" 서사를 쓰려면 zero-step actor(PPO 없음) 행이 반드시 표에 있어야 하며, PPO가 그 행보다 나아야 PPO의 존재가 정당화된다. 지연 비교(0.62 ms vs 3.6 ms)는 Python 구현값이므로 "측정된 이점"이 아니라 "offline 학습 후 feed-forward 결정"이라는 구조적 성질로만 서술해야 한다.

## 9. 2026-09-10 갱신 — matched-budget 감사 검증

`docs/training_budget_results.md`와 `results/thesis_training_budget_v2/`를 저장소에서 직접 검증한 결과다.

### 검증한 것

- **초기화 매칭이 실제로 이루어졌다.** `results/rician_ppo_20deg_csit02_matched_budget_262k_v1/runs/*/config_snapshot.yaml` 10개를 확인했다. RSMA seed *i*는 `objective_actor_rician_20deg_csit02_single_seed{i}_v2`, SDMA seed *i*는 `objective_actor_sdma_20deg_csit02_seed{i}_v1`을 받는다. 양쪽 모두 `total_timesteps: 262144`, `reset_critic: true`다.
- **결과가 매칭 이후에도 살아남았고 오히려 강해졌다.** 0-step에서 이미 2.00%(5/5), 262k에서 6.64%(5/5)다.
- **PPO 기여의 비대칭이 핵심 발견이다.** RSMA 6.02% 대 SDMA 1.35%. 이것이 "왜 PPO인가"에 대한 이 프로젝트 최강의 증거이며, 논문 서사의 중심을 6.64%가 아니라 여기에 두어야 한다.
- **critic이 예산과 함께 개선된다.** RSMA seed 0의 EV는 −2.151 → −0.377 → +0.012이고 update의 19.5%가 EV > 0이다. SDMA는 −5.218 → −0.800 → −0.158, 9.6%다.

### 문서에 반영이 필요한 것

- `docs/training_budget_results.md`가 common 사용량을 0-step 대비 262k로만 서술해 "증가했다"로 읽힌다. 실제 궤적은 비단조다.

  | | 0-step | 65k | 262k |
  |---|---:|---:|---:|
  | 평균 common power | 0.273 | 0.362 | 0.327 |
  | common-on 비율 | 0.436 | 0.461 | **0.368** |

  65k에서 과사용으로 이탈했다가 262k에서 되돌아오며 선택적으로 바뀐다. "PPO가 충분한 예산에서 스스로 교정한다"는 더 강한 서술이므로 65k → 262k 대비를 명시할 것.
- `reference_kl_coef`가 10개 run 전부 0.0이다. 드리프트 정규화는 여전히 미검증이다.

### 다음 우선순위 (2026-09-10 2차 개정)

1. **262k PPO와 CEM을 동일 초기상태·seed·horizon·burn-in에서 평가한다.** 이것 전에는 격차를 인용하지 않는다.
2. **사용자 교환 전후의 성능 감사.** 평가만 하면 되므로 저렴하다. 위 (a)/(b) 가설을 가른다.
3. **축약 actor를 처음부터 학습해 full actor와 비교한다.** common action 차원이 실제로 불필요했는지, 아니면 full actor가 좋은 상수를 탐색하는 데 역할을 했는지를 결정한다.
4. 3의 결과로 동적 common-rate 제어가 여전히 논문 중심이어야 한다면 geometry-randomized pilot.
5. 성과가 있을 때만 3~5 seed로 확대.
6. 논문 주 표를 [행동 클래스] × [0/65k/262k] 구조로 확정하고 tail 지표를 p99로 바꾼다.
7. 2차 실험(공간 regime·near/far·SNR)의 대조군 매칭 여부를 명시하거나 재실행한다.

---

## 10. 2026-09-10 2차 갱신 — two-mode 감사와 action 주장 철회

`docs/two_mode_policy_audit.md`와 `results/thesis_two_mode_audit_v4/`가 3.8절의 action-ablation 해석을 뒤집었다. calibration seed(940000–940002)와 frozen test seed(930000–930005)를 분리하고 5 checkpoint를 paired 비교한 설계이며, 결론을 수용한다.

| 대조 정책 | 평균 AoI | 원 PPO 대비 | PPO 승 |
|---|---:|---:|---:|
| gate + α·ρ 조건부 상수 고정 | **1.4539** | −0.24% | **0/5** |
| 원래 262k PPO | 1.4574 | 기준 | — |
| gate + 연속 private power만 | 1.4625 | +0.36% | 4/5 |
| gate + private argmax만 | 1.5910 | +9.17% | 5/5 |
| gate + 연속 common-rate만 | 1.7605 | +20.80% | 5/5 |
| gate 외 전부 고정 | 1.7730 | +21.66% | 5/5 |

**철회:** "전력과 common-rate의 연속적 공동 제어가 모두 필요하다"는 주장. 지지되지 않는다.

**유지:** exact common-off gate와 **사용자별 private power의 연속 배분**은 필수다. argmax로 바꾸면 9.17%를 잃는다.

세 가지를 덧붙여 기록한다.

1. **α·ρ 고정이 5/5로 이기지만 효과는 작다.** 평균 차이 −0.00349 AoI, −0.24%, paired `t≈−3.05` (`n=5`). 부호는 일관되나 여러 사후 대조군 중 하나이므로 "연속 변조가 해롭다"는 일반 결론으로 쓸 수 없다. 정확한 서술은 **"대표 20° 조건에서 α·ρ의 슬롯별 변조가 이득을 주지 않았고, 조건부 상수화가 일관되게 약간 더 좋았다"** 이며, 실질적 의미는 **동등 성능의 정책 압축 가능성**이다.
2. **이 축약은 사후적이다.** 상수는 학습된 각 checkpoint에서 추출했다. 따라서 증명된 것은 "수렴 정책이 압축 가능하다"뿐이다. **"사전학습과 PPO가 그 상수를 찾는 데 필요했다"는 주장은 근거가 없다(확인 필요).** PPO 출력에서 상수를 얻었다는 사실은 PPO가 필요했다는 증거가 아니며, 저차원 grid, CEM, 축약 actor 직접 학습으로 더 싸게 찾을 수도 있다. 검증하려면 **gate network + 상태 의존 private-power head + 학습 가능한 전역 `α,ρ,c` + 나머지 고정**인 축약 actor를 처음부터 objective pretraining과 PPO로 학습해 full actor와 비교해야 한다.
3. **기존 power-only 실패와의 관계는 부분적으로만 설명된다.** power-only는 `rho=0.3`만 잘못 고정한 실험이 아니라 gate 없음, common-rate equal 고정, common 항상 on을 함께 포함한다. 반면 `fixed_magnitude`는 학습된 gate, private-power 비율, common-rate 비율을 유지한다. 따라서 3.0426 → 1.4539를 "올바른 `rho`를 찾았기 때문"으로 단일 원인화할 수 없다. "FBL cliff 때문에 동작점 탐색이 어렵다"는 서사는 타당하지만, 현재 실험만으로 PPO가 그 탐색에 필수라고 말할 수는 없다.
4. **단일 geometry 조건부일 가능성.** regime 연구에서 학습된 common 사용량은 5°/20°/60°에 따라 0.372/0.287/0.019로 달라진다. 다만 이는 geometry별로 **따로 학습한** 정책들의 집계값이며, 하나의 정책이 geometry에 따라 `α,ρ,c`를 연속 조절했다는 증거는 아니다. geometry-randomized 학습은 "여러 정렬도를 한 정책이 처리할 때 상태 의존 common allocation이 다시 필요해지는가"를 검증하지만 실험 범위를 키운다. 축약 actor를 현재 20°에서 직접 학습하는 쪽이 더 싸고 결정적이다.

**보류된 추론:** 오라클은 gate가 켜진 상태에서도 정렬도에 따라 α를 조절한다(`r=0.501`, `results/action_state_dependence_v1/`). 그러나 이것과 `fixed_magnitude` 결과를 이어 "one-step surrogate가 보상하는 변조를 장기 목적은 필요로 하지 않는다"고 결론지을 수 없다. 두 측정의 대상이 다르다. 오라클 상관은 600개 offline state에서, `fixed_magnitude`는 PPO의 나머지 성분을 유지한 closed-loop 장기 평가에서 얻었다. 검증된 것은 **PPO의 α·ρ 변조가 불필요했다**는 사실뿐이다. 주장을 성립시키려면 동일 CEM 정책에서 원래 행동과 α·ρ만 고정한 행동을 놓고 one-step objective와 장기 AoI를 함께 측정해야 한다.

**경계 표현:** gate-off는 정확한 SDMA 경계다. gate-on은 정확한 NOMA 경계가 아니다. `c_1`이 0.88~0.92이고 `ρ<1`이며 사용자 0에게 9~42 bits의 private payload가 남고 두 private 전력 모두 정확히 0이 아니다. 따라서 **"exact SDMA boundary와 asymmetric NOMA-like RSMA corner 사이의 상태 의존 전환"** 이 정확한 서술이며, "두 정확한 경계 사이의 전환"은 과장이다.

**미해결 결함 — 사용자 permutation 비대칭.** `results/action_state_dependence_v1/permutation_test.csv` 기준 262k 정책의 교환 후 gate 일치율 0.39~0.55, common-rate 등변성 잔차 0.59~0.82, common-power 불변성 잔차 0.19~0.29다. homogeneous 설정에서 payload·AoI 가중치·경로손실이 동일하므로 사용자를 교환하면 행동도 대응 교환되어야 한다. 가능한 해석이 둘이다. (a) 두 개의 대칭적 NOMA-like 해 중 하나를 label convention으로 고른 무해한 대칭 파괴, (b) 사용자 index에 과적합되어 relabeling이나 geometry 변화에서 성능이 무너지는 일반화 문제. **행동 차이만으로는 구분할 수 없고, 교환 전후의 목적값과 장기 AoI 성능을 비교해야 한다.**

`results/action_state_dependence_v1/`와 `scripts/audit_action_state_dependence.py`는 이전 판에서 재현 불가능하다고 지적받은 정렬도 상관·교환 테스트·payload 분해를 저장소 산출물로 고정한 것이다.

---

## 참고 문헌 (검색으로 확인한 것)

- E. Kaya, "Reinforcement Learning Based Rate Splitting for Minimizing Age of Information in Finite Blocklength Systems," M.S. thesis, METU, Aug. 2025. https://open.metu.edu.tr/bitstream/handle/11511/116221/EnesKayaTez-1.pdf
- Q. Yan et al., "Age of Information Minimization for Short-Packet Communications RSMA in Satellite-based IoT," IEEE VTC-Fall 2023. https://ieeexplore.ieee.org/document/10333373
- S. R. Ali, S. Abidrabbu, M. Furqan, H. Arslan, "A Joint Framework of HARQ and RSMA for AoI Minimization," IEEE IoT J., Sep. 2025. https://ieeexplore.ieee.org/document/11053976
- "Towards Autonomous Driving with Short-Packet Rate Splitting: Age of Information Analysis and Optimization," arXiv:2604.13691, Apr. 2026. https://arxiv.org/abs/2604.13691
- O. Dizdar, Y. Wang, "RSMA for Semantic-Aware Networks: an Age of Incorrect Information Perspective," arXiv:2312.07159. https://arxiv.org/abs/2312.07159
- N. Q. Hieu et al., "Optimal Power Allocation for Rate Splitting Communications with Deep Reinforcement Learning," IEEE WCL 2021, arXiv:2107.00238. https://arxiv.org/abs/2107.00238
- J. Huang et al., "DRL-Based Power Allocation for RSMA in 6G LEO Satellite Communication System," IEEE WCL 2022. https://ieeexplore.ieee.org/document/9850358
- Y. Xu, Y. Mao, O. Dizdar, B. Clerckx, "RSMA with Finite Blocklength for Short-Packet and Low-Latency Downlink Communications," IEEE TVT 2022, arXiv:2105.06198. https://arxiv.org/abs/2105.06198
- O. N. Irkiçatal et al., "DRL Enhanced RSMA for Interference Mitigation," arXiv:2403.05974. https://arxiv.org/abs/2403.05974
- "Safe DRL for Resource Allocation with Peak AoI Violation Guarantees," arXiv:2507.08653. https://arxiv.org/abs/2507.08653
- "A Survey of Freshness-Aware Wireless Networking with Reinforcement Learning," arXiv:2512.21412, Dec. 2025. https://arxiv.org/abs/2512.21412
- "Flexible Rate-Splitting Multiple Access With Finite Blocklength," IEEE 2023. https://ieeexplore.ieee.org/document/10032203
# 2026-09-15 후속 상태

이 문서에서 지적한 CEM과 262k PPO의 평가 protocol 불일치는 해소됐다.
`configs/thesis_cem_262k_long_evaluation.yaml`을 사용해 두 정책군과 CEM을 동일한
seed 930000--930005, 8,192 slots, 2,048-slot burn-in에서 재평가했다. 최종 평균 AoI는
RSMA-PPO 1.4574, SDMA-PPO 1.5610, raw RSMA-CEM 1.2249, SDMA-CEM 1.3437,
safeguarded CEM 1.2171이다. 이후 원고에서는 기존 short-audit 수치가 아니라
`docs/cem_262k_long_results.md`의 결과를 사용한다.
