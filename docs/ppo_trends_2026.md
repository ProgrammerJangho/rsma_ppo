# PPO 최신 동향 분석과 본 프로젝트에의 적용 (2026-09-10)

이 문서는 최근 PPO 연구 동향을 조사하고, 본 프로젝트에서 관측된 세 가지 문제(critic explained variance 음수, 사전학습 정책으로부터의 드리프트, 미분가능 PHY 사전학습과 PPO의 역할 분담)에 매핑한다. 조사 시점은 2026-09-10이다.

> **갱신 (2026-09-10, matched-budget 감사 이후).** 이 문서의 조사 내용은 65k 예산 결과를 전제로 작성됐다. 이후 262k matched 감사가 완료되어 두 전제가 바뀌었다. (1) PPO 기여는 0.6~2%가 아니라 RSMA 6.02% / SDMA 1.35%다. (2) critic은 끝까지 음수가 아니라 262k에서 RSMA 기준 EV +0.012까지 올라온다. 문헌 조사 자체(1~5절)는 그대로 유효하며, 6~7절의 매핑과 권고를 아래에서 갱신했다.

---

## 0. 핵심 결론

본 프로젝트에서 관측한 문제들은 모두 2025~2026년 PPO 문헌의 **명명된 실패 모드**에 정확히 대응한다. 즉 우리 문제는 구현 버그나 특이 사례가 아니라 알려진 현상이며, 각각에 대해 검증된 처방과 인용 가능한 근거가 존재한다.

가장 중요한 발견은 **EVPO(arXiv:2604.19485)의 정리 1**이다. explained variance ≤ 0이라는 조건이 "critic이 advantage 분산을 줄이는 것이 아니라 키우고 있다"와 **동치**임을 증명했다.

262k matched 학습 로그로 측정한 본 프로젝트의 EV 궤적은 다음과 같다.

| | 첫 10% | 중간 | 마지막 10% | EV > 0 인 update |
|---|---:|---:|---:|---:|
| RSMA joint+gate (seed 0) | −2.151 | −0.377 | **+0.012** | 19.5% |
| SDMA power (seed 0) | −5.218 | −0.800 | −0.158 | 9.6% |

즉 critic은 예산과 함께 개선되며 RSMA 쪽은 끝에서 임계를 넘는다. 그러나 **두 arm 모두 학습의 80~90%를 EV ≤ 0 구간에서 보낸다.** EVPO 정리 1에 따르면 그 구간의 critic은 분산을 키우므로, 학습 대부분의 구간에서 advantage 추정이 배치 평균 기준선보다 나쁘다. 여기에 아직 회수하지 못한 여지가 있다.

동시에 주의할 점이 있다. RSMA critic이 SDMA critic보다 명확히 잘 학습되므로, **이 차이가 PPO 기여 비대칭(6.02% 대 1.35%)의 부분적 원인일 수 있다.** EV 게이팅으로 critic 문제를 제거하면 SDMA가 더 많이 개선되어 격차가 줄어들 가능성이 있다. 반대로 줄지 않으면 RSMA 이득이 critic 조건의 부산물이 아니라는 강한 방어가 된다.

---

## 1. 동향 A — critic 문제의 정식화와 critic-free 계열

### GRPO: critic 없는 PPO

2025년 LLM 정렬의 사실상 표준이 된 방식이다. 여러 샘플을 그룹으로 묶고 그룹 평균과 표준편차로 보상을 정규화해 advantage를 만든다. 학습된 value 함수가 아예 없다.

고전 RL 환경(MuJoCo, classic control)으로 GRPO를 옮긴 검증도 나왔다(arXiv:2511.03527). 프롬프트가 없으므로 rollout 배치를 부분집합으로 나눠 그룹을 만든다. 결론은 **조건부 우위**다. 환경에 따라 PPO가 여전히 낫고, 그룹 크기와 할인율에 민감하다. 무조건 도입할 대안은 아니다.

### EVPO: 언제 critic을 쓰고 언제 버릴지 판정

Pan 외, 베이징대·푸단대·상하이 AI Lab, arXiv:2604.19485 (2026-04-21).

- **문제 정식화:** 희소 보상에서 학습된 critic은 자신이 포착하는 상태 신호보다 큰 추정 잡음을 주입할 수 있다.
- **정리 1:** `EV ≤ 0 ⟺ P_A ≥ P_B ⟺ K ≥ 1/2`. 여기서 `P_A`는 critic 추정 오차의 분산, `P_B`는 critic이 포착해야 할 상태값 분산이다. EV가 0 이하면 critic이 분산을 **증가**시킨다.
- **방법:** 매 스텝 배치 단위 EV를 계산하고(추가 forward pass 없이 `EV = 1 − Var(G − V̂)/Var(G)`), EV > 0이면 critic 기준선(PPO 모드), EV ≤ 0이면 배치 평균 기준선(GRPO 모드)으로 하드 게이팅한다.
- **보장:** 매 스텝 분산이 PPO와 GRPO 중 나은 쪽 이하다.
- **한계:** 실험이 LLM 과제 4개(Sokoban, FrozenLake, WebShop, MATH)에 한정된다. **연속 제어 적용은 미탐색이라고 저자들이 명시한다.**

마지막 한계는 우리에게 기회다. 무선 연속 제어에서 EV 게이팅을 검증한 사례가 아직 없다.

---

## 2. 동향 B — 사전학습 정책 보존을 위한 KL 정규화

RLHF에서 표준으로 자리잡은 기법이다. 보상만으로 미세조정하면 초기 정책이 학습했던 것을 잊고 과적합하므로, 참조 정책과의 KL 발산에 페널티를 준다. 목적은 정확히 **policy drift 억제**다.

2025년의 정교화는 KL 가중치를 적응적으로 주는 방향이다. advantage가 큰 샘플에는 가중치를 낮추고 작은 샘플에는 높이는 방식(ADRPO 계열), 토큰별 엔트로피 기반 우선순위 부여 등이 있다.

본 프로젝트는 이 기법이 `reference_kl_coef`와 `reference_kl_anneal_steps`로 [`src/rsma_aoi/agents/ppo.py`](../src/rsma_aoi/agents/ppo.py)에 **이미 구현되어 있으나 모든 thesis run이 0.0으로 비활성화**했다.

---

## 3. 동향 C — offline-to-online RL과 value 재보정

### 세 가지 regime

Li, Ni, Sun, Bacon, arXiv:2510.01460 (v4, 2026-07).

사전학습 정책 성능 `J(π₀)`와 오프라인 데이터 성능 `J(π_D)`를 비교해 regime을 판정한다.

| Regime | 조건 | 권고 |
|---|---|---|
| Superior | `π₀`가 데이터보다 훨씬 우수 | **`π₀` 주변의 안정성을 우선**하고, 약한 오프라인 데이터에 과도하게 고정하지 말 것 |
| Inferior | 데이터가 `π₀`보다 훨씬 우수 | 오프라인 replay로 value 발산 방지, 필요시 파라미터 리셋 |
| Comparable | 비슷함 | 구현 세부가 결정 |

성능 향상을 prior knowledge / stability(열화 회피) / plasticity(개선 여력)로 분해한다.

### WSRL: 미세조정 초기의 성능 급락과 value 재보정

Zhou, Peng, Li, Levine, Kumar, ICLR 2025, arXiv:2412.07762.

핵심 진단은 **사전학습된 정책에 맞지 않는 value 함수**다. 온라인 미세조정을 시작하면 분포 이동 때문에 Q 함수가 잘못 보정되어 있고, 이것이 초기 성능 급락과 사전학습 지식의 망각을 유발한다. 처방은 사전학습 정책으로 소량의 rollout을 먼저 모아 value를 **재보정한 뒤** 본격 미세조정에 들어가는 warmup 단계다. 이렇게 하면 오프라인 데이터를 전혀 보관하지 않고도 안정적으로 미세조정된다.

---

## 4. 동향 D — 미분가능 시뮬레이터 gradient와 RL의 결합

### 1차 gradient가 언제 실패하는가

Suh 외, arXiv:2202.00817 (ICML 2022). 미분가능 시뮬레이터의 1차(해석적) gradient가 0차(REINFORCE/PPO형) 추정치를 항상 이기지는 않는다. **경직성(stiffness), 불연속성, 긴 시간지평**에서 1차 추정치는 편향이 누적되는 반면 0차는 샘플링 덕에 견고하다. 처방은 둘을 섞는 α차 추정기다. 2026년 재검토(arXiv:2604.18161)도 나왔다.

본 프로젝트에 중요한 구분이 있다. 우리의 FBL BLER은 **가파르지만 매끄럽다.** 정규근사 `ε = Q(√n(C(γ) − d/n)/√V(γ))`는 접촉 불연속과 달리 어디서나 미분 가능하다. 그래서 샘플링 기반 추정이 실패하는 영역에서도 해석적 gradient는 방향을 유지한다. 이것이 objective-aware 사전학습이 작동한 이유이며, Suh 외의 실패 조건에 우리가 해당하지 않는 이유다. 논문에서 명시할 가치가 있는 구분이다.

### APG와 on-policy RL의 결합

Chang, Yao, Liu, Yang, Chen, arXiv:2603.27317 (v2, 2026-04). "Where-to-Learn".

미분가능 모델의 해석적 gradient를 **탐색 방향 지시**로 쓰고, on-policy RL이 실제 환경 피드백으로 그 주변을 정제한다. 해결하는 문제가 정확히 두 가지다. 순수 APG는 모델 편향 때문에 국소 최적에 갇히고, 순수 PPO는 비효율적으로 탐색한다.

**이것이 본 프로젝트 구조와 직접 대응한다.** 우리는 같은 두 요소를 쓰지만 **순차적으로 한 번**(사전학습 완료 → PPO 미세조정) 결합한다. 문헌의 최신 형태는 **교차(interleaved)** 결합이다. 우리 구조는 이 계열의 가장 단순한 형태이며, 이를 명시하면 위치 설정과 향후 연구 방향이 동시에 정리된다.

---

## 5. 동향 E — 무선 통신에서의 PPO

체계적 문헌고찰(Future Internet 18(7):340, 2026, PRISMA 프로토콜, 2023~2026년 76편).

- PPO가 통신 분야에서 널리 쓰이는 이유는 **학습 안정성과 연속 변수 처리의 결합**이다. 송신 전력, 빔포밍 가중치, 라우팅 우선순위, 연산 오프로딩이 주 적용 대상이다.
- TD3와의 비교에 대한 서술이 본 프로젝트에 유용하다. TD3는 샘플 효율이 높아 빠르게 수렴하지만 **후반부에 불안정해지고 성능이 저하**되는 반면, PPO는 단조 상승하며 최종 평균 보상이 더 높다는 보고가 있다. 가장 가까운 선행연구(Kaya 2025)가 TD3를 쓰므로, PPO 선택의 방어 논거로 인용 가능하다.
- 공통 과제로 sim-to-real 전이, 보상 설계, 다중 에이전트 확장성이 지목된다. 또한 많은 DRL 프레임워크가 전역 CSI를 요구하는 중앙 제어에 의존해 밀집 배치에서 확장성이 나쁘다는 지적이 있다.

---

## 6. 본 프로젝트로의 매핑

| 우리가 관측한 것 | 측정값 (262k matched 기준) | 문헌에서의 이름 | 처방 |
|---|---|---|---|
| critic이 학습의 대부분에서 음수 EV | RSMA 80.5% / SDMA 90.4%의 update가 `EV ≤ 0` | EVPO 정리 1의 해로운 영역 | EV 게이팅. **단 SDMA를 더 개선해 격차를 줄일 위험 동반** |
| 사전학습 actor에 무작위 critic이 붙음 | checkpoint의 critic이 orthogonal 초기값, bias 전부 0, 모든 run이 `reset_critic: true` | O2O의 value 미보정 문제 (WSRL) | 사전학습 정책 rollout으로 critic만 먼저 보정 |
| 정책 드리프트가 **비단조** | 평균 `P_c` 0.273 → 0.362(65k) → 0.327(262k), common-on 0.436 → 0.461 → 0.368 | policy drift 후 자기 교정; Superior regime | `reference_kl_coef`로 65k 구간의 과잉 이탈을 줄일 수 있는지 검정 |
| PPO 기여가 행동 클래스별로 비대칭 | RSMA −6.02%, SDMA −1.35% (각 5/5) | 문헌에 대응 개념 없음 | **본 프로젝트 고유 관측. 논문 서사의 중심으로** |
| 65k에서 PPO 기여가 사라짐 | −0.21% (RSMA) | 예산 부족. 문헌의 "budget-sensitive fine-tuning" | 예산 3점(0/65k/262k)을 표로 제시 |
| 해석적 gradient가 샘플링을 이김 | 사전학습 상관 0.585 ≈ 오라클 0.589, scratch PPO 0.103 | APG vs 0차 추정기 | FBL은 가파르지만 매끄러움을 명시 |
| 순차적 사전학습 → PPO | — | Where-to-Learn의 교차 결합의 단순 형태 | 한계로 명시, 향후 연구로 제시 |

---

## 7. 실행 권고

우선순위는 262k matched 결과 이후 재정렬했다. 셋 다 새 기능 개발이 아니라 기존 기계장치의 설정 변경이다.

**1. critic을 먼저 보정한다 (WSRL 방식).** 사전학습 actor를 고정한 채 critic만 일정 스텝 학습해 EV를 0 위로 올린 뒤 actor 갱신을 시작한다. `freeze_allocation_steps`가 이미 구현되어 있어 actor head를 동결할 수 있다.

262k 결과가 이 처방의 예측을 부분적으로 검증했다. 예산을 늘리자 critic EV가 개선됐고(−2.151 → +0.012), 동시에 정책이 65k의 과잉 common 사용(`P_c` 0.362)에서 되돌아와 더 선택적으로 바뀌었다(`P_c` 0.327, common-on 0.461 → 0.368). **critic이 좋아질수록 정책이 CEM 쪽으로 이동한다는 방향성이 관측됐다.** critic warmup은 같은 효과를 더 적은 예산으로 얻으려는 시도다.

**2. EV 게이팅을 시도한다 (우선순위 상승).** EV ≤ 0인 스텝에서 배치 평균 기준선으로 전환한다. 두 arm 모두 학습의 80~90%가 이 구간이므로 회수할 여지가 크다. EVPO 저자들이 연속 제어 미탐색이라고 명시했으므로 결과가 어느 쪽이든 보고 가치가 있다.

**중요한 부수 효과:** SDMA critic이 RSMA critic보다 훨씬 나쁘므로, EV 게이팅은 SDMA를 더 많이 개선해 6.64% 격차를 줄일 수 있다. 이는 위험이 아니라 **감별 검정**이다. 격차가 유지되면 RSMA 이득이 critic 조건의 부산물이 아님이 증명되고, 줄어들면 현재 headline 해석을 수정해야 한다. 논문에 넣기 전에 확인하는 편이 안전하다.

**3. `reference_kl_coef`를 켠다 (우선순위 하락).** 65k 구간의 과잉 이탈을 억제하는 데는 여전히 유효하지만, 262k에서 정책이 스스로 교정하는 것이 관측됐으므로 긴급성은 낮아졌다. 다만 10개 run 전부 0.0이라 아직 한 번도 검증되지 않았다.

세 조치 모두 20°, SNR 10 dB, CSIT 0.2의 대표 조건에서 3 seed, 262k 예산으로 검증하면 충분하다.

---

## 8. 논문에 미치는 영향

세 가지가 바뀐다.

첫째, 현재 "critic explained variance가 음수지만 정책 학습은 정상으로 판단한다"는 서술은 EVPO 정리 1 앞에서 방어하기 어렵다. EV ≤ 0은 critic이 무해하다는 뜻이 아니라 **해롭다**는 뜻으로 정식화되었다. 다만 262k에서 EV가 임계를 넘어오므로, 현재 쓸 수 있는 정확한 서술은 "**critic은 예산과 함께 학습되며, 그 개선이 PPO 이득과 같은 방향으로 나타난다. 학습의 80~90%는 여전히 EV ≤ 0 구간이므로 회수 여지가 남아 있다**"이다. EV 궤적을 그림으로 넣고 EVPO를 인용하면 약점이 아니라 진단 도구가 된다.

둘째, 본 프로젝트의 파이프라인은 임의 조합이 아니라 **APG + on-policy RL 계열의 순차적 형태**로 정확히 위치 지을 수 있다. Where-to-Learn과 Suh 외를 인용하면 "왜 미분가능 PHY로 사전학습하는가"에 대한 이론적 근거가 생기고, FBL이 가파르지만 매끄럽다는 구분이 우리 사례가 그 계열에서 잘 작동하는 이유를 설명한다.

셋째, **PPO 기여의 행동 클래스별 비대칭(RSMA 6.02% 대 SDMA 1.35%)은 조사한 문헌 어디에도 대응 개념이 없다.** APG+RL 문헌은 단일 행동 공간에서 하이브리드가 순수 방법을 이기는지를 묻고, O2O 문헌은 사전학습 정책 품질에 따른 regime을 다룬다. "같은 초기화 방식과 같은 예산에서 행동 공간의 표현력에 따라 RL 미세조정의 가치가 달라진다"는 관측은 이 프로젝트 고유의 것이다. 규모가 작아 일반화 주장은 못 하지만, 논문에서 가장 방어 가능한 novelty 후보다.

---

## 참고문헌

- Pan et al., "EVPO: Explained Variance Policy Optimization for Adaptive Critic Utilization in LLM Post-Training," arXiv:2604.19485, 2026. https://arxiv.org/abs/2604.19485
- de Oliveira et al., "Learning Without Critics? Revisiting GRPO in Classical Reinforcement Learning Environments," arXiv:2511.03527. https://arxiv.org/abs/2511.03527
- Li, Ni, Sun, Bacon, "The Three Regimes of Offline-to-Online Reinforcement Learning," arXiv:2510.01460. https://arxiv.org/abs/2510.01460
- Zhou, Peng, Li, Levine, Kumar, "Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data," ICLR 2025, arXiv:2412.07762. https://arxiv.org/abs/2412.07762
- Suh et al., "Do Differentiable Simulators Give Better Policy Gradients?" arXiv:2202.00817, ICML 2022. https://arxiv.org/abs/2202.00817
- "Does 'Do Differentiable Simulators Give Better Policy Gradients?' Give Better Policy Gradients?" arXiv:2604.18161, 2026. https://arxiv.org/abs/2604.18161
- Chang, Yao, Liu, Yang, Chen, "Where-to-Learn: Analytical Policy Gradient Directed Exploration for On-Policy Robotic Reinforcement Learning," arXiv:2603.27317, 2026. https://arxiv.org/abs/2603.27317
- "Proximal Policy Optimization in 5G, B5G, and 6G Communication Systems: A Systematic Review," Future Internet 18(7):340, 2026. https://doi.org/10.3390/fi18070340
- Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017. https://arxiv.org/abs/1707.06347
