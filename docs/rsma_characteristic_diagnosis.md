# RSMA 특성 미활용 원인 진단 — 2026-09-15

> **후속 결과(2026-09-16):** 이 문서의 10절 Critical 1--2를 구현한 결과,
> 262k PPO의 common stream은 같은 actor의 common-off boundary보다 seen 9조건에서
> 4.15%, interpolation 4조건에서 3.49% 낮은 AoI를 냈다. 따라서 기존의 음의 common
> 기여는 구조적 필연이 아니었다. 다만 동일 프로토콜 safeguarded CEM보다 PPO AoI가
> 24.37% 높아 allocation gap은 남는다. 최신 판정은
> [`symmetric_rsma_followup_results.md`](symmetric_rsma_followup_results.md)를 따른다.

## 1. 최종 판정

현재 PHY의 common/private SINR과 decoding 순서는 표준 1-layer RSMA와 일치한다.
문제는 RSMA 식이 틀린 것이 아니라 다음 세 요인이 겹친 것이다.

1. `K=Nt=2`, fixed RZF, 모든 사용자 동시 전송 조건에서 SDMA가 이미 강하고 RSMA가
   유리한 operating region이 좁다.
2. 고정 beam과 현재 payload action이 RSMA의 joint precoder/rate-splitting 유연성을
   제한하고, common power와 common payload 사이에 매우 날카로운 결합을 만든다.
3. Objective pretraining과 PPO가 그 좁은 해를 학습하지 못한다. 특히 common gate,
   common power, payload와 사용자별 common-rate를 함께 맞추지 못하며 사용자 index
   편향도 크다.

Safeguarded CEM이 mixed 9조건 평균에서 SDMA-CEM보다 5.78% 낮은 AoI를 기록했다.
따라서 현재 action space 안에도 RSMA 이득은 존재한다. 반면 PPO common-off projection이
원 PPO보다 0.36% 좋았다. 현재 실패의 직접 원인은 `RSMA가 원래 쓸모없음`보다
`학습된 RSMA allocation이 나쁨`이다.

## 2. 이론적으로 맞는 부분

- 송신 신호는 하나의 common stream과 사용자별 private stream을 중첩한다.
- 모든 active 사용자가 common stream을 먼저 복호하고 SIC 후 private stream을 복호한다.
- Common SINR에는 모든 private stream이 interference로 들어가고, private SINR에는
  다른 private stream만 들어간다.
- Common codeword payload는 사용자 common part의 합이며 모든 사용자가 전체 codeword를
  복호한다.
- `(1-epsilon_c,k)(1-epsilon_p,k)`는 successful common decoding 이후의 private BLER로
  해석하면 유효한 conditional-chain 근사다.
- `P_c=0`, `D_c=0`은 현재 PHY에서 exact SDMA boundary다.
- Generate-at-will AoI와 성공 시 age 1 reset도 현재 slot 모델 안에서는 일관된다.

따라서 queue, HARQ 또는 stochastic arrival가 없다는 점은 연구 범위의 한계지만,
RSMA 특성이 나오지 않는 주원인은 아니다.

## 3. Operating region이 좁다

현재 시스템은 두 송신 안테나로 두 사용자를 서비스한다. RZF가 추정 채널에서 private
interference를 줄이므로 공간적으로 분리된 채널에서는 SDMA만으로 충분하다. 실제로
60°에서 safeguarded CEM은 거의 항상 SDMA boundary를 선택한다.

반대편의 5 dB에서는 256 bits를 128 channel uses에 보내야 한다. Common stream은
가장 약한 common decoder의 SINR에 묶이고, 사용자는 common과 private를 모두 성공해야
한다. 낮은 SNR에서는 두 decoding stage의 FBL cliff를 동시에 피하기 어렵다.

높은 SNR과 잘 분리된 채널에서는 AoI가 이미 1에 가까워 추가 개선 여지가 거의 없다.
따라서 RSMA가 실질적으로 유리한 곳은 `채널이 어느 정도 정렬되어 private interference가
크지만 common stream은 안정적으로 복호 가능한 중·고 SNR` 영역이다.

CEM 결과가 이를 확인한다.

| 조건 | Raw RSMA-CEM의 SDMA 대비 변화 | Safeguarded common-on |
|---|---:|---:|
| 5°, 10 dB | +11.33% | 63.0% |
| 5°, 15 dB | +22.50% | 68.4% |
| 20°, 10 dB | +7.14% | 45.4% |
| 20°, 15 dB | +8.42% | 29.4% |
| 60°, 5/10/15 dB | -33.69/-3.00/-0.16% | 약 0% |

Mixed-grid 평균만 최적화하면 RSMA가 필요한 상태가 전체 경험의 일부가 되고 common
action의 학습 신호가 희석된다.

## 4. Fixed beam이 RSMA의 핵심 자유도를 제한한다

대표 RSMA 최적화는 private/common precoder와 power, common-rate를 함께 조정한다.
현재 PPO는 RZF private direction과 principal common direction을 고정하고 power와
payload만 바꾼다.

- RZF regularization은 equal-power 기준 식이며 현재 비대칭 power allocation에 맞춰
  다시 설계되지 않는다.
- Principal common beam은 `sum_k |h_k^H u_c|^2`를 키우며 모든 common decoder의
  minimum SINR이나 FBL success를 직접 최적화하지 않는다.
- Common beam은 power, payload, AoI urgency 및 CSIT uncertainty와 무관하게 결정된다.

이는 현재 연구를 잘 정의된 fixed-beam allocation 문제로 만들지만, 일반적인 optimized
RSMA의 성능을 대표하지는 않는다. 기존 beam audit에서도 beam별 allocation을 다시
최적화해야 순위가 달라졌다. 다만 현재 principal beam에서도 CEM 이득이 존재하므로
beam만 바꾸면 문제가 모두 해결된다는 뜻은 아니다.

## 5. `alpha`, common-rate와 `rho`의 결합이 학습하기 어렵다

현재 equal-payload 두 사용자에서

```text
D_c,total = rho * D / max(c_1, c_2)
D_c,k     = c_k * D_c,total
```

이다. `c`가 균등하면 `D_c,total=2 rho D`이고 한 사용자에 치우치면 약 `rho D`가 된다.
같은 `rho`도 common-rate simplex에 따라 실제 common codeword 크기가 최대 두 배 바뀐다.

Common power `alpha`, common-rate `c`, payload scale `rho`는 별도 policy head에서
출력되지만 FBL success는 세 값의 정확한 조합에 의해 거의 계단식으로 바뀐다. 이 때문에
각 head가 개별적으로 그럴듯해도 조합이 나쁠 수 있다.

실제 5°·10 dB 결과는 다음과 같다.

| 방식 | Common-on | `alpha` when on | `rho` when on |
|---|---:|---:|---:|
| Safeguarded CEM | 63.0% | 약 0.65 | 약 0.77 |
| Mixed PPO | 10.6% | 약 0.53 | 약 0.29 |

PPO는 common을 켰을 때 전력의 절반 이상을 주면서 payload는 충분히 옮기지 않는다.
Private stream은 전력을 잃지만 private payload 감소는 작아져 common power가 낭비된다.
Common-off projection이 원 PPO보다 좋은 직접적인 이유다.

## 6. Objective-only pretraining이 CEM 정책을 distill하지 않는다

현재 `objective_only` preset은 CEM action imitation loss의 weight가 0이다. Dataset의
상태와 true-channel PHY를 사용해 differentiable one-step expected AoI를 직접 줄일 뿐,
CEM의 gate/power/rate/rho label을 직접 따라가지는 않는다.

Held-out mixed dataset에서:

| 정책 | Common-on | Common payload when on | Common power when on |
|---|---:|---:|---:|
| Safeguarded CEM label | 28.9% | 274.2 bits | 0.612 |
| 0-step seed 0 | 4.6% | 8.8 bits | 0.114 |
| 0-step seed 1 | 4.1% | 3.4 bits | 0.092 |
| 0-step seed 2 | 27.6% | 156.7 bits | 0.493 |
| 262k seed 0 | 4.1% | 4.4 bits | 0.079 |
| 262k seed 1 | 4.1% | 4.1 bits | 0.081 |
| 262k seed 2 | 12.0% | 91.5 bits | 0.450 |

두 seed는 PPO 이전부터 사실상 common-off basin에 들어갔다. PPO가 CEM-like RSMA
allocation을 망가뜨린 것만이 아니라 objective pretraining 자체가 세 seed 중 두 개에서
그 구조를 찾지 못했다.

원인은 FBL gradient saturation, gate expectation과 deterministic threshold의 차이,
single-realisation objective noise, 별도 action head의 비선형 결합으로 판단된다.

## 7. 관측과 network가 무선 채널의 대칭성을 사용하지 않는다

정책 입력은 복소 `h_hat`의 실수부와 허수부를 그대로 펼친 벡터다. 정책이 실제로 필요한
양은 channel norm, normalized alignment, RZF effective gain, cross interference,
common-beam gain처럼 global phase에 불변인 특징이다. 일반 MLP는 이를 데이터에서
스스로 근사해야 한다.

또한 equal payload/weight 조건에서는 사용자 0과 1을 교환하면 action도 같은 방식으로
교환되어야 한다. Mixed-SNR 정책의 held-out permutation 결과는 다음과 같다.

| Seed | Gate agreement | Common-rate 교환 오차 |
|---:|---:|---:|
| 0 | 95.3% | 0.973 |
| 1 | 98.8% | 0.807 |
| 2 | 71.9% | 0.500 |

Gate는 두 seed에서 비교적 불변이지만 common-rate는 사용자 교환에 거의 맞지 않는다.
Seed 0/1은 원 상태와 교환 상태 모두에서 사용자 0 index에 common rate를 몰아주는
경향을 보였다. 이는 물리적으로 의미 있는 사용자 선택보다 index shortcut에 가깝다.

## 8. PPO 수렴 실패와 정책 최적화 실패는 다르다

Mixed run은 NaN, constraint violation, KL 폭주 없이 262k까지 완료됐고 마지막 critic
explained variance도 약 0.89/0.99/0.96이었다. 따라서 학습 코드가 수치적으로 붕괴한
것은 아니다.

하지만 안정적으로 수렴한 정책이 좋은 RSMA 정책이라는 보장은 없다. Sparse gate-on
경험, sampled ACK noise, FBL cliff와 비대칭 representation 때문에 PPO는 좋은
private-power mapping을 찾고 common allocation은 포기하는 local optimum에 도달했다.

## 9. Baseline confound

별도로 학습한 RSMA-PPO와 SDMA-PPO의 차이는 PHY 효과와 optimization quality를 함께
포함한다. Mixed 평가에서 RSMA-PPO는 SDMA-PPO보다 3.32% 좋았지만, 같은 RSMA actor를
common-off boundary에 투영하면 원 RSMA보다 0.36% 더 좋아졌다.

따라서 공식 learned-SDMA control은 별도 SDMA actor와 함께 `RSMA actor common-off
projection`을 반드시 포함해야 한다. 이 control 없이 RSMA 특성을 주장하면 private
power 학습 차이를 RSMA gain으로 잘못 해석하게 된다.

## 10. 개선 우선순위

### Critical

1. **대칭·불변 observation/actor**: raw complex-flat 대신 AoI, per-user norm,
   normalized channel alignment, effective private/common gains와 cross interference를
   제공한다. 사용자별 shared encoder와 permutation-invariant pooling을 적용하거나,
   최소한 학습 중 사용자 permutation augmentation을 사용한다.
2. **Gate-on 조건부 distillation**: safeguarded CEM의 gate를 BCE로 학습하고, gate-on
   sample에서 power/rate/payload action loss와 PHY objective를 함께 사용한다.
   `objective_only` 하나에 의존하지 않는다.
3. **공유 private-power head**: RSMA와 SDMA를 별도 actor로 학습하기보다 동일 actor의
   common gate로 두 mode를 표현해 private-power 품질 confound를 제거한다.

### Major

4. **Payload action 재parameterization**: `c + rho + min(D/c)` 결합을 줄인다.
   사용자별 common bits/fractions를 직접 출력하거나, total common payload와 recipient
   simplex를 물리 단위로 명확히 분리한다. 교수님이 제안한 common-rate allocation
   제어는 유지할 수 있다.
5. **조건부 action 구조**: 먼저 gate를 결정하고, gate-off이면 private power만,
   gate-on이면 common/private power와 common-rate를 출력하는 mixture-of-experts 구조를
   사용한다.
6. **RSMA 유효 regime을 균형 있게 학습**: 5°/20°의 10/15 dB와 SDMA regime을
   condition-stratified minibatch로 균형화한다. 전체 grid 평균만으로 희귀한 common-on
   상태가 묻히지 않게 한다.

### Minor/후속 연구

7. Principal/max-min beam 선택을 discrete action 또는 offline beam table로 비교한다.
8. Posterior-CSIT SAA, precoder learning, variable blocklength는 위 학습 문제를 해결한
   뒤 검토한다.
9. Queue/HARQ/DT mismatch는 연구 범위를 넓히지만 현재 RSMA 특성 회복의 해결책은 아니다.

## 11. 학부논문 관점의 권고

현재 결과를 폐기할 필요는 없다. 논문은 다음 두 사실을 중심으로 정직하게 구성할 수 있다.

1. Safeguarded CEM은 aligned 중·고 SNR에서 RSMA의 유효 operating region을 확인한다.
2. 일반 PPO는 그 이득을 자동으로 학습하지 못하며, exact boundary, FBL-aware warm start,
   action representation과 symmetry가 중요하다.

새 구현을 하나만 선택한다면 full precoder RL보다 **사용자 대칭 feature encoder +
CEM gate/action distillation**이 우선이다. 수정량과 실험 비용이 작고, 이번 진단에서
확인된 두 직접 원인을 동시에 겨냥한다.

## 12. 근거 파일

- `src/rsma_aoi/envs/rsma_phy.py`
- `src/rsma_aoi/envs/beamforming.py`
- `src/rsma_aoi/envs/aoi_env.py`
- `src/rsma_aoi/agents/allocator_net.py`
- `src/rsma_aoi/agents/networks.py`
- `results/thesis_mixed_geometry_snr_v1/`
- `results/action_state_dependence_mixed_geometry_snr_budget_v1/`
- `docs/mixed_geometry_snr_results.md`
- `docs/common_beam_allocation_audit.md`
