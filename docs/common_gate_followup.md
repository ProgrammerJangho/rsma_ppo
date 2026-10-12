# PPO 전력·common-rate 제어 후속 검증

## 현재 결론

PPO가 전력 분할과 common payload/rate 배분을 직접 제어하는 경로는 정상 동작한다.
그러나 현재의 `K=Nt=2`, 10 dB, 2 bits/use, homogeneous 조건에서는 common stream의
추가 이득이 확인되지 않았다. 학습된 private-power SDMA가 가장 낮은 평균 AoI를 보였고,
common stream을 항상 켠 joint PPO는 성공 횟수가 더 많아도 성공 간격의 변동성이 커져
AoI가 나빠졌다. Exact common-off gate는 이 손실 대부분을 제거했다.

이 결과는 "RSMA가 구조적으로 SDMA보다 나쁘다"는 뜻이 아니다. 현재 RSMA 정책은
SDMA 경계를 정확히 포함하지만 더 큰 혼합 action space를 유한한 PPO 예산으로 학습한다.
따라서 아래 결과는 현재 beam, FBL traffic, PPO 설정과 계산 예산을 함께 평가한 것이다.

## 구현한 제어 구조

### 안전한 연속-action 초기화

독립 validation 채널 512개(seed 810000)의 PHY landscape에서 초기 대칭 배분
`f_c=1/3, rho=0.5`의 평균 common BLER가 0.9954였다. Test 평가 seed는
910000--910002로 분리했다. 이 진단을 근거로 초기 modal action을
`f_c=0.2, rho=0.05`로 설정할 수 있게 했다. Beta concentration은 20이다.

### Exact common-off gate

Joint policy에 Bernoulli 변수 `g`를 추가했다.

- `g=1`: PPO가 common/private power simplex, common payload shares, `rho`를 모두 제어한다.
- `g=0`: `f_c=0`, `rho=0`을 정확히 실행하고 PPO가 private-power simplex만 제어한다.
- Off 상태에서는 실행 action에 영향을 주지 않는 common-power latent와 rate/rho latent를
  PPO log-probability에서 제외한다.
- PPO entropy도 `H(g) + E_g[H(allocation | g)]` 형태로 계산한다.

따라서 gate가 있는 RSMA action class는 코드 수준에서 학습형 SDMA 경계를 포함한다.
`allocation_control=sdma_power`는 같은 관측, encoder, optimizer, 학습량을 사용하면서
common을 정확히 끄고 private power만 학습하는 직접 baseline이다.

## 단계별 결과

모든 수치는 deterministic policy, training seed 3개의 평균이다. 32k 결과는 초기화와
구조 진단용이고, 131k 결과도 단일 operating point pilot이다.

| 단계 | Training steps | Mean AoI | 핵심 해석 |
|---|---:|---:|---|
| 기본 joint 초기화 | 32,768 | 4.9791 | 초기 common rate가 지나치게 공격적 |
| `f_c=.2, rho=.05` 초기화 | 32,768 | 2.1402 | 초기화만으로 실패 대부분 제거 |
| 같은 초기화, entropy coefficient 0 | 32,768 | 2.1372 | entropy bonus가 주원인이라는 증거 없음 |
| Refined common gate | 32,768 | 1.8647 | exact SDMA 경계 접근으로 추가 개선 |
| Joint, common 항상 on | 131,072 | 2.0566 | 학습량을 늘려도 SDMA보다 불리 |
| Refined common gate | 131,072 | 1.7618 | 손실을 크게 줄이나 learned SDMA에는 미달 |
| Learned private-power SDMA | 131,072 | **1.7303** | 현재 조건의 가장 강한 PPO 기준 |
| Equal-power SDMA | 고정 | 1.8378 | 학습형 SDMA보다 약한 참조선 |

131k에서 gate RSMA와 learned SDMA의 평균 차이는 +0.0314 slot, 즉 SDMA 대비
약 +1.82%이다. 세 training seed 각각에서도 gate RSMA가 +0.0245, +0.0326,
+0.0372 slot 높았다. Seed가 3개뿐이므로 이를 통계적 일반성으로 해석하지 않는다.

## 평균 성공확률과 AoI 순위가 다른 이유

| 정책 | Mean AoI | ACK rate | 사용자 ACK 상관 | 동시 성공 | 동시 실패 | Inter-delivery CV | 최대 간격 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SDMA power PPO | **1.7303** | 0.4800 | -0.554 | 0.093 | 0.133 | **0.425** | 10 |
| Joint RSMA, common on | 2.0566 | **0.5298** | -0.031 | 0.272 | 0.213 | 0.806 | 17 |
| Joint RSMA, gate | 1.7618 | 0.5112 | -0.246 | 0.200 | 0.178 | 0.539 | 11 |

RSMA는 평균 ACK rate가 더 높지만 성공이 두 사용자에게 동시에 몰리고, 동시 실패와 긴
실패 구간도 증가했다. 반면 SDMA PPO는 두 사용자의 ACK를 강하게 음의 상관으로 만들어
업데이트를 고르게 분산했다. Discrete renewal 근사에서는 평균 AoI가
`E[X^2]/(2E[X]) + 1/2`에 좌우되므로, 평균 성공 횟수만 늘리는 one-step 목적은 긴 간격의
비용을 놓칠 수 있다. 여기의 interval 통계는 finite window에서 완결된 구간만 사용하므로
메커니즘 진단이며, 실제 trace에서 계산한 mean AoI가 주 지표다.

Gate는 결정론적 평가 slot의 24.4%에서 common을 켰다. Common-on slot의 평균 common
power는 0.1885, `rho`는 0.0497, common coding rate는 0.1456 bits/use, conditional common
BLER는 0.0991이었다. Common decoding 자체는 안정화됐지만 평균 AoI 이득으로 이어지지는
않았다.

초기 common-on probability를 0.5에서 0.1로 낮춘 별도 131k ablation에서는 deterministic
정책이 평가 상태 전부에서 common을 껐고 AoI 1.7259를 얻었다. Learned SDMA와의 평균
차이는 -0.00445 slot(-0.26%)이며 seed별로 -0.0079, +0.0081, -0.0136이었다. 이 차이는
training variation 범위로 보는 것이 타당하다. 이 설정은 SDMA 성능을 안전하게 회복하지만
common contribution은 0이다. 같은 test seed를 관찰한 pilot이므로 on-probability 0.1을
최종 hyperparameter로 채택하려면 새 validation/test 분리가 필요하다.

Gate 0.5 정책을 같은 seed에서 다시 실행해 사전 상태와 gate 선택을 대조했다. Common-on은
추정 채널 정렬도와 거의 상관이 없었고(correlation -0.050), pre-action 사용자 AoI 차이와는
음의 상관(-0.285)이었다. Common-on 상태의 평균 AoI gap은 0.589, off 상태는 1.213이었다.
즉 현재 gate는 채널이 평행한 RSMA 유리 구간을 찾기보다 두 사용자의 age가 비슷할 때
동시 update를 선택하는 경향이 강하다. 이것은 학습된 행동에 대한 기술이며 인과 증명은 아니다.

## 65k CSIT × near/far 방향 선별

추가 pilot은 `CSIT error std={0, 0.3}`와 5 dB path-gain disparity의 유무를 2×2로
분리했다. Error 0.3은 normalized model, near/far gain은 `[0, -5]` dB를 mean-one으로
정규화했다. 나머지 조건과 train/evaluation seed는 같으며 각 정책에 training seed 3개를
사용했다.

| 조건 | Gate RSMA AoI | SDMA-power AoI | RSMA 차이 | Common-on 비율 | On 조건 common BLER |
|---|---:|---:|---:|---:|---:|
| Perfect / homogeneous | 1.7943 | **1.7058** | +5.21% | 0.412 | 0.090 |
| Imperfect / homogeneous | 2.1341 | **1.9313** | +10.52% | 0.373 | 0.093 |
| Perfect / near-far 5 dB | 2.3368 | **2.2041** | +6.02% | 0.375 | 0.180 |
| Imperfect / near-far 5 dB | 2.6731 | **2.4342** | +9.79% | 0.396 | 0.153 |

이 작은 grid에서는 imperfect CSIT와 near/far가 RSMA 이득을 만들지 않았다. Common-on
conditional BLER가 0.09--0.18로 낮아도 장기 AoI가 개선되지 않았으므로, decoding 실패만
줄이는 것이 충분하지 않다는 증거다. Perfect/homogeneous에서 65k의 차이 5.21%가 131k에서
1.82%로 줄었다는 점 때문에, 조건 간 차이에는 물리 효과뿐 아니라 학습 속도도 섞여 있다.
본 실험은 operating-regime 방향 선별이며 최종 비교에는 동일한 더 긴 예산과 5개 이상의
training seed가 필요하다.

## 근거의 구분

- **코드 근거:** exact-off action, masked likelihood, checkpoint 재현, SDMA PHY 경계 일치.
- **실험 근거:** 위 표의 AoI, ACK correlation, inter-delivery 통계와 seed별 결과.
- **해석:** common stream이 성공을 묶어 burstiness를 키운 것이 현재 조건의 AoI 손실
  메커니즘이라는 설명. 관측 결과와 일치하지만 인과를 확정하려면 추가 ablation이 필요하다.
- **주장 불가:** RSMA 일반의 열등성, 모든 imperfect-CSIT/near-far 조건에서의 열등성,
  PPO의 전역 최적성.

## 재현 파일

- 131k joint/SDMA: `archive/results/thesis_joint_vs_sdma_131k_v2/`
- 131k refined gate: `archive/results/thesis_common_gate_refined_131k_v2/`
- 131k low-on initialization: `archive/results/thesis_common_gate_p01_131k_v1/`
- 메커니즘 CSV/그림: `archive/results/thesis_common_mechanism_131k_v1/`
- Gate-state 진단: `archive/results/thesis_gate_state_diagnostic_131k_v2/`
- 65k 요인표/그림: `archive/results/thesis_allocation_factorial_65k_v1/`
- 분석 스크립트: `scripts/analyze_common_gate_followup.py`
- 요인 분석 스크립트: `scripts/summarize_allocation_factorial.py`
- Gate-state 분석 스크립트: `scripts/diagnose_common_gate_state.py`
- 핵심 검증: `tests/test_allocation_ablation.py`

전체 테스트 510개가 통과했다. 결과 디렉터리의 manifest에는 실행 인자, source hash,
checkpoint hash, 평가 seed가 저장된다.
