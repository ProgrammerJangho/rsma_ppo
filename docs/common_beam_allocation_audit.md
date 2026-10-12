# Common beam별 allocation 재최적화 감사

**실행일:** 2026-09-07  
**목적:** `principal`, `mrt_sum`, inverse-gain `maxmin` beam을 고정 allocation이
아니라 각 beam에 맞춰 다시 최적화했을 때 exact FBL one-step AoI 목적이 어떻게
달라지는지 확인한다.

## 1. 비교 조건

- 기존 deterministic PPO의 principal-beam 궤적에서 상태를 수집했다.
- 환경별 3개 training checkpoint, checkpoint당 112개 상태, 총 336개 상태다.
- 초기 64 slot을 제외하고 4 slot 간격으로 표본을 취했다.
- PPO가 선택한 schedule은 유지하되, power split, common-rate allocation,
  common payload scale은 beam마다 CEM으로 다시 최적화했다.
- 모든 beam은 동일한 CEM 예산과 paired random seed를 사용했다.
- CEM 예산은 `3 restarts x 5 iterations x 192 population = 2,880` evaluations다.
- Estimated-CSIT mode를 사용했다. 후보 선택에는 `h_hat`만 쓰고 `h_true`는 최종
  score에만 사용했다.
- 목적은

  \[
  J=\sum_k w_k\Delta_k p_{\mathrm{succ},k}
  \]

  이며, PHY의 실제 FBL BLER와 common/private 결합 성공확률을 그대로 사용했다.
- 별도의 SDMA CEM 해를 `P_c=D_c=0` RSMA 경계 후보로 넣었다. 이것은 exact
  **PHY boundary**지만 private-power allocation 최적값은 CEM reference다.

연속된 Gauss-Markov 상태가 포함되므로 336개 상태를 독립 표본으로 취급한
유의확률은 보고하지 않는다. Training-seed별 방향과 descriptive distribution을
함께 본다.

## 2. Maxmin 대 principal 결과

`raw`는 각 beam의 CEM 해 자체다. `SDMA safeguarded`는 `h_hat`에서 beam CEM 해와
SDMA 경계 후보 중 나은 것을 선택한 뒤 `h_true`에서 점수화한 결과다.

| regime | objective | mean Δgain | median | p05 | p95 | 개선 | 악화 |
|---|---|---:|---:|---:|---:|---:|---:|
| Perfect homogeneous | raw | **+0.0773** | +0.000004 | -0.0074 | +0.4102 | 62.2% | 19.0% |
| Perfect homogeneous | SDMA safeguarded | **+0.0380** | 0 | -0.000003 | +0.2015 | 22.6% | 6.8% |
| Perfect near/far-5 | raw | **+0.1587** | +0.000078 | -0.0073 | +0.9257 | 63.4% | 16.1% |
| Perfect near/far-5 | SDMA safeguarded | **+0.0574** | 0 | -0.000005 | +0.3595 | 26.5% | 6.0% |
| Imperfect homogeneous | raw | **+0.1242** | +0.000001 | -0.4183 | +1.1157 | 54.2% | 34.5% |
| Imperfect homogeneous | SDMA safeguarded | **-0.00024** | 0 | -0.3012 | +0.4307 | 15.2% | 12.5% |
| Imperfect near/far-5 | raw | **+0.1538** | +0.000143 | -0.5079 | +1.3891 | 56.3% | 27.4% |
| Imperfect near/far-5 | SDMA safeguarded | **+0.0292** | 0 | -0.1633 | +0.8577 | 19.3% | 10.1% |

Raw mean delta는 12개 `(regime, training seed)` 조합 모두 양수였다. Perfect-CSIT의
negative tail은 작았고, imperfect-CSIT에서는 p05가 크게 음수가 되어 estimated
objective에서 선택한 beam/allocation이 true channel에서 실패할 위험이 커졌다.

## 3. Allocation이 결과를 뒤집는다

이전 fixed-allocation counterfactual에서는 near/far 상태의 다수에서 maxmin이
principal보다 나빴다. 이번 감사에서는 near/far raw mean delta가 양수가 됐다.

두 결과는 모순이 아니다.

1. Maxmin은 약한 common gain을 높이는 대신 강한 사용자의 gain을 낮출 수 있다.
2. 고정된 큰 `rho`에서는 강한 사용자가 FBL waterfall 아래로 내려가 성능이
   붕괴할 수 있다.
3. Beam에 맞춰 CEM이 common power, `rho`, rate allocation을 다시 조정하면 그
   손실을 피하는 해를 찾을 수 있다.

따라서 현재 증거는 “maxmin beam이 near/far에 나쁘다”보다 다음 결론을 지지한다.

> Common beam의 효과는 power split과 common-rate/payload allocation에 강하게
> 결합되어 있으며, 고정 allocation으로 측정한 beam 순위를 장기 PPO 성능으로
> 일반화할 수 없다.

이는 교수님이 제안한 PPO 역할, 즉 **전력 분할 비율과 common-rate allocation을
상태별로 결정하는 구조**를 유지해야 하는 직접적인 실험 근거다.

## 4. SDMA 경계가 주는 경고

SDMA boundary 선택률은 beam과 regime에 따라 약 67~78%였다. 즉 동일한 one-step
estimated objective 아래에서는 common-off 후보가 상당수 상태에서 RSMA CEM보다
낫다.

SDMA safeguard 후 maxmin–principal 차이는 raw 차이보다 크게 줄었다. 특히 imperfect
homogeneous의 평균 차이는 `-0.00024`로 사실상 0이다. 불완전 CSIT에서는 `h_hat`에서
선택한 fallback이 `h_true`에서 항상 안전하지 않으므로 “safeguard”는 정보 기준의
명칭이며 true-channel 성능 보장은 아니다.

따라서 최종 PPO에는 다음 구조를 검토할 가치가 있다.

- `common off`를 정확히 표현할 수 있는 gate 또는 boundary action
- common-on 상태에서 PPO가 power split과 common-rate allocation을 결정
- uncertainty가 큰 상태에서 common stream을 끄는 정책을 학습

## 5. 현재 판정과 다음 단계

이번 audit은 short retraining 단계로 진행할 조건을 충족한다. Maxmin은 모든 상태에서
우수하지 않지만, allocation을 재최적화했을 때 모든 regime과 training seed에서 raw
평균 개선 가능성이 확인됐다.

다음 실험은 `principal`과 `maxmin`을 동일한 PPO 설정, 3개 training seed, 공유된
evaluation seed로 각각 재학습하는 것이다. 다음을 반드시 함께 보고한다.

- long-run average/weighted/p95/peak AoI
- common activation rate
- common power fraction, `rho`, 사용자별 common-rate allocation
- SDMA boundary 선택 또는 common-off 비율
- homogeneous/near-far 및 perfect/imperfect CSIT의 interaction

이번 결과는 principal 정책의 state distribution에서 얻은 one-step counterfactual이다.
Maxmin 정책으로 재학습했을 때 생기는 future state distribution과 장기 AoI는 아직
검증하지 않았다. 또한 CEM은 approximate reference이며 전역 최적해로 표현하지
않는다.

## 6. 재현 파일

- `scripts/audit_common_beam_allocations.py`
- `src/rsma_aoi/evaluation/beam_audit.py`
- `results/common_beam_allocation_audit_homogeneous_3seed_v1`
- `results/common_beam_allocation_audit_nearfar5_3seed_v1`
- `results/common_beam_allocation_audit_imperfect_homogeneous_3seed_v1`
- `results/common_beam_allocation_audit_imperfect_nearfar5_3seed_v1`
- `results/common_beam_allocation_audit_summary_v1`

