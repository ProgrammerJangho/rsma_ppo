# Controlled Rician 채널의 RSMA–SDMA PHY 감사

**실행일:** 2026-09-07  
**목적:** PPO 학습 전에 spatial alignment가 RSMA common stream의 필요성과
power/common-rate allocation을 실제로 바꾸는지 확인한다.

## 1. 비교 설계

- `K=N_t=2`, Rician K-factor 8 dB, perfect CSIT, SNR 10 dB
- FBL `n=128`, 사용자별 payload 256 bit
- LoS angle separation `5°, 20°, 60°`
- 조건별 독립 stationary channel 192개
- 모든 angle 조건에 동일한 NLoS random draw 사용
- 각 채널에서 AoI profile 세 개 사용: `[1,1]`, `[8,1]`, `[1,8]`
- 모든 사용자를 scheduling
- `principal`, `maxmin` common beam별로 power split, common-rate allocation,
  common payload scale `rho`를 CEM으로 재최적화
- CEM 예산: `3 restarts × 5 iterations × 192 population = 2,880` evaluations
- 별도 CEM으로 SDMA private-power allocation 최적화
- RSMA의 정확한 `P_c=D_c=0` SDMA boundary와 PHY identity 확인
- 고정 allocation `f_c=0.25`, `rho=0.10`, equal private power/rate도 함께 평가

목적함수는 현재 환경의 one-step expected weighted-AoI 감소량이다.

\[
J=\sum_k w_k\Delta_kp_{\mathrm{succ},k},
\qquad
\mathbb E[\Delta(t+1)]=B-J.
\]

CEM 결과는 유한 계산 예산의 approximate reference이며 전역 최적해가 아니다.
`safe`는 perfect CSIT에서 각 beam의 CEM 해와 SDMA boundary 중 큰 `J`를 선택한 값이다.

## 2. 채널 regime이 의도대로 분리됐는가

| angle | configured LoS correlation | realized correlation mean | median | p05–p95 | mean condition number |
|---:|---:|---:|---:|---:|---:|
| 5° | 0.981 | 0.858 | 0.885 | 0.619–0.994 | 8.87 |
| 20° | 0.731 | 0.667 | 0.693 | 0.321–0.934 | 4.29 |
| 60° | ~0 | 0.139 | 0.115 | 0.014–0.349 | 1.64 |

Rician scatter 때문에 realized correlation은 LoS correlation과 같지 않지만,
aligned/intermediate/near-orthogonal 조건은 명확히 분리됐다. 특히 5° 채널의 큰
condition number는 private-only spatial separation이 어렵다는 것을 보여준다.

## 3. Equal-urgency 핵심 결과

낮은 expected next AoI가 좋다.

| angle | SDMA | RSMA principal raw | RSMA maxmin raw | raw 최선의 SDMA 대비 변화 | per-state safe 최선 |
|---:|---:|---:|---:|---:|---:|
| 5° | 2.8921 | **2.5088** | 2.5097 | **−13.25%** | 2.5080 |
| 20° | 2.4995 | 2.2762 | **2.2675** | **−9.28%** | 2.2597 |
| 60° | **2.0163** | 2.0352 | 2.0220 | +0.28% | **2.0161** |

Paired objective-gain difference의 95% confidence interval은 다음과 같다.

| angle | comparison | mean delta gain | 95% CI |
|---:|---|---:|---:|
| 5° | principal − SDMA | +0.3833 | `[+0.3285,+0.4381]` |
| 5° | maxmin − principal | −0.0009 | `[−0.0100,+0.0082]` |
| 20° | maxmin − SDMA | +0.2320 | `[+0.1826,+0.2814]` |
| 20° | maxmin − principal | +0.0087 | `[+0.0029,+0.0144]` |
| 60° | principal − SDMA | −0.0189 | `[−0.0297,−0.0082]` |
| 60° | maxmin − SDMA | −0.0057 | `[−0.0117,+0.0002]` |

## 4. Allocation이 channel alignment에 반응하는가

Equal-urgency에서 CEM이 선택한 평균은 다음과 같다.

| angle | beam | common power `f_c` | common payload `rho` | SDMA boundary 선택률 |
|---:|---|---:|---:|---:|
| 5° | principal | 0.684 | 0.725 | 35.4% |
| 5° | maxmin | 0.680 | 0.722 | 33.9% |
| 20° | principal | 0.495 | 0.490 | 47.4% |
| 20° | maxmin | 0.492 | 0.492 | 46.4% |
| 60° | principal | 0.352 | 0.024 | 100.0% |
| 60° | maxmin | 0.320 | 0.043 | 99.5% |

Angle separation이 커질수록 `rho`가 `0.72 → 0.49 → 0.02–0.04`로 감소했다.
이는 common message의 필요성이 channel alignment에 따라 달라진다는 이론적 기대와
일치한다.

60°의 raw CEM은 `rho`를 거의 0으로 만들면서도 common power를 0.32 이상 남긴다.
Softmax가 정확한 simplex boundary를 생성하지 못하고 FBL success가 포화되어 목적이
평평해졌기 때문이다. 이 해를 최적 RSMA라고 해석하면 안 된다. 현재 PPO의
`common_gate_enabled`는 gate-off 시 common power와 common payload를 모두 정확히 0으로
만들고 private power를 재정규화하므로 이 문제를 직접 처리한다.

## 5. Maxmin common beam에 대한 판정

Maxmin은 20°와 60°에서 raw common-on 해를 principal보다 개선했다. 그러나 60°에서는
두 raw RSMA 해 모두 SDMA보다 나빴고, SDMA safeguard 후 beam 차이는 사실상 사라졌다.
5°에서는 maxmin과 principal 차이의 confidence interval이 0을 포함한다.

고정 allocation만 보면 60° equal-urgency에서 maxmin−principal gain이 `+0.2465`
`[+0.2098,+0.2831]`로 매우 커 보인다. 하지만 같은 고정 allocation의 expected AoI는
principal `2.6775`, maxmin `2.4311`로 둘 다 SDMA `2.0163`보다 나쁘다. 따라서 고정
allocation beam 비교는 maxmin의 상대 개선을 크게 보이게 하면서 common stream을
써야 하는지라는 더 중요한 결정을 숨긴다.

**판정:** principal을 주 beam으로 유지하고 maxmin을 diagnostic ablation으로 둔다.
현재 논문의 핵심 제어 변수는 beam 자체보다 `common gate`, power split, common-rate
allocation이다.

## 6. AoI urgency 결과

`[8,1]`과 `[1,8]`을 모두 평가하여 사용자 index 편향을 피했다.

- 5°에서는 두 urgency 방향 모두 RSMA의 mean gain이 SDMA보다 양수였다.
- 20°에서도 mean gain은 양수였지만, state별 median은 거의 0이고 raw RSMA가 SDMA보다
  나쁜 state가 약 39–45%였다.
- 60°에서는 urgency 방향과 관계없이 raw RSMA가 SDMA보다 평균적으로 나빴고,
  boundary가 약 99–100% 선택됐다.
- Maxmin의 raw principal 대비 개선은 60°에서 가장 커졌지만, gate를 적용하면 양쪽이
  SDMA로 수렴했다.

평균 이득만 보고 모든 state에서 RSMA가 우수하다고 주장할 수 없다. 정렬된 채널에서도
상태별 common-off 결정이 필요하다는 결과다.

## 7. PPO 실험에 주는 결론

1. `common_gate_enabled=true`를 주 모델에 유지한다.
2. Scheduling은 끄고 PPO가 gate, power split, common-rate allocation에 집중하게 한다.
3. 5°, 20°, 60°를 모두 학습해 policy가 `rho`와 common-on 확률을 감소시키는지 본다.
4. Principal을 main configuration, maxmin을 beam ablation으로 사용한다.
5. SDMA는 별도 모델만 두지 말고 PPO action class 내부의 exact common-off boundary로도
   평가한다.
6. 최종 주장은 “RSMA가 항상 우수”가 아니라 “PPO가 spatial regime에 따라 RSMA와
   SDMA boundary를 선택하고 allocation을 조정할 수 있는가”로 둔다.

이번 결과는 one-step expected AoI 결과다. Long-run sampled-AoI, PPO의 state
distribution, imperfect CSIT, near/far disparity는 아직 검증하지 않았다. 따라서 다음
단계는 세 angle config에서 principal-beam PPO를 동일한 세 training seed로 학습하고,
공유 evaluation trace에서 SDMA와 비교하는 것이다.

## 8. 재현 파일

- `scripts/audit_rician_phy_regimes.py`
- `scripts/plot_rician_phy_audit.py`
- `results/rician_phy_regime_audit_v2/optimized_trace.csv.gz`
- `results/rician_phy_regime_audit_v2/fixed_trace.csv.gz`
- `results/rician_phy_regime_audit_v2/scheme_summary.csv`
- `results/rician_phy_regime_audit_v2/paired_comparisons.csv`
- `results/rician_phy_regime_audit_v2/channel_summary.csv`
- `results/rician_phy_regime_audit_v2/rician_phy_audit.png`
- `results/rician_phy_regime_audit_v2/manifest.json`
