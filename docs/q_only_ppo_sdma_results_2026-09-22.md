# Matched q-only PPO-SDMA 결과

> 2026-09-24 갱신: 아래의 기존 RSMA 행은 `action_objective`, SDMA 행은
> `objective_only` 사전학습을 사용했다. 두 방법 모두 `objective_only`로 맞춘
> 재실험 결과는 [`matched_pretraining_results_2026-09-24.md`](matched_pretraining_results_2026-09-24.md)에
> 정리했다. 조건 일치 후 paired 개선율은 16.01% ± 3.40%p로, 기존 16.05%와
> 사실상 같았다. 논문 수치에는 갱신된 결과를 사용한다.

## 비교 목적

기존 max-AoI top-2 equal-power SDMA는 사용자 선택과 전력 제어가 RSMA-PPO와
달랐고, common-off 투영은 RSMA와 함께 학습된 private 전력 q_k를 그대로
사용했다. 본 실험은 common stream의 순기여를 더 엄격하게 분리하기 위해
alpha=0과 phi_k=0을 고정하고 q_k만 학습하는 PPO-SDMA를 사용한다.

## 동일 조건

- 사용자/안테나: K=3, N_t=2, all-user transmission
- 행동: q_k만 학습하며 private 전력 simplex를 만족
- PHY: 동일 RZF private beam, FBL 모델, total power 및 payload
- 사전학습: common-off posterior-averaged one-step objective, 80 epochs
- Stage 1: 동일 mixed feasible curriculum, 524,288 steps
- Target: 20도, 15 dB, n=128, 524,288 steps
- 반복: 독립 training seed 5개
- 평가: seed당 사용자 순열 6개, 768 slots, burn-in 256 slots
- 공통 난수: RSMA-PPO와 q-only PPO-SDMA가 channel, CSIT error 및 ACK seed 공유

## 결과

| 방법 | 평균 AoI | seed SD | 95% CI half-width | 평균 p95 AoI | 성공확률 |
|---|---:|---:|---:|---:|---:|
| RSMA-PPO | 1.4626 | 0.0216 | 0.0268 | 2.0000 | 0.5675 |
| q-only PPO-SDMA | 1.7439 | 0.0597 | 0.0742 | 3.1667 | 0.4466 |
| Max-AoI top-2 equal-power SDMA | 2.8219 | - | - | 7.5000 | 0.3300 |

동일 training-seed 쌍에서 RSMA-PPO의 q-only PPO-SDMA 대비 개선율은 평균
16.05%이고 95% Student-t 신뢰구간 half-width는 4.02 percentage point다.
평균 AoI 절대 차이는 0.2813 ± 0.0825다. RSMA-PPO는 5개 training seed와
6개 사용자 순열로 이루어진 30개 paired 궤적 모두에서 q-only PPO-SDMA보다
낮은 AoI를 보였다.

| Training seed | q-only PPO-SDMA AoI | RSMA-PPO AoI | RSMA 개선율 |
|---:|---:|---:|---:|
| 0 | 1.6882 | 1.4555 | 13.78% |
| 1 | 1.7153 | 1.4436 | 15.84% |
| 2 | 1.8383 | 1.4586 | 20.66% |
| 3 | 1.7127 | 1.4999 | 12.42% |
| 4 | 1.7653 | 1.4555 | 17.55% |

## 해석

q-only PPO-SDMA가 휴리스틱 SDMA의 2.8219를 1.7439로 크게 낮춘 결과는
학습된 q_k와 all-user 전송의 결합 효과가 전체 PPO 이득의 큰 부분임을 보여준다. 따라서
48.17% 전체를 RSMA common stream의 효과로 설명할 수 없다.

동시에 RSMA-PPO가 재학습한 q-only PPO-SDMA보다 16.05% 낮은 AoI를 보였고
30/30 궤적에서 승리했다. 이 조건에서는 급한 사용자 메시지 일부를 common
stream으로 보내 간섭을 복호하는 기능이 private-power adaptation 위에 추가적인
이득을 제공한다. 기존 common-off 투영의 43.76%는 재최적화되지 않은 q 때문에
common 순기여를 과대평가한 진단값으로만 유지한다.

본 결론은 20도, 15 dB, n=128, 고정 RZF/principal beam, K=3, N_t=2의 목표
조건에 한정된다. 다른 geometry, SNR 및 blocklength에서 matched q-only 격차가
동일하다고 일반화하지 않는다.
