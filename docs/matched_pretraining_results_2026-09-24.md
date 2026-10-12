# RSMA-PPO와 SDMA-PPO의 사전학습 조건 일치 실험

## 목적

기존 목표 조건 비교에서는 RSMA-PPO가 `action_objective` 손실을, q-only
PPO-SDMA가 `objective_only` 손실을 사용했다. 이 차이가 기존 16.05% 격차를
만들었는지 확인하기 위해 RSMA도 `objective_only`로 다시 사전학습하고 이후
학습·평가 조건을 동일하게 유지했다.

## 통제한 조건

- 사용자/안테나: `K=3`, `N_t=2`, all-user transmission
- 채널·PHY: 동일 directional Rician 채널, imperfect CSIT `sigma_e=0.2`, RZF
  private beam, principal common beam, finite blocklength PHY
- 사전학습 데이터: geometry span 5/20/60도, SNR 5/10/15/20 dB,
  blocklength 64/128/256의 동일 상태 집합
- 사전학습: 두 방법 모두 posterior SAA 8개를 사용한 differentiable one-step
  FBL-AoI objective-only loss, 80 epochs
- Stage 1: 동일 mixed feasible curriculum, 524,288 steps
- Target fine-tuning: 20도, 15 dB, blocklength 128, 524,288 steps
- 반복: training seed 0--4
- 평가: seed당 6개 사용자 순열, 768 slots, burn-in 256 slots
- paired 난수: 같은 순열에서 channel, CSIT error, ACK seed를 공유

두 정책의 물리적 차이는 행동 공간이다. RSMA-PPO는 common 전력 `alpha`,
private 전력 simplex `q_k`, common payload fraction `phi_k`를 출력한다.
SDMA-PPO는 `alpha=0`, `phi_k=0`으로 고정하고 `q_k`만 출력한다.

## 결과

| 방법 | 평균 AoI | training-seed SD | 95% CI half-width | 평균 p95 AoI | 성공확률 | 평균 alpha |
|---|---:|---:|---:|---:|---:|---:|
| RSMA-PPO, objective-only | **1.4635** | 0.0119 | 0.0147 | 2.0333 | 0.5767 | 0.7607 |
| SDMA-PPO, objective-only | 1.7439 | 0.0597 | 0.0742 | 3.1667 | 0.4466 | 0 |
| RSMA-PPO, action+objective (기존) | 1.4626 | 0.0216 | 0.0268 | 2.0000 | 0.5675 | 0.7622 |

동일 training seed에서 각 방법의 6개 순열 평균을 먼저 구해 paired 개선율을
계산했다.

| Training seed | objective-only SDMA AoI | objective-only RSMA AoI | RSMA 개선율 |
|---:|---:|---:|---:|
| 0 | 1.6882 | 1.4667 | 13.12% |
| 1 | 1.7153 | 1.4689 | 14.37% |
| 2 | 1.8383 | 1.4773 | 19.64% |
| 3 | 1.7127 | 1.4586 | 14.84% |
| 4 | 1.7653 | 1.4460 | 18.09% |

- objective-only RSMA의 objective-only SDMA 대비 paired 개선율:
  **16.01% ± 3.40 percentage point** (95% Student-t CI half-width)
- paired 절대 AoI 차이: **0.2805 ± 0.0717**
- trajectory 승패: **30/30에서 RSMA의 AoI가 더 낮음**
- objective-only RSMA와 기존 action+objective RSMA의 paired 차이:
  objective-only 쪽 개선율 **-0.08% ± 2.26 percentage point**

![Matched objective-only pretraining comparison](../results/matched_pretraining_target_v1/figure_matched_pretraining.png)

## 판단

기존 16.05%와 조건 일치 후의 16.01%가 거의 같고, 두 RSMA 사전학습 손실의
최종 AoI 차이는 통계적으로 구분되지 않는다. 따라서 **이 목표 조건에서 기존
RSMA-PPO와 SDMA-PPO의 성능 차이를 사전학습 손실 불일치가 만들었다는 가설은
지지되지 않는다.** 사전학습 loss confound는 이번 실험으로 해소됐다.

이 결과는 현재 고정된 `K=3`, `N_t=2`, 20도, 15 dB, blocklength 128,
`sigma_e=0.2`, 고정 beam과 all-user transmission 조건에서 common 제어가 허용된
정책 클래스의 이점을 지지한다. 다만 RSMA와 SDMA의 행동 차원이 다르므로
"common stream 하나의 인과 효과"를 완전히 분리한 결과라고 표현하기보다,
**동일 objective-only 사전학습과 PPO 예산에서 RSMA allocation policy가 q-only
SDMA policy보다 낮은 AoI를 달성했다**고 쓰는 것이 정확하다.

또한 최종 RSMA critic의 explained variance는 약 0.04--0.12로 낮았다. Actor의
paired 장기 성능은 안정적으로 우수했지만 critic 적합도가 높았다고 주장해서는
안 된다.

## 재현 경로

- 실행: `scripts/run_matched_objective_rsma_seed_expansion.sh`
- 평가: `scripts/evaluate_matched_pretraining_target.py`
- 그림: `scripts/build_matched_pretraining_artifacts.py`
- 결과: `results/matched_pretraining_target_v1/`
- checkpoint:
  `runs/matched_objective_rsma_ppo_k3_20deg_from_stage1_seed{0..4}_524k_v1/`
