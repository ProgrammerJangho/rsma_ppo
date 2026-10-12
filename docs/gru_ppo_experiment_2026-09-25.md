# GRU-PPO 적용 실험

## 목적

현재 feed-forward PPO는 한 slot의 관측만 사용한다. 채널이 Gauss--Markov로
시간 상관되고 CSIT가 불완전하므로, 관측 이력이 실제 채널 상태를 추정하거나
AoI 제어의 장기 패턴을 학습하는 데 도움이 되는지 확인했다.

이번 실험은 GRU의 효과만 분리하기 위해 다음 조건을 고정했다.

- 시스템: `K=3`, `Nt=2`, directional Rician, 20도, SNR 15 dB,
  blocklength 128, payload 256 bits, CSIT error 0.2
- 행동: 기존과 같은 `(alpha, q_k, phi_k)`
- 출발점: feed-forward PPO와 동일한 objective-only mixed Stage-1 seed-0 actor
- target PPO 예산: 524,288 interactions
- 평가: 6개 사용자 각도 순열, 궤적당 768 slot, 앞 256 slot burn-in,
  동일 channel/CSIT/ACK seed

학습 seed가 하나뿐인 pilot이므로 아래 수치는 알고리즘의 일반적 우월성을
주장하기 위한 결과가 아니다.

## 구현

actor와 critic에 별도의 GRU memory를 두고 32-slot sequence minibatch로
truncated backpropagation through time을 수행했다. episode 종료 시 hidden
state를 초기화하고, rollout 사이에는 종료되지 않은 환경의 state를 유지한다.

기존 MLP encoder와 action head 사이에 GRU residual adapter를 넣었다. residual
projection을 0으로 초기화했으므로 Stage-1 checkpoint를 불러온 직후 GRU 정책의
출력은 기존 feed-forward actor와 정확히 같다. 따라서 새로운 정책이 다른
초기 action에서 출발하는 confound를 제거했다.

## 결과

### 학습 예산별 held-out AoI

| Target interactions | GRU-PPO 평균 AoI |
|---:|---:|
| 0 (공통 Stage-1 출발점) | 1.6762 |
| 131,072 | 1.5662 |
| 262,144 | 1.5093 |
| 393,216 | 1.4851 |
| 524,288 | 1.4816 |

GRU-PPO는 출발점보다 11.61% 개선되어 정상적으로 학습했다. 그러나 개선 폭은
후반으로 갈수록 작아졌다.

### 동일 예산 최종 비교

| 정책 | 평균 AoI | p95 AoI | 평균 성공확률 | 평균 alpha | 평균 common payload |
|---|---:|---:|---:|---:|---:|
| Feed-forward PPO | **1.4667** | 2.0 | **0.5855** | 0.7561 | 279.69 |
| GRU-PPO | 1.4816 | 2.0 | 0.5648 | 0.7429 | 264.49 |
| GRU, hidden 매 slot 초기화 | 1.4885 | 2.0 | 0.5540 | 0.7433 | 257.33 |

- GRU의 feed-forward 대비 paired gain: `-1.02 ± 1.58%p` (95% CI),
  6개 궤적 중 2개 승리
- GRU memory 유지의 매-slot 초기화 대비 gain: `+0.46 ± 1.25%p`,
  6개 궤적 중 5개 승리

GRU hidden state는 작은 양의 효과를 보였지만 표본 수가 작아 0과 구분되지
않는다. memory를 사용한 최종 정책도 feed-forward PPO보다 평균 AoI가 1.02%
높았다. feed-forward와 GRU 모두 네 학습 구간에서 non-finite core metric은
0건이었고, sampled-policy AoI도 유사하게 감소했다. 구현 실패나 발산보다
현재 문제에서 memory의 추가 정보 가치가 작고 recurrent policy의 학습 효율이
조금 낮은 결과로 해석하는 것이 타당하다.

## 판단

현재 고정 20도 target 실험의 기본 정책을 GRU-PPO로 바꿀 근거는 없다.
관측에는 현재 AoI, beam/SINR feature, 이전 action, SNR, blocklength가 이미
포함되어 있어 한 slot 관측만으로도 제어에 필요한 정보가 상당 부분 제공된다.
또한 episode마다 geometry와 SNR이 고정되므로 긴 이력으로 regime을 추정해서
얻는 이점도 제한적이다.

추가 target step만 늘리는 실험의 우선순위도 낮다. 393k에서 524k로 늘렸을 때
AoI 개선은 1.4851에서 1.4816으로 0.0036에 그쳤고 feed-forward 격차가 남았다.
GRU를 다시 연구하려면 다음처럼 memory가 실제로 필요한 POMDP를 먼저 구성해야
한다.

1. episode 안에서 geometry/SNR/CSIT quality가 변하는 환경
2. 관측 잡음 또는 delayed CSIT 아래에서 history 기반 channel filtering이 필요한 환경
3. HARQ/queue처럼 현재 관측만으로 완전히 표현되지 않는 protocol state

이는 현재 학부 논문의 필수 범위가 아니다. 본 논문의 주 결과는 검증된
feed-forward PPO를 유지하고, GRU 결과는 후속 연구 또는 negative ablation으로
보관하는 것이 적절하다.

## 산출물

- 구현: `src/rsma_aoi/agents/networks.py`, `ppo.py`, `rollout_buffer.py`
- 설정: `configs/thesis_rician_k3_nt2_20deg_sinr_gru_finetune.yaml`
- 평가: `scripts/evaluate_gru_ppo_target.py`
- 요약/그림: `results/gru_ppo_target_seed0_summary_v1/`
- 최종 checkpoint: `runs/gru_objective_rsma_ppo_k3_20deg_from_stage1_seed0_524k_v1/checkpoint.pt`
- 전체 회귀 검증: 623 tests passed
