# Gate·ρ 없는 RSMA-PPO 구현 및 예비 검증

> **상태:** 아래 성능표는 최초 sigmoid pilot의 역사 기록이다. 현재
> 구현은 [`canonical_action_definition.md`](canonical_action_definition.md)의
> endpoint-capable action으로 갱신됐다. 따라서 이 표를 새 action의 성능
> 결과로 재사용하지 않는다.

## 결론

WSL `/home/alexalex2/rsma_ppo`에 구현했다. 기존 설정과 결과는 보존하고 `configs/thesis_direct_fraction.yaml`로 새 방식을 선택한다.

최초 구현 검증은 통과했지만, 짧은 sigmoid pilot에서는 PPO 성능 개선을
얻지 못했다. 현재 구현은 near-zero를 SDMA로 간주하지 않고, action
transform 자체가 정확한 0을 출력할 수 있도록 변경했다.

## 구현

- `α = clip(1/2 + z_α/6, 0, 1)`
- `p_k = masked_softmax(z_private)_k`
- `P_c = α P_max`, `P_k = (1-α)p_k P_max`
- `phi_k = clip(1/2 + z_user,k/6, 0, 1)`, 비활성 사용자는 0
- `D_c,k = phi_k D_k`, `D_p,k = (1-phi_k)D_k`
- 실제 common-rate `C_k = D_c,k/n`, 총 common codeword bits = `Σ_k D_c,k`

새 PPO에는 common gate 및 ρ head가 없다. 환경 action에도 `common_payload_scale` 키가 없다.
Gaussian latent의 log-prob와 entropy를 사용하며, 저장된 latent로 PPO
ratio를 재계산한다. hard-sigmoid/softmax는 학습 파라미터가 없는 action
변환이다. 별도의 Bernoulli gate나 payload threshold는 없다.
기존 rollout 및 oracle 데이터 형식의 ρ 저장 칸은 호환성을 위해 남아 있지만, 새 정책은 이를 출력·샘플링·학습하지 않으며 log-prob/entropy에도 포함하지 않는다. 내부 PHY adapter의 상수는 `per_user_fraction`에서 무시된다.

새 설정명 `initial_common_fraction=0.05`는 각 사용자 `phi_k`의 초기값이다.
`initial_common_payload_scale`은 과거 checkpoint/config 재현용 alias로만
남는다. 학습 가능한 ρ가 아니다.

## 핵심 PHY 발견

128 channel uses에서 `α=1e-8`, `u_k=α²`와 같은 양수 경로의 common BLER가 약 0.5로 남는다. normal approximation의 margin은 대략 `sqrt(n)(C-D/n)/sqrt(V)`이다. 낮은 SINR에서 C와 V가 함께 감소하므로 payload가 매우 작아도 margin이 0으로 접근할 수 있고, Q(0)=0.5가 된다.

반면 **정확히 0 bits인 codeword는 기존 구현에서 decoding을 요구하지 않는다.** 따라서 near-zero와 zero 사이의 차이가 있다. 이는 낮은 SINR·극소 fractional bits에서 사용하는 근사의 경계 문제이며, 실제 시스템에서도 반드시 BLER=0.5라는 뜻은 아니다.

이번에는 BLER를 임의로 0으로 만드는 처리, integer-bit 양자화, packet decoding 가정 변경을 하지 않았다. 그런 변경은 별도 모델 변경으로 검증해야 한다.

## PPO 예비 비교

- 기존/새 구조 각각 seed 0·1·2
- seed당 **8,192 steps**, 사전학습 없는 cold start
- 기존 mixed geometry/SNR 학습 분포 유지
- 평가: 5°/20°/60° × 10/15 dB, 각 조건 2개 episode × 512 slots
- 양쪽에 같은 평가 seed 사용, 동일 외생 채널 및 ACK 난수
- 같은 policy의 common-off projection 포함: payload를 전부 private로 복원하고 private 비중에 전체 전력을 재배분

| 정책 | Seed 0 | Seed 1 | Seed 2 | 평균 AoI |
|---|---:|---:|---:|---:|
| legacy | 2.5625 | 2.7314 | 2.5970 | 2.6303 |
| legacy common-off projection | 2.4364 | 2.4094 | 2.4365 | 2.4275 |
| direct | 2.9646 | 3.2181 | 2.9930 | 3.0586 |
| direct common-off projection | 2.4409 | 2.6855 | 2.4086 | 2.5117 |

새 구조 평균 AoI는 기존보다 약 16.28% 높았다. 이는 **짧은 cold-start 예비 결과**다. 기존 262,144-step 및 objective 사전학습 결과와 직접 비교할 수 없다. 3개 seed와 조건당 2개 평가 episode만으로 통계적 우월성을 주장하지 않는다. warm start 후 장기 성능은 미확인이다.

## CEM 비교

6개 조건 × 16개 동일 상태 = 96 states. 양쪽 모두 RSMA-CEM과 SDMA-CEM을 별도로 구하는 safeguard를 사용했다. 각 solve는 1 restart × 5 iterations × 192 candidates이다. 즉 safeguard 전체 예산은 두 solve의 합이다.
현재 추정 채널로 후보를 선택하고, 실제 채널로 결과를 평가했다. 채널·AoI·전력 예산·beam family를 동일하게 유지했다.

- 기존 표현: 평균 expected next-slot AoI **1.2193**
- 새 표현: 평균 expected next-slot AoI **1.1824**
- 약 **3.03% 감소**

이는 새 표현으로도 좋은 allocation을 찾을 수 있다는 증거다. 장기 AoI나 PPO 개선의 증거는 아니다. 표현에 따라 탐색 차원과 후보가 달라지므로 CEM의 전역 최적성도 주장하지 않는다.

## 실행한 검증

- 전체 pytest: **594 passed**
- 전력 제약, 사용자별 payload 보존, fraction 비정규화 유지
- NumPy/PyTorch SINR·BLER·성공확률 일치 및 finite gradient
- exact SDMA boundary, zero private bits, 사용자별 decoding chain
- near-zero 잔여 BLER regression
- PPO update 전 log-prob replay 일치, 비활성 ρ 저장값이 score에 영향 없음
- PPO update, checkpoint 저장/복원
- 다른 action 표현의 pretrained checkpoint 로드 거부
- CEM raw/SDMA/safeguard, beam-audit boundary
- 기존 oracle label의 payload-preserving 변환: 각 split 32개 smoke 데이터
- objective_only 사전학습 2 epochs → checkpoint → PPO 512 steps 실행

마지막 사전학습 실험은 연결 경로 검증이며 성능 비교표에는 사용하지 않았다.
초기 rollout에서 episode가 아직 종료되지 않아 출력되는 `ret nan`은 기존 logger의 빈 episode 평균 표기다. 실행된 PPO loss가 NaN으로 붕괴했다는 의미는 아니다.

## 재현

WSL에서 실행:

```bash
cd /home/alexalex2/rsma_ppo
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q

# 독립된 새 결과 디렉터리를 선택할 것
OMP_NUM_THREADS=1 .venv/bin/python scripts/run_direct_fraction_study.py --steps 8192 --out results/direct_fraction_pilot_v2

# 긴 학습용 설정. 이번 작업에서는 262k 실험을 실행하지 않았다.
OMP_NUM_THREADS=1 .venv/bin/python scripts/train_ppo.py --env-config configs/thesis_direct_fraction.yaml --seed 0 --run-dir runs/direct_fraction_full_seed0

# 기존 oracle label을 새 표현으로 변환 (기존 데이터는 보존)
.venv/bin/python scripts/convert_direct_fraction_dataset.py --source datasets/rician_mixed_geometry_snr_csit02_oracle_v1 --source-config configs/thesis_rician_mixed_geometry_snr_csit02.yaml --target-config configs/thesis_direct_fraction.yaml --out datasets/direct_fraction_full_v1

OMP_NUM_THREADS=1 .venv/bin/python scripts/pretrain_objective_actor.py --dataset datasets/direct_fraction_full_v1 --env-config configs/thesis_direct_fraction.yaml --agent-config configs/thesis_direct_fraction.yaml --preset objective_only --epochs 60 --out runs/direct_fraction_objective_full_seed0
```

## 한계 및 다음 판단

새 PPO·CEM·objective actor 경로를 지원한다. 기존 standalone AllocatorNet 및 TD3는 이번 새 action 모드를 지원하지 않으며 잘못 실행되지 않도록 오류를 낸다. 예전 분석 스크립트 중 simplex·ρ를 전제로 하는 스크립트는 새 모드에 그대로 적용하지 말고 신규 runner를 사용한다.
실행된 실험에 대칭 encoder, 새 distillation loss, full precoder RL은 추가하지 않았다.

현재 direct action을 새 연구 세대의 기준으로 채택했다. 기존 sigmoid
pilot과 gate 기반 checkpoint는 action class가 다르므로 그대로 이어서
학습하거나 headline 표에서 직접 비교하지 않는다. 다음 단계는 새 action으로
oracle data를 생성하고 matched objective warm start와 장기 PPO를 다시 실행하는 것이다.

## 파일

- 새 설정: `configs/thesis_direct_fraction.yaml`
- 비교 runner: `scripts/run_direct_fraction_study.py`
- 데이터 변환: `scripts/convert_direct_fraction_dataset.py`
- 새 테스트: `tests/test_direct_fraction.py`
- 원본 실험 결과: `results/direct_fraction_pilot_v1/`
- 학습 checkpoints: `runs/direct_fraction_pilot_v1/`
- 기존 파일 백업: `archive/direct_fraction_before_*/`
- 전달 bundle: 변경 source/config/tests, patch, 이 보고서, JSON 실험 결과 및 검증 로그. 큰 학습 checkpoint와 기존 전체 저장소는 포함하지 않음.
