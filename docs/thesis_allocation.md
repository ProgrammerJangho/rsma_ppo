# 졸업논문 진행 기준: PPO의 전력 및 common rate 공동 배분

> **역사 기록:** 이 문서는 2026-09-06의 simplex + `rho` + gate 실험을
> 설명한다. 현재 action 정의로 사용하지 않는다. 새 정의는
> [`canonical_action_definition.md`](canonical_action_definition.md)이며,
> 새 학습에서는 `alpha`, private-power simplex `q_k`, 사용자별 payload
> fraction `phi_k`만 사용한다.

**현재 진행:** [exact common-off gate 및 learned-SDMA 후속 검증](common_gate_followup.md).
[최초 3-seed ablation](allocation_ablation_results.md)은 초기 실패 원인 진단으로 보존한다.
`configs/thesis_ablation.yaml`과 `scripts/run_allocation_ablation.py`에서
전력-only / split-only / joint를 실행할 수 있다. 신규 pilot은 물리적 AoI cap을 끈다.
아래 1단계 설정과 결과는 최초 4096-step 점검 기록으로 보존한다.

## 연구 범위 (2026-09-06)

교수님 제안에 따라 PPO가 전력과 common 데이터 배분을 직접 학습한다.
첫 실험은 K=Nt=2, 모든 사용자 활성화, 고정 RZF/private 및 principal/common beam,
고정 총전력, 고정 blocklength, generate-at-will이다. Scheduling, queue, HARQ,
DT 상태, full precoder 학습은 이번 단계에 포함하지 않는다.

- PPO 행동: `power_fractions`, `rate_fractions`, `common_payload_scale`.
- `scheduling_enabled: false`, `allocator_mode: none`, `power_scale_action: false`.
- CEM은 향후 별도 baseline이다. PPO 출력을 환경 안에서 덮어쓰지 않는다.
- 관측: AoI, 추정 채널, 이전 ACK 및 이전 전력/선택 정보.
- 보상: feasible action에서 전송 후 사용자 평균 AoI의 음수. PPO의 gamma=0.99
  discounted objective는 정확한 average-cost 최적화와 같다고 주장하지 않는다.

## Rate 정의

`c_k = rate_fractions[k]`는 전체 common payload에서 사용자 k의 구성 비율이다.
`rho = common_payload_scale`는 그 구성에서 가능한 최대 common payload의 사용 비율이다.

```
D_common_total = rho * min_k(D_k / c_k)   (c_k > 0인 사용자)
D_common_k = c_k * D_common_total
C_k = D_common_k / n
r_common = sum_k(C_k)
r_private_k = (D_k - D_common_k) / n
```

단위는 bits/channel use. 이는 전송 시도 coding rate이며 성공이 보장된 goodput이 아니다.
목표 BLER hard constraint 없이 FBL BLER → ACK → AoI를 통해 실패 비용을 반영한다.
연속 bits 완화, perfect CSIR, successful common decoding 이후 perfect SIC를 가정한다.

## 1단계 실행 (WSL, 프로젝트 루트)

```bash
.venv/bin/python -m pytest -q tests/test_thesis_allocation.py tests/test_action_feasibility.py tests/test_smoke_training.py tests/test_rsma_phy.py tests/test_fbl.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/train_ppo.py --env-config configs/thesis_allocation.yaml --total-timesteps 4096 --seed 0 --run-dir runs/thesis_allocation_smoke_new --eval-episodes 0
.venv/bin/python scripts/audit_allocation.py --checkpoint runs/thesis_allocation_smoke_new/checkpoint.pt --out-dir results/thesis_allocation_smoke_new
```

재실행 시 새 run/output 디렉터리를 사용한다. 기존 학습 결과를 덮어쓰지 않는다.
Audit은 설정과 checkpoint의 환경 일치를 확인하고, 모든 사용자가 활성화되며
요청한 전력/분할이 PHY에 전달되는지 slot마다 검증한다. 실제 채널로 행동을 선택하지 않는다.
Fixed RSMA (common power 0.25, rho 0.3)와 equal-power common-off/SDMA는 단순 참조이며
최적 baseline이 아니다. 동일 evaluation seeds를 사용한다.

`allocation_trace.csv`에는 사용자별 common bits, C_k, 전체 common coding rate,
private coding rate, 전력 비율, BLER, ACK, AoI가 기록된다.
`manifest.json`에는 소스와 checkpoint hash가 저장된다. 현재 폴더에는 Git metadata가 없다.

이 smoke run은 학습/전송 경로 점검이다. 4096 steps, 단일 training seed, burn-in 없는
512-slot 평가로 알고리즘 우열이나 steady-state AoI를 주장하지 않는다.
현재 설정의 cap=1024는 초기 age=1인 512-slot episode에서 도달 불가능하다.
이는 uncapped long-run 구현을 대신하지 않는다. 평가 길이 증가 전 cap 처리를 수정해야 한다.

## 이후 순서

1. 전력만 / 분할만 / 공동 학습 ablation을 정의한다. Freeze한 무작위 head를
   고정 배분 baseline으로 취급하지 말고, 실행 action을 명시적으로 고정하고
   PPO log-prob에서도 실제 학습 차원을 일치시킨다.
2. Fixed allocation, 동일 정보의 CEM allocation, optimized SDMA를 비교한다.
   CEM에는 정확한 common-off 후보와 계산 예산/시간을 명시한다.
3. Perfect CSIT에서 pilot 3 seeds → imperfect CSIT → near/far 순으로 확장한다.
   SNR=10 dB의 첫 조건은 진단용이며 RSMA가 유리하다고 선별한 조건이 아니다.
4. Uncapped physical age, 긴 평가/burn-in, 사용자별 tail과 inter-delivery 통계를 준비한다.
   학습/validation/test seeds 및 조건을 분리하고 본 실험은 가능하면 5 training seeds 이상.
5. 성능 역전을 숨기지 않는다. 전력·split·BLER·서비스 간격으로 원인을 분석한다.

기존 OMA 정수 자원 배분, NOMA 정보 조건, screening/long-run 정책 불일치는
별도 수정 항목이다. 본 단계는 해당 비교 경로를 사용하지 않는다.

## 1단계 실행 결과 (2026-09-06)

- 지정한 테스트 78개 통과. Power/rate/rho head의 실제 parameter update 확인.
  Scheduling head는 업데이트되지 않으며 모든 사용자가 활성화됨.
- Seed 0, 4096 environment steps 학습 완료 (8 PPO updates).
- Evaluation seeds 900000, 900001, 900002; 각 512 slots; deterministic PPO.
- 모든 평가 slot에서 action→실제 전력/payload 일치 및 allocator 미사용 확인.
- 소스/체크포인트 hash와 원시 결과: `archive/results/thesis_allocation_smoke_seed0/`.

| 방법 | 사용자 평균 AoI (slots) | Pooled p95 | 최대 관측 AoI |
|---|---:|---:|---:|
| 고정 RSMA (common power=0.25, rho=0.3) | 22.640625 | 64 | 93 |
| PPO 공동 배분, 4096-step smoke | 27.458008 | 76 | 136 |
| 균등 전력 SDMA/common-off | 1.836263 | 5 | 14 |

두 RSMA 정책이 이 조건에서 좋다는 증거는 없다. PPO의 짧은 점검 학습 결과를
최종 알고리즘 성능으로 해석하지 않는다. 현재 기준의 고정 RSMA도 최적화된
baseline이 아니다. 다음은 전력-only / split-only / joint ablation과
common-off 경계의 접근성을 확인하는 단계다. 초기 episode가 완료되기 전
학습 로그의 episode return은 NaN(집계 표본 없음)이며, 최종 loss는 유한하다.
