# RSMA–AoI PPO 졸업논문 프로젝트

유한 블록길이 환경에서 1계층 RSMA를 활용해 하향링크 정보의 나이(Age of Information, AoI)를 최소화하는 연구 프로젝트입니다. 안테나 2개를 갖춘 기지국이 방향성을 가진 Rician 채널을 통해 고정 크기의 상태 갱신 패킷을 전송하며, PPO가 공통 전력 비율 `alpha`, 개별 전력 분배 벡터 `q_k`, 사용자별 공통 페이로드 비율 `phi_k`를 결정합니다. RZF 개별 빔과 주성분 기반 공통 빔을 고정하여 자원 할당의 효과를 분석합니다.

핵심 연구 결과는 목적함수 기반 사전학습, 인과적인 물리계층 관측, 단계별 PPO 학습을 통해 과부하 시스템인 `K=3, N_t=2`에서 상태에 따른 RSMA 자원 할당을 학습할 수 있다는 것입니다. 목표 조건과 실행 가능한 조건의 결과는 독립적인 학습 시드 5개로 평가했습니다. RSMA가 항상 SDMA보다 우수하다고 가정하지 않으며, 전송이 거의 불가능한 짧은 블록길이와 낮은 SNR 조건은 아직 해결되지 않은 극한 조건으로 남겨 두었습니다.

## 현재 실험 구성

- 시스템: `K=3`, `N_t=2`인 과부하 MU-MISO 하향링크, 1계층 RSMA.
- 채널: 평균 전력을 보존하는 Rician ULA 채널과 Gauss–Markov 시간 변화 모델.
- 물리계층: 정규 근사 기반 유한 블록길이(FBL) 블록 오류율(BLER), 완전한 CSIR, 설정 가능한 불완전 CSIT.
- 트래픽: 필요할 때 상태 갱신 패킷을 생성하며, 전송 성공 시 AoI를 1슬롯으로 초기화.
- PPO 행동: `alpha`, 개별 전력 분배 벡터 `q_k`, 독립적인 페이로드 비율 `phi_k`. 학습되는 게이트와 `rho` 행동은 사용하지 않음.
- PPO 최적화: 1단계 액터/크리틱 학습률 `3e-4/1e-3`, 목표 조건 미세조정 학습률 `1e-4/5e-4`, 보상 정규화, 클리핑된 가치함수 손실.
- 주요 평가: 학습에 사용하지 않은 공통 난수 궤적에서 결정론적 정책을 비교. 샘플링 롤아웃은 학습 진단에 사용.

현재 기준 설정 파일은 다음과 같습니다.

- `configs/thesis_rician_k3_nt2_mixed_phy_sinr.yaml`
- `configs/thesis_rician_k3_nt2_mixed_phy_sinr_stage1.yaml`
- `configs/thesis_rician_k3_nt2_20deg_sinr_finetune.yaml`

확정된 표, 그림, 체크포인트 해시는 로컬의 `results/thesis_mixed_phy_final_v1/`에 있습니다. 이전의 `simplex_scaled`, 게이트, `rho` 설정은 행동 구조를 재설계하기 전에 생성한 결과를 재현하기 위해 보존했습니다.

## 프로젝트 구조

| 경로 | 용도 |
|---|---|
| `src/rsma_aoi/` | 환경, 채널, 물리계층, PPO/TD3 및 평가 라이브러리 |
| `configs/` | 실행 가능한 실험 설정. 현재 설정은 `thesis_*` 계열 |
| `scripts/` | 학습, 평가 및 재현 가능한 분석 실행 스크립트 |
| `tests/` | 단위 테스트와 통합 테스트 |
| `docs/` | 현재 모델의 설계 결정과 실험 결론 |
| `datasets/` | 목적함수 기반 사전학습에 사용하는 오라클 데이터셋 |
| `runs/` | 재사용 가능한 사전학습 체크포인트 — 로컬 보관 |
| `results/` | 논문 실험 결과와 연구별 요약 — 로컬 보관 |
| `references/` | 연구자가 제공한 원본 프로젝트 문서 |
| `archive/` | 이전 프로젝트, 대체된 문서와 과거 실험 — 로컬 보관 |

문서를 읽을 때는 [연구 문서 안내](docs/README.md)부터 시작하세요. `archive/`는 과거 의사결정을 추적하기 위한 자료이며, 현재 모델의 구현 기준은 최신 문서에 있습니다.

가상환경(`.venv/`), 학습 체크포인트(`runs/`), 실험 결과(`results/`), 과거 자료(`archive/`), 임시 파일(`tmp/`), 생성된 문서와 그림(`output/`)은 `.gitignore`에 따라 GitHub 업로드에서 제외됩니다. 저장소를 복제하면 소스 코드, 설정, 연구 문서와 포함된 데이터셋을 받을 수 있으며, 위의 로컬 산출물은 별도로 준비하거나 실험을 실행해 생성해야 합니다.

## 프로젝트 검증

WSL 환경에서 다음 명령어를 실행합니다.

```bash
cd /home/alexalex2/rsma_ppo
.venv/bin/python -m pytest -q
```

기존 검증 기록에 따르면 행동 구조 재설계 후 전체 구현 테스트 616개가 통과했습니다. 학습 시드 5개에 대한 산출물 생성 후에는 환경·설정 중심 테스트 159개가 통과했습니다.

## 확정된 학습 파이프라인 재현

```bash
SEEDS="0 1 2 3 4" scripts/run_mixed_phy_seed_expansion.sh pretrain
SEEDS="0 1 2 3 4" scripts/run_mixed_phy_seed_expansion.sh stage1
SEEDS="0 1 2 3 4" scripts/run_mixed_phy_seed_expansion.sh target

.venv/bin/python scripts/evaluate_mixed_phy_seed_expansion.py \
  --out-dir results/<new_target_result_directory>
.venv/bin/python scripts/evaluate_mixed_phy_multiseed_grid.py \
  --out-dir results/<new_grid_result_directory>
```

실행 스크립트는 이미 완료된 체크포인트를 건너뜁니다. 평가 결과는 새 출력 디렉터리에 저장해야 합니다. 모든 공식 비교에는 독립적인 학습 시드와 동일한 미사용 채널·CSIT·ACK 시드를 적용합니다.

## 연구 범위

현재 핵심 실험의 범위에는 스케줄링, 확률적 패킷 도착, 대기열, HARQ, 블록길이 제어, 학습 기반 프리코더, 명시적인 디지털 트윈 상태 불일치가 포함되지 않습니다. OMA/NOMA, TD3 및 이전의 이질적 사용자 조건 연구는 보조 비교 또는 과거 실험으로 보관되어 있습니다.
