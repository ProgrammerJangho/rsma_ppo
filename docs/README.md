# 연구 문서 안내

아래 순서로 문서를 읽으세요. 현재 행동 정의 문서는 이전의 게이트, 전력 분배 심플렉스, `rho` 설명보다 우선합니다. 다른 주제에서도 최신 결과 문서와 이전 진단의 가정이 충돌하면 최신 결과를 기준으로 해석합니다.

0. [현재 행동 정의](canonical_action_definition.md) — 게이트를 사용하지 않는 현재 자원 할당 행동을 정의합니다. 아래 문서에 남아 있는 이전 게이트, 심플렉스, `rho` 행동 설명을 대체합니다.

[행동 구조 재설계 구현 기록](action_redesign_implementation_2026-09-17.md)에는 변경된 코드 경로, 호환성 범위와 검증 결과가 정리되어 있습니다.

**최신 PPO 결과:** [물리계층 관측과 SINR 기반 단계별 학습](mixed_phy_sinr_curriculum_2026-09-20.md)은 인과적인 물리계층 관측, 사용자 배치·SNR·블록길이를 바꾸는 단계별 학습, 전체 조건 격자로의 확장 실패, 선택된 1단계→목표 조건 체크포인트를 기록합니다. 주요 `K=3, N_t=2` PPO 결과의 현재 기준 문서입니다.

**논문 결과 장:** [논문 실험 결과](thesis_results_chapter_2026-09-22.md)에는 학습 시드 5개의 확정된 표, 통계적 해석, 주장 가능한 범위, 한계와 그림 설명이 있습니다. 관련 산출물 목록은 로컬의 `results/thesis_mixed_phy_final_v1/manifest.json`에 있습니다.

**통합 논문 초안:** [논문 원고 초안](thesis_manuscript_draft_2026-09-22.md)은 현재 시스템 모델, 관련 연구, PPO 방법, 실험 절차, 확정된 결과와 한계를 연결합니다. 참고문헌 데이터는 [BibTeX 파일](thesis_references.bib)에 있습니다. 남은 학교 양식 적용과 주장 검토에는 [논문 제출 점검표](thesis_submission_checklist_2026-09-22.md)를 사용하세요.

**동일 조건 사전학습 분석:** [사전학습 조건을 맞춘 실험 결과](matched_pretraining_results_2026-09-24.md)는 목적함수 기반 사전학습 손실, 데이터, 에포크, PPO 학습 단계와 학습 예산을 동일하게 맞춰 목표 조건의 RSMA/SDMA 비교를 반복합니다. 이전 q-only 결과 문서에서 사전학습 조건의 차이가 섞여 있던 해석을 대체합니다.

1. [프로젝트 중간 진행 기록](project_midterm_status.md) — 이전 게이트·`rho` 행동을 사용하던 시기의 기록입니다. 2026-09-22 안내에서 현재 행동과 결과 문서를 가리킵니다.
2. [최종 평가 절차](final_evaluation_protocol.md) — 확정된 장시간 주요 비교, 불확실성 평가 단위와 해석 가능한 범위.
3. [최종 실험 구성표](final_experiment_matrix.md) — 미완료 학습 실행과 주요 조건·공간 배치·근거리/원거리 연구의 통계적 범위.
4. [최종 조건별 결과](final_regime_results.md) — 학습 시드 5개의 주요 결과, 동일 조건 공간 배치 비교와 수정된 근거리/원거리 극한 조건 평가.
5. [SNR 민감도 결과](snr_sensitivity_results.md) — 동일 조건의 5/10/15 dB 재학습과 장시간 평가.
6. [262k PPO와 CEM의 장시간 비교](cem_262k_long_results.md) — 확정된 평가 절차에서 262k PPO와 계산량을 제한한 CEM을 비교한 결과.
7. [이전 CEM 기준 결과](cem_reference_results.md) — 추적 가능성과 구현 비용 분석을 위해 보관한 단시간 CEM 평가.
8. [초기화 조건을 맞춘 결과](matched_initialization_results.md) — RSMA/SDMA 목적함수 기반 사전학습 조건을 맞추고, PPO 학습 전과 학습 후의 효과를 비교.
9. [학습 예산별 결과](training_budget_results.md) — 동일 조건의 0/65k/262k PPO 비교와 학습 시드 5개의 최종 정책 결과.
10. [두 가지 정책 모드 분석](two_mode_policy_audit.md) — 수렴한 연속 행동에서 어떤 요소가 상태에 따라 달라지는지 확인하는 단계적 비교.
11. [혼합 사용자 배치 정책 분석](mixed_geometry_policy_audit.md) — 하나의 정책으로 5/20/60도 조건을 평가하여 경계 집중이 고정 배치에서만 나타나는 현상인지 확인.
12. [혼합 사용자 배치·SNR 결과](mixed_geometry_snr_results.md) — 하나의 정책의 배치·SNR 일반화, CEM 비교와 공통 전송을 끄는 투영 분석.
13. [과부하 시스템 결과](k3_nt2_overloaded_results.md) — `K=3, N_t=2`에서 전체 사용자/선택된 사용자 대상 CEM, 사용자 3명의 자원 할당만 학습하는 PPO, 수치적 기준 해법과의 남은 차이.
14. [대칭성을 활용한 RSMA 후속 결과](symmetric_rsma_followup_results.md) — 대칭성을 반영한 관측·행동 사전학습, 동일 조건 쌍에서의 공통 전송 기여도와 CEM 비교.
15. [과부하 조건의 PPO 성능 차이 진단](k3_ppo_gap_diagnosis.md) — 동일 상태 CEM과의 성능 차이, 행동 헤드별 기여도, 목적함수 기반 사전학습과 정책 성능을 떨어뜨리는 PPO 탐색 분석.
16. [RSMA 특성 진단](rsma_characteristic_diagnosis.md) — 현재 학습 정책이 RSMA를 충분히 활용하지 못하는 이유와 실험 근거에 따른 개선 방향.
17. [논문 자원 할당 정의](thesis_allocation.md) — 논문 범위, PPO 행동과 공통 전송률 할당의 정확한 의미.
18. [채널 모델 선정 근거](channel_model_recommendation.md) — 선택한 방향성 Rician/Gauss–Markov 채널과 비교 조건.
19. [Rician 물리계층 조건 분석](rician_phy_regime_audit.md) — 공간 배치 조건에 따른 고정 빔 FBL RSMA/SDMA 동작.
20. [공통 빔 할당 분석](common_beam_allocation_audit.md) — 주성분 빔을 기본값으로 유지하고 max-min 빔을 비교 실험으로 사용하는 이유.
21. [공통 전송 게이트 후속 분석](common_gate_followup.md) — 공통 전송을 완전히 끄는 경계와 이전 게이트 메커니즘.
22. [목적함수·행동 요소별 비교](objective_action_ablation.md) — 전력, 전송률 분할과 공통 전송 게이트의 기여도.
23. [불완전 CSIT와 SAA 사전학습](imperfect_csit_saa_pretraining.md) — 사후분포 SAA의 개선 실패 결과와 동일 조건 CSIT 비교.
24. [Rician PPO 학습 진단](rician_ppo_training_diagnosis.md) — PPO 결과의 종합 결론과 수렴 분석.

대체된 제안, 초기 자원 할당 예비 실험, 외부에서 가져온 이전 코드와의 비교는 로컬의 `archive/docs/`에 보관되어 있습니다. 의사결정 과정을 추적할 때 참고할 수 있으며, 현재 구현의 명세는 최신 문서를 기준으로 삼습니다. `results/`와 `archive/`는 GitHub 업로드에서 제외된 로컬 산출물입니다.
