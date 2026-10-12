# 졸업논문 제출 전 체크리스트

현재 연구 결과와 five-seed 실험은 논문 초안을 작성할 수 있는 상태다. 아래는
추가 학습 목록이 아니라, 현재 증거를 과장 없이 제출 원고로 바꾸기 위한 작업
목록이다.

## 제출 전 필수

- [ ] 학교의 한글/영문 제목, 초록, 장 번호, 여백 및 참고문헌 양식을 적용한다.
- [ ] 지도교수와 최종 제목 및 contribution 세 문장을 확정한다.
- [ ] `thesis_manuscript_draft_2026-09-22.md`의 모든 코드 표기를 논문식 기호로
  통일한다. 특히 `alpha`, `q_k`, `phi_k`, `H_hat`을 각각
  $\alpha$, $q_k$, $\phi_k$, $\hat{\mathbf H}$로 표기한다.
- [ ] 표 1--3과 그림 1--3을 학교 템플릿에 삽입하고 본문에서 번호로 참조한다.
- [ ] 참고문헌 12개의 저자, 권·호, 페이지, DOI를 최종 출력본에서 대조한다.
- [ ] SDMA 대비 48.17%는 target 조건의 결과, 26.29%는 feasible 18-regime
  평균임을 모든 초록·결론·발표자료에서 구분한다.
- [ ] Stress 결과의 부호가 확정되지 않았다는 문장을 삭제하지 않는다.
- [ ] 연구 범위를 `K=3, N_t=2`, 고정 RZF/principal beam, generate-at-will,
  fixed payload, imperfect CSIT의 allocation policy로 명시한다.
- [ ] 제출본 생성 직전에 `scripts/build_thesis_mixed_phy_artifacts.py`를 실행하고
  manifest의 모든 SHA-256을 다시 검사한다.

## 지도교수 선택 사항

- [ ] 본문의 알고리즘 절에 PPO 학습 pipeline pseudocode를 학교 양식으로 넣는다.
- [ ] 부록에 seed별 target 표와 18개 feasible regime의 상세 표를 넣는다.
- [ ] 발표자료에는 target 성능, `alpha` response, stress limitation의 세 그림만
  우선 사용한다.
- [ ] 제목에서 “상태 적응형” 대신 “PHY-aware” 또는 “혼합 채널 조건”을 쓸지
  결정한다.

## 현재 논문을 위해 추가할 필요가 없는 대규모 작업

- Full-precoder WMMSE 구현
- SAC/TD3/DDPG 전체 재학습 비교
- Stochastic arrival, queue 및 HARQ 동시 도입
- `K>=4` 대규모 확장
- Blocklength와 scheduling을 PPO action에 추가
- 10 training seeds 확대

이 항목들은 후속 연구로 의미가 있지만 지금 함께 넣으면 각 요소의 효과를
분리하기 어렵고, 현재 five-seed 핵심 결과의 완성도를 높이지 않는다. 지도교수가
명시적으로 요구하면 하나씩 독립 실험으로 추가한다.

## 현재 제출 가능한 핵심 주장

1. Causal PHY feature를 관측하는 gate-free PPO가 common/private 전력과
   사용자별 common payload를 직접 배분한다.
2. Objective-aware pretraining과 staged curriculum을 거친 정책은 target에서
   SDMA 대비 48.17%, feasible 18-regime에서 평균 26.29% 낮은 AoI를 보였다.
3. 학습된 common-power fraction은 geometry와 SNR에 따라 변하며, 저 SNR 또는
   매우 짧은 blocklength에서는 성능 우열이 확정되지 않는다.

“RSMA가 모든 조건에서 SDMA보다 우수하다”, “PPO가 전역 최적이다”, “본 연구가
RSMA+AoI+DRL의 최초 연구다”는 현재 증거로 주장하지 않는다.
