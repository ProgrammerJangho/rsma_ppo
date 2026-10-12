#!/usr/bin/env python3
"""Write the Markdown report for the 5-training x 5-evaluation-seed study.

Reads results/kics_2m_5train_5eval_v1/ plus the 2.10M grid files and writes
docs/kics_2m_5train_5eval_results_2026-09-25.md. The evaluation-seed table is
the appendix referred to by the KICS paper: it is kept out of the two pages
because the paper's CIs are over training seeds only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "rsma_ppo_2m": "RSMA-PPO 2.10M", "sdma_ppo_2m": "SDMA-PPO 2.10M",
    "rsma_ppo_524k": "RSMA-PPO 0.52M", "sdma_ppo_524k": "SDMA-PPO 0.52M",
    "fixed_rule_rsma": "고정 규칙 RSMA (α=0.70)", "max_aoi_single_user": "Max-AoI 단일 사용자 MRT",
}


def table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", default=str(ROOT / "results/kics_2m_5train_5eval_v1"))
    ap.add_argument("--single-grid", default=str(ROOT / "results/kics_2m_single_user_grid_v1"))
    ap.add_argument("--out", default=str(ROOT / "docs/kics_2m_5train_5eval_results_2026-09-25.md"))
    args = ap.parse_args()
    ev = Path(args.eval_dir)
    agg = pd.read_csv(ev / "target_aggregate.csv")
    paired = pd.read_csv(ev / "paired_summary.csv")
    seed_means = pd.read_csv(ev / "training_seed_means.csv")
    eval_means = pd.read_csv(ev / "evaluation_seed_means.csv")
    roles = pd.read_csv(ev / "action_roles_by_training_seed.csv")
    role_summary = pd.read_csv(ev / "action_roles_summary.csv")
    cf = pd.read_csv(ev / "counterfactual_summary.csv")
    stratum = pd.read_csv(Path(args.single_grid) / "stratum_aggregate.csv")

    main_rows = []
    for label in LABELS:
        row = agg[agg.label == label].iloc[0]
        unit = "학습 seed 5개" if row.ci_unit == "training_seed" else "평가 seed 5개"
        main_rows.append({
            "방법": LABELS[label], "평균 AoI": f"{row.mean_aoi:.4f}",
            "95% CI": f"±{row.mean_aoi_ci95:.4f}", "CI 단위": unit,
            "p95": f"{row.p95_aoi:.2f}", "성공확률": f"{row.success_probability:.3f}",
            "평균 α": f"{row.alpha_mean:.3f}",
        })

    by_seed = seed_means[seed_means.label.isin(["rsma_ppo_2m", "sdma_ppo_2m", "rsma_ppo_524k", "sdma_ppo_524k"])]
    seed_wide = by_seed.pivot(index="training_seed", columns="label", values="mean_aoi").reset_index()
    seed_wide = seed_wide.rename(columns={k: LABELS[k] for k in LABELS})
    seed_wide = seed_wide.round(4)

    eval_wide = eval_means.pivot(index="eval_repeat", columns="label", values="mean_aoi").reset_index()
    eval_wide = eval_wide.rename(columns={k: LABELS[k] for k in LABELS}).round(4)

    paired_view = paired[["comparison", "gain_pct", "gain_ci95_training_seed", "abs_diff",
                          "abs_diff_ci95_training_seed", "training_seed_wins", "seed_eval_pair_wins",
                          "trajectory_wins"]].copy()
    paired_view = paired_view.round(3)

    role_view = role_summary[["statistic", "mean", "ci95_halfwidth", "min", "max"]].round(4)
    roles_view = roles.drop(columns=["slots"]).round(4)
    cf_view = cf.round(3)
    stratum_view = stratum.round(3)

    text = f"""# KICS 2.10M 결과: 학습 seed 5개 × 평가 seed 5개

작성일: 2026-09-25

## 무엇을 했나

- RSMA-PPO와 SDMA-PPO(q-only) 모두 seed 0~4를 목표 조건에서 누적 2,097,152 step까지 학습했다.
  동일한 objective-only 사전학습과 혼합 조건 Stage 1(524,288 step) 이후, 목표 조건 524,288 step
  체크포인트에서 (A) 524,288 step, (B) 1,048,576 step을 이어 학습했다. 각 단계는 actor, critic,
  optimizer, 관측·보상 정규화 상태를 복원하고 learning-rate annealing은 단계마다 새로 시작했다.
  실행: `scripts/run_2m_staged_continuation.sh`, 검증: `scripts/verify_2m_checkpoints.py`.
- SDMA 연장은 `allocation_control=sdma_power`를 명시했다. `load_checkpoint`는 가중치만 복원하므로
  이 설정 없이 이어 학습하면 q-only 체크포인트가 오류 없이 joint(α>0) 정책으로 바뀐다.
- 평가: 20°, 15 dB, n=128, CSIT 오차 0.2. 학습 seed마다 평가 seed 5개(eval_seed = 990278 +
  10000·repeat + 100·permutation) × 사용자 순열 6개 = 30개 궤적, 768 slot, 처음 256 slot 제외.
  모든 방법이 같은 채널·CSIT·ACK 난수를 쓴다. 스크립트: `scripts/evaluate_kics_2m_5train_5eval.py`.
- 집계: 순열 평균 → 평가 seed별 값 → 평가 seed 5개 평균 = 학습 seed별 성능. 학습된 정책의 95% CI는
  학습 seed 5개 평균에 대한 Student-t 구간(df=4)이다. 150개 궤적을 독립 표본으로 쓰지 않았다.
  규칙 기준선은 학습 seed가 없으므로 평가 seed 5개에 대한 구간이다.

## 목표 조건 성능

{table(pd.DataFrame(main_rows))}

## 학습 seed별 평균 AoI (평가 seed 5개 × 순열 6개 평균)

{table(seed_wide)}

## 부록: 평가 seed별 평균 AoI (학습 seed 5개 × 순열 6개 평균)

학습 seed 변동과 별개로, 같은 정책 집합이 평가 난수에 따라 얼마나 달라지는지 보여 준다.

{table(eval_wide)}

## 쌍별 비교 (같은 학습 seed·같은 평가 seed끼리 대응)

gain = 100 × (기준 − 방법) / 기준. 학습 seed별로 평가 seed 5개 평균을 낸 뒤 학습 seed 5개로 CI를 구했다.
wins는 학습 seed 5개, 학습·평가 seed 쌍 25개, 궤적 150개 중 방법이 더 낮은 수다.

{table(paired_view)}

## 행동 역할 통계 (RSMA-PPO 2.10M, 학습 seed별)

{table(roles_view)}

요약 (학습 seed 5개 평균, 95% CI):

{table(role_view)}

## 행동 대체 실험 (closed-loop, RSMA-PPO 2.10M)

대체 상수와 AoI 순위 규칙은 별도 보정 궤적(eval_seed = 980278 + 100·permutation)에서 정책별로 구했다.
평가 궤적은 위 5×6과 같다.

{table(cf_view)}

## 36개 조건 격자 (2.10M RSMA-PPO, 조건당 평가 seed 1개 × 순열 6개)

{table(stratum_view)}

## 해석상 주의

- 학습량은 두 방법이 같다(누적 2.10M). 다만 행동 차원과 최적화 난도가 달라 RSMA-PPO와 SDMA-PPO의
  차이를 common stream만의 인과효과로 단정하지 않는다.
- 목표 조건 학습은 세 단계로 나뉘어 learning-rate annealing이 두 번 재시작됐다. 하나의 연속 스케줄로
  학습한 것과 같지 않다.
- 격자 평가는 목표 조건으로 미세조정한 최종 정책을 적용한 것이며, stress 조건은 PPO 학습에 포함되지
  않았다(사전학습에만 포함). feasible 평균 AoI는 0.52M 정책보다 약간 높아졌다(목표 조건 과적합 가능성).
- 행동 역할 분리는 seed에 따라 다르다. seed 0, 2, 3, 4는 q_k–상대 AoI 상관 0.94~0.96, φ_k–정렬도
  Spearman 0.68~0.82, α의 slot 간 SD 0.05~0.09로 역할이 분리됐지만, seed 1은 α를 넓게(SD 0.24,
  5~95%: 0.29~1.00) 조절하는 다른 해를 찾았고 AoI는 비슷했다(1.404). 그래서 논문 표 2는 평균±CI 대신
  학습 seed 중앙값 [최소–최대]를 쓴다. 대체 실험의 AoI 증가도 seed 1에서 크게 튄다(+51~191%).
- seed-0 전용 감사(`docs/feedforward_policy_state_adaptation_audit_2026-09-25.md`)의 수치(α 고정 +5.44%
  등)는 seed 0 진단값이다. 5개 seed 결과는 위 표를 따른다.
- 논문 v2(2026-09-28)부터 stress 조건(SNR 5 dB 또는 n=64)의 수치와 "단일 사용자 전송이 우세했다"는
  결론을 뺐다. PPO가 학습하지 않은 조건이라 분포 밖 일반화 실패, 전체 사용자 활성화, RSMA 구조, 물리적
  난도 중 원인을 구분할 수 없기 때문이다. 논문에는 "학습에 포함된 SNR·blocklength 범위에 한정"이라는
  범위 한계로만 남겼다. 위 격자 표의 stress 행은 내부 기록이다.
"""
    Path(args.out).write_text(text, encoding="utf-8")
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
