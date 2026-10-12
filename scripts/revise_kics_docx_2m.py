#!/usr/bin/env python3
"""Revise the user-edited KICS two-page DOCX with the 2.10M-step results.

The source DOCX is edited in place of a copy: paragraphs are located by their
opening words, existing inline equations are reused, new text runs copy the
paragraph's own run formatting, table cells keep their cell formatting and the
two figures are replaced at their current width. Every number is read from the
result files below, never typed by hand.

Inputs
    results/kics_2m_5train_5eval_v1/            5 training x 5 evaluation seeds
    results/kics_2m_rsma_grid_v1/               36-regime grid, 2.10M RSMA-PPO
    results/kics_2m_single_user_grid_v1/        paired single-user grid baseline
    output/kics/figure_return_curve_2m*.{png,csv}
    output/kics/figure_alpha_feasible_2m.png

Extra dependencies (not in requirements.txt): python-docx, latex2mathml.
"""
from __future__ import annotations

import argparse
import copy
import io
import re
import sys
from pathlib import Path

import docx
import numpy as np
import pandas as pd
from docx.oxml.ns import qn
from lxml import etree
from PIL import Image
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_kics_short_paper import MathConverter  # noqa: E402

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PATHS = {
    "eval": ROOT / "results/kics_2m_5train_5eval_v1",
    "grid": ROOT / "results/kics_2m_rsma_grid_v1",
    "single_grid": ROOT / "results/kics_2m_single_user_grid_v1",
    "single_grid_524k": ROOT / "results/matched_objective_single_user_grid_v1",
    "curve_summary": ROOT / "output/kics/figure_return_curve_2m_summary.csv",
    "curve_figure": ROOT / "output/kics/figure_return_curve_2m.png",
    "alpha_figure": ROOT / "output/kics/figure_alpha_feasible_2m.png",
}


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------
def ci95(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def load_numbers() -> dict:
    ev_dir = PATHS["eval"]
    agg = pd.read_csv(ev_dir / "target_aggregate.csv").set_index("label")
    paired = pd.read_csv(ev_dir / "paired_summary.csv").set_index("comparison")
    roles = pd.read_csv(ev_dir / "action_roles_summary.csv").set_index("statistic")
    cf = pd.read_csv(ev_dir / "counterfactual_summary.csv").set_index("mode")
    evalvar = pd.read_csv(ev_dir / "evaluation_seed_variation.csv").set_index("label")
    curve = pd.read_csv(PATHS["curve_summary"])
    stratum = pd.read_csv(PATHS["single_grid"] / "stratum_aggregate.csv").set_index("stratum")
    stratum_524k = pd.read_csv(PATHS["single_grid_524k"] / "stratum_aggregate.csv").set_index("stratum")
    grid = pd.read_csv(PATHS["grid"] / "regime_by_training_seed.csv")
    feasible = grid[(grid.snr_db >= 10) & (grid.blocklength >= 128)]

    def marginal(axis):
        per_seed = feasible.groupby(["training_seed", axis]).alpha_mean.mean()
        return per_seed.groupby(axis).mean()

    per_seed_roles = pd.read_csv(ev_dir / "action_roles_by_training_seed.csv").set_index("training_seed")
    traj = pd.read_csv(ev_dir / "target_trajectories.csv")
    cfw = traj[traj.label == "rsma_ppo_2m"].pivot_table(
        index=["training_seed", "eval_repeat", "permutation"], columns="mode", values="mean_aoi")
    cf_seed = pd.DataFrame({
        mode: (100.0 * (cfw[mode] - cfw["original"]) / cfw["original"]).groupby(level="training_seed").mean()
        for mode in ("fixed_alpha", "rank_q", "fixed_phi_level")
    })
    seed_perf = pd.read_csv(ev_dir / "training_seed_means.csv")
    n = {"agg": agg, "paired": paired, "roles": roles, "cf": cf, "evalvar": evalvar,
         "per_seed_roles": per_seed_roles, "cf_seed": cf_seed, "seed_perf": seed_perf,
         "stratum_524k": stratum_524k,
         "stratum": stratum, "alpha_snr": marginal("snr_db"), "alpha_geo": marginal("geometry_deg")}
    for family in ("RSMA-PPO", "SDMA-PPO"):
        rows = curve[curve.family == family].set_index("stage_end")
        n[f"curve_{family}"] = (float(rows.loc[0, "seed_mean_last"]),
                                float(rows.loc[2_097_152, "seed_mean_last"]))
    return n


def f3(x: float) -> str:
    return f"{x:.3f}"


def f2(x: float) -> str:
    return f"{x:.2f}"


# ---------------------------------------------------------------------------
# DOCX helpers
# ---------------------------------------------------------------------------
def find(doc, prefix: str):
    matches = [p for p in doc.paragraphs if p.text.strip().startswith(prefix)]
    if len(matches) != 1:
        raise LookupError(f"{len(matches)} paragraphs start with {prefix!r}")
    return matches[0]


def set_size(rpr, half_points: int) -> None:
    for tag in ("w:sz", "w:szCs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rpr, qn(tag))
        el.set(qn("w:val"), str(half_points))


def rewrite(paragraph, parts, math: MathConverter, size: int | None = None) -> None:
    """Replace a paragraph's runs/equations with ``parts``.

    ``parts`` items: ``str`` (text in the paragraph's first-run format),
    ``("m", i)`` (the paragraph's i-th existing inline equation) or
    ``("tex", latex)`` (a new inline equation).
    """
    p = paragraph._p
    maths = [child for child in p if child.tag == f"{{{M_NS}}}oMath"]
    runs = [child for child in p if child.tag == qn("w:r") and child.find(qn("w:t")) is not None]
    proto = copy.deepcopy(runs[0].find(qn("w:rPr")))
    if size is not None:
        set_size(proto, size)
    for child in list(p):
        if child.tag in (qn("w:r"), f"{{{M_NS}}}oMath", qn("w:proofErr")):
            p.remove(child)
    for part in parts:
        if isinstance(part, str):
            run = etree.SubElement(p, qn("w:r"))
            if proto is not None:
                run.append(copy.deepcopy(proto))
            text = etree.SubElement(run, qn("w:t"))
            text.text = part
            text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        elif part[0] == "m":
            p.append(copy.deepcopy(maths[part[1]]))
        else:
            p.append(math(part[1]))


def rewrite_caption(paragraph, text: str) -> None:
    """Keep the bold label run (e.g. "그림 1. ") and replace the remaining text."""
    runs = [r for r in paragraph.runs if r.text]
    runs[1].text = text
    for run in runs[2:]:
        run._r.getparent().remove(run._r)


def set_cell(cell, parts, math: MathConverter, bold: bool | None = None,
             size: int | None = None) -> None:
    paragraph = cell.paragraphs[0]
    p = paragraph._p
    runs = [child for child in p if child.tag == qn("w:r")]
    proto = copy.deepcopy(runs[0].find(qn("w:rPr")))
    if bold is not None:
        b = proto.find(qn("w:b"))
        if bold and b is None:
            proto.insert(1, etree.Element(qn("w:b")))
        if not bold and b is not None:
            proto.remove(b)
    if size is not None:
        set_size(proto, size)
    for child in list(p):
        if child.tag in (qn("w:r"), f"{{{M_NS}}}oMath"):
            p.remove(child)
    for part in ([parts] if isinstance(parts, str) else parts):
        if isinstance(part, str):
            run = etree.SubElement(p, qn("w:r"))
            run.append(copy.deepcopy(proto))
            text = etree.SubElement(run, qn("w:t"))
            text.text = part
            text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        else:
            p.append(math(part[1]))


def set_column_widths(table, widths) -> None:
    grid = table._tbl.tblGrid
    for col, width in zip(grid.iter(qn("w:gridCol")), widths):
        col.set(qn("w:w"), str(width))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            tcw = cell._tc.get_or_add_tcPr().find(qn("w:tcW"))
            tcw.set(qn("w:w"), str(width))
            tcw.set(qn("w:type"), "dxa")


def replace_picture(doc, paragraph, image_path: Path) -> None:
    p = paragraph._p
    blip = p.find(f".//{{{A_NS}}}blip")
    part = doc.part.related_parts[blip.get(qn("r:embed"))]
    data = image_path.read_bytes()
    part._blob = data
    width, height = Image.open(io.BytesIO(data)).size
    extent = p.find(f".//{{{WP_NS}}}extent")
    cx = int(extent.get("cx"))
    cy = int(round(cx * height / width))
    extent.set("cy", str(cy))
    p.find(f".//{{{A_NS}}}ext").set("cy", str(cy))


# ---------------------------------------------------------------------------
# revision
# ---------------------------------------------------------------------------
def keep_table_together(table) -> None:
    """Keep every row with the next one so a table never splits across columns."""
    for row in table.rows[:-1]:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                ppr = paragraph._p.get_or_add_pPr()
                if ppr.find(qn("w:keepNext")) is None:
                    ppr.insert(0, etree.Element(qn("w:keepNext")))


def minus(value: float) -> str:
    return f"{value:.0f}".replace("-", "−")


def revise(source: Path, out: Path, xsl: Path) -> dict:
    n = load_numbers()
    agg, paired, roles, cf, stratum = n["agg"], n["paired"], n["roles"], n["cf"], n["stratum"]
    rsma, sdma = agg.loc["rsma_ppo_2m"], agg.loc["sdma_ppo_2m"]
    rule, single = agg.loc["fixed_rule_rsma"], agg.loc["max_aoi_single_user"]
    g_sdma = paired.loc["rsma_ppo_2m_vs_sdma_ppo_2m"]
    g_rule = paired.loc["rsma_ppo_2m_vs_fixed_rule_rsma"]
    g_single = paired.loc["rsma_ppo_2m_vs_max_aoi_single_user"]
    g_rsma_len = paired.loc["rsma_ppo_2m_vs_rsma_ppo_524k"]
    g_sdma_len = paired.loc["sdma_ppo_2m_vs_sdma_ppo_524k"]
    g_524 = paired.loc["rsma_ppo_524k_vs_sdma_ppo_524k"]
    feas = stratum.loc["feasible"]
    ev_rsma = n["evalvar"].loc["rsma_ppo_2m"]
    a_snr, a_geo = n["alpha_snr"], n["alpha_geo"]
    r0, r1 = n["curve_RSMA-PPO"]
    s0, s1 = n["curve_SDMA-PPO"]
    psr, cfs = n["per_seed_roles"], n["cf_seed"]
    # One policy found a different solution; the text names it, so check the data agree.
    exception = int(psr.alpha_sd.idxmax())
    if exception != int(psr.phi_alignment_spearman.idxmin()):
        raise ValueError("the alpha-variable seed is not the weak phi-alignment seed; revise the text")
    typical = psr.drop(index=exception)
    seed_perf = n["seed_perf"]
    exception_aoi = float(seed_perf[(seed_perf.label == "rsma_ppo_2m")
                                    & (seed_perf.training_seed == exception)].mean_aoi.iloc[0])
    feas_524k = n["stratum_524k"].loc["feasible"]

    def med_range(series, digits=2) -> str:
        return f"{series.median():.{digits}f} [{series.min():.{digits}f}–{series.max():.{digits}f}]"

    def cf_cell(mode) -> str:
        values = cfs[mode]
        return f"+{values.median():.1f}% [{values.min():.1f}–{values.max():.1f}]"

    def pct_ci(row) -> str:
        return f"{f2(row.gain_pct)}%(±{f2(row.gain_ci95_training_seed)}%p)"

    doc = docx.Document(str(source))
    math = MathConverter(xsl)
    table_math = MathConverter(xsl, size_half_points=18)    # tables at body size (9 pt)
    note_math = MathConverter(xsl, size_half_points=16)     # table notes at 8 pt
    table_pt, note_pt = 18, 16

    # Abstract --------------------------------------------------------------
    rewrite(find(doc, "본 논문은 사용자 3명"), [
        "본 논문은 사용자 3명, 송신 안테나 2개의 과부하 MU-MISO 하향링크에서 유한 블록길이(FBL) RSMA의 "
        "전력과 common payload 분할을 PPO로 학습하여 장기 평균 AoI를 낮춘다. 동일한 사전학습과 누적 "
        "2,097,152 step의 PPO 학습 후 목표 조건(SNR 15 dB, blocklength 128)에서 5개 학습 seed(seed당 5개 "
        f"평가 seed)의 평균 AoI는 {f3(rsma.mean_aoi)}로 SDMA-PPO, 고정 규칙 RSMA, 최대 AoI 단일 사용자 "
        "전송보다 각각 "
        f"{f2(g_sdma.gain_pct)}%, {f2(g_rule.gain_pct)}%, {f2(g_single.gain_pct)}% 낮았다. 5개 중 "
        f"{len(typical)}개 학습 seed에서 정책은 private 전력을 AoI가 큰 사용자에게 집중하고 common payload를 "
        "채널 정렬도에 따라 조절하며 common 전력을 높은 운용점 주변에서 보정했다."
    ], math)

    # Section III: training procedure ---------------------------------------
    rewrite(find(doc, "행동 "), [
        "행동 ", ("m", 0), "에서 ", ("m", 1), "와 ", ("m", 2), "는 Gaussian 잠재변수 ", ("m", 3), "를 ",
        ("m", 4), "로, ", ("m", 5), "는 softmax로 변환해 SDMA 경계(", ("m", 6), ")를 정확히 표현한다. "
        "상태는 정규화 AoI, 이전 ACK와 전력 할당, 추정 채널 기반 이득·상관·beam 이득·nominal SINR, SNR과 ",
        ("m", 7), "이다. 보상은 ", ("m", 8), "이며 PPO[6](GAE, 할인율 0.99, clip 0.2, 2×128 은닉층)로 "
        "학습한다. RSMA-PPO와 SDMA-PPO 모두 동일한 학습 상태, posterior channel 표본 8개, 80 epochs 및 "
        "one-step 기대 AoI 목적함수로 사전학습하고, 혼합 조건(각도 폭 5/20/60°, SNR 10/15/20 dB, ",
        ("m", 9), "=128/256)에서 524,288 step 학습한다. 목표 조건(20°, 15 dB, ", ("m", 10),
        "=128)에서는 524,288 step 뒤 체크포인트에서 524,288 step과 1,048,576 step을 차례로 이어 누적 "
        "2,097,152 step까지 학습한다. 각 단계는 actor, critic, optimizer와 정규화 상태를 복원하되 learning "
        "rate annealing을 새로 시작한다. 에피소드 보상은 두 방법 모두 모든 단계에서 증가했다(그림 1)."
    ], math)
    rewrite_caption(find(doc, "그림 1."),
                    "목표 조건 에피소드(512 slot) 보상. 5개 학습 seed 평균 ±1 SD(65,536 step 이동평균). "
                    "점선: 학습률 annealing 재시작.")

    # Section IV: protocol -----------------------------------------------------
    rewrite(find(doc, "평가는 결정적 정책을"), [
        "평가는 독립 학습한 5개 학습 seed의 결정적 정책마다 평가 seed 5개×사용자 순열 6개의 30개 궤적(방법당 "
        "150개, 768 slot 중 처음 256 slot 제외)을 실행하며, 모든 방법이 같은 채널, CSIT 오차, ACK 난수를 "
        "공유한다. 학습 seed별 30개 궤적 평균을 표본으로 5개 학습 seed의 95% Student-t CI를 구했고, 평가 "
        f"seed 간 변동(RSMA-PPO {f3(ev_rsma['min'])}–{f3(ev_rsma['max'])})은 CI에 포함하지 않았다. 기준선은 "
        "(i) ", ("m", 0), "으로 고정하고 ", ("m", 1),
        "만 학습하는 SDMA-PPO(사전학습, curriculum, 누적 step 동일), (ii) 최대 AoI 사용자의 패킷 전체를 "
        "common으로 보내고 ", ("m", 2), "를 검증으로 0.70으로 정한 고정 규칙 RSMA, (iii) 최대 AoI 사용자 "
        "한 명에게 전체 전력을 주는 단일 사용자 MRT(평균 AoI 하한 2)다."
    ], math)

    # Table 1 -------------------------------------------------------------------
    rewrite_caption(find(doc, "표 1."), "목표 조건(20°, 15 dB, n=128)의 성능(누적 2.10M step)")
    table1 = doc.tables[0]
    rows = [("RSMA-PPO", rsma, "", True), ("SDMA-PPO", sdma, "", False),
            ("고정 규칙 RSMA", rule, "*", False), ("단일 사용자 MRT", single, "*", False)]
    rows[1:] = sorted(rows[1:], key=lambda item: item[1].mean_aoi)
    for i, (name, row, mark, bold) in enumerate(rows, start=1):
        ci = "–" if row.mean_aoi_ci95 < 5e-4 else f"±{row.mean_aoi_ci95:.3f}{mark}"
        values = [name, f3(row.mean_aoi), ci, f2(row.p95_aoi), f3(row.success_probability)]
        for c, value in enumerate(values):
            set_cell(table1.cell(i, c), value, table_math, bold=bold, size=table_pt)
    for c in range(len(table1.columns)):
        header = table1.cell(0, c)
        set_cell(header, header.text, table_math, bold=True, size=table_pt)
    set_column_widths(table1, [1700, 850, 720, 600, 880])
    keep_table_together(table1)
    rewrite(find(doc, "*규칙 정책은"), [
        "평균 AoI: 학습 seed 5×평가 seed 5×순열 6(150개 궤적) 평균. 95% CI: 학습 seed 5개 기준(*규칙: "
        "평가 seed 5개 기준). p95: 궤적별 AoI 95백분위수, 성공확률: 사용자·slot별 PHY 성공확률 평균."
    ], note_math, size=note_pt)

    rewrite(find(doc, "동일한 사전학습과 PPO 조건의"), [
        f"RSMA-PPO는 SDMA-PPO보다 평균 AoI가 {pct_ci(g_sdma)} 낮았고 학습 seed "
        f"{int(g_sdma.training_seed_wins)}/5, 학습·평가 seed 쌍 {int(g_sdma.seed_eval_pair_wins)}/25, 궤적 "
        f"{int(g_sdma.trajectory_wins)}/150에서 더 낮았다. 행동 차원과 최적화 난도가 달라 common stream만의 "
        f"인과효과로 단정하지 않는다. 고정 규칙 RSMA와 단일 사용자 대비 개선은 각각 {pct_ci(g_rule)}, "
        f"{pct_ci(g_single)}였다. 0.52M→2.10M 학습으로 AoI는 RSMA-PPO {f2(g_rsma_len.gain_pct)}%, SDMA-PPO "
        f"{f2(g_sdma_len.gain_pct)}% 줄었고, 0.52M 동일 예산에서도 차이는 {f2(g_524.gain_pct)}%였다."
    ], math)

    # Table 2: action roles -----------------------------------------------------
    rewrite_caption(find(doc, "표 2."), "목표 조건의 행동별 상태 대응(RSMA-PPO)")
    table2 = doc.tables[1]
    set_column_widths(table2, [480, 2600, 1670])
    table2_rows = [
        (["행동"], ["상태–행동 통계"], ["대체 시 AoI 증가"]),
        ([("tex", r"q_k")], [f"상대 AoI r {med_range(psr.q_relative_aoi_pearson)}"], [cf_cell("rank_q")]),
        ([("tex", r"\phi_k")], [f"채널 정렬도 ρ {med_range(psr.phi_alignment_spearman)}"],
         [cf_cell("fixed_phi_level")]),
        ([("tex", r"\alpha")], [f"평균 {psr.alpha_mean.median():.2f}, SD {med_range(psr.alpha_sd)}"],
         [cf_cell("fixed_alpha")]),
    ]
    for i, cells in enumerate(table2_rows):
        for c, parts in enumerate(cells):
            set_cell(table2.cell(i, c), parts, table_math, bold=(i == 0), size=table_pt)
    keep_table_together(table2)
    rewrite(find(doc, "괄호는 5개 seed의"), [
        "5개 학습 seed 중앙값 [최소–최대]. r: Pearson, ρ: Spearman(사용자 평균 ", ("tex", r"\phi_k"),
        "), SD: slot 간 ", ("tex", r"\alpha"), " 표준편차. 대체: 순위 규칙, 수준 고정, 평균값 고정(closed-loop)."
    ], note_math, size=note_pt)

    rewrite(find(doc, "표 2에서 정책은"), [
        f"표 2의 역할 분리는 5개 학습 seed 중 {len(typical)}개(seed "
        f"{', '.join(str(int(s)) for s in typical.index)})에서 관찰되었다. 이들 seed에서 ", ("tex", r"q_k"),
        "는 private 전력을 최대 AoI 사용자에게 집중하고(선택률 "
        f"{typical.q_argmax_is_oldest_unique_pct.min():.1f}–{typical.q_argmax_is_oldest_unique_pct.max():.1f}%), ",
        ("tex", r"\phi_k"), "는 채널 정렬도가 높을수록 common payload를 늘리며(ρ "
        f"{typical.phi_alignment_spearman.min():.2f}–{typical.phi_alignment_spearman.max():.2f}), ", ("tex", r"\alpha"),
        f"는 높은 운용점(평균 {typical.alpha_mean.min():.2f}–{typical.alpha_mean.max():.2f}) 주변에서 작게(SD "
        f"≤{typical.alpha_sd.max():.2f}) 보정했다. 나머지 seed {exception}은 ", ("tex", r"\alpha"),
        f"를 크게 바꾸는(SD {psr.alpha_sd.loc[exception]:.2f}) 다른 해로 비슷한 AoI({exception_aoi:.3f})를 "
        "얻었으므로, 역할 분리는 모든 해의 공통 성질이 아닌 다수 seed의 경향이다. 모든 seed에서 각 행동을 "
        "상수나 규칙으로 바꾸면 AoI가 증가했다. 그림 2의 "
        "feasible 조건(SNR≥10 dB, ", ("tex", "n"), "≥128)에서 ", ("tex", r"\alpha"),
        f"는 SNR과 함께 {a_snr.loc[10.0]:.2f}에서 {a_snr.loc[20.0]:.2f}로 증가했고, LoS 방향이 더 분리된 "
        f"60°에서 {a_geo.loc[60]:.2f}로 감소했다."
    ], math)
    replace_picture(doc, doc.paragraphs[[p.text for p in doc.paragraphs].index(
        find(doc, "그림 2.").text) - 1], PATHS["alpha_figure"])
    rewrite_caption(find(doc, "그림 2."),
                    "학습 범위의 feasible 조건(18개)에서 평균 α. (a) SNR별, (b) 각도 폭별. Error bar: 5개 학습 "
                    "seed의 95% CI.")

    rewrite(find(doc, "이 feasible 조건의"), [
        f"혼합 조건 학습에 포함된 이 feasible 조건의 평균 AoI는 {f3(feas.ppo_mean_aoi)}로 단일 사용자"
        f"({f3(feas.single_user_mean_aoi)})보다 {f2(feas.gain_mean_pct)}%(±{f2(feas.gain_ci95_halfwidth_pct)}%p) "
        f"낮았으나 0.52M 정책({f3(feas_524k.ppo_mean_aoi)})보다는 높았다."
    ], math)

    rewrite(find(doc, "본 논문은 SDMA 경계를"), [
        "본 논문은 SDMA 경계를 포함하는 연속 행동으로 FBL RSMA의 전력·payload 분할을 PPO로 학습하였다. "
        "목표 조건에서 제안 정책은 사전학습과 누적 학습 step이 같은 SDMA-PPO보다 "
        f"{f2(g_sdma.gain_pct)}%, 두 AoI 규칙보다 {f2(g_rule.gain_pct)}~{f2(g_single.gain_pct)}% 낮은 AoI를 "
        f"보였고, 5개 중 {len(typical)}개 학습 seed에서 ", ("tex", r"q_k"), ", ", ("tex", r"\phi_k"), ", ",
        ("tex", r"\alpha"), "가 각각 AoI 긴급도, 채널 정렬도, common 전력 보정을 담당했다. 결과는 CSIT 오차 ",
        ("tex", r"\sigma_e=0.2"), ", 고정 beam 규칙 및 학습에 포함된 SNR·blocklength 범위에 한정된다. 향후 "
        "사용자 선택과 더 넓은 채널 조건을 학습에 포함해 일반화 성능을 검증할 예정이다."
    ], math)

    replace_picture(doc, doc.paragraphs[[p.text for p in doc.paragraphs].index(
        find(doc, "그림 1.").text) - 1], PATHS["curve_figure"])
    doc.save(str(out))
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xsl", required=True, help="Office MML2OMML.XSL")
    for key, default in PATHS.items():
        ap.add_argument(f"--{key.replace('_', '-')}", default=str(default))
    args = ap.parse_args()
    for key in PATHS:
        PATHS[key] = Path(getattr(args, key))
    revise(Path(args.source), Path(args.out), Path(args.xsl))
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
