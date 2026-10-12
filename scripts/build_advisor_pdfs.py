#!/usr/bin/env python3
"""Build the Korean thesis draft and advisor result-summary PDFs from Markdown."""
from __future__ import annotations

import html
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp/pdfs/build"
OUT = ROOT / "output/pdf"
FONT_DIR = Path("/mnt/c/Windows/Fonts")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("NotoKR", str(FONT_DIR / "NotoSansKR-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("NotoKR-Medium", str(FONT_DIR / "NotoSansKR-Medium.ttf")))
    pdfmetrics.registerFont(TTFont("NotoKR-Bold", str(FONT_DIR / "NotoSansKR-Bold.ttf")))
    pdfmetrics.registerFontFamily(
        "NotoKR", normal="NotoKR", bold="NotoKR-Bold",
        italic="NotoKR", boldItalic="NotoKR-Bold",
    )


BIB_KEYS = [
    "mao2018bridging", "joudeh2016partial", "mao2022survey",
    "polyanskiy2010fbl", "xu2022fblrsma", "kaul2012realtime",
    "yates2021aoi", "kadota2018scheduling", "schulman2017ppo",
    "yan2023satellite", "kaya2025thesis", "hieu2021drl",
    "ali2025harqrsma",
]
BIB_NUM = {key: index + 1 for index, key in enumerate(BIB_KEYS)}


def inline(text: str) -> str:
    def citation(match: re.Match[str]) -> str:
        keys = re.findall(r"@([\w-]+)", match.group(0))
        nums = [str(BIB_NUM[k]) for k in keys if k in BIB_NUM]
        return "[" + ", ".join(nums) + "]" if nums else ""

    text = re.sub(r"\[@[^\]]+\]", citation, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<font name='NotoKR-Medium'>\1</font>", text)
    text = text.replace("±", "&#177;").replace("<br>", "<br/>")
    return text


def styles():
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="NotoKR", fontSize=8.7,
            leading=13.4, alignment=TA_JUSTIFY, textColor=colors.HexColor("#263238"),
            spaceAfter=3.8 * mm, wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="NotoKR-Bold", fontSize=16,
            leading=22, textColor=colors.HexColor("#0d5068"), spaceBefore=5 * mm,
            spaceAfter=3 * mm, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="NotoKR-Bold", fontSize=12,
            leading=17, textColor=colors.HexColor("#17627a"), spaceBefore=4 * mm,
            spaceAfter=2 * mm, keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3", parent=base["Heading3"], fontName="NotoKR-Bold", fontSize=9.7,
            leading=14, textColor=colors.HexColor("#205a70"), spaceBefore=3 * mm,
            spaceAfter=1.5 * mm, keepWithNext=True,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["BodyText"], fontName="NotoKR", fontSize=7.2,
            leading=10.5, alignment=TA_CENTER, textColor=colors.HexColor("#54656d"),
            spaceBefore=1.2 * mm, spaceAfter=3.5 * mm, wordWrap="CJK",
        ),
        "quote": ParagraphStyle(
            "Quote", parent=base["BodyText"], fontName="NotoKR-Medium", fontSize=8.5,
            leading=13.2, leftIndent=5 * mm, rightIndent=4 * mm,
            borderColor=colors.HexColor("#7da9b7"), borderWidth=0.8,
            borderPadding=4 * mm, backColor=colors.HexColor("#edf6f8"),
            textColor=colors.HexColor("#21434e"), spaceAfter=4 * mm, wordWrap="CJK",
        ),
        "list": ParagraphStyle(
            "List", parent=base["BodyText"], fontName="NotoKR", fontSize=8.5,
            leading=13.2, leftIndent=5 * mm, firstLineIndent=-3.5 * mm,
            spaceAfter=1.4 * mm, textColor=colors.HexColor("#263238"), wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "Table", parent=base["BodyText"], fontName="NotoKR", fontSize=6.5,
            leading=8.3, alignment=TA_CENTER, wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "TableHeader", parent=base["BodyText"], fontName="NotoKR-Bold", fontSize=6.5,
            leading=8.3, alignment=TA_CENTER, textColor=colors.white, wordWrap="CJK",
        ),
    }


def render_math(expr: str, index: int) -> Path:
    out = TMP / f"eq_{index:03d}.png"
    cleaned = expr.strip().replace("\\operatorname", "\\mathrm")
    cleaned = cleaned.replace("\\mathbb", "\\mathrm")
    cleaned = re.sub(r"\\mathbf\s+([A-Za-z0-9])", r"\\mathbf{\1}", cleaned)
    cleaned = re.sub(r"\\boldsymbol\s*(\\[A-Za-z]+|[A-Za-z])", r"\\boldsymbol{\1}", cleaned)
    cleaned = re.sub(r"\\mathrm\s+([A-Za-z])", r"\\mathrm{\1}", cleaned)
    cleaned = re.sub(r"\\sqrt\s+([A-Za-z0-9])", r"\\sqrt{\1}", cleaned)
    cleaned = cleaned.replace("\\frac1K", "\\frac{1}{K}")
    cleaned = cleaned.replace("\\le ", "\\leq ")
    cleaned = cleaned.replace("\\begin{cases}", "\\left\\{").replace("\\end{cases}", "")
    fig = plt.figure(figsize=(9.0, 0.65), dpi=220)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    try:
        ax.text(0.5, 0.5, f"${cleaned}$", ha="center", va="center", fontsize=14, color="#263238")
        fig.canvas.draw()
        fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05)
    except Exception as exc:
        print(f"math fallback {index}: {exc}: {cleaned}")
        ax.clear(); ax.axis("off")
        ax.text(0.5, 0.5, cleaned, ha="center", va="center", fontsize=10, color="#263238")
        fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05)
    finally:
        plt.close(fig)
    return out


def parse_bib() -> list[str]:
    raw = (ROOT / "docs/thesis_references.bib").read_text(encoding="utf-8")
    entries: dict[str, str] = {}
    positions = list(re.finditer(r"@[A-Za-z]+\{([^,]+),", raw))
    for i, match in enumerate(positions):
        block = raw[match.start(): positions[i + 1].start() if i + 1 < len(positions) else len(raw)]
        entries[match.group(1)] = block

    def field(block: str, name: str) -> str:
        match = re.search(rf"\b{name}\s*=\s*\{{(.*?)\}}\s*,?\s*$", block, re.I | re.M | re.S)
        if not match:
            match = re.search(rf"\b{name}\s*=\s*\"(.*?)\"\s*,?\s*$", block, re.I | re.M | re.S)
        return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""

    refs = []
    for key in BIB_KEYS:
        block = entries.get(key, "")
        authors = field(block, "author").replace(" and ", ", ")
        title = field(block, "title").replace("{", "").replace("}", "")
        venue = field(block, "journal") or field(block, "booktitle") or field(block, "school")
        year = field(block, "year")
        doi = field(block, "doi")
        parts = [authors, f'“{title}”' if title else "", venue, year]
        text = ", ".join(p for p in parts if p)
        if doi:
            text += f", DOI: {doi}"
        refs.append(text + ".")
    return refs


def markdown_story(path: Path, *, thesis: bool) -> list:
    st = styles()
    lines = path.read_text(encoding="utf-8").splitlines()
    story: list = []
    eq_index = 0
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), st["body"]))
            paragraph.clear()

    if thesis:
        title = lines[0].removeprefix("# ").strip()
        story.extend([
            Spacer(1, 58 * mm),
            Paragraph(inline(title), ParagraphStyle(
                "Cover", fontName="NotoKR-Bold", fontSize=20, leading=30,
                alignment=TA_CENTER, textColor=colors.HexColor("#174f65"), wordWrap="CJK",
            )),
            Spacer(1, 13 * mm),
            Table([[""]], colWidths=[56 * mm], rowHeights=[0.8 * mm],
                  style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#77aabd"))])),
            Spacer(1, 18 * mm),
            Paragraph("RSMA + Reinforcement Learning + Age of Information", ParagraphStyle(
                "CoverSub", fontName="NotoKR", fontSize=8, leading=11,
                alignment=TA_CENTER, textColor=colors.HexColor("#78909c"),
            )),
            PageBreak(),
        ])
        lines = lines[1:]
    else:
        lines = lines[1:]

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph(); i += 1; continue
        if stripped == "---":
            flush_paragraph(); story.append(Spacer(1, 1.5 * mm)); i += 1; continue
        if stripped == "\\[":
            flush_paragraph(); eq: list[str] = []; i += 1
            while i < len(lines) and lines[i].strip() != "\\]":
                eq.append(lines[i].strip()); i += 1
            eq_index += 1
            img_path = render_math(" ".join(eq), eq_index)
            img = Image(str(img_path)); img._restrictSize(160 * mm, 25 * mm)
            story.extend([Spacer(1, 1 * mm), img, Spacer(1, 2 * mm)])
            i += 1; continue
        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if image_match:
            flush_paragraph()
            image_path = (path.parent / image_match.group(2)).resolve()
            if image_path.is_file():
                img = Image(str(image_path)); img._restrictSize(168 * mm, 88 * mm)
                story.extend([Spacer(1, 1.5 * mm), img, Spacer(1, 1 * mm)])
            i += 1; continue
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?\s*:?-+", lines[i + 1].strip()):
            flush_paragraph(); rows: list[list[str]] = []
            rows.append([c.strip() for c in stripped.strip("|").split("|")])
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            ncols = max(len(r) for r in rows)
            rows = [r + [""] * (ncols - len(r)) for r in rows]
            pdata = []
            for ridx, row in enumerate(rows):
                sty = st["table_header"] if ridx == 0 else st["table"]
                pdata.append([Paragraph(inline(c), sty) for c in row])
            widths = [174 * mm / ncols] * ncols
            table = Table(pdata, colWidths=widths, repeatRows=1, hAlign="CENTER")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14566d")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f7f8")]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c7d6db")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 1.6 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6 * mm),
            ]))
            story.extend([table, Spacer(1, 3 * mm)]); continue
        if stripped.startswith("### "):
            flush_paragraph(); story.append(Paragraph(inline(stripped[4:]), st["h3"])); i += 1; continue
        if stripped.startswith("## "):
            flush_paragraph(); story.append(Paragraph(inline(stripped[3:]), st["h1"])); i += 1; continue
        if stripped.startswith("# "):
            flush_paragraph(); story.append(Paragraph(inline(stripped[2:]), st["h1"])); i += 1; continue
        if stripped.startswith("> "):
            flush_paragraph(); story.append(Paragraph(inline(stripped[2:]), st["quote"])); i += 1; continue
        list_match = re.match(r"^(\d+\.|-)\s+(.*)$", stripped)
        if list_match:
            flush_paragraph()
            bullet = list_match.group(1)
            symbol = f"{bullet}" if bullet != "-" else "•"
            story.append(Paragraph(f"{symbol}&nbsp;&nbsp;{inline(list_match.group(2))}", st["list"]))
            i += 1; continue
        if stripped.startswith("**그림") or stripped.startswith("**표"):
            flush_paragraph(); story.append(Paragraph(inline(stripped), st["caption"])); i += 1; continue
        paragraph.append(stripped); i += 1
    flush_paragraph()

    if thesis:
        story.append(PageBreak())
        story.append(Paragraph("참고문헌", st["h1"]))
        for idx, ref in enumerate(parse_bib(), 1):
            story.append(Paragraph(f"[{idx}] {inline(ref)}", ParagraphStyle(
                f"Ref{idx}", parent=st["body"], fontSize=6.8, leading=10.2,
                leftIndent=5 * mm, firstLineIndent=-5 * mm, spaceAfter=1.5 * mm,
            )))
    return story


class ResearchDoc(BaseDocTemplate):
    def __init__(self, filename: str, *, title: str, thesis: bool):
        super().__init__(
            filename, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
            topMargin=18 * mm, bottomMargin=17 * mm,
            title=title, author="RSMA-AoI 연구 프로젝트",
        )
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates(PageTemplate(id="all", frames=[frame], onPage=self._decorate))
        self.thesis = thesis
        self.header = "RSMA-PPO 기반 AoI 최소화"

    def _decorate(self, canvas, doc):
        if self.thesis and doc.page == 1:
            return
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d7e2e6")); canvas.setLineWidth(0.35)
        canvas.line(18 * mm, A4[1] - 12 * mm, A4[0] - 18 * mm, A4[1] - 12 * mm)
        canvas.setFont("NotoKR", 6.5); canvas.setFillColor(colors.HexColor("#8b9aa1"))
        canvas.drawString(18 * mm, A4[1] - 9 * mm, self.header)
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, str(doc.page if not self.thesis else doc.page - 1))
        canvas.restoreState()


def build(src: Path, dest: Path, *, title: str, thesis: bool) -> None:
    doc = ResearchDoc(str(dest), title=title, thesis=thesis)
    doc.build(markdown_story(src, thesis=thesis))


def main() -> int:
    register_fonts()
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    OUT.mkdir(parents=True, exist_ok=True)
    thesis_tmp = TMP / "졸업논문_중간초안_RSMA_PPO_AoI.pdf"
    results_tmp = TMP / "RSMA_PPO_상세실험결과.pdf"
    build(
        ROOT / "docs/thesis_manuscript_draft_2026-09-22.md", thesis_tmp,
        title="RSMA-PPO 기반 AoI 최소화 - 논문 중간 초안", thesis=True,
    )
    build(
        ROOT / "docs/advisor_results_summary_2026-09-22.md", results_tmp,
        title="RSMA-PPO 기반 AoI 최소화 - 상세 실험 결과", thesis=False,
    )
    for src in (thesis_tmp, results_tmp):
        shutil.copy2(src, OUT / src.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
