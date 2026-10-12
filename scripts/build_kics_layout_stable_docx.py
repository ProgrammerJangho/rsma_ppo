#!/usr/bin/env python3
"""Build a two-page, layout-stable DOCX from the verified KICS PDF pages."""
from __future__ import annotations

from pathlib import Path
import shutil

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Mm, Pt


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "output/kics/RSMA_PPO_AoI_KICS_2p.pdf"
OUTPUT = ROOT / "output/kics/RSMA_PPO_AoI_KICS_2p.docx"
TMP = ROOT / "tmp/kics_docx_pages"


def configure_section(section) -> None:
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    # A small printable margin keeps Word/PDF drivers from moving the image.
    section.top_margin = Mm(5)
    section.bottom_margin = Mm(5)
    section.left_margin = Mm(5)
    section.right_margin = Mm(5)
    section.header_distance = Mm(0)
    section.footer_distance = Mm(0)


def add_page_image(doc: Document, image_path: Path, page_number: int) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run()
    # Preserve the A4 aspect ratio inside the printable 200 x 287 mm area.
    shape = run.add_picture(str(image_path), width=Mm(200), height=Mm(282.857))
    doc_pr = shape._inline.docPr
    doc_pr.set("name", f"KICS paper page {page_number}")
    doc_pr.set("descr", f"RSMA-PPO AoI KICS paper page {page_number}")


def main() -> int:
    pages = [TMP / "page-1.png", TMP / "page-2.png"]
    if not all(path.is_file() for path in pages):
        raise FileNotFoundError("Render the verified two-page PDF to tmp/kics_docx_pages first")

    doc = Document()
    configure_section(doc.sections[0])
    normal = doc.styles["Normal"]
    normal.font.size = Pt(1)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)

    add_page_image(doc, pages[0], 1)
    doc.add_page_break()
    add_page_image(doc, pages[1], 2)

    # Remove blank paragraphs created around the explicit section break where safe.
    for paragraph in doc.paragraphs:
        ppr = paragraph._p.pPr
        if not paragraph.text and not paragraph._p.xpath(".//w:drawing") and ppr is None:
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(0)

    settings = doc.settings.element
    compat = settings.find(qn("w:compat"))
    if compat is not None:
        for child in list(compat):
            if child.tag == qn("w:compatSetting") and child.get(qn("w:name")) == "compatibilityMode":
                child.set(qn("w:val"), "15")
    doc.core_properties.title = "PPO-Based RSMA Resource Allocation for AoI Minimization"
    doc.core_properties.subject = "Two-page KICS paper with matched-pretraining results"
    doc.core_properties.author = "Park Jang Ho, Kwak Do Hyun"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
