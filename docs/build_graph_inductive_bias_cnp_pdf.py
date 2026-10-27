"""Build ``graph-inductive-bias-CNP.pdf`` from its Markdown source."""

from __future__ import annotations

import html
import re
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "graph-inductive-bias-CNP.md"
OUTPUT = ROOT / "docs" / "graph-inductive-bias-CNP.pdf"

INK = colors.HexColor("#14231d")
GREEN = colors.HexColor("#1d6b50")
GREEN_TINT = colors.HexColor("#edf3ee")
AMBER = colors.HexColor("#a86a1b")
MUTED = colors.HexColor("#66736d")
RULE = colors.HexColor("#d8d4c7")
PAPER = colors.HexColor("#fdfcf7")
CODE = colors.HexColor("#f2efe5")


def register_fonts() -> None:
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    pdfmetrics.registerFont(TTFont("Body", font_dir / "DejaVuSerif.ttf"))
    pdfmetrics.registerFont(TTFont("BodyBold", font_dir / "DejaVuSerif-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("Sans", font_dir / "DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("SansBold", font_dir / "DejaVuSans-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("Mono", font_dir / "DejaVuSansMono.ttf"))
    pdfmetrics.registerFontFamily(
        "Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold"
    )
    pdfmetrics.registerFontFamily(
        "Sans", normal="Sans", bold="SansBold", italic="Sans", boldItalic="SansBold"
    )


def make_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "DocBody",
            parent=sample["BodyText"],
            fontName="Body",
            fontSize=9.5,
            leading=14.2,
            textColor=INK,
            spaceAfter=6.5,
            alignment=TA_LEFT,
            allowWidows=0,
            allowOrphans=0,
        ),
        "section": ParagraphStyle(
            "Section",
            fontName="SansBold",
            fontSize=15.5,
            leading=18.5,
            textColor=INK,
            spaceBefore=5,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "subsection": ParagraphStyle(
            "Subsection",
            fontName="SansBold",
            fontSize=8.8,
            leading=12,
            textColor=GREEN,
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "bullet": ParagraphStyle(
            "BulletText",
            fontName="Body",
            fontSize=9.3,
            leading=13.7,
            textColor=INK,
            spaceAfter=2.5,
        ),
        "code": ParagraphStyle(
            "CodeBlock",
            fontName="Mono",
            fontSize=7.5,
            leading=11.2,
            textColor=INK,
            backColor=CODE,
            borderColor=RULE,
            borderWidth=0.5,
            borderPadding=7,
            spaceBefore=5,
            spaceAfter=8,
        ),
        "table_head": ParagraphStyle(
            "TableHead", fontName="SansBold", fontSize=7.2, leading=9, textColor=INK
        ),
        "table_cell": ParagraphStyle(
            "TableCell", fontName="Sans", fontSize=7.6, leading=10, textColor=INK
        ),
        "eyebrow": ParagraphStyle(
            "CoverEyebrow",
            fontName="SansBold",
            fontSize=8,
            leading=10,
            textColor=AMBER,
            spaceAfter=14,
        ),
        "title": ParagraphStyle(
            "CoverTitle",
            fontName="BodyBold",
            fontSize=30,
            leading=32,
            textColor=INK,
            spaceAfter=18,
        ),
        "deck": ParagraphStyle(
            "CoverDeck",
            fontName="Body",
            fontSize=12.4,
            leading=18,
            textColor=colors.HexColor("#425149"),
            spaceAfter=24,
        ),
        "flow": ParagraphStyle(
            "Flow",
            fontName="Sans",
            fontSize=7.2,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "cover_note": ParagraphStyle(
            "CoverNote",
            fontName="SansBold",
            fontSize=8.2,
            leading=11,
            textColor=colors.white,
            backColor=GREEN,
            borderPadding=9,
        ),
    }


def plain_math(value: str) -> str:
    replacements = {
        r"\theta": "θ",
        r"\phi": "φ",
        r"\psi": "ψ",
        r"\sigma": "σ",
        r"\in": "∈",
        r"\subseteq": "⊆",
        r"\cap": "∩",
        r"\varnothing": "∅",
        r"\le": "≤",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value.replace(r"\{", "{").replace(r"\}", "}")


def rich(value: str) -> str:
    value = html.escape(value, quote=True)
    value = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda match: f'<font color="#1d6b50">{match.group(1)}</font>',
        value,
    )
    value = re.sub(
        r"`([^`]+)`",
        lambda match: f'<font name="Mono" backColor="#f2efe5"> {match.group(1)} </font>',
        value,
    )
    value = re.sub(
        r"\\\((.*?)\\\)",
        lambda match: f'<font name="Mono">{html.escape(plain_math(html.unescape(match.group(1))))}</font>',
        value,
    )
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    return value


def equation_images(raw_lines: list[str]) -> list:
    raw = " ".join(line.strip() for line in raw_lines).strip()
    expressions = [part.strip() for part in re.split(r"\\qquad", raw) if part.strip()]
    flowables = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        for expression in expressions:
            figure = Figure(figsize=(6.8, 0.42), dpi=220, facecolor="none")
            FigureCanvasAgg(figure)
            figure.text(
                0.5,
                0.5,
                f"${expression}$",
                ha="center",
                va="center",
                fontsize=12,
                color="#14231d",
            )
            buffer = BytesIO()
            figure.savefig(
                buffer,
                format="png",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.04,
                dpi=220,
            )
            buffer.seek(0)
            image = Image(buffer)
            scale = min(1.0, 155 * mm / image.imageWidth)
            image.drawWidth = image.imageWidth * scale
            image.drawHeight = image.imageHeight * scale
            image.hAlign = "CENTER"
            image._source_buffer = buffer
            flowables.extend([Spacer(1, 2), image, Spacer(1, 3)])
        return flowables
    except Exception:
        return [Paragraph(html.escape(raw.replace(r"\qquad", "   ")), STYLES["code"])]


def markdown_table(lines: list[str]) -> Table:
    rows = []
    for index, line in enumerate(lines):
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if index == 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        style = STYLES["table_head"] if index == 0 else STYLES["table_cell"]
        rows.append([Paragraph(rich(cell), style) for cell in cells])
    table = Table(rows, colWidths=[170 * mm / len(rows[0])] * len(rows[0]), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), GREEN_TINT),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
                ("LINEBELOW", (0, 1), (-1, -1), 0.35, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def body_paragraph(value: str, style: str = "body") -> Paragraph:
    return Paragraph(rich(value), STYLES[style])


def markdown_story(source: str) -> list:
    lines = source.splitlines()
    story = []
    index = 1 if lines and lines[0].startswith("# ") else 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line.strip():
            index += 1
            continue
        if line.startswith("## "):
            if story:
                story.extend([Spacer(1, 6), HRFlowable(width="100%", thickness=0.5, color=RULE)])
            story.append(Paragraph(rich(line[3:].strip()), STYLES["section"]))
            index += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(rich(line[4:].strip().upper()), STYLES["subsection"]))
            index += 1
            continue
        if line.strip() == r"\[":
            equation = []
            index += 1
            while index < len(lines) and lines[index].strip() != r"\]":
                equation.append(lines[index])
                index += 1
            index += 1
            story.append(KeepTogether(equation_images(equation)))
            continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.extend([markdown_table(table_lines), Spacer(1, 4)])
            continue
        if re.match(r"^- ", line):
            items = []
            while index < len(lines) and re.match(r"^- ", lines[index]):
                items.append(
                    ListItem(body_paragraph(lines[index][2:].strip(), "bullet"), leftIndent=12)
                )
                index += 1
            story.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    start="circle",
                    leftIndent=16,
                    bulletFontName="Body",
                    bulletFontSize=7,
                    spaceAfter=6,
                )
            )
            continue
        if re.match(r"^\d+\. ", line):
            items = []
            start = int(line.split(".", 1)[0])
            while index < len(lines) and re.match(r"^\d+\. ", lines[index]):
                value = re.sub(r"^\d+\.\s+", "", lines[index])
                items.append(ListItem(body_paragraph(value, "bullet"), leftIndent=15))
                index += 1
            story.append(
                ListFlowable(
                    items,
                    bulletType="1",
                    start=start,
                    leftIndent=19,
                    bulletFontName="Sans",
                    bulletFontSize=8,
                    spaceAfter=7,
                )
            )
            continue
        paragraph = [line.strip()]
        index += 1
        while (
            index < len(lines)
            and lines[index].strip()
            and not re.match(r"^(#{1,3} |[-] |\d+\. |\||\\\[)", lines[index])
        ):
            paragraph.append(lines[index].strip())
            index += 1
        story.append(body_paragraph(" ".join(paragraph)))
    return story


def page_decoration(canvas, document) -> None:
    canvas.saveState()
    canvas.setTitle("Graph inductive bias in a Conditional Neural Process")
    canvas.setAuthor("AIMS PAIEC")
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    if document.page > 1:
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.35)
        canvas.line(20 * mm, A4[1] - 13 * mm, A4[0] - 20 * mm, A4[1] - 13 * mm)
        canvas.line(20 * mm, 13 * mm, A4[0] - 20 * mm, 13 * mm)
        canvas.setFont("Sans", 6.7)
        canvas.setFillColor(MUTED)
        canvas.drawString(20 * mm, A4[1] - 10.5 * mm, "GRAPH INDUCTIVE BIAS IN CNP")
        canvas.drawRightString(A4[0] - 20 * mm, A4[1] - 10.5 * mm, "AIMS · RESEARCH NOTE")
        canvas.drawString(20 * mm, 9.2 * mm, "agrov/aims-neurips")
        canvas.drawRightString(A4[0] - 20 * mm, 9.2 * mm, str(document.page - 1))
    canvas.restoreState()


def build() -> None:
    story = [
        Spacer(1, 28 * mm),
        Paragraph("AIMS · RESEARCH DESIGN NOTE", STYLES["eyebrow"]),
        Paragraph("Graph inductive bias in a Conditional Neural Process", STYLES["title"]),
        Paragraph(
            "A paper-faithful CNP baseline, a clean graph ablation, and a "
            "benchmark-held-out evaluation plan for online model testing.",
            STYLES["deck"],
        ),
        Spacer(1, 8 * mm),
    ]
    cells = []
    for title, subtitle in (
        ("Context", "item embedding + response"),
        ("Encoder", "shared MLP"),
        ("Mean pool", "task representation"),
        ("Decoder", "target probability"),
    ):
        cells.append(
            Paragraph(
                f'<font name="SansBold" color="#1d6b50">{title}</font><br/>{subtitle}',
                STYLES["flow"],
            )
        )
    flow = Table([cells], colWidths=[42.5 * mm] * 4, rowHeights=[25 * mm])
    flow.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BOX", (0, 0), (-1, -1), 0.6, RULE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, RULE),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f6f8f4")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend(
        [
            flow,
            Spacer(1, 19 * mm),
            Paragraph(
                "Baseline first. Graph structure only after the pure CNP is verified.",
                STYLES["cover_note"],
            ),
            PageBreak(),
        ]
    )
    story.extend(markdown_story(SOURCE.read_text(encoding="utf-8")))

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Graph inductive bias in a Conditional Neural Process",
        author="AIMS PAIEC",
        subject="CNP and graph inductive bias design note",
    )
    document.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)


register_fonts()
STYLES = make_styles()

if __name__ == "__main__":
    build()
    print(f"Wrote {OUTPUT}")
