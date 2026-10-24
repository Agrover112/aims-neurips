"""Build the AIMS field guide PDF.

Usage:
    python build.py

Pipeline:
  1. concatenate the HTML fragments in src/
  2. extract part/section headings and generate a contents page
  3. render a first pass, locate each section's page by its lede sentence
  4. re-render with page numbers, merge the cover, add PDF bookmarks
"""

from __future__ import annotations

import html
import re
import shutil
import sys
import unicodedata
from pathlib import Path

from playwright.sync_api import sync_playwright
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)

PARTS = [
    "00_head.html",
    "01_cover.html",
    "02_front.html",
    "03_part1.html",
    "04_part2.html",
    "05_part34.html",
    "06_part5.html",
    "07_part6.html",
]

HEADER = """
<div style="width:100%;font-family:Arial,Helvetica,sans-serif;font-size:7pt;color:#9a9587;
            padding:0 26mm;display:flex;justify-content:space-between;">
  <span>From Statistics to Neural Processes</span>
  <span>AIMS PAIEC · 2026</span>
</div>
"""

FOOTER = """
<div style="width:100%;font-family:Arial,Helvetica,sans-serif;font-size:7pt;color:#9a9587;
            padding:0 26mm;display:flex;justify-content:space-between;">
  <span>agrov/aims-neurips · field guide</span>
  <span>p. <span class="pageNumber"></span> / <span class="totalPages"></span></span>
</div>
"""


def strip_tags(value: str) -> str:
    value = re.sub(r"<span class=\"num\">.*?</span>", "", value, flags=re.S)
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value).strip()


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"[^a-z0-9]", "", value.lower())


def read_body() -> str:
    chunks = []
    for name in PARTS:
        text = (SRC / name).read_text()
        if name == "00_head.html":
            chunks.append(text)
        elif name == "01_cover.html":
            continue
        else:
            chunks.append(text)
    chunks.append("""
</body>
</html>
""")
    return "\n".join(chunks)


def read_cover() -> str:
    head = (SRC / "00_head.html").read_text()
    cover = (SRC / "01_cover.html").read_text()
    extra = """
    <style>
      @page { size: A4; margin: 0; }
      html, body { margin: 0; padding: 0; background: #0f201a; }
      .cover { height: 296.6mm; }
    </style>
    """
    return head.replace("</head>", extra + "</head>") + cover + "\n</body>\n</html>\n"


def extract_structure(body: str):
    pattern = re.compile(
        r'<h1 class="part-title"[^>]*data-part="([^"]+)"[^>]*>(.*?)</h1>'
        r'|<h2 class="sec"[^>]*id="([^"]+)"[^>]*>(.*?)</h2>',
        re.S,
    )
    items = []
    for match in pattern.finditer(body):
        if match.group(1):
            items.append(
                {
                    "kind": "part",
                    "label": match.group(1),
                    "title": strip_tags(match.group(2)),
                }
            )
            continue
        sec_id = match.group(3)
        raw = match.group(4)
        num_match = re.search(r'<span class="num">(.*?)</span>', raw, flags=re.S)
        number = strip_tags(num_match.group(1)) if num_match else ""
        title = strip_tags(raw)
        tail = body[match.end() : match.end() + 900]
        lede_match = re.search(r'<p class="lede">(.*?)</p>', tail, flags=re.S)
        lede = ""
        if lede_match:
            lede = re.sub(r"\\\(.*?\\\)", " ", lede_match.group(1), flags=re.S)
            lede = re.sub(r"\\\[.*?\\\]", " ", lede, flags=re.S)
            lede = strip_tags(lede)
        items.append(
            {
                "kind": "section",
                "id": sec_id,
                "number": number,
                "title": title,
                "fingerprint": normalize(lede)[:60] if lede else "",
            }
        )
    return items


def build_toc(items, page_map: dict[str, int] | None):
    page_map = page_map or {}
    rows = ['<div class="kicker">AIMS · PAIEC 2026</div>', "<h1>Contents</h1>"]
    group_open = False
    current_part = None
    for item in items:
        if item["kind"] == "part":
            label = item["label"]
            title = item["title"]
            rows.append(f'<div class="toc-part">{html.escape(label)} · {html.escape(title)}</div>')
            group_open = True
            continue
        if not group_open:
            rows.append('<div class="toc-part">Start here</div>')
            group_open = True
        number = item["number"]
        page = page_map.get(item["id"], "··")
        rows.append(
            '<div class="toc-row">'
            f'<span class="n">{html.escape(number)}</span>'
            f'<span class="t">{html.escape(item["title"])}</span>'
            '<span class="dots"></span>'
            f'<span class="pg" data-target="{item["id"]}">{page}</span>'
            "</div>"
        )
    rows.append(
        '<p class="toc-note">Boxes: <b>Intuition</b> gives the picture, '
        '<b>Solved</b> is a 60-second numeric check, <b>Trap</b> is the mistake, '
        '<b>In this repo</b> points at the code.</p>'
    )
    return "\n".join(rows)


def assemble(body: str, toc_html: str) -> str:
    return body.replace('<nav class="toc" id="toc"></nav>', f'<nav class="toc" id="toc">{toc_html}</nav>')


def wait_for_math(page, require_math: bool = True) -> None:
    page.wait_for_function("() => document.fonts.status === 'loaded'", timeout=30000)
    if require_math:
        page.evaluate("() => window.MathJax.startup.promise")
        page.wait_for_function(
            "() => document.querySelectorAll('mjx-container').length > 40", timeout=30000
        )
    page.wait_for_timeout(400)


def render(pw, html_text: str, out_pdf: Path, cover: bool = False) -> None:
    browser = pw.chromium.launch(
        env={
            "LD_LIBRARY_PATH": "/tmp/opencode/guidework/libasound/usr/lib/x86_64-linux-gnu",
        }
    )
    page = browser.new_page(viewport={"width": 794, "height": 1123})
    page.emulate_media(media="print")
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    tmp = BUILD / ("cover.html" if cover else "body.html")
    html_text = html_text.replace("<head>", f'<head><base href="{ROOT.as_uri()}/">', 1)
    tmp.write_text(html_text)
    page.goto(tmp.as_uri())
    wait_for_math(page, require_math=not cover)
    if cover:
        page.pdf(
            path=str(out_pdf),
            format="A4",
            print_background=True,
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
        )
    else:
        page.pdf(
            path=str(out_pdf),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template=HEADER,
            footer_template=FOOTER,
            margin={"top": "18mm", "bottom": "16mm", "left": "22mm", "right": "22mm"},
        )
    if errors:
        print("PAGE ERRORS:", errors[:5])
    browser.close()


def page_texts(pdf: Path) -> list[str]:
    reader = PdfReader(str(pdf))
    return [normalize(page.extract_text() or "") for page in reader.pages]


def locate(items, texts: list[str]) -> dict[str, int]:
    # Find where the body starts: the how-to lede is unique and not in the TOC.
    start = 0
    marker = normalize("You do not need to read this cover to cover")
    for index, text in enumerate(texts):
        if marker in text:
            start = index
            break
    found: dict[str, int] = {}
    missing = []
    for item in items:
        if item["kind"] != "section":
            continue
        fingerprint = item.get("fingerprint") or normalize(item["title"])[:40]
        page_index = None
        for index in range(start, len(texts)):
            if fingerprint and fingerprint in texts[index]:
                page_index = index
                break
        if page_index is None:
            fallback = normalize(item["title"])[:30]
            for index in range(start, len(texts)):
                if fallback and fallback in texts[index]:
                    page_index = index
                    break
        if page_index is None:
            missing.append(item["title"])
        else:
            found[item["id"]] = page_index + 1  # printed page number, cover unnumbered
    if missing:
        print("WARNING: could not locate:", ", ".join(missing))
    return found


def add_bookmarks(writer: PdfWriter, items, page_map: dict[str, int]) -> None:
    parent = None
    for item in items:
        if item["kind"] == "part":
            parent = writer.add_outline_item(
                f'{item["label"]} · {item["title"]}', 0
            )
        else:
            page_number = page_map.get(item["id"], 1)
            title = f'{item["number"]} · {item["title"]}' if item["number"] else item["title"]
            writer.add_outline_item(title, page_number, parent=parent)


def main() -> None:
    body = read_body()
    cover_html = read_cover()
    items = extract_structure(body)

    toc_empty = build_toc(items, None)
    first_body = BUILD / "body-pass1.pdf"
    cover_pdf = BUILD / "cover.pdf"
    second_body = BUILD / "body-pass2.pdf"
    final = ROOT / "aims-field-guide.pdf"

    with sync_playwright() as pw:
        render(pw, assemble(body, toc_empty), first_body)
        texts = page_texts(first_body)
        page_map = locate(items, texts)
        print(f"located {len(page_map)} sections across {len(texts)} pages")
        render(pw, assemble(body, build_toc(items, page_map)), second_body)
        render(pw, cover_html, cover_pdf, cover=True)

    writer = PdfWriter()
    for pdf in (cover_pdf, second_body):
        reader = PdfReader(str(pdf))
        for page in reader.pages:
            writer.add_page(page)

    add_bookmarks(writer, items, page_map)
    writer.add_metadata(
        {
            "/Title": "From Statistics to Neural Processes — AIMS PAIEC field guide",
            "/Author": "AIMS PAIEC working notes",
            "/Subject": "Statistics, IRT, and Neural Processes for the AIMS Predictive AI Evaluation Challenge",
            "/Keywords": "Brier score, calibration, IRT, amortized IRT, Neural Processes, CNP, NP, ANP",
        }
    )
    with open(final, "wb") as handle:
        writer.write(handle)

    total = len(PdfReader(str(final)).pages)
    print(f"wrote {final} ({total} pages)")
    (BUILD / "pagemap.json").write_text(
        "\n".join(f"{key} {value}" for key, value in sorted(page_map.items()))
    )


if __name__ == "__main__":
    main()
