#!/usr/bin/env python
"""Convert DeepSeek OCR responses into Markdown.

The OCR API returns per-page model output only.  `rich_layout_recognition`
keeps inline layout markers of the form

    <｜ref｜>label<｜/ref｜><｜box｜>[[x1,y1,x2,y2]]<｜/box｜> content

with coordinates normalised to 0-1000 against the displayed page.  Image
blocks carry no pixels, so this script crops them from the source PDF with
PyMuPDF and writes a Markdown file that links to the crops.

Usage:
    python ocr_to_markdown.py RESPONSE.json SOURCE.pdf OUT_DIR
"""
from __future__ import annotations

import argparse
import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import fitz
from PIL import Image

REF = re.compile(r"<｜ref｜>(.*?)<｜/ref｜><｜box｜>\[\[([-\d.,\s]+)\]\]<｜/box｜>")
LABELS = {"title", "sub_title", "text", "image", "image_caption", "table",
          "table_caption", "equation"}


def html_table_to_md(fragment: str) -> str:
    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows, self.row, self.cell, self.in_cell = [], [], [], False
            self.preamble = []

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self.row = []
            elif tag in ("td", "th"):
                self.in_cell, self.cell = True, []

        def handle_endtag(self, tag):
            if tag in ("td", "th") and self.in_cell:
                self.row.append("".join(self.cell).strip())
                self.in_cell = False
            elif tag == "tr" and self.row:
                self.rows.append(self.row)
                self.row = []

        def handle_data(self, data):
            if self.in_cell:
                self.cell.append(data)
            else:
                self.preamble.append(data)

    p = P()
    p.feed(fragment)
    if p.row:
        p.rows.append(p.row)
    if not p.rows:
        return fragment
    width = max(len(r) for r in p.rows)
    rows = [r + [""] * (width - len(r)) for r in p.rows]

    def esc(cell):
        return cell.replace("|", "\\|")

    lines = []
    preamble = " ".join("".join(p.preamble).split())
    if preamble:
        lines.append(f"*{preamble}*\n")
    lines += ["| " + " | ".join(esc(c) for c in rows[0]) + " |",
              "| " + " | ".join("---" for _ in range(width)) + " |"]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def strip_tags(s: str) -> str:
    s = re.sub(r"</?center>", "", s, flags=re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    return s.strip()


def parse_page(text: str):
    """Yield (label, box, content) for each layout element on a page."""
    matches = list(REF.finditer(text))
    for i, m in enumerate(matches):
        label = m.group(1).strip()
        box = [float(v) for v in m.group(2).split(",")]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield label, box, text[m.end():end].strip()
    # Text before the first marker (if any) is kept as a plain paragraph.
    prefix = text[:matches[0].start()].strip() if matches else text.strip()
    if prefix:
        yield "text", None, prefix


def render_page(doc: fitz.Document, index: int, dpi: int):
    page = doc[index]
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return page, pix


def crop_norm(pix: fitz.Pixmap, box, out: Path):
    x1, y1, x2, y2 = box
    w, h = pix.width, pix.height
    clip = (
        max(0, int(x1 / 1000 * w)), max(0, int(y1 / 1000 * h)),
        min(w, int(x2 / 1000 * w + 0.5)), min(h, int(y2 / 1000 * h + 0.5)),
    )
    im = Image.open(io.BytesIO(pix.tobytes("png")))
    im.crop(clip).save(out)
    return out.name


def convert_rich(data: dict, doc: fitz.Document, out_dir: Path, dpi: int = 200) -> str:
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    md = []
    for page in data["pages"]:
        idx, text = page["index"], page["text"]
        md.append(f"\n<!-- page {idx + 1} -->\n")
        if "<｜ref｜>" not in text:
            md.append(strip_tags(text) + "\n")
            continue
        pix = None
        fig_no = 0
        for label, box, content in parse_page(text):
            content = strip_tags(content)
            if label == "image":
                if pix is None:
                    _, pix = render_page(doc, idx, dpi)
                fig_no += 1
                name = f"page_{idx + 1:03d}_img_{fig_no}.png"
                crop_norm(pix, box, img_dir / name)
                md.append(f"![page {idx + 1} figure {fig_no}](images/{name})\n")
            elif label == "image_caption":
                md.append(f"*{content}*\n" if content else "")
            elif label == "title":
                md.append((content if content.startswith("#") else "# " + content) + "\n")
            elif label == "sub_title":
                md.append((content if content.startswith("#") else "## " + content) + "\n")
            elif label == "equation":
                md.append(content + "\n")
            elif label == "table":
                body = content
                start = body.lower().find("<table")
                md.append(html_table_to_md(body[start:] if start >= 0 else body) + "\n")
            else:  # text, table_caption, unknown
                md.append(content + "\n")
    return "\n".join(md)


def convert_text(data: dict) -> str:
    parts = []
    for page in data["pages"]:
        parts.append(f"\n<!-- page {page['index'] + 1} -->\n")
        parts.append(page["text"].strip() + "\n")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("response")
    ap.add_argument("pdf")
    ap.add_argument("out_dir")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    data = json.loads(Path(args.response).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(args.pdf)

    if data.get("purpose") == "rich_layout_recognition":
        md = convert_rich(data, doc, out_dir, args.dpi)
        name = "rich_layout.md"
    else:
        md = convert_text(data)
        name = "text_recognition.md"
    (out_dir / name).write_text(md, encoding="utf-8")
    print(f"wrote {out_dir / name} ({len(md)} chars)")
    n = len(list((out_dir / "images").glob("*.png"))) if (out_dir / "images").exists() else 0
    print(f"images: {n}")


if __name__ == "__main__":
    main()
