"""Convert DeepSeek layout markers, HTML tables and PDF crops to MinerU artifacts.

Adapted from reference/deepseek_ocr_rich.py. API indices refer to the uploaded
PDF; the page map records the corresponding indices in the original PDF.
"""

from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import re

from ..normalize.markdown import MARKDOWN_IMAGE_RE

REF = re.compile(r"<｜ref｜>(.*?)<｜/ref｜>\s*<｜box｜>\s*\[\[([-\d.,\s]+)\]\]\s*<｜/box｜>", re.S)


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
    lines += [
        "| " + " | ".join(esc(c) for c in rows[0]) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def strip_tags(s: str) -> str:
    s = re.sub(r"</?center>", "", s, flags=re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    return s.strip()


def parse_page(text: str):
    matches = list(REF.finditer(text))
    prefix = text[: matches[0].start()] if matches else text
    if prefix.strip():
        yield "text", None, prefix.strip()
    for i, match in enumerate(matches):
        label = match[1].strip().lower()
        box = [float(value) for value in match[2].split(",")]
        if len(box) != 4 or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
            raise ValueError("Invalid DeepSeek OCR layout box")
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield label, box, text[match.end() : end].strip()


def _document_text(content: str, image_targets: set[str]) -> str:
    # The normalizer treats all image references as local file paths. Only the
    # converter can create a legitimate image reference from an OCR response.
    content = MARKDOWN_IMAGE_RE.sub(lambda m: m[0] if m[2] in image_targets else m[1][2:-2], content)
    return re.sub(r"<img\b[^>]*(?:>|$)", "", content, flags=re.IGNORECASE)


def convert_pages(pages: list[dict], doc, source_indices: list[int], out_dir: Path, dpi: int) -> None:
    """Write full.md, crops, and page-local MinerU content blocks in reading order."""
    import fitz

    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    markdown = []
    blocks = []
    image_targets: set[str] = set()
    for page in pages:
        idx = page["index"]
        original_idx = source_indices[idx]
        number = original_idx + 1
        figure = 0
        for label, box, content in parse_page(page["text"]):
            content = strip_tags(content)
            block = {"page_idx": idx, "source_page_idx": original_idx, "type": "text"}
            if box is not None:
                block["bbox"] = box
            if label == "image":
                figure += 1
                name = f"page_{number:03d}_img_{figure}.png"
                # Crop on the displayed page, including its PDF rotation.
                pdf_page = doc[idx]
                rect = pdf_page.rect
                clip = fitz.Rect(
                    box[0] * rect.width / 1000,
                    box[1] * rect.height / 1000,
                    box[2] * rect.width / 1000,
                    box[3] * rect.height / 1000,
                )
                pdf_page.get_pixmap(dpi=dpi, clip=clip, alpha=False).save(images / name)
                target = f"images/{name}"
                image_targets.add(target)
                block.update(type="image", img_path=target)
                paragraph = f"![page {number} figure {figure}]({target})"
            elif label in {"title", "sub_title"}:
                level = 1 if label == "title" else 2
                block.update(text=content.lstrip("# "), text_level=level)
                paragraph = content if content.startswith("#") else f"{'#' * level} {content}"
            elif label == "table":
                block.update(type="table", table_body=content)
                paragraph = html_table_to_md(content)
            elif label == "equation":
                block.update(type="equation", text=content)
                paragraph = content
            else:
                block["text"] = content
                paragraph = f"*{content}*" if label == "image_caption" and content else content
            markdown.append(paragraph)
            blocks.append(block)
    (out_dir / "deepseek_content_list.json").write_text(
        json.dumps(blocks, ensure_ascii=False), encoding="utf-8"
    )
    text = _document_text("\n\n".join(markdown), image_targets)
    (out_dir / "full.md").write_text(text + "\n", encoding="utf-8", newline="\n")
