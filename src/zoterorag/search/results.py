from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal


Consumer = Literal["manual", "llm_text", "llm_multimodal"]
ImageReturn = Literal["file_ref", "base64", "none"]


@dataclass(frozen=True)
class SearchResult:
    document_id: str
    chunk_id: str
    title: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    images: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "text": self.text,
            "score": self.score,
            "metadata": dict(self.metadata),
            "images": list(self.images),
        }


def sanitize_results_for_consumer(
    results: list[SearchResult | dict[str, Any]],
    consumer: Consumer,
    image_return: ImageReturn = "none",
    max_images: int = 5,
) -> list[dict[str, Any]]:
    """Apply output safety rules for manual and LLM consumers."""

    sanitized: list[dict[str, Any]] = []
    for result in results:
        item = result.to_dict() if isinstance(result, SearchResult) else dict(result)
        images = list(item.get("images") or [])
        if consumer == "llm_text" or image_return == "none":
            item["text"] = strip_image_references_from_text(str(item.get("text", "")))
            # Pure-text LLM consumers must never receive image bytes or file
            # handles. They can see that images exist and read captions, but the
            # payload remains text-only by construction.
            item.pop("images", None)
            item["has_images"] = bool(images)
            item["image_count"] = len(images)
            captions = [img.get("caption") for img in images if img.get("caption")]
            if captions:
                item["image_captions"] = captions[:max_images]
        elif consumer == "manual":
            item["images"] = [
                {
                    "image_id": img.get("image_id"),
                    "caption": img.get("caption"),
                    "file_ref": img.get("file_ref"),
                    "thumbnail_ref": img.get("thumbnail_ref"),
                }
                for img in images[:max_images]
            ]
        elif consumer == "llm_multimodal":
            if image_return == "base64":
                item["images"] = [
                    {
                        "image_id": img.get("image_id"),
                        "caption": img.get("caption"),
                        "base64": img.get("base64"),
                        "mime_type": img.get("mime_type"),
                    }
                    for img in images[:max_images]
                    if img.get("base64")
                ]
            else:
                item["images"] = [
                    {
                        "image_id": img.get("image_id"),
                        "caption": img.get("caption"),
                        "file_ref": img.get("file_ref"),
                    }
                    for img in images[:max_images]
                ]
        else:
            raise ValueError(f"unknown consumer: {consumer}")
        sanitized.append(item)
    return sanitized


MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
HTML_IMAGE_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)


def strip_image_references_from_text(text: str) -> str:
    def replace_markdown(match: re.Match[str]) -> str:
        caption = match.group(1).strip()
        return f"[Image: {caption}]" if caption else "[Image]"

    text = MARKDOWN_IMAGE_RE.sub(replace_markdown, text)
    return HTML_IMAGE_RE.sub("[Image]", text)
