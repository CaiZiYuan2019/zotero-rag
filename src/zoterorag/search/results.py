from __future__ import annotations

import base64
from dataclasses import dataclass, field
import mimetypes
from pathlib import Path
import re
from typing import Any, Literal


Consumer = Literal["manual", "llm_text", "llm_multimodal"]
ImageReturn = Literal["file_ref", "base64", "none"]


class RerankNotSupportedError(NotImplementedError):
    """Raised when rerank is requested but no RerankProvider is implemented."""


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
    max_image_bytes: int = 256 * 1024,
) -> list[dict[str, Any]]:
    """Apply output safety rules for manual and LLM consumers."""

    sanitized: list[dict[str, Any]] = []
    for result in results:
        item = result.to_dict() if isinstance(result, SearchResult) else dict(result)
        item.setdefault("rerank_score", None)
        images = list(item.get("images") or [])
        if consumer == "llm_text" or image_return == "none":
            item["text"] = strip_image_references_from_text(str(item.get("text", "")))
            item["metadata"] = strip_image_payload_metadata(dict(item.get("metadata") or {}))
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
                    base64_image_payload(img, max_image_bytes=max_image_bytes)
                    for img in images[:max_images]
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


def strip_image_payload_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove image handle fields from metadata sent to text-only consumers."""

    blocked_keys = {
        "image_path",
        "image_embedding_path",
        "embedding_relative_path",
        "image_return",
        "file_ref",
        "thumbnail_ref",
        "base64",
        "mime_type",
    }
    return {key: value for key, value in metadata.items() if key not in blocked_keys}


def base64_image_payload(image: dict[str, Any], *, max_image_bytes: int) -> dict[str, Any]:
    """Return a bounded base64 payload or an explicit omission reason."""

    payload = {
        "image_id": image.get("image_id"),
        "caption": image.get("caption"),
    }
    if image.get("base64"):
        encoded = str(image["base64"])
        decoded_size = decoded_base64_size(encoded)
        if decoded_size > max_image_bytes:
            return {
                **payload,
                "omitted_reason": f"image_bytes_exceed_limit:{decoded_size}>{max_image_bytes}",
            }
        return {
            **payload,
            "base64": encoded,
            "mime_type": image.get("mime_type") or "application/octet-stream",
            "byte_count": decoded_size,
        }

    source = image.get("image_abs_path")
    if not source:
        return {**payload, "omitted_reason": "no_base64_or_local_file"}
    path = Path(str(source))
    if not path.is_file():
        return {**payload, "omitted_reason": "local_file_missing"}
    size = path.stat().st_size
    if size > max_image_bytes:
        return {**payload, "omitted_reason": f"image_bytes_exceed_limit:{size}>{max_image_bytes}"}
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mime_type = image.get("mime_type") or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {**payload, "base64": encoded, "mime_type": mime_type, "byte_count": size}


def decoded_base64_size(value: str) -> int:
    try:
        return len(base64.b64decode(value, validate=True))
    except Exception:
        return len(value.encode("utf-8"))


def ensure_rerank_disabled(rerank: bool) -> None:
    """Reject rerank requests until a real RerankProvider is implemented."""

    if rerank:
        raise RerankNotSupportedError("rerank is reserved but not implemented; use rerank=false")
