from __future__ import annotations

from dataclasses import dataclass
import base64
import binascii
from pathlib import Path
from typing import Any


DEFAULT_MAX_QUERY_IMAGE_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class QueryImage:
    kind: str
    value: str
    mime_type: str | None = None

    @property
    def file_path(self) -> str | None:
        return self.value if self.kind == "file_path" else None

    @property
    def base64_data(self) -> str | None:
        return self.value if self.kind == "base64" else None


def normalize_query_image(
    payload: dict[str, Any] | None,
    *,
    allowed_roots: list[str | Path] | None = None,
    max_bytes: int = DEFAULT_MAX_QUERY_IMAGE_BYTES,
) -> QueryImage | None:
    """Validate a multimodal query image payload.

    API callers can otherwise turn future embedding providers into arbitrary
    local-file readers. When `allowed_roots` is supplied, file queries must stay
    under one of those roots. Base64 input is size-checked but not persisted.
    """

    if payload is None:
        return None
    kind = str(payload.get("type") or payload.get("kind") or "")
    value = str(payload.get("value") or "")
    mime_type = payload.get("mime_type")
    if kind not in {"file_path", "base64"}:
        raise ValueError("query_image.type must be 'file_path' or 'base64'")
    if not value:
        raise ValueError("query_image.value is required")
    if kind == "file_path":
        return QueryImage(
            kind="file_path",
            value=str(validate_query_image_file(value, allowed_roots=allowed_roots)),
            mime_type=str(mime_type) if mime_type else None,
        )
    validate_base64_image(value, max_bytes=max_bytes)
    return QueryImage(kind="base64", value=value, mime_type=str(mime_type) if mime_type else None)


def validate_query_image_file(
    path: str | Path,
    *,
    allowed_roots: list[str | Path] | None = None,
) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"query image file not found: {path}")
    if allowed_roots:
        roots = [Path(root).expanduser().resolve() for root in allowed_roots]
        if not any(is_relative_to(resolved, root) for root in roots):
            raise ValueError("query image file is outside the allowed runtime roots")
    return resolved


def validate_base64_image(value: str, *, max_bytes: int = DEFAULT_MAX_QUERY_IMAGE_BYTES) -> int:
    try:
        decoded = base64.b64decode(value, validate=True)
    except binascii.Error as exc:
        raise ValueError("query_image.value is not valid base64") from exc
    size = len(decoded)
    if size > max_bytes:
        raise ValueError(f"query image exceeds max_bytes={max_bytes}")
    return size


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
