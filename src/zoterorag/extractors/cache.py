from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_options_hash(options: dict[str, Any] | None = None) -> str:
    encoded = json.dumps(options or {}, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_selected_pages(selected_pages: str | Iterable[int] | None) -> str:
    if selected_pages is None:
        return ""
    if isinstance(selected_pages, str):
        return selected_pages.strip()
    return ",".join(str(page) for page in sorted(set(selected_pages)))


def extractor_cache_key(
    *,
    pdf_sha256: str,
    selected_pages: str | Iterable[int] | None,
    extractor_name: str,
    extractor_version: str,
    options_hash: str,
) -> str:
    payload = {
        "pdf_sha256": pdf_sha256,
        "selected_pages": canonical_selected_pages(selected_pages),
        "extractor_name": extractor_name,
        "extractor_version": extractor_version,
        "options_hash": options_hash,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def recommended_mineru_timeout_seconds(selected_page_count: int) -> int:
    """Conservative MinerU timeout.

    MinerU conversion latency varies heavily by document layout and service
    load. The project uses pages*6+30 rather than the older pages*2+30 rule so
    long-running jobs are not marked failed while the API is still working.
    """

    if selected_page_count <= 0:
        raise ValueError("selected_page_count must be positive")
    return selected_page_count * 6 + 30
