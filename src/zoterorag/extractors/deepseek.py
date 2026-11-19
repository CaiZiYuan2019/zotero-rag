"""DeepSeek's synchronous OCR endpoint, adapted to the extraction job lifecycle."""

from __future__ import annotations

from itertools import groupby
import json
import math
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any

from filelock import FileLock

from .base import ExtractArtifact, ExtractJobState
from .cache import canonical_selected_pages, sha256_file, stable_options_hash
from .deepseek_converter import convert_pages

ENDPOINT_URL = "https://api.deepseek.com/v1/tools/ocr"
DEFAULT_PURPOSE = "rich_layout_recognition"
MAX_PAGES = 1000
MAX_BYTES = 100 * 1024 * 1024
# PyMuPDF does not support concurrent access even to independent documents.
# Only local PDF work is serialized; separate OCR requests may run concurrently.
_pdf_lock = Lock()


class DeepSeekOCRError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class DeepSeekIncompleteResult(DeepSeekOCRError):
    """A persisted OCR result is truncated and requires manual review."""


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _source_indices(selection: str, count: int) -> list[int]:
    if not selection:
        if not 1 <= count <= MAX_PAGES:
            raise ValueError("DeepSeek OCR accepts 1 to 1000 pages per upload")
        return list(range(count))
    pages = set()
    for part in selection.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", part)
        if not match:
            raise ValueError("Invalid selected PDF page range")
        first = int(match[1])
        last = int(match[2] or first)
        if not 1 <= first <= last <= count:
            raise ValueError("Selected pages outside source PDF")
        if last - first + 1 > MAX_PAGES:
            raise ValueError("DeepSeek OCR accepts at most 1000 pages per upload")
        pages.update(range(first - 1, last))
        if len(pages) > MAX_PAGES:
            raise ValueError("DeepSeek OCR accepts at most 1000 pages per upload")
    return sorted(pages)


def _prepare_upload(source: Path, upload: Path, selection: str, source_hash: str) -> list[int]:
    import fitz

    tmp = upload.with_suffix(".tmp")
    with _pdf_lock, fitz.open(source) as doc:
        if not doc.is_pdf or doc.needs_pass:
            raise ValueError("DeepSeek OCR requires an unlocked PDF")
        indices = _source_indices(selection, doc.page_count)
        if selection:
            # Copy only the requested pages. Do not inherit document-level attachments.
            with fitz.open() as subset:
                for _, run in groupby(enumerate(indices), key=lambda item: item[1] - item[0]):
                    contiguous = [index for _, index in run]
                    subset.insert_pdf(doc, from_page=contiguous[0], to_page=contiguous[-1])
                subset.save(tmp, garbage=4, deflate=True)
        else:
            shutil.copyfile(source, tmp)
    if tmp.stat().st_size > MAX_BYTES:
        raise ValueError("DeepSeek OCR upload exceeds 100 MiB")
    if sha256_file(source if selection else tmp) != source_hash:
        raise ValueError("Source PDF changed after job creation")
    tmp.replace(upload)
    return indices


def _checked_pages(body: Any, purpose: str, count: int) -> list[dict]:
    if not isinstance(body, dict) or "error" in body:
        raise DeepSeekOCRError("DeepSeek OCR returned an error response")
    if body.get("purpose") != purpose:
        raise DeepSeekOCRError("DeepSeek OCR response purpose mismatch")
    pages = body.get("pages")
    if (
        type(body.get("page_count")) is not int
        or body["page_count"] != count
        or not isinstance(pages, list)
        or len(pages) != count
    ):
        raise DeepSeekOCRError("DeepSeek OCR returned incomplete page coverage")
    seen = set()
    for page in pages:
        if not isinstance(page, dict) or type(page.get("index")) is not int:
            raise DeepSeekOCRError("Invalid DeepSeek OCR page index")
        idx = page["index"]
        if not 0 <= idx < count or idx in seen:
            raise DeepSeekOCRError("Duplicate or out-of-range DeepSeek OCR page index")
        if not isinstance(page.get("text"), str) or page.get("finish_reason") not in ("stop", "length"):
            raise DeepSeekOCRError("Invalid DeepSeek OCR page result")
        seen.add(idx)
    return sorted(pages, key=lambda p: p["index"])


class DeepSeekOCRProvider:
    name = "deepseek"
    version = "ocr-v1-rich-1"

    def __init__(self, *, api_key: str | None = None, endpoint_url: str = ENDPOINT_URL, client=None):
        self.api_key = api_key
        self.endpoint = endpoint_url
        self.client = client

    def fingerprint(self, input_file: Path, options_hash: str) -> str:
        return f"{self.name}:{sha256_file(input_file)}:{options_hash}"

    def submit(self, input_file: Path, options_hash: str, *, options=None, api_key=None) -> ExtractJobState:
        opts = options or {}
        selection = canonical_selected_pages(opts.get("page_ranges") or opts.get("selected_pages"))
        if selection and not re.fullmatch(r"[\d\s,\-]+", selection):
            raise ValueError("Invalid selected PDF page range")
        # The legacy manager persists the page range separately from download options.
        job_id = f"{self.fingerprint(input_file, options_hash)}:{selection}"
        return ExtractJobState(external_job_id=job_id, state="completed")

    def poll(self, external_job_id: str, *, api_key=None, options=None) -> ExtractJobState:
        return ExtractJobState(external_job_id=external_job_id, state="completed")

    def _post(self, upload: Path, purpose: str, key: str | None, timeout: float):
        key = (key or self.api_key or "").strip()
        if not key:
            raise DeepSeekOCRError("DeepSeek OCR API key is required")
        client = self.client
        if client is None:
            import requests as client
        with upload.open("rb") as fp:
            response = client.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {key}"},
                files={"file": ("document.pdf", fp, "application/pdf")},
                data={"purpose": purpose},
                timeout=(30, timeout),
            )
        if response.status_code != 200:
            raise DeepSeekOCRError(
                f"DeepSeek OCR HTTP {response.status_code}", status_code=response.status_code
            )
        try:
            return response.json()
        except ValueError:
            raise DeepSeekOCRError("DeepSeek OCR response is not valid JSON") from None

    def download(
        self, external_job_id: str, output_dir: Path, *, api_key=None, source_pdf=None, options=None
    ) -> ExtractArtifact:
        if source_pdf is None:
            raise ValueError("DeepSeek OCR requires a source PDF path")
        name, source_hash, _, selection = external_job_id.split(":", 3)
        if name != self.name or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
            raise ValueError("Invalid DeepSeek OCR job identifier")
        opts = options or {}
        purpose = opts.get("purpose", DEFAULT_PURPOSE)
        dpi = opts.get("dpi", 200)
        timeout = float(opts.get("timeout_seconds", 210))
        if (
            purpose not in (DEFAULT_PURPOSE, "text_recognition")
            or type(dpi) is not int
            or dpi <= 0
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Invalid DeepSeek OCR request options")
        source = Path(source_pdf)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with FileLock(str(output / ".deepseek.lock")):
            request_path = output / "request.json"
            response_path = output / "response.json"
            upload = output / "upload.pdf"
            identity = {
                "external_job_id": external_job_id,
                "source_pdf_sha256": source_hash,
                "endpoint_hash": stable_options_hash({"endpoint": self.endpoint}),
                "purpose": purpose,
            }
            if request_path.is_file():
                request = json.loads(request_path.read_text(encoding="utf-8"))
                if not all(request.get(k) == v for k, v in identity.items()):
                    raise DeepSeekOCRError("Cached DeepSeek OCR request differs from this job")
                if sha256_file(upload) != request["uploaded_pdf_sha256"]:
                    raise DeepSeekOCRError("Cached DeepSeek OCR upload checksum mismatch")
            else:
                if response_path.exists():
                    raise DeepSeekOCRError("Missing request provenance for cached DeepSeek OCR response")
                indices = _prepare_upload(source, upload, selection, source_hash)
                request = {
                    **identity,
                    "source_pdf": str(source),
                    "uploaded_pdf": str(upload),
                    "uploaded_pdf_sha256": sha256_file(upload),
                    "selected_pages": selection,
                    "selected_original_page_indices": indices,
                }
                _write_json(request_path, request)
            indices = request["selected_original_page_indices"]
            if response_path.is_file():
                body = json.loads(response_path.read_text(encoding="utf-8"))
                pages = _checked_pages(body, purpose, len(indices))
            else:
                body = self._post(upload, purpose, api_key, timeout)
                pages = _checked_pages(body, purpose, len(indices))
                _write_json(response_path, body)
            incomplete = [indices[p["index"]] + 1 for p in pages if p["finish_reason"] == "length"]
            if incomplete:
                raise DeepSeekIncompleteResult(
                    f"DeepSeek OCR finish_reason=length on source pages {incomplete}"
                )

            import fitz

            extract_dir = output / "extract"
            with TemporaryDirectory(dir=output, prefix=".convert-") as staging:
                stage = Path(staging)
                with _pdf_lock, fitz.open(upload) as doc:
                    convert_pages(pages, doc, indices, stage, dpi)
                extract_dir.mkdir(exist_ok=True)
                images_dir = extract_dir / "images"
                images_dir.mkdir(exist_ok=True)
                for image in (stage / "images").iterdir():
                    image.replace(images_dir / image.name)
                for filename in ("deepseek_content_list.json", "full.md"):
                    (stage / filename).replace(extract_dir / filename)
            manifest_path = output / "manifest.json"
            _write_json(
                manifest_path,
                {
                    **request,
                    "provider": self.name,
                    "provider_version": self.version,
                    "response_id": body.get("id"),
                    "request_path": str(request_path),
                    "response_path": str(response_path),
                    "state": "downloaded",
                    "dpi": dpi,
                    "extract_dir": str(extract_dir),
                    "markdown_path": str(extract_dir / "full.md"),
                    "content_list_path": str(extract_dir / "deepseek_content_list.json"),
                },
            )
            return ExtractArtifact(source_pdf=source, artifact_dir=output, manifest_path=manifest_path)
