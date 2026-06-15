from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import shutil
import time
import zipfile
from typing import Any, Callable, Protocol

from .base import ExtractArtifact, ExtractJobState

try:
    from requests.exceptions import ConnectionError as _RequestsConnectionError
    from requests.exceptions import Timeout as _RequestsTimeout
except Exception:  # pragma: no cover - requests is an optional runtime dependency
    _RequestsTimeout = None
    _RequestsConnectionError = None


APPLY_UPLOAD_URL = "https://mineru.net/api/v4/file-urls/batch"
BATCH_RESULT_URL = "https://mineru.net/api/v4/extract-results/batch/{batch_id}"

REQUEST_TIMEOUT_SECONDS = 60
TRANSFER_TIMEOUT_SECONDS = 300

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_INITIAL_SECONDS = 1.0
DEFAULT_RETRY_MAX_SECONDS = 60.0
DEFAULT_RETRY_BACKOFF_FACTOR = 2.0
DEFAULT_RETRY_JITTER = 0.1

_TRANSIENT_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 520})


class MinerUResponse(Protocol):
    status_code: int
    text: str
    headers: dict[str, str]

    def json(self) -> dict[str, Any]:
        ...

    def iter_content(self, chunk_size: int) -> Any:
        ...


class MinerUHTTPClient(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        timeout: int | float | None = None,
    ) -> MinerUResponse:
        ...

    def put(
        self,
        url: str,
        *,
        data: Any = None,
        timeout: int | float | None = None,
    ) -> MinerUResponse:
        ...

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        stream: bool = False,
        timeout: int | float | None = None,
    ) -> MinerUResponse:
        ...


@dataclass(frozen=True)
class MinerUJobSnapshot:
    batch_id: str
    state: str
    full_zip_url: str | None = None
    error_message: str = ""
    progress: dict[str, Any] | None = None


class MinerUAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        batch_id: str | None = None,
        status_code: int | None = None,
        code: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.batch_id = batch_id
        self.status_code = status_code
        self.code = code


class MinerUProvider:
    """MinerU API v4 extractor provider.

    Secrets are accepted only as method arguments or constructor arguments and
    are never written into manifests. The manager stores key aliases separately
    so multi-key scheduling can be audited without leaking API keys.
    """

    name = "mineru"
    version = "api-v4-vlm"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: MinerUHTTPClient | None = None,
        apply_upload_url: str = APPLY_UPLOAD_URL,
        batch_result_url: str = BATCH_RESULT_URL,
        request_timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
        transfer_timeout_seconds: int = TRANSFER_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_initial_seconds: float = DEFAULT_RETRY_INITIAL_SECONDS,
        retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
        retry_backoff_factor: float = DEFAULT_RETRY_BACKOFF_FACTOR,
        retry_jitter: float = DEFAULT_RETRY_JITTER,
    ) -> None:
        self.api_key = api_key
        self.client = client or _load_requests_client()
        self.apply_upload_url = apply_upload_url
        self.batch_result_url = batch_result_url
        self.request_timeout_seconds = request_timeout_seconds
        self.transfer_timeout_seconds = transfer_timeout_seconds
        self.max_retries = max_retries
        self.retry_initial_seconds = retry_initial_seconds
        self.retry_max_seconds = retry_max_seconds
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_jitter = retry_jitter

    def _request_timeout_seconds(self, options: dict[str, Any] | None = None) -> int | float:
        options_timeout = int((options or {}).get("timeout_seconds", 0) or 0)
        return max(self.request_timeout_seconds, options_timeout) if options_timeout > 0 else self.request_timeout_seconds

    def _transfer_timeout_seconds(self, options: dict[str, Any] | None = None) -> int | float:
        options_timeout = int((options or {}).get("timeout_seconds", 0) or 0)
        return max(self.transfer_timeout_seconds, options_timeout) if options_timeout > 0 else self.transfer_timeout_seconds

    def _is_retryable_exception(self, exc: BaseException) -> bool:
        if isinstance(exc, (TimeoutError, OSError)):
            return True
        if _RequestsTimeout is not None and isinstance(exc, (_RequestsTimeout, _RequestsConnectionError)):
            return True
        return False

    def _sleep_backoff(self, delay: float) -> None:
        jitter = random.uniform(-self.retry_jitter, self.retry_jitter) * delay
        time.sleep(max(0.0, delay + jitter))

    def _request_with_retry(
        self,
        request_fn: Callable[[], MinerUResponse],
        *,
        stage: str,
        batch_id: str | None = None,
    ) -> MinerUResponse:
        """Execute a request with exponential backoff for transient failures.

        Non-transient HTTP status codes are returned to the caller so that the
        normal error handling path can raise without retrying. Only the stage,
        status code and batch id are surfaced; request URLs and response bodies
        are never included in exception messages.
        """

        last_exception: BaseException | None = None
        delay = self.retry_initial_seconds
        for attempt in range(self.max_retries):
            try:
                response = request_fn()
            except Exception as exc:
                last_exception = exc
                if not self._is_retryable_exception(exc) or attempt == self.max_retries - 1:
                    raise
                self._sleep_backoff(delay)
                delay = min(delay * self.retry_backoff_factor, self.retry_max_seconds)
                continue
            if response.status_code in _TRANSIENT_HTTP_STATUSES and attempt < self.max_retries - 1:
                self._sleep_backoff(delay)
                delay = min(delay * self.retry_backoff_factor, self.retry_max_seconds)
                continue
            return response
        # All retries exhausted on a retryable exception.
        if last_exception is not None:
            raise last_exception
        raise MinerUAPIError("request failed after retries", stage=stage, batch_id=batch_id)

    def _upload_file(self, path: Path, url: str, options: dict[str, Any] | None = None) -> MinerUResponse:
        with path.open("rb") as handle:
            return self.client.put(
                url,
                data=handle,
                timeout=self._transfer_timeout_seconds(options),
            )

    def fingerprint(self, input_file: Path, options_hash: str) -> str:
        return f"{self.name}:{self.version}:{Path(input_file).name}:{options_hash}"

    def submit(
        self,
        input_file: Path,
        options_hash: str,
        *,
        options: dict[str, Any] | None = None,
        api_key: str | None = None,
    ) -> ExtractJobState:
        key = self._require_api_key(api_key)
        path = Path(input_file)
        options = options or {}
        payload = self._build_submit_payload(path, options)
        headers = self._headers(key)

        # Step 1: ask MinerU for a one-time upload URL and a durable batch id.
        response = self._request_with_retry(
            lambda: self.client.post(
                self.apply_upload_url,
                headers=headers,
                json=payload,
                timeout=self._request_timeout_seconds(options),
            ),
            stage="request-upload",
        )
        body = raise_for_mineru_api_error(response, "request-upload")
        data = body.get("data") or {}
        batch_id = str(data.get("batch_id") or "")
        upload_urls = data.get("file_urls") or []
        if not batch_id:
            raise MinerUAPIError("MinerU did not return batch_id.", stage="request-upload")
        if not upload_urls:
            raise MinerUAPIError("MinerU did not return file_urls.", stage="request-upload", batch_id=batch_id)

        # Step 2: upload the PDF bytes to the signed URL. This URL is transient;
        # only batch_id should be persisted for resume/poll/download.
        upload_response = self._request_with_retry(
            lambda: self._upload_file(path, str(upload_urls[0]), options),
            stage="upload",
            batch_id=batch_id,
        )
        if upload_response.status_code not in (200, 201):
            raise MinerUAPIError(
                "PDF upload failed",
                stage="upload",
                batch_id=batch_id,
                status_code=upload_response.status_code,
            )

        return ExtractJobState(external_job_id=batch_id, state="running", message="uploaded")

    def poll(
        self,
        external_job_id: str,
        *,
        api_key: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ExtractJobState:
        snapshot = self._poll_snapshot(external_job_id, api_key=api_key, options=options)
        if snapshot.state == "done":
            state = "completed"
        elif snapshot.state == "failed":
            state = "failed"
        else:
            state = snapshot.state or "running"
        # Only desensitized fields are persisted; never the upstream error text
        # or any URLs returned by the MinerU API.
        return ExtractJobState(
            external_job_id=external_job_id,
            state=state,
            message=json.dumps(
                {
                    "mineru_state": snapshot.state,
                    "progress": snapshot.progress,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

    def download(
        self,
        external_job_id: str,
        output_dir: Path,
        *,
        api_key: str | None = None,
        source_pdf: Path | str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ExtractArtifact:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot = self._poll_snapshot(external_job_id, api_key=api_key, options=options)
        if snapshot.state != "done":
            raise MinerUAPIError(
                "MinerU batch is not ready for download",
                stage="download",
                batch_id=external_job_id,
            )
        if not snapshot.full_zip_url:
            raise MinerUAPIError("MinerU completed but full_zip_url is missing.", stage="download", batch_id=external_job_id)

        zip_path = output_dir / "mineru_result.zip"
        extract_dir = output_dir / "extract"
        self._download_zip(snapshot.full_zip_url, zip_path, external_job_id, options=options)
        extract_dir.mkdir(parents=True, exist_ok=True)
        safe_extract_zip(zip_path, extract_dir)

        markdown_path = find_first_file(extract_dir, "full.md")
        source_pdf_path = Path(source_pdf) if source_pdf is not None else Path(external_job_id)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "provider": self.name,
                    "provider_version": self.version,
                    "external_job_id": external_job_id,
                    "state": "downloaded",
                    "zip_path": str(zip_path),
                    "extract_dir": str(extract_dir),
                    "markdown_path": str(markdown_path) if markdown_path else None,
                    "source_pdf": str(source_pdf_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return ExtractArtifact(source_pdf=source_pdf_path, artifact_dir=output_dir, manifest_path=manifest_path)

    def _build_submit_payload(self, input_file: Path, options: dict[str, Any]) -> dict[str, Any]:
        page_ranges = str(options.get("page_ranges") or options.get("selected_pages") or "").strip()
        file_payload: dict[str, Any] = {
            "name": input_file.name,
            "data_id": sanitize_data_id(input_file.stem),
        }
        if page_ranges:
            file_payload["page_ranges"] = page_ranges
        return {
            "files": [file_payload],
            "model_version": options.get("model_version", "vlm"),
            "enable_formula": bool(options.get("enable_formula", True)),
            "enable_table": bool(options.get("enable_table", True)),
        }

    def _poll_snapshot(
        self,
        batch_id: str,
        *,
        api_key: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> MinerUJobSnapshot:
        url = self.batch_result_url.format(batch_id=batch_id)
        response = self._request_with_retry(
            lambda: self.client.get(
                url,
                headers=self._headers(self._require_api_key(api_key)),
                timeout=self._request_timeout_seconds(options),
            ),
            stage="poll",
            batch_id=batch_id,
        )
        body = raise_for_mineru_api_error(response, "poll", batch_id=batch_id)
        result = body.get("data") or {}
        extract_results = result.get("extract_result") or []
        if not extract_results:
            return MinerUJobSnapshot(batch_id=batch_id, state="empty-result")
        item = extract_results[0] or {}
        state = str(item.get("state") or "unknown")
        return MinerUJobSnapshot(
            batch_id=batch_id,
            state=state,
            full_zip_url=item.get("full_zip_url"),
            error_message=str(item.get("err_msg") or ""),
            progress=item.get("extract_progress") or None,
        )

    def _download_zip(
        self,
        url: str,
        save_path: Path,
        batch_id: str,
        *,
        options: dict[str, Any] | None = None,
    ) -> None:
        response = self._request_with_retry(
            lambda: self.client.get(
                url,
                stream=True,
                timeout=self._transfer_timeout_seconds(options),
            ),
            stage="download",
            batch_id=batch_id,
        )
        if response.status_code != 200:
            raise MinerUAPIError(
                "ZIP download failed",
                stage="download",
                batch_id=batch_id,
                status_code=response.status_code,
            )
        expected_length = response.headers.get("Content-Length")
        written = 0
        with save_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    written += len(chunk)
        if expected_length is not None and int(expected_length) != written:
            raise MinerUAPIError(
                "ZIP download size mismatch",
                stage="download",
                batch_id=batch_id,
            )
        if not zipfile.is_zipfile(save_path):
            raise MinerUAPIError(
                "ZIP download is not a valid zip file",
                stage="download",
                batch_id=batch_id,
            )

    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _require_api_key(self, api_key: str | None) -> str:
        key = (api_key or self.api_key or "").strip()
        if not key:
            raise MinerUAPIError("MinerU API key is required.", stage="config")
        return key


def raise_for_mineru_api_error(
    response: MinerUResponse,
    stage: str,
    *,
    batch_id: str | None = None,
) -> dict[str, Any]:
    if response.status_code != 200:
        raise MinerUAPIError(
            "MinerU API request failed",
            stage=stage,
            batch_id=batch_id,
            status_code=response.status_code,
        )
    try:
        body = response.json()
    except Exception as exc:
        raise MinerUAPIError("MinerU API response is not valid JSON.", stage=stage, batch_id=batch_id) from exc
    if body.get("code") != 0:
        raise MinerUAPIError(
            "MinerU API returned an error code",
            stage=stage,
            batch_id=batch_id,
            status_code=response.status_code,
            code=body.get("code"),
        )
    return body


def _zip_member_is_symlink(member: zipfile.ZipInfo) -> bool:
    # ZipInfo stores Unix st_mode in the high 16 bits of external_attr.
    # S_IFLNK == 0o120000, so the file-type nibble is 0o12.
    if member.create_system != 3:  # not Unix
        return False
    st_mode = member.external_attr >> 16
    return (st_mode & 0o170000) == 0o120000


def safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            if _zip_member_is_symlink(member):
                raise MinerUAPIError(
                    f"ZIP member is a symlink and not allowed: {member.filename}",
                    stage="extract",
                )
            target = (output_dir / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise MinerUAPIError(
                    f"Unsafe ZIP member path from MinerU result: {member.filename}",
                    stage="extract",
                ) from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def sanitize_data_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text[:128] if text else "pdf_task"


def find_first_file(root: Path, filename: str) -> Path | None:
    for path in root.rglob(filename):
        return path
    return None


def _load_requests_client() -> MinerUHTTPClient:
    try:
        import requests
    except Exception as exc:  # pragma: no cover - depends on optional runtime dependency
        raise MinerUAPIError("requests is required for MinerUProvider.", stage="config") from exc
    return requests
