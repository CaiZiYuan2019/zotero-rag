from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import zipfile
from typing import Any, Protocol

from .base import ExtractArtifact, ExtractJobState


APPLY_UPLOAD_URL = "https://mineru.net/api/v4/file-urls/batch"
BATCH_RESULT_URL = "https://mineru.net/api/v4/extract-results/batch/{batch_id}"

REQUEST_TIMEOUT_SECONDS = 60
TRANSFER_TIMEOUT_SECONDS = 300


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
    def __init__(self, message: str, *, stage: str, batch_id: str | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.batch_id = batch_id


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
    ) -> None:
        self.api_key = api_key
        self.client = client or _load_requests_client()
        self.apply_upload_url = apply_upload_url
        self.batch_result_url = batch_result_url
        self.request_timeout_seconds = request_timeout_seconds
        self.transfer_timeout_seconds = transfer_timeout_seconds
        self._source_by_batch_id: dict[str, Path] = {}

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
        payload = self._build_submit_payload(path, options or {})
        headers = self._headers(key)

        # Step 1: ask MinerU for a one-time upload URL and a durable batch id.
        response = self.client.post(
            self.apply_upload_url,
            headers=headers,
            json=payload,
            timeout=self.request_timeout_seconds,
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
        with path.open("rb") as handle:
            upload_response = self.client.put(
                str(upload_urls[0]),
                data=handle,
                timeout=self.transfer_timeout_seconds,
            )
        if upload_response.status_code not in (200, 201):
            raise MinerUAPIError(
                f"PDF upload failed: HTTP {upload_response.status_code} - {upload_response.text[:300]}",
                stage="upload",
                batch_id=batch_id,
            )

        self._source_by_batch_id[batch_id] = path
        return ExtractJobState(external_job_id=batch_id, state="running", message="uploaded")

    def poll(self, external_job_id: str, *, api_key: str | None = None) -> ExtractJobState:
        snapshot = self._poll_snapshot(external_job_id, api_key=api_key)
        if snapshot.state == "done":
            state = "completed"
        elif snapshot.state == "failed":
            state = "failed"
        else:
            state = snapshot.state or "running"
        return ExtractJobState(
            external_job_id=external_job_id,
            state=state,
            message=json.dumps(
                {
                    "mineru_state": snapshot.state,
                    "progress": snapshot.progress,
                    "error_message": snapshot.error_message,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

    def download(self, external_job_id: str, output_dir: Path, *, api_key: str | None = None) -> ExtractArtifact:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot = self._poll_snapshot(external_job_id, api_key=api_key)
        if snapshot.state != "done":
            raise MinerUAPIError(
                f"MinerU batch is not ready for download: {snapshot.state}",
                stage="download",
                batch_id=external_job_id,
            )
        if not snapshot.full_zip_url:
            raise MinerUAPIError("MinerU completed but full_zip_url is missing.", stage="download", batch_id=external_job_id)

        zip_path = output_dir / "mineru_result.zip"
        extract_dir = output_dir / "extract"
        self._download_zip(snapshot.full_zip_url, zip_path, external_job_id)
        extract_dir.mkdir(parents=True, exist_ok=True)
        safe_extract_zip(zip_path, extract_dir)

        markdown_path = find_first_file(extract_dir, "full.md")
        source_pdf = self._source_by_batch_id.get(external_job_id, Path(external_job_id))
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
                    "source_pdf": str(source_pdf),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return ExtractArtifact(source_pdf=source_pdf, artifact_dir=output_dir, manifest_path=manifest_path)

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

    def _poll_snapshot(self, batch_id: str, *, api_key: str | None = None) -> MinerUJobSnapshot:
        url = self.batch_result_url.format(batch_id=batch_id)
        response = self.client.get(
            url,
            headers=self._headers(self._require_api_key(api_key)),
            timeout=self.request_timeout_seconds,
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

    def _download_zip(self, url: str, save_path: Path, batch_id: str) -> None:
        response = self.client.get(url, stream=True, timeout=self.transfer_timeout_seconds)
        if response.status_code != 200:
            raise MinerUAPIError(f"ZIP download failed: HTTP {response.status_code}", stage="download", batch_id=batch_id)
        with save_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

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
            f"MinerU API request failed: HTTP {response.status_code} - {response.text[:500]}",
            stage=stage,
            batch_id=batch_id,
        )
    try:
        body = response.json()
    except Exception as exc:
        raise MinerUAPIError("MinerU API response is not valid JSON.", stage=stage, batch_id=batch_id) from exc
    if body.get("code") != 0:
        raise MinerUAPIError(
            f"MinerU API request failed: {body.get('msg', 'unknown error')}",
            stage=stage,
            batch_id=batch_id,
        )
    return body


def safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    root = output_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise MinerUAPIError(
                    f"Unsafe ZIP member path from MinerU result: {member.filename}",
                    stage="extract",
                ) from exc
        archive.extractall(output_dir)


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
