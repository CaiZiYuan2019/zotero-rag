# -*- coding: utf-8 -*-
"""Shared core for PaddleOCR-VL-1.6 GUI/CLI.

Hard-coded token/model are kept in sync with `ref.py`.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

# In frozen builds, make sure bundled OpenSSL DLLs are on PATH before loading _ssl.
if hasattr(sys, "_MEIPASS"):
    _meipass = sys._MEIPASS
    _path = os.environ.get("PATH", "")
    if _meipass not in _path.split(os.pathsep):
        os.environ["PATH"] = _meipass + os.pathsep + _path

import requests  # noqa: E402

# Ensure charset detection dependency is available for requests in frozen builds.
try:
    import charset_normalizer  # noqa: F401
except Exception:  # pragma: no cover
    pass


JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
TOKEN = "TOKEN"
MODEL = "PaddleOCR-VL-1.6"

REQUEST_TIMEOUT = 60
UPLOAD_TIMEOUT = 300
POLL_TIMEOUT = 60
DOWNLOAD_TIMEOUT = 300


class PaddleOCRError(RuntimeError):
    def __init__(self, message: str, stage: str, job_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.job_id = job_id


def load_api_key() -> str:
    """Return the hard-coded token (matching ref.py style)."""
    return TOKEN


def build_optional_payload(
    orientation: bool = False,
    unwarp: bool = False,
    chart: bool = False,
) -> dict[str, Any]:
    return {
        "useDocOrientationClassify": orientation,
        "useDocUnwarping": unwarp,
        "useChartRecognition": chart,
    }


def submit_job(
    file_path_or_url: str,
    optional_payload: dict[str, Any],
    is_url: bool = False,
    api_key: Optional[str] = None,
) -> str:
    """Submit a job and return jobId."""
    token = api_key or load_api_key()
    headers = {"Authorization": f"bearer {token}"}

    if is_url:
        headers["Content-Type"] = "application/json"
        payload = {
            "fileUrl": file_path_or_url,
            "model": MODEL,
            "optionalPayload": optional_payload,
        }
        try:
            resp = requests.post(JOB_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        except Exception as exc:
            raise PaddleOCRError(f"Submit URL job failed: {exc}", "submit") from exc
    else:
        local_path = Path(file_path_or_url)
        if not local_path.exists():
            raise PaddleOCRError(f"File not found: {local_path}", "validation")

        data = {
            "model": MODEL,
            "optionalPayload": json.dumps(optional_payload),
        }
        try:
            with open(local_path, "rb") as f:
                files = {"file": (local_path.name, f)}
                resp = requests.post(
                    JOB_URL,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=UPLOAD_TIMEOUT,
                )
        except Exception as exc:
            raise PaddleOCRError(f"Submit local job failed: {exc}", "submit") from exc

    if resp.status_code != 200:
        raise PaddleOCRError(
            f"Submit failed: HTTP {resp.status_code} - {resp.text[:500]}",
            "submit",
        )

    try:
        body = resp.json()
    except Exception as exc:
        raise PaddleOCRError(f"Submit response is not valid JSON: {exc}", "submit") from exc

    data = body.get("data") or {}
    job_id = data.get("jobId")
    if not job_id:
        raise PaddleOCRError(f"Submit did not return jobId: {body}", "submit")
    return job_id


def poll_job(
    job_id: str,
    timeout: int = 300,
    interval: int = 5,
    api_key: Optional[str] = None,
    stop_event: Optional[Any] = None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Poll job until done/failed/timeout and return the data dict."""
    token = api_key or load_api_key()
    headers = {"Authorization": f"bearer {token}"}
    url = f"{JOB_URL}/{job_id}"
    deadline = time.monotonic() + timeout

    def log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    while time.monotonic() < deadline:
        if stop_event and getattr(stop_event, "is_set", lambda: False)():
            raise PaddleOCRError("Polling stopped by user.", "poll", job_id)

        try:
            resp = requests.get(url, headers=headers, timeout=POLL_TIMEOUT)
        except Exception as exc:
            raise PaddleOCRError(f"Poll failed: {exc}", "poll", job_id) from exc

        if resp.status_code != 200:
            raise PaddleOCRError(
                f"Poll failed: HTTP {resp.status_code} - {resp.text[:500]}",
                "poll",
                job_id,
            )

        try:
            body = resp.json()
        except Exception as exc:
            raise PaddleOCRError(f"Poll response is not valid JSON: {exc}", "poll", job_id) from exc

        data = body.get("data") or {}
        state = data.get("state", "")

        if state == "pending":
            log("状态：pending（排队中）")
        elif state == "running":
            progress = data.get("extractProgress") or {}
            total = progress.get("totalPages", "?")
            extracted = progress.get("extractedPages", 0)
            log(f"状态：running（已解析 {extracted}/{total} 页）")
        elif state == "done":
            log("状态：done")
            return data
        elif state == "failed":
            error_msg = data.get("errorMsg") or "未知错误"
            raise PaddleOCRError(f"Job failed: {error_msg}", "convert", job_id)
        else:
            log(f"状态：{state or 'unknown'}")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))

    raise PaddleOCRError(f"Timed out after {timeout}s waiting for conversion.", "timeout", job_id)


def download_jsonl(jsonl_url: str) -> list[dict[str, Any]]:
    """Download and parse JSONL result."""
    try:
        resp = requests.get(jsonl_url, timeout=DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        raise PaddleOCRError(f"Download JSONL failed: {exc}", "download") from exc

    results: list[dict[str, Any]] = []
    for line in resp.text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as exc:
            raise PaddleOCRError(f"Parse JSONL failed: {exc}", "parse") from exc
        results.append(obj)
    return results


def save_jsonl_results(results: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    """Save markdown and images from JSONL results, and concatenate into full.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_md: list[str] = []
    saved_images: list[str] = []
    failed_images: list[str] = []
    page_num = 0
    full_md_parts: list[str] = []

    for obj in results:
        result = obj.get("result") or {}
        layout_results = result.get("layoutParsingResults") or []
        if not layout_results:
            continue

        for res in layout_results:
            markdown = res.get("markdown") or {}
            text = markdown.get("text", "")
            images = markdown.get("images") or {}
            output_images = res.get("outputImages") or {}

            md_path = output_dir / f"doc_{page_num:04d}.md"
            md_path.write_text(text, encoding="utf-8")
            saved_md.append(str(md_path))
            full_md_parts.append(text)

            for img_path, img_url in images.items():
                try:
                    full_img_path = output_dir / img_path
                    full_img_path.parent.mkdir(parents=True, exist_ok=True)
                    img_resp = requests.get(img_url, timeout=DOWNLOAD_TIMEOUT)
                    img_resp.raise_for_status()
                    full_img_path.write_bytes(img_resp.content)
                    saved_images.append(str(full_img_path))
                except Exception as exc:
                    failed_images.append(f"{img_path}: {exc}")

            for img_name, img_url in output_images.items():
                try:
                    filename = output_dir / f"{img_name}_{page_num}.jpg"
                    img_resp = requests.get(img_url, timeout=DOWNLOAD_TIMEOUT)
                    img_resp.raise_for_status()
                    filename.write_bytes(img_resp.content)
                    saved_images.append(str(filename))
                except Exception as exc:
                    failed_images.append(f"{img_name}: {exc}")

            page_num += 1

    full_md_path: Optional[Path] = None
    if full_md_parts:
        full_md_path = output_dir / "full.md"
        full_md_path.write_text("\n\n".join(full_md_parts), encoding="utf-8")

    return {
        "markdown_files": saved_md,
        "full_md_path": str(full_md_path) if full_md_path else None,
        "saved_images": saved_images,
        "failed_images": failed_images,
        "page_count": page_num,
    }


def process_job(
    file_path_or_url: str,
    output_dir: Path,
    optional_payload: dict[str, Any],
    is_url: bool = False,
    timeout: int = 300,
    interval: int = 5,
    api_key: Optional[str] = None,
    stop_event: Optional[Any] = None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """End-to-end: submit, poll, download JSONL, save results."""
    log = log_cb or (lambda _: None)

    log(f"提交任务：{file_path_or_url}")
    job_id = submit_job(file_path_or_url, optional_payload, is_url=is_url, api_key=api_key)
    log(f"job_id = {job_id}")

    log("轮询解析状态...")
    data = poll_job(job_id, timeout=timeout, interval=interval, api_key=api_key, stop_event=stop_event, log_cb=log)

    result_url = data.get("resultUrl") or {}
    jsonl_url = result_url.get("jsonUrl")
    if not jsonl_url:
        raise PaddleOCRError("Job done but no jsonUrl returned.", "download", job_id)

    log(f"下载结果：{jsonl_url}")
    results = download_jsonl(jsonl_url)
    log(f"解析到 {len(results)} 行结果")

    log("保存结果...")
    save_info = save_jsonl_results(results, output_dir)

    return {
        "ok": True,
        "job_id": job_id,
        "state": "done",
        "output_dir": str(output_dir),
        **save_info,
    }
