#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal MinerU PDF-to-Markdown CLI.

Stdout is reserved for a single JSON object. Progress and diagnostics go to
stderr so agents can parse the result reliably.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

try:
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover - reported at runtime
    fitz = None
    FITZ_IMPORT_ERROR = exc
else:
    FITZ_IMPORT_ERROR = None


APPLY_UPLOAD_URL = "https://mineru.net/api/v4/file-urls/batch"
BATCH_RESULT_URL = "https://mineru.net/api/v4/extract-results/batch/{batch_id}"

MAX_SELECTED_PAGES = 200
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_POLL_INTERVAL_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 60
TRANSFER_TIMEOUT_SECONDS = 300


@dataclass
class ConversionConfig:
    pdf_path: Path
    output_dir: Path
    page_ranges: str
    total_pages: int
    selected_pages: list[int]
    api_key: str
    timeout_seconds: int
    poll_interval_seconds: int


class MinerUCLIError(RuntimeError):
    def __init__(self, message: str, stage: str, batch_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.batch_id = batch_id


class ProgressFile:
    def __init__(self, file_path: Path, chunk_size: int = 1024 * 256) -> None:
        self.file_path = file_path
        self.fp = open(file_path, "rb")
        self.total = file_path.stat().st_size
        self.sent = 0
        self.chunk_size = chunk_size
        self.last_percent = -1

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            size = self.chunk_size
        data = self.fp.read(size)
        if data:
            self.sent += len(data)
            self._log_progress()
        return data

    def __len__(self) -> int:
        return self.total

    def close(self) -> None:
        self.fp.close()

    def _log_progress(self) -> None:
        if self.total <= 0:
            return
        percent = int(self.sent * 100 / self.total)
        if percent >= 100 or percent - self.last_percent >= 10:
            self.last_percent = percent
            log(f"upload {percent}% ({format_bytes(self.sent)}/{format_bytes(self.total)})")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_api_key(cli_api_key: Optional[str]) -> str:
    if cli_api_key:
        return cli_api_key.strip()

    env_key = os_getenv("MINERU_API_KEY")
    if env_key:
        return env_key.strip()

    mineru_py = Path(__file__).with_name("MinerU.py")
    if mineru_py.exists():
        try:
            tree = ast.parse(mineru_py.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                if not any(isinstance(t, ast.Name) and t.id == "MINERU_API_KEY" for t in node.targets):
                    continue
                value = ast.literal_eval(node.value)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        except Exception:
            pass

    raise MinerUCLIError(
        "Missing API key. Pass --api-key, set MINERU_API_KEY, or define MINERU_API_KEY in scripts/MinerU.py.",
        "config",
    )


def os_getenv(name: str) -> Optional[str]:
    import os

    return os.environ.get(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a local PDF to Markdown through MinerU API. Uses vlm, auto language, formula/table enabled.",
    )
    parser.add_argument("pdf", help="Path to the PDF file.")
    parser.add_argument("--out", default=".", help="Output directory. Default: current directory.")
    parser.add_argument("--pages", default="", help='Optional page ranges, e.g. "1-5,8,10--1". Empty means all pages.')
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Seconds to wait for MinerU conversion after upload. Recommended: selected_pages*2+30. Default: 300.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help="Seconds between status checks. Default: 5.",
    )
    parser.add_argument("--api-key", default=None, help="MinerU API key. Defaults to MINERU_API_KEY or scripts/MinerU.py.")
    return parser


def validate_config(args: argparse.Namespace) -> ConversionConfig:
    if fitz is None:
        raise MinerUCLIError(f"PyMuPDF import failed: {FITZ_IMPORT_ERROR}", "config")

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise MinerUCLIError(f"PDF does not exist: {pdf_path}", "validation")
    if pdf_path.suffix.lower() != ".pdf":
        raise MinerUCLIError(f"Only PDF files are supported: {pdf_path}", "validation")

    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with fitz.open(str(pdf_path)) as doc:
            total_pages = doc.page_count
    except Exception as exc:
        raise MinerUCLIError(f"Failed to read PDF page count: {exc}", "validation") from exc

    selected_pages = parse_page_ranges(args.pages, total_pages)
    if not selected_pages:
        raise MinerUCLIError("Page range selected no pages.", "validation")
    if len(selected_pages) > MAX_SELECTED_PAGES:
        raise MinerUCLIError(
            f"Selected {len(selected_pages)} pages, above {MAX_SELECTED_PAGES}. Use --pages to split the PDF.",
            "validation",
        )
    if total_pages > MAX_SELECTED_PAGES and not args.pages.strip():
        raise MinerUCLIError(
            f"PDF has {total_pages} pages, above {MAX_SELECTED_PAGES}. Use --pages to select a smaller range.",
            "validation",
        )
    if args.timeout <= 0:
        raise MinerUCLIError("--timeout must be positive.", "validation")
    if args.poll_interval <= 0:
        raise MinerUCLIError("--poll-interval must be positive.", "validation")

    return ConversionConfig(
        pdf_path=pdf_path,
        output_dir=output_dir,
        page_ranges=args.pages.strip(),
        total_pages=total_pages,
        selected_pages=selected_pages,
        api_key=load_api_key(args.api_key),
        timeout_seconds=args.timeout,
        poll_interval_seconds=args.poll_interval,
    )


def convert(cfg: ConversionConfig) -> dict[str, Any]:
    started = time.monotonic()
    batch_id: Optional[str] = None
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "files": [
            {
                "name": cfg.pdf_path.name,
                "data_id": sanitize_data_id(cfg.pdf_path.stem),
            }
        ],
        "model_version": "vlm",
        "enable_formula": True,
        "enable_table": True,
    }
    if cfg.page_ranges:
        payload["files"][0]["page_ranges"] = cfg.page_ranges

    log("1/4 requesting upload URL")
    try:
        response = requests.post(APPLY_UPLOAD_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except Exception as exc:
        raise MinerUCLIError(f"Upload URL request failed: {exc}", "request-upload") from exc
    raise_for_api_error(response, "Upload URL request failed", "request-upload")
    data = response.json()["data"]
    batch_id = data["batch_id"]
    upload_urls = data.get("file_urls") or []
    if not upload_urls:
        raise MinerUCLIError("MinerU did not return file_urls.", "request-upload", batch_id)

    log(f"batch_id={batch_id}")
    log("2/4 uploading PDF")
    progress_file = ProgressFile(cfg.pdf_path)
    try:
        try:
            put_response = requests.put(upload_urls[0], data=progress_file, timeout=TRANSFER_TIMEOUT_SECONDS)
        except Exception as exc:
            raise MinerUCLIError(f"PDF upload failed: {exc}", "upload", batch_id) from exc
        if put_response.status_code not in (200, 201):
            raise MinerUCLIError(
                f"PDF upload failed: HTTP {put_response.status_code} - {put_response.text[:300]}",
                "upload",
                batch_id,
            )
    finally:
        progress_file.close()

    log("3/4 polling conversion")
    result_url = BATCH_RESULT_URL.format(batch_id=batch_id)
    deadline = time.monotonic() + cfg.timeout_seconds
    state = "unknown"
    full_zip_url: Optional[str] = None

    while time.monotonic() < deadline:
        try:
            poll_response = requests.get(result_url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        except Exception as exc:
            raise MinerUCLIError(f"Status request failed: {exc}", "poll", batch_id) from exc
        raise_for_api_error(poll_response, "Status request failed", "poll", batch_id)
        result = poll_response.json()["data"]
        extract_result = result.get("extract_result") or []
        if not extract_result:
            log("state=empty-result")
            sleep_until_next_poll(cfg.poll_interval_seconds, deadline)
            continue

        item = extract_result[0]
        state = item.get("state") or "unknown"
        if state == "running":
            progress = item.get("extract_progress") or {}
            log(f"state=running pages={progress.get('extracted_pages', 0)}/{progress.get('total_pages', '?')}")
        else:
            log(f"state={state}")

        if state == "done":
            full_zip_url = item.get("full_zip_url")
            break
        if state == "failed":
            raise MinerUCLIError(f"MinerU conversion failed: {item.get('err_msg') or 'unknown error'}", "convert", batch_id)

        sleep_until_next_poll(cfg.poll_interval_seconds, deadline)

    if state != "done":
        raise MinerUCLIError(f"Timed out after {cfg.timeout_seconds}s waiting for conversion. Last state: {state}", "timeout", batch_id)
    if not full_zip_url:
        raise MinerUCLIError("Conversion completed but full_zip_url is missing.", "download", batch_id)

    log("4/4 downloading and extracting result")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = sanitize_data_id(cfg.pdf_path.stem)
    zip_path = cfg.output_dir / f"{base_name}_mineru_{timestamp}.zip"
    extract_dir = cfg.output_dir / f"{base_name}_mineru_{timestamp}"
    download_file(full_zip_url, zip_path, batch_id)
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except Exception as exc:
        raise MinerUCLIError(f"Failed to extract ZIP: {exc}", "extract", batch_id) from exc

    markdown_path = find_first_file(extract_dir, "full.md")
    elapsed = time.monotonic() - started
    return {
        "ok": True,
        "batch_id": batch_id,
        "state": "done",
        "page_count": cfg.total_pages,
        "selected_pages": len(cfg.selected_pages),
        "zip_path": str(zip_path),
        "extract_dir": str(extract_dir),
        "markdown_path": str(markdown_path) if markdown_path else None,
        "elapsed_seconds": round(elapsed, 2),
        "message": "completed" if markdown_path else "completed but full.md was not found",
    }


def raise_for_api_error(response: requests.Response, context: str, stage: str, batch_id: Optional[str] = None) -> None:
    if response.status_code != 200:
        raise MinerUCLIError(f"{context}: HTTP {response.status_code} - {response.text[:500]}", stage, batch_id)
    try:
        body = response.json()
    except Exception as exc:
        raise MinerUCLIError(f"{context}: response is not valid JSON.", stage, batch_id) from exc
    if body.get("code") != 0:
        raise MinerUCLIError(f"{context}: {body.get('msg', 'unknown error')}", stage, batch_id)


def sleep_until_next_poll(interval: int, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(min(interval, remaining))


def download_file(url: str, save_path: Path, batch_id: str) -> None:
    try:
        with requests.get(url, stream=True, timeout=TRANSFER_TIMEOUT_SECONDS) as response:
            if response.status_code != 200:
                raise MinerUCLIError(f"ZIP download failed: HTTP {response.status_code}", "download", batch_id)
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            last_percent = -1
            with open(save_path, "wb") as fp:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    fp.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        percent = int(downloaded * 100 / total)
                        if percent >= 100 or percent - last_percent >= 10:
                            last_percent = percent
                            log(f"download {percent}% ({format_bytes(downloaded)}/{format_bytes(total)})")
    except MinerUCLIError:
        raise
    except Exception as exc:
        raise MinerUCLIError(f"ZIP download failed: {exc}", "download", batch_id) from exc


def deep_remove_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: deep_remove_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [deep_remove_none(v) for v in obj if v is not None]
    return obj


def sanitize_data_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text[:128] if text else "pdf_task"


def find_first_file(root: Path, filename: str) -> Optional[Path]:
    for path in root.rglob(filename):
        return path
    return None


def format_bytes(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{num}B"


def expand_signed_page_index(value: str, total_pages: int) -> int:
    n = int(value)
    if n == 0:
        raise MinerUCLIError("Page number cannot be 0.", "validation")
    page = n if n > 0 else total_pages + n + 1
    if page < 1 or page > total_pages:
        raise MinerUCLIError(f"Page number out of range: {value} (PDF has {total_pages} pages)", "validation")
    return page


def parse_page_ranges(page_ranges: str, total_pages: int) -> list[int]:
    if total_pages <= 0:
        return []

    text = (page_ranges or "").strip()
    if not text:
        return list(range(1, total_pages + 1))

    selected: set[int] = set()
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens:
        raise MinerUCLIError("Invalid page range.", "validation")

    range_pattern = re.compile(r"^([+-]?\d+)\s*-\s*([+-]?\d+)$")
    single_pattern = re.compile(r"^[+-]?\d+$")

    for token in tokens:
        if single_pattern.fullmatch(token):
            selected.add(expand_signed_page_index(token, total_pages))
            continue

        match = range_pattern.fullmatch(token)
        if match:
            start = expand_signed_page_index(match.group(1), total_pages)
            end = expand_signed_page_index(match.group(2), total_pages)
            low, high = sorted((start, end))
            selected.update(range(low, high + 1))
            continue

        raise MinerUCLIError(f"Cannot parse page range fragment: {token}", "validation")

    return sorted(selected)


def emit_result(result: dict[str, Any]) -> None:
    print(json.dumps(deep_remove_none(result), ensure_ascii=False), flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    started = time.monotonic()
    parser = build_parser()
    args = parser.parse_args(argv)
    batch_id: Optional[str] = None
    try:
        cfg = validate_config(args)
        result = convert(cfg)
        emit_result(result)
        return 0
    except MinerUCLIError as exc:
        batch_id = exc.batch_id
        emit_result(
            {
                "ok": False,
                "error": str(exc),
                "stage": exc.stage,
                "batch_id": batch_id,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
        )
        return 1
    except KeyboardInterrupt:
        emit_result(
            {
                "ok": False,
                "error": "Interrupted.",
                "stage": "interrupt",
                "batch_id": batch_id,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
        )
        return 130
    except Exception as exc:
        emit_result(
            {
                "ok": False,
                "error": str(exc),
                "stage": "unexpected",
                "batch_id": batch_id,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
