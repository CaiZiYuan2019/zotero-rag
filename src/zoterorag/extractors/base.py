from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ExtractJobState:
    external_job_id: str
    state: str
    message: str = ""


@dataclass(frozen=True)
class ExtractArtifact:
    source_pdf: Path
    artifact_dir: Path
    manifest_path: Path


class ExtractorProvider(Protocol):
    name: str
    version: str

    def fingerprint(self, input_file: Path, options_hash: str) -> str:
        ...

    def submit(
        self,
        input_file: Path,
        options_hash: str,
        *,
        options: dict[str, Any] | None = None,
        api_key: str | None = None,
    ) -> ExtractJobState:
        ...

    def poll(
        self,
        external_job_id: str,
        *,
        api_key: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ExtractJobState:
        ...

    def download(
        self,
        external_job_id: str,
        output_dir: Path,
        *,
        api_key: str | None = None,
        source_pdf: Path | str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ExtractArtifact:
        ...


class StubExtractorProvider:
    """Non-network extractor for tests and dry runs."""

    name = "stub"
    version = "0"

    def fingerprint(self, input_file: Path, options_hash: str) -> str:
        return f"stub:{input_file.name}:{options_hash}"

    def submit(
        self,
        input_file: Path,
        options_hash: str,
        *,
        options: dict[str, Any] | None = None,
        api_key: str | None = None,
    ) -> ExtractJobState:
        return ExtractJobState(external_job_id=self.fingerprint(input_file, options_hash), state="completed")

    def poll(
        self,
        external_job_id: str,
        *,
        api_key: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ExtractJobState:
        return ExtractJobState(external_job_id=external_job_id, state="completed")

    def download(
        self,
        external_job_id: str,
        output_dir: Path,
        *,
        api_key: str | None = None,
        source_pdf: Path | str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ExtractArtifact:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = output_dir / "manifest.json"
        source_pdf_path = Path(source_pdf) if source_pdf is not None else Path(external_job_id)
        manifest.write_text(
            json.dumps(
                {
                    "provider": self.name,
                    "provider_version": self.version,
                    "external_job_id": external_job_id,
                    "state": "completed",
                    "source_pdf": str(source_pdf_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return ExtractArtifact(source_pdf=source_pdf_path, artifact_dir=output_dir, manifest_path=manifest)
