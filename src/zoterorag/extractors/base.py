from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol


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

    def submit(self, input_file: Path, options_hash: str) -> ExtractJobState:
        ...

    def poll(self, external_job_id: str) -> ExtractJobState:
        ...

    def download(self, external_job_id: str, output_dir: Path) -> ExtractArtifact:
        ...


class StubExtractorProvider:
    """Non-network extractor for tests and dry runs."""

    name = "stub"
    version = "0"

    def fingerprint(self, input_file: Path, options_hash: str) -> str:
        return f"stub:{input_file.name}:{options_hash}"

    def submit(self, input_file: Path, options_hash: str) -> ExtractJobState:
        return ExtractJobState(external_job_id=self.fingerprint(input_file, options_hash), state="completed")

    def poll(self, external_job_id: str) -> ExtractJobState:
        return ExtractJobState(external_job_id=external_job_id, state="completed")

    def download(self, external_job_id: str, output_dir: Path) -> ExtractArtifact:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = output_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "provider": self.name,
                    "provider_version": self.version,
                    "external_job_id": external_job_id,
                    "state": "completed",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return ExtractArtifact(source_pdf=Path(external_job_id), artifact_dir=output_dir, manifest_path=manifest)
