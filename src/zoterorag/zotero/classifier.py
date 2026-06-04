from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .shadow import ZoteroAttachment


PDF_CONTENT_TYPE = "application/pdf"


STRONG_TRANSLATION_MARKERS = (
    "_zh-cn_dual.pdf",
    "_zh_cn_dual.pdf",
    "dual.pdf",
    "双语",
    "中英对照",
    "scholaread",
)

WEAK_TRANSLATION_MARKERS = (
    "immersive",
    "沉浸式",
)


@dataclass(frozen=True)
class ClassifiedAttachment:
    attachment: ZoteroAttachment
    classification: str
    source_quality: str
    reasons: list[str]
    file_path: Path | None
    file_exists: bool
    file_size: int | None
    file_mtime: float | None

    def to_record(self) -> dict[str, Any]:
        return {
            "attachment_key": self.attachment.attachment_key,
            "parent_key": self.attachment.parent_key,
            "content_type": self.attachment.content_type,
            "relative_path": self.attachment.relative_path,
            "title": self.attachment.title,
            "abstract": self.attachment.abstract,
            "date": self.attachment.date,
            "url": self.attachment.url,
            "classification": self.classification,
            "source_quality": self.source_quality,
            "reasons": self.reasons,
            "file_path": str(self.file_path) if self.file_path else None,
            "file_exists": self.file_exists,
            "file_size": self.file_size,
            "file_mtime": self.file_mtime,
            "metadata": {},
        }


def storage_relative_name(relative_path: str | None) -> str | None:
    if not relative_path or not relative_path.startswith("storage:"):
        return None
    return relative_path.removeprefix("storage:")


def resolve_storage_file(storage_dir: str | Path, attachment_key: str, relative_path: str | None) -> Path | None:
    filename = storage_relative_name(relative_path)
    if filename is None:
        return None
    return Path(storage_dir) / attachment_key / filename


def translation_reasons(text: str) -> tuple[list[str], list[str]]:
    lowered = text.lower()
    strong = [f"translation_marker:{marker}" for marker in STRONG_TRANSLATION_MARKERS if marker in lowered]
    weak = [f"weak_translation_marker:{marker}" for marker in WEAK_TRANSLATION_MARKERS if marker in lowered]
    return strong, weak


def classify_attachment(
    attachment: ZoteroAttachment,
    storage_dir: str | Path,
    review_rules: dict[str, dict[str, Any]] | None = None,
) -> ClassifiedAttachment:
    review_rules = review_rules or {}
    reasons: list[str] = []
    source_quality = "unknown"
    # Zotero stores local attachment paths as "storage:<filename>"; the actual
    # file is under storage/<attachment_key>/<filename>. We only inspect file
    # metadata here, not PDF content, so this scan cannot trigger conversion.
    file_path = resolve_storage_file(storage_dir, attachment.attachment_key, attachment.relative_path)
    file_exists = bool(file_path and file_path.is_file())
    file_size = file_path.stat().st_size if file_exists and file_path else None
    file_mtime = file_path.stat().st_mtime if file_exists and file_path else None

    # Manual rules are explicit user choices and intentionally override weak or
    # strong filename heuristics. They are still persisted with file facts so a
    # later ingest stage can refuse impossible work, such as a missing file.
    manual_rule = review_rules.get(attachment.attachment_key)
    if manual_rule:
        decision = manual_rule["decision"]
        reasons.append(f"manual_{decision}:{manual_rule.get('reason', '')}")
        classification = "included_manual" if decision == "include" else "excluded_manual"
        return ClassifiedAttachment(
            attachment=attachment,
            classification=classification,
            source_quality="manual_override",
            reasons=reasons,
            file_path=file_path,
            file_exists=file_exists,
            file_size=file_size,
            file_mtime=file_mtime,
        )

    if attachment.content_type != PDF_CONTENT_TYPE:
        return ClassifiedAttachment(
            attachment=attachment,
            classification="report_only",
            source_quality="non_pdf",
            reasons=["non_pdf_attachment"],
            file_path=file_path,
            file_exists=file_exists,
            file_size=file_size,
            file_mtime=file_mtime,
        )

    if file_path is None:
        return ClassifiedAttachment(
            attachment=attachment,
            classification="missing_file",
            source_quality="storage_path_unresolved",
            reasons=["not_a_storage_path"],
            file_path=None,
            file_exists=False,
            file_size=None,
            file_mtime=None,
        )

    if not file_exists:
        return ClassifiedAttachment(
            attachment=attachment,
            classification="missing_file",
            source_quality="missing_pdf",
            reasons=["storage_file_missing"],
            file_path=file_path,
            file_exists=False,
            file_size=None,
            file_mtime=None,
        )

    marker_text = " ".join(
        part
        for part in (attachment.relative_path, attachment.title, attachment.url)
        if part
    )
    strong, weak = translation_reasons(marker_text)
    reasons.extend(strong)
    reasons.extend(weak)

    # Translation heuristics never auto-exclude. Even strong matches are sent to
    # review so false positives such as "immersive" or unusual file names can be
    # manually included without editing the database or code.
    if strong:
        classification = "needs_review"
        source_quality = "suspected_translation_pdf"
    elif weak:
        classification = "needs_review"
        source_quality = "weak_translation_signal"
    elif attachment.parent_key is None:
        classification = "orphan_metadata_only"
        source_quality = "orphan_pdf"
        reasons.append("orphan_pdf_no_parent_item")
    else:
        classification = "included_auto"
        source_quality = "primary_candidate"
        reasons.append("pdf_with_parent")

    return ClassifiedAttachment(
        attachment=attachment,
        classification=classification,
        source_quality=source_quality,
        reasons=reasons,
        file_path=file_path,
        file_exists=file_exists,
        file_size=file_size,
        file_mtime=file_mtime,
    )
