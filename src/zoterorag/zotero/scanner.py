from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..db import JobEvent, StateLedger
from .classifier import ClassifiedAttachment, classify_attachment
from .shadow import ZoteroShadow


@dataclass(frozen=True)
class ScanReport:
    scanned: int
    summary: dict[str, int]
    attachments: list[ClassifiedAttachment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "summary": self.summary,
            "attachments": [item.to_record() for item in self.attachments],
        }


def scan_shadow_to_ledger(
    shadow_db: str | Path,
    storage_dir: str | Path,
    ledger: StateLedger,
    limit: int | None = None,
) -> ScanReport:
    job_id = ledger.create_job("zotero_scan", {"shadow_db": str(shadow_db), "limit": limit})
    ledger.set_job_status(job_id, "running")
    # The scanner only opens the already-copied shadow database. The live Zotero
    # database is never touched in this phase; refreshing the shadow is handled
    # by runtime.copy_zotero_shadow with a read-only source connection.
    shadow = ZoteroShadow(shadow_db)
    try:
        review_rules = ledger.review_rule_map()
        attachments = shadow.list_attachments(limit=limit)
        classified = [
            classify_attachment(attachment, storage_dir, review_rules=review_rules)
            for attachment in attachments
        ]
        records = [item.to_record() for item in classified]
        # Persist scan results as a checkpointed review queue. MinerU and Qwen
        # providers are deliberately not invoked during scanning.
        ledger.upsert_attachments(records)
        # A complete scan is authoritative for attachment presence. Limited
        # scans are diagnostics only and must not mark unscanned rows deleted.
        deleted_count = 0
        if limit is None:
            deleted_count = ledger.mark_absent_attachments_deleted(
                item.attachment.attachment_key for item in classified
            )
        summary = dict(Counter(item.classification for item in classified))
        if deleted_count:
            summary["deleted"] = deleted_count
        report = ScanReport(scanned=len(classified), summary=summary, attachments=classified)
        ledger.add_scan_report({"scanned": report.scanned, "summary": report.summary}, job_id=job_id)
        ledger.checkpoint(
            "zotero_shadow",
            "scan_zotero",
            "completed",
            {"scanned": report.scanned, "summary": report.summary},
        )
        ledger.add_event(
            JobEvent(
                job_id=job_id,
                stage="scan_zotero",
                status="completed",
                message=f"scanned {report.scanned} attachments",
                payload={"summary": report.summary},
            )
        )
        ledger.set_job_status(job_id, "completed")
        return report
    except Exception as exc:
        ledger.add_event(
            JobEvent(
                job_id=job_id,
                stage="scan_zotero",
                status="failed",
                message=str(exc),
            )
        )
        ledger.set_job_status(job_id, "failed")
        raise
    finally:
        shadow.close()
