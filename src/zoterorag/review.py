from __future__ import annotations

from typing import Any

from .db import StateLedger


def explain_attachment_review(ledger: StateLedger, attachment_key: str) -> dict[str, Any]:
    """Explain the persisted review decision for one scanned attachment.

    The explanation is built only from the local state ledger. It does not open
    the live Zotero database, inspect PDF contents, or call extraction providers;
    this keeps manual review safe while long-running ingest workers are paused
    or unavailable.
    """

    attachment = ledger.get_attachment(attachment_key)
    if attachment is None:
        raise KeyError(f"attachment not found in local review queue: {attachment_key}")
    rule = ledger.review_rule_map().get(attachment_key)
    reasons = list(attachment.get("reasons") or [])
    return {
        "attachment_key": attachment_key,
        "attachment": public_attachment_facts(attachment),
        "classification": attachment["classification"],
        "source_quality": attachment["source_quality"],
        "scan_status": attachment["scan_status"],
        "reasons": reasons,
        "manual_rule": rule,
        "policy_notes": policy_notes_for_attachment(attachment, reasons, rule),
        "suggested_actions": suggested_review_actions(attachment_key, attachment, rule),
    }


def public_attachment_facts(attachment: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_key": attachment.get("parent_key"),
        "content_type": attachment.get("content_type"),
        "relative_path": attachment.get("relative_path"),
        "title": attachment.get("title"),
        "date": attachment.get("date_value"),
        "url": attachment.get("url"),
        "file_path": attachment.get("file_path"),
        "file_exists": bool(attachment.get("file_exists")),
        "file_size": attachment.get("file_size"),
        "file_mtime": attachment.get("file_mtime"),
        "first_seen_at": attachment.get("first_seen_at"),
        "last_seen_at": attachment.get("last_seen_at"),
        "updated_at": attachment.get("updated_at"),
    }


def policy_notes_for_attachment(
    attachment: dict[str, Any],
    reasons: list[str],
    rule: dict[str, Any] | None,
) -> list[str]:
    notes: list[str] = []
    if rule is not None:
        notes.append("manual_rule_overrides_automatic_classification_on_next_scan")
    if any(reason.startswith("translation_marker:") for reason in reasons):
        notes.append("strong_translation_markers_go_to_review_not_auto_exclusion")
    if any(reason.startswith("weak_translation_marker:") for reason in reasons):
        notes.append("weak_translation_markers_like_immersive_only_request_review")
    if attachment["classification"] == "orphan_metadata_only":
        notes.append("orphan_pdf_is_metadata_only_until_manually_included")
    if attachment["classification"] == "missing_file" or not attachment.get("file_exists"):
        notes.append("missing_file_blocks_pdf_extraction_until_storage_is_available")
    if attachment["classification"] == "report_only":
        notes.append("non_pdf_attachment_is_reported_but_not_queued_for_pdf_conversion")
    if attachment["classification"] in {"included_auto", "included_manual"} and attachment.get("file_exists"):
        notes.append("attachment_is_eligible_for_ingest_planning")
    return notes


def suggested_review_actions(
    attachment_key: str,
    attachment: dict[str, Any],
    rule: dict[str, Any] | None,
) -> list[dict[str, str]]:
    actions = [
        {
            "action": "include",
            "cli": f"zoterorag review include --attachment-key {attachment_key} --reason \"...\"",
            "effect": "force_include_on_next_scan",
        },
        {
            "action": "exclude",
            "cli": f"zoterorag review exclude --attachment-key {attachment_key} --reason \"...\"",
            "effect": "force_exclude_on_next_scan",
        },
    ]
    if rule is not None:
        actions.append(
            {
                "action": "rescan",
                "cli": "zoterorag scan",
                "effect": "apply_existing_manual_rule_to_classification",
            }
        )
    elif attachment["classification"] == "needs_review":
        actions.append(
            {
                "action": "decide",
                "cli": f"zoterorag review include|exclude --attachment-key {attachment_key} --reason \"...\"",
                "effect": "resolve_review_queue_entry",
            }
        )
    return actions
