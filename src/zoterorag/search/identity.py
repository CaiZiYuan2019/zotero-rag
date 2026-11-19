"""Resolve public IDs against the local ledger without changing stored indexes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..db import StateLedger


@dataclass(frozen=True)
class SourceIdentity:
    document_id: str
    attachment_key: str | None
    parent_item_key: str | None

    @property
    def source_id(self) -> str:
        return self.parent_item_key or self.attachment_key or self.document_id

    def fields(self) -> dict[str, str | None]:
        return {
            "source_id": self.source_id,
            "document_id": self.document_id,
            "attachment_key": self.attachment_key,
            "parent_item_key": self.parent_item_key,
            "parent_key": self.parent_item_key,
        }

    def enrich_result(self, result: dict[str, Any]) -> dict[str, Any]:
        fields = self.fields()
        return {**result, **fields, "metadata": {**result.get("metadata", {}), **fields}}


def document_identities(ledger: StateLedger, document_ids: Iterable[str]) -> dict[str, SourceIdentity]:
    """Resolve exact artifact IDs first, then legacy rows keyed by attachment."""
    keys = list(dict.fromkeys(document_ids))
    identities = {
        row["document_id"]: SourceIdentity(row["document_id"], row["attachment_key"], row["parent_key"])
        for row in ledger.artifact_identity_rows(keys)
    }
    missing = [key for key in keys if key not in identities]
    identities.update(attachment_identities(ledger, missing))
    return identities


def attachment_identities(ledger: StateLedger, attachment_keys: Iterable[str]) -> dict[str, SourceIdentity]:
    """Prefer the exact-ID artifact, then the most recent artifact of this attachment."""
    keys = list(dict.fromkeys(attachment_keys))
    identities = {}
    for row in ledger.artifact_identity_rows(keys, by_attachment=True):
        key = row["attachment_key"]
        identities.setdefault(key, SourceIdentity(row["document_id"], key, row["parent_key"]))
    missing = [key for key in keys if key not in identities]
    parents = ledger.attachment_parents(missing)
    for key in missing:
        identities[key] = SourceIdentity(key, key if key in parents else None, parents.get(key))
    return identities
