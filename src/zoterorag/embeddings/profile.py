from __future__ import annotations

import hashlib
import json
from typing import Any


VECTOR_SEMANTIC_PROFILE_FIELDS = (
    "name",
    "provider",
    "model",
    "dimension",
    "modality",
    "query_role_mode",
    "document_role_mode",
    "instruction_template",
)


def embedding_profile_hash(profile: dict[str, Any]) -> str:
    """Hash the fields that change embedding semantics.

    Routing flags such as `enabled` or `default_for_text` are intentionally
    excluded: changing the default model affects which profile a query selects,
    but it does not make vectors already built for the profile stale. Role and
    instruction fields are included because qwen-style query/document handling
    changes the embedding space and must trigger re-indexing.
    """

    embedded_profile = dict(profile.get("profile") or {})
    payload = {
        field: profile.get(field, embedded_profile.get(field))
        for field in VECTOR_SEMANTIC_PROFILE_FIELDS
    }
    payload["image_policy"] = embedded_profile.get("image_policy", profile.get("image_policy"))
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
