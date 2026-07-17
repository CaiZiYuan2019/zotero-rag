"""Zotero collection (folder) browsing and partitioned search.

Collection filtering flows exclusively through the ``attachment_collections``
association table. The ``collections_json`` stored on each attachment is only
used for display/FTS text, never for filtering here.
"""

from __future__ import annotations

from typing import Any

from ..db import StateLedger
from .metadata import metadata_search


def list_collections(ledger: StateLedger) -> list[dict[str, Any]]:
    """Return a flat list of collections including their parent key."""

    return ledger.list_ledger_collections()


def get_collection_tree(ledger: StateLedger) -> list[dict[str, Any]]:
    """Build a nested tree from ``parent_collection_key`` links.

    Collections whose parent is missing (or ``None``) become roots. Each node
    gains a ``children`` list sorted by collection name then key.
    """

    flat = ledger.list_ledger_collections()
    nodes: dict[str, dict[str, Any]] = {}
    for row in flat:
        nodes[row["collection_key"]] = {**row, "children": []}

    roots: list[dict[str, Any]] = []
    for key, node in nodes.items():
        parent_key = node.get("parent_collection_key")
        parent = nodes.get(parent_key) if parent_key else None
        if parent is not None and parent_key != key:
            parent["children"].append(node)
        else:
            roots.append(node)

    def _sort(items: list[dict[str, Any]]) -> None:
        items.sort(key=lambda item: (str(item.get("collection_name") or "").casefold(), item["collection_key"]))
        for item in items:
            _sort(item["children"])

    _sort(roots)
    return roots


def _descendant_keys(ledger: StateLedger, collection_key: str) -> list[str]:
    """Collect ``collection_key`` and all descendants via BFS over parent links."""

    flat = ledger.list_ledger_collections()
    children_by_parent: dict[str, list[str]] = {}
    for row in flat:
        parent = row.get("parent_collection_key")
        if parent:
            children_by_parent.setdefault(parent, []).append(row["collection_key"])

    collected: list[str] = []
    seen: set[str] = set()
    queue = [collection_key]
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        collected.append(current)
        queue.extend(children_by_parent.get(current, []))
    return collected


def search_by_collection(
    ledger: StateLedger,
    collection_key: str,
    query: str | None = None,
    *,
    include_children: bool = False,
    limit: int = 10,
    consumer: str = "llm_text",
    **filters: Any,
) -> list[dict[str, Any]]:
    """Search (or list) attachments within a Zotero collection.

    ``include_children`` defaults to ``False`` (only the target collection).
    When ``True`` it expands to all descendant collections. An empty ``query``
    lists the partition through the metadata pure-filter path.
    """

    if include_children:
        keys = _descendant_keys(ledger, collection_key)
    else:
        keys = [collection_key]

    attachment_keys = ledger.attachment_keys_for_collections(keys)
    if not attachment_keys:
        return []

    return metadata_search(
        ledger,
        query or "",
        attachment_keys=attachment_keys,
        limit=limit,
        consumer=consumer,
        **filters,
    )
