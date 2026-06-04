from __future__ import annotations

from pathlib import Path

from tests._support import OptionalModuleTestCase, call_with_known_kwargs, workspace_tmpdir


class LocalVectorStoreTests(OptionalModuleTestCase):
    def _load_store_class(self):
        module = self.import_first_available(
            (
                "zoterorag.vector_store.local",
                "zoterorag.vectorstore.local",
                "zoterorag.vector.local",
                "zoterorag.index.local_vector",
            )
        )
        store_class = self.get_first_attr(module, ("LocalVectorStore", "VectorStore", "InMemoryVectorStore"))
        return module, store_class

    def _build_store(self, store_class, module, root: Path):
        for candidate in (
            {"root_dir": root, "profile_name": "test-profile", "dimension": 3},
            {"store_dir": root, "profile_name": "test-profile", "dimension": 3},
            {"path": root / "vectors.sqlite", "profile_name": "test-profile", "dimension": 3},
            {"index_path": root / "vectors.sqlite", "profile_name": "test-profile", "dimension": 3},
            {"base_path": root / "vectors.sqlite", "profile_name": "test-profile", "dimension": 3},
            {},
        ):
            try:
                return store_class(**candidate)
            except TypeError:
                continue
        self.fail(f"could not construct vector store from {store_class!r}")

    def _add_records(self, module, store) -> None:
        vector_record_class = getattr(module, "VectorRecord", None)
        records = [
            {
                "record_id": "record-a",
                "chunk_id": "chunk-a",
                "document_id": "doc-1",
                "text": "alpha topic",
                "vector": [1.0, 0.0, 0.0],
                "modality": "text",
                "metadata": {"kind": "note"},
            },
            {
                "record_id": "record-b",
                "chunk_id": "chunk-b",
                "document_id": "doc-2",
                "text": "beta topic",
                "vector": [0.0, 1.0, 0.0],
                "modality": "text",
                "metadata": {"kind": "note"},
            },
        ]
        if vector_record_class is not None:
            records = [vector_record_class(**record) for record in records]

        add_method = self.get_first_attr(store, ("add", "add_many", "upsert", "upsert_many", "index"))
        errors: list[str] = []
        for payload in (
            {"records": records},
            {"items": records},
            {"chunks": records},
            {"documents": records},
            {"entries": records},
            {"record": records[0]},
            {"item": records[0]},
            {"chunk": records[0]},
        ):
            try:
                result = call_with_known_kwargs(add_method, **payload)
                if payload.keys() & {"record", "item", "chunk"}:
                    second_payload = next(
                        p for p in ({"record": records[1]}, {"item": records[1]}, {"chunk": records[1]}) if next(iter(p)) in payload
                    )
                    call_with_known_kwargs(add_method, **second_payload)
                return result
            except Exception as exc:  # pragma: no cover - exercised only for API probing
                errors.append(f"{tuple(payload.keys())}: {exc}")
        self.fail("could not add records to vector store: " + "; ".join(errors))

    def _search(self, store):
        search_method = self.get_first_attr(store, ("search", "query", "nearest", "similarity_search"))
        errors: list[str] = []
        for payload in (
            {"vector": [1.0, 0.0, 0.0], "limit": 2},
            {"query_vector": [1.0, 0.0, 0.0], "limit": 2},
            {"embedding": [1.0, 0.0, 0.0], "limit": 2},
            {"vector": [1.0, 0.0, 0.0], "top_k": 2},
            {"query_vector": [1.0, 0.0, 0.0], "top_k": 2},
        ):
            try:
                return call_with_known_kwargs(search_method, **payload)
            except Exception as exc:  # pragma: no cover - exercised only for API probing
                errors.append(f"{tuple(payload.keys())}: {exc}")
        self.fail("could not search vector store: " + "; ".join(errors))

    def _as_mapping(self, result):
        if isinstance(result, dict):
            return result
        if hasattr(result, "__dict__"):
            return dict(result.__dict__)
        self.fail(f"search result is not mapping-like: {result!r}")

    def test_add_and_search_ranks_exact_vector_first(self) -> None:
        module, store_class = self._load_store_class()
        with workspace_tmpdir("vector-store-") as tmpdir:
            store = self._build_store(store_class, module, tmpdir)
            self._add_records(module, store)
            results = self._search(store)

            self.assertIsInstance(results, list)
            self.assertGreaterEqual(len(results), 1)

            first = self._as_mapping(results[0])
            winner = first.get("chunk_id") or first.get("id") or first.get("key")
            self.assertEqual("chunk-a", winner)
            if len(results) > 1:
                second = self._as_mapping(results[1])
                first_score = first.get("score")
                second_score = second.get("score")
                if first_score is not None and second_score is not None:
                    self.assertGreaterEqual(first_score, second_score)
            close_method = getattr(store, "close", None)
            if callable(close_method):
                close_method()
