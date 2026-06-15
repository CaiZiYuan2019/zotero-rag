from __future__ import annotations

import unittest

from zoterorag.search.results import (
    RerankNotSupportedError,
    SearchResult,
    ensure_rerank_disabled,
    sanitize_results_for_consumer,
)


class SearchResultConsumerTests(unittest.TestCase):
    def test_sanitize_strips_embedding_path_fields_for_text_consumer(self) -> None:
        result = SearchResult(
            document_id="DOC1",
            chunk_id="DOC1:image:00001",
            title="Figure",
            text="caption text",
            score=0.5,
            metadata={
                "image_path": "images/img001.png",
                "image_embedding_path": "embedding_images/img001.png",
                "embedding_relative_path": "embedding_images/img001.png",
                "safe_key": "kept",
            },
            images=[],
        )
        sanitized = sanitize_results_for_consumer([result], consumer="llm_text")
        metadata = sanitized[0]["metadata"]
        self.assertNotIn("image_path", metadata)
        self.assertNotIn("image_embedding_path", metadata)
        self.assertNotIn("embedding_relative_path", metadata)
        self.assertEqual("kept", metadata["safe_key"])


class RerankErrorTests(unittest.TestCase):
    def test_rerank_disabled_by_default(self) -> None:
        ensure_rerank_disabled(False)

    def test_rerank_raises_custom_not_implemented_error(self) -> None:
        with self.assertRaises(RerankNotSupportedError):
            ensure_rerank_disabled(True)

        # Existing callers that catch NotImplementedError still work.
        with self.assertRaises(NotImplementedError):
            ensure_rerank_disabled(True)


if __name__ == "__main__":
    unittest.main()
