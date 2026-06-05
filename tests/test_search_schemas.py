from __future__ import annotations

from tests._support import OptionalModuleTestCase, call_with_known_kwargs


class SearchSchemaTests(OptionalModuleTestCase):
    def _load_module(self):
        return self.import_first_available(
            (
                "zoterorag.search.schemas",
                "zoterorag.search.results",
                "zoterorag.search.schema",
            )
        )

    def test_llm_text_sanitization_excludes_image_payloads(self) -> None:
        module = self._load_module()
        sanitizer = self.get_first_attr(
            module,
            (
                "sanitize_text_results_for_llm",
                "sanitize_results_for_llm_text",
                "sanitize_llm_text_results",
                "llm_text_results",
                "to_llm_text_results",
                "sanitize_results_for_consumer",
            ),
        )
        raw_results = [
            {
                "document_id": "doc-1",
                "chunk_id": "text-1",
                "title": "Text chunk",
                "score": 0.9,
                "text": "plain text chunk ![chart](images/img001.png)",
                "metadata": {"content_type": "text/plain"},
                "images": [
                    {
                        "image_id": "img-a",
                        "caption": "chart caption",
                        "base64": "abc",
                        "file_ref": "file-a.png",
                        "thumbnail_ref": "thumb-a.png",
                        "mime_type": "image/png",
                    }
                ],
            },
            {
                "document_id": "doc-2",
                "chunk_id": "image-1",
                "title": "Figure chunk",
                "score": 0.8,
                "text": "figure caption",
                "metadata": {
                    "content_type": "image/png",
                    "image_path": "images/img002.png",
                    "image_return": "file_ref",
                },
                "images": [
                    {
                        "image_id": "img-b",
                        "caption": "microscopy image",
                        "base64": "def",
                        "file_ref": "file-b.png",
                        "thumbnail_ref": "thumb-b.png",
                        "mime_type": "image/png",
                    }
                ],
            },
        ]

        sanitized = call_with_known_kwargs(
            sanitizer,
            results=raw_results,
            items=raw_results,
            raw_results=raw_results,
            consumer="llm_text",
            image_return="none",
        )
        self.assertIsInstance(sanitized, list)
        self.assertGreaterEqual(len(sanitized), 1)

        for item in sanitized:
            payload = item if isinstance(item, dict) else dict(item.__dict__)
            llm_text = payload.get("llm_text", payload.get("text", ""))
            self.assertIsInstance(llm_text, str)
            self.assertIsNone(payload["rerank_score"])
            self.assertNotIn("images", payload)
            self.assertNotIn("base64", repr(payload))
            self.assertNotIn("file_ref", repr(payload))
            self.assertNotIn("thumbnail_ref", repr(payload))
            self.assertNotIn("image_path", repr(payload))
            self.assertNotIn("images/img002.png", repr(payload))
            self.assertNotIn("images/img001.png", llm_text)
