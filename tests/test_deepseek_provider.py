from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from tests._support import workspace_tmpdir
from zoterorag.db import StateLedger
from zoterorag.extractors.manager import ExtractionManager, ExtractionRequest
from zoterorag.normalize.markdown import normalize_markdown_document
from zoterorag.extractors.deepseek import (
    DEFAULT_PURPOSE,
    DeepSeekIncompleteResult,
    DeepSeekOCRError,
    DeepSeekOCRProvider,
)
from zoterorag.normalize.merge_extract_segments import merge_extract_segments

try:
    import fitz
except ImportError:
    fitz = None


def page(index, text="body", reason="stop"):
    return {"index": index, "text": text, "finish_reason": reason}


def result(pages, purpose=DEFAULT_PURPOSE):
    return {"id": "fixture", "purpose": purpose, "page_count": len(pages), "pages": pages}


def marker(label, content="", bbox="0,0,500,500"):
    return f"<｜ref｜>{label}<｜/ref｜><｜box｜>[[{bbox}]]<｜/box｜>{content}"


@unittest.skipIf(fitz is None, "requires the pdf extra")
class DeepSeekProviderTests(unittest.TestCase):
    def setUp(self):
        self.root = self.enterContext(workspace_tmpdir("deepseek-provider-"))
        self.source = self.root / "original.pdf"
        self.cache = self.root / "cache"
        with fitz.open() as doc:
            for number in range(1, 5):
                p = doc.new_page(width=144, height=72)
                p.insert_text((5, 15), f"source page {number}")
                p.draw_rect(fitz.Rect(0, 36, 72, 72), color=(0, 1, 0), fill=(0, 1, 0))
            doc.save(self.source)
        self.client = Mock()
        self.provider = DeepSeekOCRProvider(api_key="fixture-only", client=self.client)
        self.client.post.return_value.status_code = 200

    def job(self, selection="", **options):
        job = self.provider.submit(self.source, "options-hash", options={"page_ranges": selection, **options})
        self.assertEqual("completed", self.provider.poll(job.external_job_id).state)
        return job.external_job_id

    def fetch(self, job, **options):
        return self.provider.download(job, self.cache, source_pdf=self.source, options=options)

    def test_rich_subset_matches_wire_pdf_crops_and_merge(self):
        text = (
            "preface\n"
            + marker("Title", "Study")
            + marker("image", bbox="0,500,500,1000")
            + marker("image_caption", "sample")
            + marker("table", "<table><tr><td>A|B</td></tr></table>")
        )
        body = result([page(1, "plain page"), page(0, text)])

        def post(url, *, files, headers, data, timeout):
            self.assertEqual("https://api.deepseek.com/v1/tools/ocr", url)
            self.assertEqual({"purpose": DEFAULT_PURPOSE}, data)
            self.assertEqual("application/pdf", files["file"][2])
            with fitz.open(stream=files["file"][1].read(), filetype="pdf") as doc:
                self.assertEqual(["source page 2", "source page 4"], [p.get_text().strip() for p in doc])
            return Mock(status_code=200, json=Mock(return_value=body))

        self.client.post.side_effect = post
        job = self.job("2,4")
        self.client.post.assert_not_called()
        artifact = self.fetch(job, dpi=72)
        manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual([1, 3], manifest["selected_original_page_indices"])
        self.assertNotEqual(manifest["source_pdf_sha256"], manifest["uploaded_pdf_sha256"])
        markdown = Path(manifest["markdown_path"]).read_text(encoding="utf-8")
        self.assertLess(markdown.index("preface"), markdown.index("# Study"))
        self.assertNotIn("<!-- page", markdown)
        self.assertTrue(markdown.rstrip().endswith("plain page"))
        self.assertIn("*sample*", markdown)
        self.assertIn(r"| A\|B |", markdown)
        img = fitz.Pixmap(str(self.cache / "extract/images/page_002_img_1.png"))
        self.assertEqual((72, 36, (0, 255, 0)), (img.width, img.height, img.pixel(10, 10)))
        blocks = json.loads(Path(manifest["content_list_path"]).read_text(encoding="utf-8"))
        self.assertEqual((1, 3), (blocks[-1]["page_idx"], blocks[-1]["source_page_idx"]))
        merged = merge_extract_segments([artifact.artifact_dir], self.root / "merged")
        self.assertIn("images/img001.png", merged.read_text(encoding="utf-8"))
        self.assertTrue((merged.parent / "images/img001.png").is_file())

    def test_manager_reuses_completed_job_and_normalizes_chunks(self):
        self.client.post.return_value.json.return_value = result(
            [page(0, marker("image")), page(1, "sample")]
        )
        ledger = StateLedger(self.root / "state.sqlite")
        self.addCleanup(ledger.close)
        manager = ExtractionManager(ledger=ledger, provider=self.provider, cache_dir=self.cache)
        request = ExtractionRequest(input_file=self.source, selected_pages="1,4", selected_page_count=2)
        first = manager.ensure_extraction(request)
        second = manager.ensure_extraction(request)
        self.assertTrue(second.cache_hit)
        self.assertEqual("downloaded", first.job["state"])
        self.client.post.assert_called_once()
        normalized = normalize_markdown_document(
            source_markdown=Path(first.job["artifact_dir"]) / "extract/full.md",
            output_root=self.root / "normalized",
            document_id="fixture",
        )
        self.assertEqual(1, len(normalized.images))
        self.assertTrue(any("sample" in c["text"] for c in normalized.chunks))
        self.assertTrue(all("<!-- page" not in c["text"] for c in normalized.chunks))

    def test_manager_records_permanent_truncation(self):
        self.client.post.return_value.json.return_value = result([page(0, reason="length")])
        ledger = StateLedger(self.root / "state.sqlite")
        self.addCleanup(ledger.close)
        manager = ExtractionManager(ledger=ledger, provider=self.provider, cache_dir=self.cache)
        job = manager.submit_only(ExtractionRequest(input_file=self.source, selected_pages="3")).job
        with self.assertRaises(DeepSeekIncompleteResult):
            manager.harvest(job["job_id"])
        failed = ledger.get_extract_job(job_id=job["job_id"])
        self.assertEqual("failed_manual_review", failed["state"])
        self.assertEqual("DeepSeekIncompleteResult", failed["error_code"])
        self.assertFalse((self.cache / job["cache_key"] / "manifest.json").exists())

    def test_full_text_restarts_after_local_failure(self):
        purpose = "text_recognition"
        self.client.post.return_value.json.return_value = result([page(i) for i in range(4)], purpose)
        job = self.job(purpose=purpose)
        with patch("zoterorag.extractors.deepseek.convert_pages", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.fetch(job, purpose=purpose)
        self.assertFalse((self.cache / "manifest.json").exists())
        self.assertTrue((self.cache / "response.json").is_file())
        self.assertEqual(self.source.read_bytes(), (self.cache / "upload.pdf").read_bytes())
        self.source.unlink()
        # A new provider can finish conversion from disk without credentials.
        offline = DeepSeekOCRProvider(client=Mock())
        artifact = offline.download(job, self.cache, source_pdf=self.source, options={"purpose": purpose})
        offline.client.post.assert_not_called()
        self.assertTrue(artifact.manifest_path.is_file())

    def test_incomplete_retained_without_publishing_or_resubmitting(self):
        self.client.post.return_value.json.return_value = result([page(0, reason="length")])
        job = self.job("3")
        for _ in range(2):
            with self.assertRaisesRegex(DeepSeekIncompleteResult, r"length on source pages \[3\]"):
                self.fetch(job)
        self.client.post.assert_called_once()
        self.assertFalse((self.cache / "manifest.json").exists())
        self.assertTrue((self.cache / "response.json").exists())

    def test_rejects_bad_page_coverage_purpose_type_and_api_errors(self):
        job = self.job("1-2")
        for malformed in (
            result([page(0), page(0)]),
            result([page(0), page(2)]),
            result([page(0)]),
            result([page(0), page(True)]),
            result([page(0), page(1, None)]),
            result([page(0), page(1, reason="unknown")]),
            result([page(0), page(1)], "text_recognition"),
            {"error": {"message": "upstream fixture"}},
        ):
            with self.subTest(body=malformed):
                self.client.post.return_value.json.return_value = malformed
                with self.assertRaises(DeepSeekOCRError) as raised:
                    self.fetch(job)
                self.assertNotIn("upstream fixture", str(raised.exception))
                self.assertFalse((self.cache / "manifest.json").exists())
        self.client.post.return_value.status_code = 429
        with self.assertRaises(DeepSeekOCRError) as raised:
            self.fetch(job)
        self.assertEqual(429, raised.exception.status_code)

    def test_transport_failure_propagates_and_can_retry(self):
        job = self.job("2")
        self.client.post.side_effect = TimeoutError("request timed out")
        with self.assertRaisesRegex(TimeoutError, "request timed out"):
            self.fetch(job)
        self.assertFalse((self.cache / "response.json").exists())
        self.client.post.side_effect = None
        self.client.post.return_value.json.return_value = result([page(0)])
        self.assertTrue(self.fetch(job).manifest_path.is_file())

    def test_parallel_downloads_share_one_response(self):
        self.client.post.return_value.json.return_value = result([page(0)])
        job = self.job("3")
        with ThreadPoolExecutor(max_workers=2) as pool:
            artifacts = list(pool.map(self.fetch, [job, job]))
        self.client.post.assert_called_once()
        self.assertEqual(artifacts[0], artifacts[1])

    def test_rejects_invalid_selection_before_post(self):
        for selection in ("0", "3-1", "1-5", "1,"):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                self.fetch(self.job(selection))
        self.client.post.assert_not_called()
