from __future__ import annotations

from io import BytesIO
import json
import unittest
import zipfile

from tests._support import workspace_tmpdir
from zoterorag.extractors import ApiKeyRef, ExtractionManager, ExtractionRequest, ExtractorKeyPool, MinerUAPIError, MinerUProvider
from zoterorag.extractors.mineru import safe_extract_zip
from zoterorag.db import StateLedger


class MinerUProviderTests(unittest.TestCase):
    def test_submit_poll_download_with_fake_client(self) -> None:
        with workspace_tmpdir("mineru-provider-") as tmpdir:
            pdf_path = tmpdir / "paper one.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 fake body")
            client = FakeMinerUClient(zip_bytes=build_zip({"result/full.md": "# Title\n"}))
            provider = MinerUProvider(api_key="sk-test-secret", client=client)

            submitted = provider.submit(
                pdf_path,
                "options-hash",
                options={"page_ranges": "1-2", "model_version": "vlm", "enable_formula": True},
            )
            polled = provider.poll(submitted.external_job_id)
            artifact = provider.download(submitted.external_job_id, tmpdir / "artifact")

            self.assertEqual("batch-001", submitted.external_job_id)
            self.assertEqual("completed", polled.state)
            self.assertTrue((tmpdir / "artifact" / "extract" / "result" / "full.md").is_file())

            payload = client.posts[0]["json"]
            self.assertEqual("paper one.pdf", payload["files"][0]["name"])
            self.assertEqual("1-2", payload["files"][0]["page_ranges"])
            self.assertEqual("paper_one", payload["files"][0]["data_id"])
            self.assertEqual("Bearer sk-test-secret", client.posts[0]["headers"]["Authorization"])
            self.assertEqual(b"%PDF-1.4 fake body", client.uploaded_bytes)

            manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("mineru", manifest["provider"])
            self.assertEqual("batch-001", manifest["external_job_id"])
            self.assertNotIn("sk-test-secret", json.dumps(manifest, ensure_ascii=False))

    def test_manager_passes_secret_to_provider_without_persisting_it(self) -> None:
        with workspace_tmpdir("mineru-manager-") as tmpdir:
            pdf_path = tmpdir / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 fake body")
            ledger = StateLedger(tmpdir / "state.sqlite")
            client = FakeMinerUClient(zip_bytes=build_zip({"full.md": "# Done\n"}))
            try:
                manager = ExtractionManager(
                    ledger=ledger,
                    cache_dir=tmpdir / "extract_cache",
                    provider=MinerUProvider(client=client),
                    key_pool=ExtractorKeyPool([ApiKeyRef(alias="mineru_a", secret="sk-test-secret")]),
                )
                result = manager.ensure_extraction(
                    ExtractionRequest(
                        input_file=pdf_path,
                        attachment_key="ATTACH1",
                        selected_pages="3-4",
                        selected_page_count=2,
                    )
                )

                self.assertEqual("downloaded", result.job["state"])
                self.assertEqual("mineru_a", result.job["api_key_alias"])
                self.assertEqual("Bearer sk-test-secret", client.posts[0]["headers"]["Authorization"])
                serialized_jobs = json.dumps(ledger.list_extract_jobs(), ensure_ascii=False, default=str)
                self.assertIn("mineru_a", serialized_jobs)
                self.assertNotIn("sk-test-secret", serialized_jobs)
                self.assertEqual("3-4", client.posts[0]["json"]["files"][0]["page_ranges"])
            finally:
                ledger.close()

    def test_safe_extract_rejects_zip_path_traversal(self) -> None:
        with workspace_tmpdir("mineru-zip-") as tmpdir:
            zip_path = tmpdir / "bad.zip"
            zip_path.write_bytes(build_zip({"../escape.md": "bad"}))
            with self.assertRaises(MinerUAPIError):
                safe_extract_zip(zip_path, tmpdir / "out")

    def test_safe_extract_rejects_zip_symlink(self) -> None:
        with workspace_tmpdir("mineru-zip-") as tmpdir:
            zip_path = tmpdir / "bad.zip"
            zip_path.write_bytes(build_zip_with_symlink("secret.txt", "/etc/passwd"))
            with self.assertRaises(MinerUAPIError) as ctx:
                safe_extract_zip(zip_path, tmpdir / "out")
            self.assertIn("symlink", str(ctx.exception))


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        body: dict | None = None,
        content: bytes = b"",
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = body or {}
        self._content = content
        self.text = text or json.dumps(self._body)
        self.headers = headers or {}

    def json(self) -> dict:
        return self._body

    def iter_content(self, chunk_size: int):
        for index in range(0, len(self._content), chunk_size):
            yield self._content[index : index + chunk_size]


class FakeMinerUClient:
    def __init__(self, *, zip_bytes: bytes) -> None:
        self.zip_bytes = zip_bytes
        self.posts: list[dict] = []
        self.gets: list[dict] = []
        self.puts: list[dict] = []
        self.uploaded_bytes = b""

    def post(self, url, *, headers=None, json=None, timeout=None):
        self.posts.append({"url": url, "headers": headers or {}, "json": json or {}, "timeout": timeout})
        return FakeResponse(
            body={
                "code": 0,
                "data": {
                    "batch_id": "batch-001",
                    "file_urls": ["https://upload.example.test/put"],
                },
            }
        )

    def put(self, url, *, data=None, timeout=None):
        self.puts.append({"url": url, "timeout": timeout})
        self.uploaded_bytes = data.read() if data is not None else b""
        return FakeResponse(status_code=200, body={"ok": True})

    def get(self, url, *, headers=None, stream=False, timeout=None):
        self.gets.append({"url": url, "headers": headers or {}, "stream": stream, "timeout": timeout})
        if url == "https://download.example.test/result.zip":
            return FakeResponse(status_code=200, content=self.zip_bytes)
        return FakeResponse(
            body={
                "code": 0,
                "data": {
                    "extract_result": [
                        {
                            "state": "done",
                            "full_zip_url": "https://download.example.test/result.zip",
                            "extract_progress": {"extracted_pages": 2, "total_pages": 2},
                        }
                    ]
                },
            }
        )


def build_zip(files: dict[str, str]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def build_zip_with_symlink(link_name: str, target: str) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        info = zipfile.ZipInfo(link_name)
        info.create_system = 3  # Unix
        info.external_attr = 0o120777 << 16  # symlink
        archive.writestr(info, target)
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
