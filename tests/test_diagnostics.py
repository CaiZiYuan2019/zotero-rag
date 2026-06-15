from __future__ import annotations

import unittest

from tests._support import workspace_tmpdir
from tests.test_zotero_shadow import build_minimal_zotero_db
from zoterorag.config import EmbeddingProfile
from zoterorag.config import load_config
from zoterorag.db import StateLedger
from zoterorag.diagnostics import run_runtime_diagnostics
from zoterorag.runtime import initialize_runtime


class RuntimeDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_reports_local_readiness_without_refreshing_shadow(self) -> None:
        with workspace_tmpdir("diagnostics-") as tmpdir:
            config_path = tmpdir / "config.toml"
            source_db = tmpdir / "zotero.sqlite"
            storage = tmpdir / "storage"
            data_dir = tmpdir / "data"
            storage.mkdir()
            source_db.write_bytes(b"source-placeholder")
            (tmpdir / ".env").write_text("BAILIAN_KEY=test-key\n", encoding="utf-8")
            config_path.write_text(
                f"""
[paths]
zotero_db = "{source_db.as_posix()}"
zotero_storage = "{storage.as_posix()}"
data_dir = "{data_dir.as_posix()}"

[server]
require_api_token = true

[[embedding_profiles]]
name = "stub_text"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
enabled = true
default_for_text = true
default_for_multimodal = false
""",
                encoding="utf-8",
            )
            config, ledger = initialize_runtime(config_path)
            try:
                build_minimal_zotero_db(config.paths.shadow_db)

                report = run_runtime_diagnostics(config, ledger)

                self.assertTrue(report["ok"])
                self.assertTrue(report["checks"]["state"]["ok"])
                self.assertEqual("wal", report["checks"]["state"]["journal_mode"].casefold())
                self.assertGreaterEqual(report["checks"]["state"]["busy_timeout_ms"], 30000)
                self.assertTrue(report["checks"]["zotero_source"]["source_exists"])
                self.assertFalse(report["checks"]["zotero_source"]["opened_source_db"])
                self.assertEqual("readable", report["checks"]["shadow"]["status"])
                self.assertEqual(1, report["checks"]["shadow"]["pdf_count"])
                self.assertEqual("skipped", report["checks"]["vector_staging_self_test"]["status"])
                self.assertTrue(report["checks"]["api_access"]["external_without_token_denied"])
                self.assertTrue(report["checks"]["providers"]["ok"])
                self.assertIn("mineru", report["checks"]["providers"])
                self.assertIn("qwen3vl_embedding", report["checks"]["providers"])
                self.assertFalse(report["checks"]["external_execution"]["mineru_executed"])
                self.assertFalse(report["checks"]["external_execution"]["embedding_executed"])
            finally:
                ledger.close()

    def test_vector_verification_is_explicit(self) -> None:
        with workspace_tmpdir("diagnostics-vectors-") as tmpdir:
            config_path = tmpdir / "config.toml"
            source_db = tmpdir / "zotero.sqlite"
            storage = tmpdir / "storage"
            data_dir = tmpdir / "data"
            storage.mkdir()
            source_db.write_bytes(b"source-placeholder")
            config_path.write_text(
                f"""
[paths]
zotero_db = "{source_db.as_posix()}"
zotero_storage = "{storage.as_posix()}"
data_dir = "{data_dir.as_posix()}"

[[embedding_profiles]]
name = "stub_text"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
enabled = true
default_for_text = true
default_for_multimodal = false
""",
                encoding="utf-8",
            )
            config, ledger = initialize_runtime(config_path)
            try:
                build_minimal_zotero_db(config.paths.shadow_db)

                shallow = run_runtime_diagnostics(config, ledger, verify_vectors=False)
                deep = run_runtime_diagnostics(config, ledger, verify_vectors=True)

                self.assertTrue(shallow["checks"]["vectors"]["ok"])
                self.assertFalse(shallow["checks"]["vectors"]["verify_vectors"])
                self.assertFalse(deep["checks"]["vectors"]["ok"])
                self.assertTrue(deep["checks"]["vectors"]["verify_vectors"])
                self.assertIn("missing_vector_store", deep["checks"]["vectors"]["indexes"][0]["verification"]["errors"][0])
            finally:
                ledger.close()

    def test_vector_staging_self_test_is_explicit_and_temporary(self) -> None:
        with workspace_tmpdir("diagnostics-vector-self-test-") as tmpdir:
            config = make_config(tmpdir)
            ledger = StateLedger(config.paths.state_db)
            try:
                build_minimal_zotero_db(config.paths.shadow_db)

                report = run_runtime_diagnostics(config, ledger, self_test_vector_store=True)
                check = report["checks"]["vector_staging_self_test"]

                self.assertTrue(report["ok"])
                self.assertTrue(check["ok"])
                self.assertEqual("passed", check["status"])
                self.assertTrue(check["checks"]["staged_invisible_before_publish"])
                self.assertTrue(check["checks"]["published_visible"])
                self.assertTrue(check["checks"]["replacement_invisible_before_republish"])
                self.assertTrue(check["checks"]["replacement_visible_after_republish"])
                self.assertEqual({"documents": 1, "chunks": 1}, check["checks"]["counts_after_republish"])
                self.assertIn("zoterorag-diagnostics", check["temp_root"])
            finally:
                ledger.close()

    def test_vector_provenance_reports_unattributed_and_tracked_indexes(self) -> None:
        with workspace_tmpdir("diagnostics-provenance-") as tmpdir:
            ledger = StateLedger(tmpdir / "state.sqlite")
            try:
                ledger.upsert_embedding_profiles(
                    [
                        EmbeddingProfile(
                            name="stub_text",
                            provider="stub",
                            model="stub",
                            dimension=8,
                            modality="text",
                            enabled=True,
                            default_for_text=True,
                            backend="sqlite-local",
                        )
                    ]
                )
                ledger.register_vector_index(
                    profile_name="stub_text",
                            backend="sqlite-local",
                    path=tmpdir / "vectors" / "stub_text" / "vectors.sqlite",
                    document_count=1,
                    chunk_count=3,
                    active=True,
                    active_version="batch-live",
                )
                config = make_config(tmpdir)
                build_minimal_zotero_db(config.paths.shadow_db)

                unattributed = run_runtime_diagnostics(config, ledger)["checks"]["vectors"]["indexes"][0]
                self.assertEqual("unattributed", unattributed["provenance"]["status"])
                self.assertFalse(unattributed["provenance"]["active_version_has_completed_batch"])

                ledger.upsert_embedding_batch(
                    batch_hash="batch-live",
                    profile_name="stub_text",
                    profile_hash="profile-hash",
                    document_id="DOC1",
                    chunk_type="text",
                    chunk_count=3,
                    status="completed",
                    provider="stub",
                    model="stub",
                )
                tracked = run_runtime_diagnostics(config, ledger)["checks"]["vectors"]["indexes"][0]
                self.assertEqual("tracked_active_version", tracked["provenance"]["status"])
                self.assertTrue(tracked["provenance"]["active_version_has_completed_batch"])
            finally:
                ledger.close()


def make_config(tmpdir):
    config_path = tmpdir / "config.toml"
    source_db = tmpdir / "zotero.sqlite"
    storage = tmpdir / "storage"
    storage.mkdir(exist_ok=True)
    source_db.write_bytes(b"source-placeholder")
    (tmpdir / ".env").write_text("BAILIAN_KEY=test-key\n", encoding="utf-8")
    config_path.write_text(
        f"""
[paths]
zotero_db = "{source_db.as_posix()}"
zotero_storage = "{storage.as_posix()}"
data_dir = "{(tmpdir / 'data').as_posix()}"

[[embedding_profiles]]
name = "stub_text"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
enabled = true
default_for_text = true
default_for_multimodal = false
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    config.ensure_runtime_dirs()
    return config


if __name__ == "__main__":
    unittest.main()
