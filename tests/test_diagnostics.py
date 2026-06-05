from __future__ import annotations

import unittest

from tests._support import workspace_tmpdir
from tests.test_zotero_shadow import build_minimal_zotero_db
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
                self.assertTrue(report["checks"]["zotero_source"]["source_exists"])
                self.assertFalse(report["checks"]["zotero_source"]["opened_source_db"])
                self.assertEqual("readable", report["checks"]["shadow"]["status"])
                self.assertEqual(1, report["checks"]["shadow"]["pdf_count"])
                self.assertTrue(report["checks"]["api_access"]["external_without_token_denied"])
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


if __name__ == "__main__":
    unittest.main()
