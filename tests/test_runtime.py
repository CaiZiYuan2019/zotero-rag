from __future__ import annotations

from pathlib import Path
import unittest

from tests._support import workspace_tmpdir
from zoterorag.db import StateLedger
from zoterorag.runtime import initialize_runtime


class RuntimeInitializationTests(unittest.TestCase):
    def test_vector_index_counts_are_not_reset_by_profile_bootstrap(self) -> None:
        with workspace_tmpdir("runtime-init-") as tmpdir:
            config_path = tmpdir / "config.toml"
            data_dir = tmpdir / "data"
            config_path.write_text(
                f"""
[paths]
zotero_db = "{(tmpdir / 'zotero.sqlite').as_posix()}"
zotero_storage = "{(tmpdir / 'storage').as_posix()}"
data_dir = "{data_dir.as_posix()}"

[[embedding_profiles]]
name = "profile-a"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
backend = "sqlite-local"
enabled = true
default_for_text = true
default_for_multimodal = false
""",
                encoding="utf-8",
            )
            ledger = StateLedger(data_dir / "state" / "state.sqlite")
            try:
                ledger.register_vector_index(
                    profile_name="profile-a",
                    backend="sqlite-local",
                    path=data_dir / "vector_store" / "profile-a" / "vectors.sqlite",
                    document_count=3,
                    chunk_count=9,
                    active=True,
                    active_version="batch-live",
                )
            finally:
                ledger.close()

            _, reopened = initialize_runtime(config_path)
            try:
                after = reopened.list_vector_indexes()[0]
                self.assertEqual(3, after["document_count"])
                self.assertEqual(9, after["chunk_count"])
                self.assertEqual("batch-live", after["active_version"])
            finally:
                reopened.close()

    def test_default_backend_is_lancedb_with_directory_path(self) -> None:
        with workspace_tmpdir("runtime-backend-") as tmpdir:
            config_path = tmpdir / "config.toml"
            data_dir = tmpdir / "data"
            config_path.write_text(
                f"""
[paths]
zotero_db = "{(tmpdir / 'zotero.sqlite').as_posix()}"
zotero_storage = "{(tmpdir / 'storage').as_posix()}"
data_dir = "{data_dir.as_posix()}"

[[embedding_profiles]]
name = "profile-lance"
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
                indexes = {idx["profile_name"]: idx for idx in ledger.list_vector_indexes()}
                self.assertIn("profile-lance", indexes)
                self.assertEqual("lancedb", indexes["profile-lance"]["backend"])
                self.assertEqual(
                    Path(data_dir / "vector_store" / "profile-lance").resolve(),
                    Path(indexes["profile-lance"]["path"]).resolve(),
                )
            finally:
                ledger.close()

    def test_sqlite_local_backend_uses_file_path(self) -> None:
        with workspace_tmpdir("runtime-sqlite-") as tmpdir:
            config_path = tmpdir / "config.toml"
            data_dir = tmpdir / "data"
            config_path.write_text(
                f"""
[paths]
zotero_db = "{(tmpdir / 'zotero.sqlite').as_posix()}"
zotero_storage = "{(tmpdir / 'storage').as_posix()}"
data_dir = "{data_dir.as_posix()}"

[[embedding_profiles]]
name = "profile-sqlite"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
backend = "sqlite-local"
enabled = true
default_for_text = true
default_for_multimodal = false
""",
                encoding="utf-8",
            )
            config, ledger = initialize_runtime(config_path)
            try:
                indexes = {idx["profile_name"]: idx for idx in ledger.list_vector_indexes()}
                self.assertEqual("sqlite-local", indexes["profile-sqlite"]["backend"])
                self.assertEqual(
                    Path(data_dir / "vector_store" / "profile-sqlite" / "vectors.sqlite").resolve(),
                    Path(indexes["profile-sqlite"]["path"]).resolve(),
                )
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
