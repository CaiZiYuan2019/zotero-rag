from __future__ import annotations

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
                )
            finally:
                ledger.close()

            _, reopened = initialize_runtime(config_path)
            try:
                after = reopened.list_vector_indexes()[0]
                self.assertEqual(3, after["document_count"])
                self.assertEqual(9, after["chunk_count"])
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
