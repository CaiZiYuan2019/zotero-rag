from __future__ import annotations

from pathlib import Path
import json
import unittest

from tests._support import workspace_tmpdir
from zoterorag.backup import create_backup, verify_manifest_files
from zoterorag.config import AppConfig, EmbeddingProfile, PathsConfig, ServerConfig
from zoterorag.db import StateLedger


def build_config(root: Path) -> AppConfig:
    return AppConfig(
        paths=PathsConfig(
            zotero_db=root / "zotero-source" / "zotero.sqlite",
            zotero_storage=root / "zotero-source" / "storage",
            data_dir=root / "data",
        ),
        server=ServerConfig(require_api_token=True),
        embedding_profiles=(
            EmbeddingProfile(
                name="test_text",
                provider="stub",
                model="stub",
                dimension=3,
                modality="text",
                default_for_text=True,
            ),
        ),
    )


class BackupTests(unittest.TestCase):
    def test_snapshot_backup_copies_state_config_shadow_and_manifest(self) -> None:
        with workspace_tmpdir("backup-") as root:
            config = build_config(root)
            config.ensure_runtime_dirs()
            ledger = StateLedger(config.paths.state_db)
            try:
                ledger.upsert_review_rule("ATTACH", "include", "keep")
                config_file = root / "config.toml"
                config_file.write_text("[paths]\n", encoding="utf-8")
                config.paths.shadow_db.write_bytes(b"shadow-copy")

                result = create_backup(
                    config,
                    ledger,
                    mode="snapshot",
                    out_dir=root / "backup-out",
                    config_path=config_file,
                )

                self.assertTrue(result.manifest_path.is_file())
                self.assertEqual([], verify_manifest_files(result.manifest_path))
                self.assertTrue((result.backup_dir / "state" / "state.sqlite").is_file())
                self.assertTrue((result.backup_dir / "config" / "config.toml").is_file())
                self.assertTrue((result.backup_dir / "shadow" / "zotero.sqlite").is_file())

                manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
                self.assertEqual("snapshot", manifest["mode"])
                self.assertIn("zotero_db_not_copied", manifest["source"])
                self.assertFalse((result.backup_dir / "zotero-source").exists())

                backups = ledger.list_backups()
                self.assertEqual(1, len(backups))
                self.assertEqual(result.backup_id, backups[0]["backup_id"])
                self.assertIn("source", backups[0]["manifest"])
                self.assertIn("zotero_storage_not_copied", backups[0]["manifest"]["source"])
            finally:
                ledger.close()

    def test_full_backup_copies_vector_store_runtime_files(self) -> None:
        with workspace_tmpdir("backup-") as root:
            config = build_config(root)
            config.ensure_runtime_dirs()
            ledger = StateLedger(config.paths.state_db)
            try:
                vector_file = config.paths.vector_store_dir / "profile" / "vectors.sqlite"
                vector_file.parent.mkdir(parents=True, exist_ok=True)
                vector_file.write_bytes(b"vector-data")

                result = create_backup(
                    config,
                    ledger,
                    mode="full",
                    out_dir=root / "backup-out",
                    config_path=root / "missing-config.toml",
                )

                self.assertTrue((result.backup_dir / "vector_store" / "profile" / "vectors.sqlite").is_file())
                self.assertEqual([], verify_manifest_files(result.manifest_path))
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
