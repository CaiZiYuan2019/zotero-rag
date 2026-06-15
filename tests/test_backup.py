from __future__ import annotations

from contextlib import suppress
from pathlib import Path
import json
import sqlite3
import unittest
from unittest.mock import patch

from tests._support import workspace_tmpdir
from zoterorag.backup import (
    create_backup,
    plan_restore_backup,
    resolve_backup_manifest,
    restore_backup,
    restore_file,
    verify_manifest_files,
    _checkpoint_state_db,
    _checkpoint_sqlite,
)
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
                backend="sqlite-local",
            ),
        ),
    )


def backup_root_for(config: AppConfig) -> Path:
    return config.paths.data_dir / "backups"


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
                    out_dir=backup_root_for(config),
                    config_path=config_file,
                )

                self.assertTrue(result.manifest_path.is_file())
                self.assertEqual([], verify_manifest_files(result.manifest_path))
                self.assertTrue(
                    (result.backup_dir / "state" / "state.sqlite").is_file()
                )
                self.assertTrue(
                    (result.backup_dir / "config" / "config.toml").is_file()
                )
                self.assertTrue(
                    (result.backup_dir / "shadow" / "zotero.sqlite").is_file()
                )

                manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
                self.assertEqual("snapshot", manifest["mode"])
                self.assertIn("zotero_db_not_copied", manifest["source"])
                self.assertFalse((result.backup_dir / "zotero-source").exists())

                backups = ledger.list_backups()
                self.assertEqual(1, len(backups))
                self.assertEqual(result.backup_id, backups[0]["backup_id"])
                self.assertIn("source", backups[0]["manifest"])
                self.assertIn(
                    "zotero_storage_not_copied", backups[0]["manifest"]["source"]
                )
            finally:
                ledger.close()

    def test_full_backup_copies_vector_store_runtime_files(self) -> None:
        with workspace_tmpdir("backup-") as root:
            config = build_config(root)
            config.ensure_runtime_dirs()
            ledger = StateLedger(config.paths.state_db)
            try:
                vector_file = (
                    config.paths.vector_store_dir / "profile" / "vectors.sqlite"
                )
                vector_file.parent.mkdir(parents=True, exist_ok=True)
                vector_file.write_bytes(b"vector-data")

                result = create_backup(
                    config,
                    ledger,
                    mode="full",
                    out_dir=backup_root_for(config),
                    config_path=root / "missing-config.toml",
                )

                self.assertTrue(
                    (
                        result.backup_dir
                        / "vector_store"
                        / "profile"
                        / "vectors.sqlite"
                    ).is_file()
                )
                self.assertEqual([], verify_manifest_files(result.manifest_path))
            finally:
                ledger.close()

    def test_restore_plan_maps_only_runtime_files(self) -> None:
        with workspace_tmpdir("backup-restore-plan-") as root:
            config = build_config(root)
            config.ensure_runtime_dirs()
            ledger = StateLedger(config.paths.state_db)
            try:
                config_file = root / "config.toml"
                config_file.write_text("[paths]\n", encoding="utf-8")
                config.paths.shadow_db.write_bytes(b"shadow-copy")

                result = create_backup(
                    config,
                    ledger,
                    mode="snapshot",
                    out_dir=backup_root_for(config),
                    config_path=config_file,
                )
                plan = plan_restore_backup(config, result.manifest_path)

                self.assertEqual([], plan.errors)
                by_path = {item["path"]: item for item in plan.files}
                self.assertTrue(by_path["state/state.sqlite"]["restorable"])
                self.assertTrue(by_path["shadow/zotero.sqlite"]["restorable"])
                self.assertFalse(by_path["config/config.toml"]["restorable"])
                self.assertNotIn("backup_manifest.json", by_path)
                self.assertNotIn(
                    "zotero-source", json.dumps(plan.to_dict(), ensure_ascii=False)
                )
            finally:
                ledger.close()

    def test_confirmed_restore_creates_pre_restore_snapshot_and_restores_runtime_files(
        self,
    ) -> None:
        with workspace_tmpdir("backup-restore-") as root:
            config = build_config(root)
            config.ensure_runtime_dirs()
            ledger = StateLedger(config.paths.state_db)
            try:
                ledger.upsert_review_rule("ATTACH", "include", "before")
                vector_file = (
                    config.paths.vector_store_dir / "profile" / "vectors.sqlite"
                )
                vector_file.parent.mkdir(parents=True, exist_ok=True)
                vector_file.write_bytes(b"vector-before")
                config_file = root / "config.toml"
                config_file.write_text("[paths]\n", encoding="utf-8")

                backup = create_backup(
                    config,
                    ledger,
                    mode="full",
                    out_dir=backup_root_for(config),
                    config_path=config_file,
                )

                ledger.upsert_review_rule("ATTACH", "exclude", "after")
                vector_file.write_bytes(b"vector-after")

                restored = restore_backup(
                    config,
                    ledger,
                    manifest_path=backup.manifest_path,
                    pre_restore_out_dir=backup_root_for(config) / "pre-restore",
                    config_path=config_file,
                    confirm=True,
                    close_ledger_before_apply=True,
                )

                self.assertTrue(restored.applied)
                self.assertIsNotNone(restored.pre_restore_backup)
                self.assertTrue(
                    Path(restored.pre_restore_backup["manifest_path"]).is_file()
                )
                self.assertEqual(b"vector-before", vector_file.read_bytes())
            finally:
                with suppress(Exception):
                    ledger.close()

            reopened = StateLedger(config.paths.state_db)
            try:
                rules = reopened.list_review_rules()
                self.assertEqual(1, len(rules))
                self.assertEqual("include", rules[0]["decision"])
                self.assertEqual("before", rules[0]["reason"])
            finally:
                reopened.close()

    def test_create_backup_rejects_out_dir_outside_backup_root(self) -> None:
        with workspace_tmpdir("backup-out-dir-") as root:
            config = build_config(root)
            config.ensure_runtime_dirs()
            ledger = StateLedger(config.paths.state_db)
            try:
                for bad in (root / "outside", ".."):
                    with self.subTest(out_dir=bad):
                        with self.assertRaises(ValueError):
                            create_backup(
                                config,
                                ledger,
                                mode="snapshot",
                                out_dir=bad,
                            )
            finally:
                ledger.close()

    def test_create_backup_accepts_subdirectory_under_backup_root(self) -> None:
        with workspace_tmpdir("backup-subdir-") as root:
            config = build_config(root)
            config.ensure_runtime_dirs()
            ledger = StateLedger(config.paths.state_db)
            try:
                result = create_backup(
                    config,
                    ledger,
                    mode="snapshot",
                    out_dir=backup_root_for(config) / "sub",
                )
                self.assertTrue(
                    result.backup_dir.is_relative_to(backup_root_for(config))
                )
            finally:
                ledger.close()

    def test_restore_plan_rejects_manifest_path_escape(self) -> None:
        with workspace_tmpdir("backup-escape-") as root:
            config = build_config(root)
            config.ensure_runtime_dirs()
            ledger = StateLedger(config.paths.state_db)
            try:
                result = create_backup(
                    config,
                    ledger,
                    mode="snapshot",
                    out_dir=backup_root_for(config),
                )
                manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
                manifest["files"].append(
                    {
                        "path": "../escape.txt",
                        "size": 1,
                        "sha256": "a" * 64,
                    }
                )
                result.manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                plan = plan_restore_backup(config, result.manifest_path)
                self.assertTrue(any("invalid_path" in e for e in plan.errors))
                escape_items = [
                    item for item in plan.files if item["path"] == "../escape.txt"
                ]
                self.assertEqual(1, len(escape_items))
                self.assertFalse(escape_items[0]["restorable"])
            finally:
                ledger.close()

    def test_resolve_backup_manifest_rejects_outside_backup_root(self) -> None:
        with workspace_tmpdir("backup-resolve-") as root:
            config = build_config(root)
            config.ensure_runtime_dirs()
            ledger = StateLedger(config.paths.state_db)
            outside = root / "outside_manifest.json"
            outside.write_text("{}", encoding="utf-8")
            try:
                with self.assertRaises(ValueError):
                    resolve_backup_manifest(
                        ledger,
                        str(outside),
                        backup_root=backup_root_for(config),
                    )
            finally:
                ledger.close()

    def test_plan_restore_backup_rejects_manifest_with_dotdot(self) -> None:
        with workspace_tmpdir("backup-dotdot-") as root:
            config = build_config(root)
            config.ensure_runtime_dirs()
            ledger = StateLedger(config.paths.state_db)
            try:
                with self.assertRaises(ValueError):
                    plan_restore_backup(
                        config,
                        "../escape/backups/backup_manifest.json",
                        backup_root=backup_root_for(config),
                    )
            finally:
                ledger.close()

    def test_verify_manifest_files_detects_tampered_file(self) -> None:
        with workspace_tmpdir("backup-tamper-") as root:
            config = build_config(root)
            config.ensure_runtime_dirs()
            ledger = StateLedger(config.paths.state_db)
            try:
                result = create_backup(
                    config,
                    ledger,
                    mode="snapshot",
                    out_dir=backup_root_for(config),
                )
                state_backup = result.backup_dir / "state" / "state.sqlite"
                original = bytearray(state_backup.read_bytes())
                original[0] ^= 1
                state_backup.write_bytes(bytes(original))
                errors = verify_manifest_files(result.manifest_path)
                self.assertTrue(any("sha256" in e for e in errors))
            finally:
                ledger.close()

    def test_restore_file_verifies_sha256_before_replacing_target(self) -> None:
        with workspace_tmpdir("restore-file-") as root:
            source = root / "source.txt"
            target = root / "target.txt"
            source.write_text("hello", encoding="utf-8")
            expected = (
                verify_manifest_files  # avoid unused import warning; not used here
            )
            expected = (
                "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
            )

            restore_file(source, target, expected_sha256=expected)
            self.assertEqual("hello", target.read_text(encoding="utf-8"))

            source.write_text("world", encoding="utf-8")
            with self.assertRaises(ValueError):
                restore_file(source, target, expected_sha256=expected)
            self.assertEqual("hello", target.read_text(encoding="utf-8"))

    def test_restore_backup_fails_when_source_file_is_tampered(self) -> None:
        with workspace_tmpdir("backup-restore-tamper-") as root:
            config = build_config(root)
            config.ensure_runtime_dirs()
            ledger = StateLedger(config.paths.state_db)
            try:
                config_file = root / "config.toml"
                config_file.write_text("[paths]\n", encoding="utf-8")
                backup = create_backup(
                    config,
                    ledger,
                    mode="snapshot",
                    out_dir=backup_root_for(config),
                    config_path=config_file,
                )
                # Tamper the backed-up state file in-place while preserving its
                # size so the manifest size check passes and the sha256 mismatch
                # is detected.
                state_backup = backup.backup_dir / "state" / "state.sqlite"
                original = bytearray(state_backup.read_bytes())
                original[0] ^= 1
                state_backup.write_bytes(bytes(original))

                restored = restore_backup(
                    config,
                    ledger,
                    manifest_path=backup.manifest_path,
                    pre_restore_out_dir=backup_root_for(config) / "pre-restore",
                    config_path=config_file,
                    confirm=True,
                    close_ledger_before_apply=True,
                )

                self.assertFalse(restored.applied)
                self.assertTrue(any("sha256" in e for e in restored.errors))
            finally:
                with suppress(Exception):
                    ledger.close()

    def test_create_backup_defaults_to_runtime_config_path(self) -> None:
        with workspace_tmpdir("backup-default-config-") as root:
            config = build_config(root)
            config.ensure_runtime_dirs()
            ledger = StateLedger(config.paths.state_db)
            try:
                (root / "config").mkdir(parents=True, exist_ok=True)
                (root / "config" / "config.toml").write_text("[paths]\n", encoding="utf-8")
                result = create_backup(
                    config,
                    ledger,
                    mode="snapshot",
                    out_dir=backup_root_for(config),
                )
                self.assertTrue(
                    (result.backup_dir / "config" / "config.toml").is_file()
                )
            finally:
                ledger.close()

    def test_create_backup_calls_checkpoint_helpers(self) -> None:
        with workspace_tmpdir("backup-checkpoint-") as root:
            config = build_config(root)
            config.ensure_runtime_dirs()
            ledger = StateLedger(config.paths.state_db)
            config.paths.shadow_db.write_bytes(b"shadow")
            vector_file = config.paths.vector_store_dir / "vectors.sqlite"
            vector_file.parent.mkdir(parents=True, exist_ok=True)
            vector_file.write_bytes(b"vectors")
            try:
                calls = {"state": False, "shadow": False, "runtime": []}

                def fake_state(ledger_arg):
                    calls["state"] = True

                def fake_shadow(path):
                    calls["shadow"] = True

                def fake_runtime(source_root):
                    calls["runtime"].append(str(source_root))

                with (
                    patch("zoterorag.backup._checkpoint_state_db", fake_state),
                    patch("zoterorag.backup._checkpoint_sqlite", fake_shadow),
                    patch(
                        "zoterorag.backup._checkpoint_runtime_sqlite_files",
                        fake_runtime,
                    ),
                ):
                    create_backup(
                        config,
                        ledger,
                        mode="full",
                        out_dir=backup_root_for(config),
                    )

                self.assertTrue(calls["state"])
                self.assertTrue(calls["shadow"])
                self.assertIn(str(config.paths.vector_store_dir), calls["runtime"])
            finally:
                ledger.close()

    def test_checkpoint_state_db_truncates_wal(self) -> None:
        with workspace_tmpdir("checkpoint-") as root:
            db_path = root / "state.sqlite"
            ledger = StateLedger(db_path)
            try:
                ledger.upsert_review_rule("ATTACH", "include", "x")
                # Force WAL frames by writing through a separate connection.
                conn = sqlite3.connect(db_path)
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY)"
                    )
                    conn.execute("INSERT INTO test_table VALUES (1)")
                    conn.commit()
                finally:
                    conn.close()

                _checkpoint_state_db(ledger)

                # After TRUNCATE the WAL file should be reset (zero bytes).
                wal_path = root / "state.sqlite-wal"
                if wal_path.exists():
                    self.assertEqual(0, wal_path.stat().st_size)
            finally:
                ledger.close()

    def test_checkpoint_sqlite_skips_non_wal_database(self) -> None:
        with workspace_tmpdir("checkpoint-nonwal-") as root:
            db_path = root / "plain.sqlite"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            finally:
                conn.close()
            # Should complete without error even though journal_mode is DELETE.
            _checkpoint_sqlite(db_path)


if __name__ == "__main__":
    unittest.main()
