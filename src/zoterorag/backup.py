from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any
import uuid

from .config import AppConfig
from .db import StateLedger


BACKUP_MANIFEST = "backup_manifest.json"


@dataclass(frozen=True)
class BackupResult:
    backup_id: str
    mode: str
    backup_dir: Path
    manifest_path: Path
    files: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "mode": self.mode,
            "backup_dir": str(self.backup_dir),
            "manifest_path": str(self.manifest_path),
            "files": self.files,
        }


@dataclass(frozen=True)
class RestoreResult:
    manifest_path: Path
    mode: str
    applied: bool
    files: list[dict[str, Any]]
    errors: list[str]
    pre_restore_backup: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "mode": self.mode,
            "applied": self.applied,
            "files": self.files,
            "errors": self.errors,
            "pre_restore_backup": self.pre_restore_backup,
        }


def create_backup(
    config: AppConfig,
    ledger: StateLedger,
    *,
    mode: str,
    out_dir: str | Path,
    config_path: str | Path = "config/config.example.toml",
) -> BackupResult:
    if mode not in {"snapshot", "full"}:
        raise ValueError("mode must be 'snapshot' or 'full'")

    backup_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    root = Path(out_dir).expanduser().resolve() / backup_id
    root.mkdir(parents=True, exist_ok=False)

    files: list[dict[str, Any]] = []
    state_target = root / "state" / "state.sqlite"
    copy_sqlite_database(config.paths.state_db, state_target)
    files.append(file_manifest_entry(state_target, "state/state.sqlite"))

    config_source = Path(config_path)
    if config_source.is_file():
        config_target = root / "config" / config_source.name
        copy_file(config_source, config_target)
        files.append(file_manifest_entry(config_target, f"config/{config_source.name}"))

    if config.paths.shadow_db.is_file():
        shadow_target = root / "shadow" / "zotero.sqlite"
        copy_file(config.paths.shadow_db, shadow_target)
        files.append(file_manifest_entry(shadow_target, "shadow/zotero.sqlite"))

    if mode == "full":
        files.extend(copy_runtime_tree(config.paths.vector_store_dir, root / "vector_store", root))
        for dirname in ("extract_cache", "normalized", "embedding_cache"):
            source = config.paths.data_dir / dirname
            files.extend(copy_runtime_tree(source, root / dirname, root))

    manifest = {
        "backup_id": backup_id,
        "mode": mode,
        "backup_dir": str(root),
        "manifest_path": str(root / BACKUP_MANIFEST),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "data_dir": str(config.paths.data_dir),
            "state_db": str(config.paths.state_db),
            "shadow_db": str(config.paths.shadow_db),
            "vector_store_dir": str(config.paths.vector_store_dir),
            # Intentionally record but do not copy Zotero source paths. The
            # project may read Zotero through a shadow, but backups must not
            # duplicate or mutate the live Zotero library unless a future
            # explicit include_zotero_source option is added.
            "zotero_db_not_copied": str(config.paths.zotero_db),
            "zotero_storage_not_copied": str(config.paths.zotero_storage),
        },
        "files": files,
    }
    manifest_path = root / BACKUP_MANIFEST
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(file_manifest_entry(manifest_path, BACKUP_MANIFEST))

    result = BackupResult(
        backup_id=backup_id,
        mode=mode,
        backup_dir=root,
        manifest_path=manifest_path,
        files=files,
    )
    ledger.add_backup_record(backup_id, mode, root, manifest)
    return result


def plan_restore_backup(config: AppConfig, manifest_path: str | Path) -> RestoreResult:
    """Validate a backup manifest and map restorable files to runtime targets.

    This function is read-only. It never mutates the current runtime and never
    restores config files or Zotero source data. Config files are included in
    backups for auditability, but restoring them implicitly could repoint the
    application at an unexpected Zotero library.
    """

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = read_manifest(manifest_file)
    errors = verify_manifest_files(manifest_file)
    files = build_restore_file_plan(config, manifest_file, manifest)
    return RestoreResult(
        manifest_path=manifest_file,
        mode=str(manifest.get("mode", "unknown")),
        applied=False,
        files=files,
        errors=errors,
    )


def restore_backup(
    config: AppConfig,
    ledger: StateLedger,
    *,
    manifest_path: str | Path,
    pre_restore_out_dir: str | Path,
    config_path: str | Path = "config/config.example.toml",
    confirm: bool = False,
    close_ledger_before_apply: bool = False,
) -> RestoreResult:
    """Restore runtime files from a verified backup.

    Actual restore is intentionally opt-in through `confirm=True`. When applying
    inside the CLI, pass `close_ledger_before_apply=True` so Windows can replace
    the current state SQLite file after the pre-restore snapshot is created.
    """

    planned = plan_restore_backup(config, manifest_path)
    if planned.errors or not confirm:
        return planned

    pre_restore = create_backup(
        config,
        ledger,
        mode="snapshot",
        out_dir=pre_restore_out_dir,
        config_path=config_path,
    )
    if close_ledger_before_apply:
        ledger.close()

    applied_files: list[dict[str, Any]] = []
    for item in planned.files:
        if not item.get("restorable", False):
            applied_files.append({**item, "applied": False})
            continue
        source = Path(item["source_path"])
        target = Path(item["target_path"])
        restore_file(source, target)
        applied_files.append({**item, "applied": True})

    return RestoreResult(
        manifest_path=planned.manifest_path,
        mode=planned.mode,
        applied=True,
        files=applied_files,
        errors=[],
        pre_restore_backup=pre_restore.to_dict(),
    )


def resolve_backup_manifest(ledger: StateLedger, backup_ref: str | Path) -> Path:
    ref_path = Path(backup_ref)
    if ref_path.is_file():
        return ref_path
    for backup in ledger.list_backups():
        if backup["backup_id"] == str(backup_ref):
            manifest_path = Path(backup["manifest"].get("manifest_path") or Path(backup["path"]) / BACKUP_MANIFEST)
            return manifest_path
    raise KeyError(f"backup not found: {backup_ref}")


def copy_sqlite_database(source: str | Path, target: str | Path) -> None:
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.is_file():
        raise FileNotFoundError(f"SQLite source not found: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    uri = f"file:{source_path.resolve().as_posix()}?mode=ro"
    source_conn = sqlite3.connect(uri, uri=True, timeout=5)
    try:
        target_conn = sqlite3.connect(target_path)
        try:
            source_conn.backup(target_conn)
        finally:
            target_conn.close()
    finally:
        source_conn.close()


def copy_file(source: str | Path, target: str | Path) -> None:
    source_path = Path(source)
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def copy_runtime_tree(source: Path, target: Path, backup_root: Path) -> list[dict[str, Any]]:
    if not source.exists():
        return []
    entries: list[dict[str, Any]] = []
    source_root = source.resolve()
    backup_root_resolved = backup_root.resolve()
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if backup_root_resolved in resolved.parents:
            continue
        relative = path.relative_to(source_root)
        target_path = target / relative
        copy_file(path, target_path)
        entries.append(file_manifest_entry(target_path, str(target_path.relative_to(backup_root))))
    return entries


def read_manifest(manifest_path: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"backup manifest not found: {manifest_file}")
    return json.loads(manifest_file.read_text(encoding="utf-8"))


def build_restore_file_plan(
    config: AppConfig,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    root = manifest_path.parent
    planned: list[dict[str, Any]] = []
    for item in manifest.get("files", []):
        relative_path = str(item["path"]).replace("\\", "/")
        source = (root / relative_path).resolve()
        target = restore_target_for_relative_path(config, relative_path)
        if target is None:
            planned.append(
                {
                    "path": relative_path,
                    "source_path": str(source),
                    "target_path": None,
                    "restorable": False,
                    "reason": "not_runtime_restore_target",
                }
            )
            continue
        planned.append(
            {
                "path": relative_path,
                "source_path": str(source),
                "target_path": str(target),
                "restorable": True,
            }
        )
    return planned


def restore_target_for_relative_path(config: AppConfig, relative_path: str) -> Path | None:
    if relative_path == "state/state.sqlite":
        return config.paths.state_db
    if relative_path == "shadow/zotero.sqlite":
        return config.paths.shadow_db
    runtime_prefixes = {
        "vector_store/": config.paths.vector_store_dir,
        "extract_cache/": config.paths.extract_cache_dir,
        "normalized/": config.paths.normalized_dir,
        "embedding_cache/": config.paths.embedding_cache_dir,
    }
    for prefix, target_root in runtime_prefixes.items():
        if relative_path.startswith(prefix):
            suffix = relative_path.removeprefix(prefix)
            target = target_root / Path(suffix)
            ensure_restore_target_allowed(config, target)
            return target
    return None


def ensure_restore_target_allowed(config: AppConfig, target: Path) -> None:
    resolved = target.resolve()
    data_dir = config.paths.data_dir.resolve()
    if resolved == data_dir or data_dir in resolved.parents:
        return
    raise ValueError(f"restore target escapes runtime data directory: {target}")


def restore_file(source: str | Path, target: str | Path) -> None:
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.is_file():
        raise FileNotFoundError(f"backup file missing during restore: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_target = target_path.with_name(f"{target_path.name}.restore_tmp")
    shutil.copy2(source_path, temporary_target)
    temporary_target.replace(target_path)


def file_manifest_entry(path: str | Path, relative_path: str) -> dict[str, Any]:
    file_path = Path(path)
    stat = file_path.stat()
    return {
        "path": relative_path.replace("\\", "/"),
        "size": stat.st_size,
        "sha256": sha256_file(file_path),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_files(manifest_path: str | Path) -> list[str]:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    root = manifest_file.parent
    errors = []
    for item in manifest.get("files", []):
        path = root / item["path"]
        if not path.is_file():
            errors.append(f"missing:{item['path']}")
            continue
        if path.stat().st_size != item["size"]:
            errors.append(f"size:{item['path']}")
            continue
        if sha256_file(path) != item["sha256"]:
            errors.append(f"sha256:{item['path']}")
    return errors
