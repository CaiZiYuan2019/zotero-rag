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


def _resolve_backup_root(config: AppConfig) -> Path:
    return (config.paths.data_dir / "backups").resolve()


def _validate_out_dir(out_dir: str | Path, backup_root: Path) -> Path:
    parsed = Path(out_dir).expanduser()
    if not parsed.is_absolute():
        parsed = backup_root / parsed
    resolved = parsed.resolve()
    root_resolved = backup_root.resolve()
    if resolved != root_resolved and not resolved.is_relative_to(root_resolved):
        raise ValueError(
            f"backup out_dir must resolve under the configured backup root: {root_resolved}"
        )
    return resolved


def _validate_backup_relative_path(relative_path: str) -> None:
    normalized = relative_path.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"backup path must be relative and contain no '..': {relative_path}"
        )


def _checkpoint_state_db(ledger: StateLedger) -> None:
    """Freeze in-process writes and checkpoint the ledger WAL.

    Must be called while no other thread in this process is writing to the
    ledger. The ledger's internal lock serializes access for the duration of the
    checkpoint.
    """
    with ledger.conn as raw_conn:
        cur = raw_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        busy, _log, _checkpointed = cur.fetchone()
        if busy:
            raise RuntimeError("state database checkpoint is busy; backup aborted")


_SQLITE_MAGIC = b"SQLite format 3\x00"


def _is_sqlite_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < len(_SQLITE_MAGIC):
        return False
    with path.open("rb") as handle:
        return handle.read(len(_SQLITE_MAGIC)) == _SQLITE_MAGIC


def _checkpoint_sqlite(path: str | Path) -> None:
    """Checkpoint a standalone SQLite file if it is using WAL mode."""
    db_path = Path(path)
    if not _is_sqlite_file(db_path):
        return
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        if journal != "wal":
            return
        cur = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        busy, _log, _checkpointed = cur.fetchone()
        if busy:
            raise RuntimeError(f"sqlite checkpoint is busy for {db_path}")
    finally:
        conn.close()


def _checkpoint_runtime_sqlite_files(source_root: Path) -> None:
    """Checkpoint any *.sqlite files under a runtime directory before copying.

    Runtime directories may contain opaque files with a ``.sqlite`` suffix, so
    checkpoint errors are treated as best-effort and do not abort the backup.
    """
    if not source_root.exists():
        return
    for path in source_root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".sqlite":
            try:
                _checkpoint_sqlite(path)
            except sqlite3.Error:
                continue


def create_backup(
    config: AppConfig,
    ledger: StateLedger,
    *,
    mode: str,
    out_dir: str | Path,
    config_path: str | Path = "config/config.toml",
) -> BackupResult:
    if mode not in {"snapshot", "full"}:
        raise ValueError("mode must be 'snapshot' or 'full'")

    backup_root = _resolve_backup_root(config)
    backup_root.mkdir(parents=True, exist_ok=True)
    validated_out = _validate_out_dir(out_dir, backup_root)

    # Freeze writes to the ledger and checkpoint WAL so the state copy is
    # internally consistent. Other SQLite runtime files are checkpointed below
    # just before they are copied.
    _checkpoint_state_db(ledger)

    backup_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    root = validated_out / backup_id
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
        _checkpoint_sqlite(config.paths.shadow_db)
        shadow_target = root / "shadow" / "zotero.sqlite"
        copy_file(config.paths.shadow_db, shadow_target)
        files.append(file_manifest_entry(shadow_target, "shadow/zotero.sqlite"))

    if mode == "full":
        _checkpoint_runtime_sqlite_files(config.paths.vector_store_dir)
        files.extend(
            copy_runtime_tree(
                config.paths.vector_store_dir, root / "vector_store", root
            )
        )
        for dirname in ("extract_cache", "normalized", "embedding_cache"):
            source = config.paths.data_dir / dirname
            _checkpoint_runtime_sqlite_files(source)
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
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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


def plan_restore_backup(
    config: AppConfig,
    manifest_path: str | Path,
    *,
    backup_root: Path | None = None,
) -> RestoreResult:
    """Validate a backup manifest and map restorable files to runtime targets.

    This function is read-only. It never mutates the current runtime and never
    restores config files or Zotero source data. Config files are included in
    backups for auditability, but restoring them implicitly could repoint the
    application at an unexpected Zotero library.
    """

    root = backup_root or _resolve_backup_root(config)
    manifest_file = _validate_manifest_path(manifest_path, root)
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
    config_path: str | Path = "config/config.toml",
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

    manifest = read_manifest(planned.manifest_path)
    expected_by_path = {
        item["path"]: item.get("sha256") for item in manifest.get("files", [])
    }
    backup_root = planned.manifest_path.parent

    applied_files: list[dict[str, Any]] = []
    errors: list[str] = []
    for item in planned.files:
        if not item.get("restorable", False):
            applied_files.append({**item, "applied": False})
            continue
        source = Path(item["source_path"])
        target = Path(item["target_path"])
        try:
            restore_file(
                source,
                target,
                expected_sha256=expected_by_path.get(item["path"]),
                allowed_root=backup_root,
            )
            applied_files.append({**item, "applied": True})
        except (FileNotFoundError, ValueError) as exc:
            applied_files.append({**item, "applied": False, "error": str(exc)})
            errors.append(f"{item['path']}: {exc}")

    return RestoreResult(
        manifest_path=planned.manifest_path,
        mode=planned.mode,
        applied=confirm and not errors,
        files=applied_files,
        errors=errors,
        pre_restore_backup=pre_restore.to_dict(),
    )


def _validate_manifest_path(manifest_path: str | Path, backup_root: Path) -> Path:
    parsed = Path(manifest_path)
    if ".." in parsed.parts:
        raise ValueError(f"backup manifest path must not contain '..': {manifest_path}")
    if parsed.is_absolute():
        resolved = parsed.resolve()
    else:
        resolved = (backup_root / parsed).resolve()
    root_resolved = backup_root.resolve()
    if resolved != root_resolved and not resolved.is_relative_to(root_resolved):
        raise ValueError(
            f"backup manifest path must resolve under the configured backup root: {root_resolved}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"backup manifest not found: {manifest_path}")
    return resolved


def resolve_backup_manifest(
    ledger: StateLedger,
    backup_ref: str | Path,
    *,
    backup_root: Path | None = None,
) -> Path:
    root = backup_root.resolve() if backup_root else None
    ref_path = Path(backup_ref)
    if ref_path.is_file():
        if root is None:
            root = _resolve_backup_root_from_path(ref_path)
        return _validate_manifest_path(ref_path, root)
    if root is None:
        raise ValueError("backup_root is required when resolving by backup_id")
    for backup in ledger.list_backups():
        if backup["backup_id"] == str(backup_ref):
            manifest_path = Path(
                backup["manifest"].get("manifest_path")
                or Path(backup["path"]) / BACKUP_MANIFEST
            )
            return _validate_manifest_path(manifest_path, root)
    raise KeyError(f"backup not found: {backup_ref}")


def _resolve_backup_root_from_path(path: Path) -> Path:
    """Infer a likely backup root from an existing manifest path.

    Walks up the tree looking for a directory named ``backups``; otherwise
    returns the manifest's parent directory so callers can still validate.
    """
    resolved = path.resolve()
    for parent in resolved.parents:
        if parent.name == "backups":
            return parent
    return resolved.parent


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


def copy_runtime_tree(
    source: Path, target: Path, backup_root: Path
) -> list[dict[str, Any]]:
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
        entries.append(
            file_manifest_entry(target_path, str(target_path.relative_to(backup_root)))
        )
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
        try:
            _validate_backup_relative_path(relative_path)
        except ValueError as exc:
            planned.append(
                {
                    "path": relative_path,
                    "source_path": None,
                    "target_path": None,
                    "restorable": False,
                    "reason": "invalid_relative_path",
                    "error": str(exc),
                }
            )
            continue
        source = (root / relative_path).resolve()
        if not source.is_relative_to(root.resolve()):
            planned.append(
                {
                    "path": relative_path,
                    "source_path": str(source),
                    "target_path": None,
                    "restorable": False,
                    "reason": "path_escapes_backup_root",
                }
            )
            continue
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


def restore_target_for_relative_path(
    config: AppConfig, relative_path: str
) -> Path | None:
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


def restore_file(
    source: str | Path,
    target: str | Path,
    *,
    expected_sha256: str | None = None,
    allowed_root: Path | None = None,
) -> None:
    source_path = Path(source)
    target_path = Path(target)
    if allowed_root is not None:
        resolved_source = source_path.resolve()
        resolved_root = allowed_root.resolve()
        if not resolved_source.is_relative_to(resolved_root):
            raise ValueError(f"restore source escapes allowed root: {source_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"backup file missing during restore: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_target = target_path.with_name(f"{target_path.name}.restore_tmp")
    shutil.copy2(source_path, temporary_target)
    if expected_sha256 is not None:
        if sha256_file(temporary_target) != expected_sha256:
            temporary_target.unlink(missing_ok=True)
            raise ValueError(f"sha256 mismatch for {source_path}")
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
        relative_path = str(item["path"]).replace("\\", "/")
        try:
            _validate_backup_relative_path(relative_path)
        except ValueError:
            errors.append(f"invalid_path:{item['path']}")
            continue
        path = root / relative_path
        resolved = path.resolve()
        if not resolved.is_relative_to(root.resolve()):
            errors.append(f"escape:{item['path']}")
            continue
        if not resolved.is_file():
            errors.append(f"missing:{item['path']}")
            continue
        if resolved.stat().st_size != item["size"]:
            errors.append(f"size:{item['path']}")
            continue
        if sha256_file(resolved) != item["sha256"]:
            errors.append(f"sha256:{item['path']}")
    return errors
