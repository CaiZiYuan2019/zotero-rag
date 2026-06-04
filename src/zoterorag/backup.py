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
