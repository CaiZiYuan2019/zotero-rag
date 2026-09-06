from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any
import tomllib


@dataclass(frozen=True)
class PathsConfig:
    zotero_db: Path
    zotero_storage: Path
    data_dir: Path = Path("data")

    @property
    def state_db(self) -> Path:
        return self.data_dir / "state" / "state.sqlite"

    @property
    def shadow_db(self) -> Path:
        return self.data_dir / "shadow" / "zotero.sqlite"

    @property
    def vector_store_dir(self) -> Path:
        return self.data_dir / "vector_store"

    @property
    def extract_cache_dir(self) -> Path:
        return self.data_dir / "extract_cache"

    @property
    def normalized_dir(self) -> Path:
        return self.data_dir / "normalized"

    @property
    def embedding_cache_dir(self) -> Path:
        return self.data_dir / "embedding_cache"


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    require_api_token: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.port, int) or not (1 <= self.port <= 65535):
            raise ValueError(
                f"server.port must be an integer between 1 and 65535, got {self.port!r}"
            )


@dataclass(frozen=True)
class EmbeddingProfile:
    name: str
    provider: str
    model: str
    dimension: int
    modality: str
    enabled: bool = True
    default_for_text: bool = False
    default_for_multimodal: bool = False
    query_role_mode: str = "instruction"
    document_role_mode: str = "plain"
    instruction_template: str = ""
    image_policy: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 16
    rate_limit: dict[str, Any] = field(default_factory=dict)
    backend: str = "lancedb"
    # Phase 1: concurrent embedding workers per profile. 1 (the default)
    # keeps the original fully serial embed path.
    embed_workers: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ValueError(
                f"embedding profile {self.name!r} dimension must be a positive integer, got {self.dimension!r}"
            )
        if self.modality not in {"text", "multimodal"}:
            raise ValueError(
                f"embedding profile {self.name!r} modality must be 'text' or 'multimodal', got {self.modality!r}"
            )
        if self.backend not in {"sqlite-local", "lancedb"}:
            raise ValueError(
                f"embedding profile {self.name!r} backend must be 'sqlite-local' or 'lancedb', got {self.backend!r}"
            )
        if isinstance(self.embed_workers, bool) or not isinstance(self.embed_workers, int) or self.embed_workers < 1:
            raise ValueError(
                f"embedding profile {self.name!r} embed_workers must be a positive integer, got {self.embed_workers!r}"
            )


@dataclass(frozen=True)
class McpConfig:
    tool_timeout: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.tool_timeout, (int, float)) or self.tool_timeout <= 0:
            raise ValueError(
                f"mcp.tool_timeout must be a positive number, got {self.tool_timeout!r}"
            )


@dataclass(frozen=True)
class IngestConfig:
    # Phase 3: batch vector publishing. True (the default) stages all
    # documents of a run and publishes once per profile; False falls back to
    # the historical per-document publish.
    batch_publish: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.batch_publish, bool):
            raise TypeError(
                f"ingest.batch_publish must be a boolean, got {self.batch_publish!r}"
            )


@dataclass(frozen=True)
class ExtractConfig:
    split_large_pdfs: bool = True
    max_pages_per_extraction: int = 200
    # 决策1: total lifetime of one extract job, measured from submitted_at.
    # Jobs exceeding it are flipped to failed_manual_review (no auto-resubmit).
    job_max_lifetime_seconds: int = 21600
    # Phase 2: extraction concurrency and pacing. The ExtractionManager
    # constructor defaults keep limiters off (0/unlimited); the effective
    # production values live here and are injected via cli/app wiring.
    per_key_submit_concurrency: int = 1
    submit_min_interval_seconds: float = 1.5
    harvest_workers: int = 2
    poll_max_per_minute: int = 600
    poll_interval_seconds: float = 30.0
    # 决策2: max automatic resubmissions of one extract job (counted across
    # rounds in payload["resubmit_attempts"]); beyond it the job is flipped
    # to failed_manual_review.
    max_resubmit_attempts: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.max_pages_per_extraction, int) or self.max_pages_per_extraction < 1:
            raise ValueError(
                f"extract.max_pages_per_extraction must be a positive integer, "
                f"got {self.max_pages_per_extraction!r}"
            )
        if (
            not isinstance(self.job_max_lifetime_seconds, int)
            or isinstance(self.job_max_lifetime_seconds, bool)
            or self.job_max_lifetime_seconds < 1
        ):
            raise ValueError(
                f"extract.job_max_lifetime_seconds must be a positive integer, "
                f"got {self.job_max_lifetime_seconds!r}"
            )
        if (
            not isinstance(self.per_key_submit_concurrency, int)
            or isinstance(self.per_key_submit_concurrency, bool)
            or self.per_key_submit_concurrency < 1
        ):
            raise ValueError(
                f"extract.per_key_submit_concurrency must be a positive integer, "
                f"got {self.per_key_submit_concurrency!r}"
            )
        if (
            not isinstance(self.submit_min_interval_seconds, (int, float))
            or isinstance(self.submit_min_interval_seconds, bool)
            or self.submit_min_interval_seconds < 0
        ):
            raise ValueError(
                f"extract.submit_min_interval_seconds must be a non-negative number "
                f"(0 disables pacing), got {self.submit_min_interval_seconds!r}"
            )
        if (
            not isinstance(self.harvest_workers, int)
            or isinstance(self.harvest_workers, bool)
            or self.harvest_workers < 1
        ):
            raise ValueError(
                f"extract.harvest_workers must be a positive integer, "
                f"got {self.harvest_workers!r}"
            )
        if (
            not isinstance(self.poll_max_per_minute, int)
            or isinstance(self.poll_max_per_minute, bool)
            or self.poll_max_per_minute < 0
        ):
            raise ValueError(
                f"extract.poll_max_per_minute must be a non-negative integer "
                f"(0 disables the limit), got {self.poll_max_per_minute!r}"
            )
        if (
            not isinstance(self.poll_interval_seconds, (int, float))
            or isinstance(self.poll_interval_seconds, bool)
            or self.poll_interval_seconds <= 0
        ):
            raise ValueError(
                f"extract.poll_interval_seconds must be a positive number, "
                f"got {self.poll_interval_seconds!r}"
            )
        if (
            not isinstance(self.max_resubmit_attempts, int)
            or isinstance(self.max_resubmit_attempts, bool)
            or self.max_resubmit_attempts < 0
        ):
            raise ValueError(
                f"extract.max_resubmit_attempts must be a non-negative integer, "
                f"got {self.max_resubmit_attempts!r}"
            )


@dataclass(frozen=True)
class VectorsConfig:
    # M4: embedding batches stuck "running" longer than this are recycled to
    # "failed" by `vectors prune` so crashed runs cannot block pruning forever.
    # Keep in sync with DEFAULT_STALE_BATCH_SECONDS in embeddings/indexer.py.
    stale_embedding_batch_hours: float = 24.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.stale_embedding_batch_hours, (int, float))
            or isinstance(self.stale_embedding_batch_hours, bool)
            or self.stale_embedding_batch_hours <= 0
        ):
            raise ValueError(
                f"vectors.stale_embedding_batch_hours must be a positive number, "
                f"got {self.stale_embedding_batch_hours!r}"
            )


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    server: ServerConfig = field(default_factory=ServerConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    vectors: VectorsConfig = field(default_factory=VectorsConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    embedding_profiles: tuple[EmbeddingProfile, ...] = ()

    def ensure_runtime_dirs(self) -> None:
        for path in (
            self.paths.state_db.parent,
            self.paths.shadow_db.parent,
            self.paths.vector_store_dir,
            self.paths.extract_cache_dir,
            self.paths.normalized_dir,
            self.paths.embedding_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _parse_bool(value: Any, field_name: str) -> bool:
    """Parse a configuration boolean, rejecting ambiguous string values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        raise ValueError(f"{field_name} must be 'true' or 'false', got {value!r}")
    raise TypeError(f"{field_name} must be a boolean, got {type(value).__name__}")


def _validate_profiles(
    profiles_data: list[dict[str, Any]],
) -> tuple[EmbeddingProfile, ...]:
    """Validate embedding profile entries and ensure default uniqueness per modality."""
    required_fields = ("name", "provider", "model", "dimension", "modality")
    seen_names: set[str] = set()
    for entry in profiles_data:
        missing = [field for field in required_fields if field not in entry]
        if missing:
            raise ValueError(f"embedding profile missing required fields: {missing}")
        name = entry["name"]
        if name in seen_names:
            raise ValueError(f"duplicate embedding profile name: {name!r}")
        seen_names.add(name)

    profiles: list[EmbeddingProfile] = []
    for entry in profiles_data:
        # Normalize boolean-like values that may come from manual edits or callers.
        for bool_field in ("enabled", "default_for_text", "default_for_multimodal"):
            if bool_field in entry:
                entry[bool_field] = _parse_bool(
                    entry[bool_field],
                    f"embedding_profiles.{entry['name']}.{bool_field}",
                )
        profiles.append(EmbeddingProfile(**entry))

    for modality in ("text", "multimodal"):
        modality_profiles = [p for p in profiles if p.modality == modality]
        if not modality_profiles:
            continue
        enabled_profiles = [p for p in modality_profiles if p.enabled]
        if not enabled_profiles:
            continue
        default_flag = (
            "default_for_text" if modality == "text" else "default_for_multimodal"
        )
        defaults = [p for p in enabled_profiles if getattr(p, default_flag)]
        if len(defaults) != 1:
            raise ValueError(
                f"expected exactly one enabled {modality} embedding profile with {default_flag}=true, found {len(defaults)}"
            )

    return tuple(profiles)


_PLACEHOLDER_RE = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>")


def _expand_path_placeholders(value: str) -> str:
    """Expand environment variables and ``<ENV_NAME>`` placeholders.

    Supports standard shell forms (``$VAR``, ``${VAR}``, Windows ``%VAR%``)
    via :func:`os.path.expandvars`, plus project-specific angle-bracket
    placeholders such as ``<ZOTERO_DB_PATH>``.
    """
    expanded = os.path.expandvars(value)
    for match in _PLACEHOLDER_RE.finditer(expanded):
        env_name = match.group(1)
        env_value = os.environ.get(env_name)
        if env_value is None:
            continue
        expanded = expanded.replace(match.group(0), env_value)
    return expanded


def _check_unresolved_placeholders(value: str, field_name: str) -> None:
    """Raise a clear error if a path still contains ``<ENV_NAME>`` placeholders."""
    placeholders = _PLACEHOLDER_RE.findall(value)
    if placeholders:
        names = ", ".join(sorted(set(placeholders)))
        raise ValueError(
            f"{field_name} contains unresolved placeholder(s): {names}.\n"
            f"Either edit config/config.toml and replace the placeholder with an absolute path, "
            f"or set the environment variable in the same shell before running the command:\n"
            f"  PowerShell: $env:{placeholders[0]}='C:\\Path\\To\\Zotero'\n"
            f"  cmd.exe:    set {placeholders[0]}=C:\\Path\\To\\Zotero"
        )


def _project_root(config_path: Path) -> Path:
    """Return the project root for a configuration file.

    When the configuration lives inside a ``config/`` directory, the project
    root is its parent so that ``data_dir = "data"`` resolves to the
    project-level ``data/`` directory. Otherwise the project root is the
    directory containing the configuration file itself.
    """
    parent = config_path.parent
    if parent.name == "config":
        return parent.parent
    return parent


def load_config(path: str | Path = "config/config.toml") -> AppConfig:
    """Load and validate the application configuration from a TOML file.

    Relative paths in ``paths.data_dir`` are resolved relative to the project
    root (the directory containing ``config/`` when the file is under
    ``config/config.toml``), not the current working directory.
    """
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")

    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    paths_data = data.get("paths", {})
    server_data = data.get("server", {})
    mcp_data = data.get("mcp", {})
    extract_data = data.get("extract", {})
    vectors_data = data.get("vectors", {})
    ingest_data = data.get("ingest", {})
    profiles_data = data.get("embedding_profiles", [])

    missing_paths = [
        field for field in ("zotero_db", "zotero_storage") if field not in paths_data
    ]
    if missing_paths:
        raise ValueError(f"missing required paths config: {missing_paths}")

    raw_data_dir = Path(paths_data.get("data_dir", "data"))
    if raw_data_dir.is_absolute():
        data_dir = raw_data_dir.resolve()
    else:
        data_dir = (_project_root(config_path) / raw_data_dir).resolve()

    zotero_db = _expand_path_placeholders(str(paths_data["zotero_db"]))
    zotero_storage = _expand_path_placeholders(str(paths_data["zotero_storage"]))
    _check_unresolved_placeholders(zotero_db, "paths.zotero_db")
    _check_unresolved_placeholders(zotero_storage, "paths.zotero_storage")

    paths = PathsConfig(
        zotero_db=Path(zotero_db),
        zotero_storage=Path(zotero_storage),
        data_dir=data_dir,
    )
    server = ServerConfig(
        host=server_data.get("host", "127.0.0.1"),
        port=int(server_data.get("port", 8765)),
        require_api_token=_parse_bool(
            server_data.get("require_api_token", True),
            "server.require_api_token",
        ),
    )
    profiles = _validate_profiles(profiles_data)
    mcp = McpConfig(
        tool_timeout=float(mcp_data.get("tool_timeout", 30.0)),
    )
    extract = ExtractConfig(
        split_large_pdfs=_parse_bool(
            extract_data.get("split_large_pdfs", True),
            "extract.split_large_pdfs",
        ),
        max_pages_per_extraction=int(extract_data.get("max_pages_per_extraction", 200)),
        job_max_lifetime_seconds=int(extract_data.get("job_max_lifetime_seconds", 21600)),
        per_key_submit_concurrency=int(extract_data.get("per_key_submit_concurrency", 1)),
        submit_min_interval_seconds=float(extract_data.get("submit_min_interval_seconds", 1.5)),
        harvest_workers=int(extract_data.get("harvest_workers", 2)),
        poll_max_per_minute=int(extract_data.get("poll_max_per_minute", 600)),
        poll_interval_seconds=float(extract_data.get("poll_interval_seconds", 30.0)),
        max_resubmit_attempts=int(extract_data.get("max_resubmit_attempts", 3)),
    )
    vectors = VectorsConfig(
        stale_embedding_batch_hours=float(vectors_data.get("stale_embedding_batch_hours", 24.0)),
    )
    ingest = IngestConfig(
        batch_publish=_parse_bool(ingest_data.get("batch_publish", True), "ingest.batch_publish"),
    )
    return AppConfig(
        paths=paths,
        server=server,
        mcp=mcp,
        extract=extract,
        vectors=vectors,
        ingest=ingest,
        embedding_profiles=profiles,
    )
