from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
            raise ValueError(f"server.port must be an integer between 1 and 65535, got {self.port!r}")


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

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ValueError(f"embedding profile {self.name!r} dimension must be a positive integer, got {self.dimension!r}")
        if self.modality not in {"text", "multimodal"}:
            raise ValueError(f"embedding profile {self.name!r} modality must be 'text' or 'multimodal', got {self.modality!r}")
        if self.backend not in {"sqlite-local", "lancedb"}:
            raise ValueError(f"embedding profile {self.name!r} backend must be 'sqlite-local' or 'lancedb', got {self.backend!r}")


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    server: ServerConfig = field(default_factory=ServerConfig)
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


def _validate_profiles(profiles_data: list[dict[str, Any]]) -> tuple[EmbeddingProfile, ...]:
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
                entry[bool_field] = _parse_bool(entry[bool_field], f"embedding_profiles.{entry['name']}.{bool_field}")
        profiles.append(EmbeddingProfile(**entry))

    for modality in ("text", "multimodal"):
        modality_profiles = [p for p in profiles if p.modality == modality]
        if not modality_profiles:
            continue
        enabled_profiles = [p for p in modality_profiles if p.enabled]
        if not enabled_profiles:
            continue
        default_flag = "default_for_text" if modality == "text" else "default_for_multimodal"
        defaults = [p for p in enabled_profiles if getattr(p, default_flag)]
        if len(defaults) != 1:
            raise ValueError(
                f"expected exactly one enabled {modality} embedding profile with {default_flag}=true, found {len(defaults)}"
            )

    return tuple(profiles)


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
    profiles_data = data.get("embedding_profiles", [])

    missing_paths = [field for field in ("zotero_db", "zotero_storage") if field not in paths_data]
    if missing_paths:
        raise ValueError(f"missing required paths config: {missing_paths}")

    raw_data_dir = Path(paths_data.get("data_dir", "data"))
    if raw_data_dir.is_absolute():
        data_dir = raw_data_dir.resolve()
    else:
        data_dir = (_project_root(config_path) / raw_data_dir).resolve()

    paths = PathsConfig(
        zotero_db=Path(paths_data["zotero_db"]),
        zotero_storage=Path(paths_data["zotero_storage"]),
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
    return AppConfig(paths=paths, server=server, embedding_profiles=profiles)
