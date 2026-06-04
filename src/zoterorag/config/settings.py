from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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


def load_config(path: str | Path = "config/config.example.toml") -> AppConfig:
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    paths_data = data.get("paths", {})
    server_data = data.get("server", {})
    profiles_data = data.get("embedding_profiles", [])

    paths = PathsConfig(
        zotero_db=Path(paths_data["zotero_db"]),
        zotero_storage=Path(paths_data["zotero_storage"]),
        data_dir=Path(paths_data.get("data_dir", "data")),
    )
    server = ServerConfig(
        host=server_data.get("host", "127.0.0.1"),
        port=int(server_data.get("port", 8765)),
        require_api_token=bool(server_data.get("require_api_token", True)),
    )
    profiles = tuple(EmbeddingProfile(**profile) for profile in profiles_data)
    return AppConfig(paths=paths, server=server, embedding_profiles=profiles)
