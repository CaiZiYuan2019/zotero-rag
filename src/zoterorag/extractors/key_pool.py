from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


ENV_KEY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")


@dataclass(frozen=True)
class ApiKeyRef:
    alias: str
    secret: str

    def public_dict(self) -> dict[str, str]:
        return {"alias": self.alias, "redacted": redact_secret(self.secret)}


class ExtractorKeyPool:
    """Round-robin API key selector that exposes aliases to the state ledger.

    Worker code can hold the secret in memory long enough to call the provider,
    but progress rows, logs, and backups should only store `alias`.
    """

    def __init__(self, keys: list[ApiKeyRef] | None = None) -> None:
        self._keys = list(keys or [])
        self._next_index = 0

    @classmethod
    def from_env_file(
        cls,
        env_path: str | Path = ".env",
        *,
        prefixes: tuple[str, ...] = ("MINERU_KEY", "MINERU_API_KEY"),
    ) -> "ExtractorKeyPool":
        values = load_dotenv_values(env_path)
        keys: list[ApiKeyRef] = []
        candidates = [
            name
            for name in values
            if any(name == prefix or name.startswith(f"{prefix}_") for prefix in prefixes)
        ]
        candidates.sort(key=lambda name: key_sort_tuple(name, prefixes))
        for name in candidates:
            secret = values[name].strip()
            if not secret:
                continue
            keys.append(ApiKeyRef(alias=alias_from_env_name(name, len(keys) + 1), secret=secret))
        return cls(keys)

    def has_keys(self) -> bool:
        return bool(self._keys)

    def list_public_keys(self) -> list[dict[str, str]]:
        return [key.public_dict() for key in self._keys]

    def next_key(self) -> ApiKeyRef | None:
        if not self._keys:
            return None
        key = self._keys[self._next_index % len(self._keys)]
        self._next_index += 1
        return key


def load_dotenv_values(env_path: str | Path) -> dict[str, str]:
    path = Path(env_path)
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ENV_KEY_RE.match(line)
        if not match or line.lstrip().startswith("#"):
            continue
        name, raw_value = match.groups()
        value = raw_value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        values[name] = value
    return values


def alias_from_env_name(name: str, fallback_index: int) -> str:
    suffix = name.removeprefix("MINERU_API_KEY").removeprefix("MINERU_KEY").strip("_")
    return f"mineru_{suffix.lower()}" if suffix else f"mineru_{fallback_index}"


def key_sort_tuple(name: str, prefixes: tuple[str, ...]) -> tuple[int, int, str]:
    for prefix_index, prefix in enumerate(prefixes):
        if name == prefix:
            return (prefix_index, 0, "")
        if name.startswith(f"{prefix}_"):
            return (prefix_index, 1, name.removeprefix(f"{prefix}_"))
    return (len(prefixes), 1, name)


def redact_secret(secret: str) -> str:
    return "<redacted>"
