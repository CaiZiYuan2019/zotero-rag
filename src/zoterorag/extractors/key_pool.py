from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable


ENV_KEY_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ApiKeyRef:
    alias: str
    secret: str

    def public_dict(self) -> dict[str, str]:
        return {"alias": self.alias, "redacted": redact_secret(self.secret)}

    def __repr__(self) -> str:
        return f"ApiKeyRef(alias={self.alias!r}, secret={redact_secret(self.secret)})"

    def __str__(self) -> str:
        return f"ApiKeyRef(alias={self.alias}, secret={redact_secret(self.secret)})"


@dataclass
class ApiKeyRuntimeState:
    alias: str
    in_flight: int = 0
    cooldown_until: float = 0.0


class ExtractorKeyPool:
    """Round-robin API key selector that exposes aliases to the state ledger.

    Worker code can hold the secret in memory long enough to call the provider,
    but progress rows, logs, and backups should only store `alias`.
    """

    def __init__(
        self,
        keys: list[ApiKeyRef] | None = None,
        *,
        per_key_submit_concurrency: int = 1,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if per_key_submit_concurrency < 1:
            raise ValueError("per_key_submit_concurrency must be >= 1")
        self._keys = list(keys or [])
        self._next_index = 0
        self.per_key_submit_concurrency = per_key_submit_concurrency
        self._clock = clock or time.monotonic
        self._states = {key.alias: ApiKeyRuntimeState(alias=key.alias) for key in self._keys}
        self._lock = threading.RLock()

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
        seen_aliases: set[str] = set()
        for name in candidates:
            raw_value = values[name].strip()
            if not raw_value:
                continue
            secrets = [s.strip() for s in raw_value.split(",") if s.strip()]
            if not secrets:
                continue
            base_alias = base_alias_from_env_name(name)
            for idx, secret in enumerate(secrets, start=1):
                if len(secrets) == 1:
                    # Single key: keep legacy alias naming so existing tests/users
                    # see the same aliases as before.
                    alias = alias_from_env_name(name, len(keys) + 1)
                else:
                    alias = f"{base_alias}_{idx}"
                # Defensive: ensure aliases are unique across all env variables.
                original_alias = alias
                dedup = 1
                while alias in seen_aliases:
                    dedup += 1
                    alias = f"{original_alias}_{dedup}"
                seen_aliases.add(alias)
                keys.append(ApiKeyRef(alias=alias, secret=secret))
        return cls(keys)

    def has_keys(self) -> bool:
        return bool(self._keys)

    def list_public_keys(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    **key.public_dict(),
                    "in_flight": self._states[key.alias].in_flight,
                    "cooldown_remaining_seconds": round(
                        self._cooldown_remaining_unsafe(key.alias), 3
                    ),
                }
                for key in self._keys
            ]

    def next_key(self) -> ApiKeyRef | None:
        with self._lock:
            if not self._keys:
                return None
            key = self._keys[self._next_index % len(self._keys)]
            self._next_index += 1
            return key

    def acquire_key(self) -> ApiKeyRef | None:
        """Reserve a key for a submit/upload worker.

        Selection is round-robin but skips keys at their per-key concurrency
        limit or in cooldown. Only aliases should be persisted outside memory.
        """

        with self._lock:
            if not self._keys:
                return None
            now = self._clock()
            for offset in range(len(self._keys)):
                index = (self._next_index + offset) % len(self._keys)
                key = self._keys[index]
                state = self._states[key.alias]
                if state.cooldown_until > now:
                    continue
                if state.in_flight >= self.per_key_submit_concurrency:
                    continue
                state.in_flight += 1
                self._next_index = index + 1
                return key
            return None

    def release_key(self, alias: str, *, cooldown_seconds: float = 0.0) -> None:
        with self._lock:
            state = self._require_state(alias)
            state.in_flight = max(0, state.in_flight - 1)
            if cooldown_seconds > 0:
                state.cooldown_until = max(
                    state.cooldown_until, self._clock() + cooldown_seconds
                )

    def mark_key_cooldown(self, alias: str, *, cooldown_seconds: float) -> None:
        if cooldown_seconds <= 0:
            return
        with self._lock:
            state = self._require_state(alias)
            state.cooldown_until = max(
                state.cooldown_until, self._clock() + cooldown_seconds
            )

    def cooldown_remaining(self, alias: str) -> float:
        with self._lock:
            return self._cooldown_remaining_unsafe(alias)

    def _cooldown_remaining_unsafe(self, alias: str) -> float:
        state = self._require_state(alias)
        return max(0.0, state.cooldown_until - self._clock())

    def _require_state(self, alias: str) -> ApiKeyRuntimeState:
        try:
            return self._states[alias]
        except KeyError as exc:
            raise KeyError(f"unknown extractor key alias: {alias}") from exc


def load_dotenv_values(env_path: str | Path) -> dict[str, str]:
    path = Path(env_path)
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        name, value = parsed
        values[name] = value
    return values


def _parse_env_line(line: str) -> tuple[str, str] | None:
    """Parse a single KEY=VALUE line from a .env file.

    Handles inline comments and escaped quotes inside single/double-quoted
    values without depending on third-party libraries.
    """

    if "=" not in line:
        return None
    key, rest = line.split("=", 1)
    key = key.strip()
    if not ENV_KEY_NAME_RE.match(key):
        return None
    value, _ = _parse_env_value(rest.strip())
    return key, value


def _parse_env_value(rest: str) -> tuple[str, str]:
    """Return (value, trailing_rest) respecting quotes and inline comments."""

    if not rest:
        return "", ""
    quote = rest[0]
    if quote in ('"', "'"):
        parsed = []
        i = 1
        while i < len(rest):
            char = rest[i]
            if char == "\\" and i + 1 < len(rest):
                next_char = rest[i + 1]
                if next_char == quote or next_char == "\\":
                    parsed.append(next_char)
                    i += 2
                    continue
            elif char == quote:
                trailing = rest[i + 1 :].strip()
                if trailing.startswith("#"):
                    trailing = ""
                return "".join(parsed), trailing
            parsed.append(char)
            i += 1
        # Unterminated quoted value: return everything up to the end.
        return "".join(parsed), ""

    # Unquoted value: inline comment starts at '#' preceded by whitespace.
    for i, char in enumerate(rest):
        if char == "#" and (i == 0 or rest[i - 1].isspace()):
            return rest[:i].rstrip(), ""
    return rest.rstrip(), ""


def base_alias_from_env_name(name: str) -> str:
    """Return the alias prefix for a MinerU env variable without an index."""
    suffix = name.removeprefix("MINERU_API_KEY").removeprefix("MINERU_KEY").strip("_")
    return f"mineru_{suffix.lower()}" if suffix else "mineru"


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
