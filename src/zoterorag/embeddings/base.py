from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol


@dataclass(frozen=True)
class EmbeddingInput:
    input_id: str
    text: str
    image_path: str | None = None
    role: str = "document"


@dataclass(frozen=True)
class EmbeddingVector:
    input_id: str
    vector: list[float]


class EmbeddingProvider(Protocol):
    name: str
    model: str
    dimension: int

    def embed(self, inputs: list[EmbeddingInput]) -> list[EmbeddingVector]:
        ...


class StubEmbeddingProvider:
    """Deterministic non-network embedding provider for tests."""

    def __init__(self, dimension: int = 8) -> None:
        self.name = "stub"
        self.model = "stub"
        self.dimension = dimension

    def embed(self, inputs: list[EmbeddingInput]) -> list[EmbeddingVector]:
        return [EmbeddingVector(item.input_id, self._vector_for(item)) for item in inputs]

    def _vector_for(self, item: EmbeddingInput) -> list[float]:
        seed = f"{item.role}\0{item.text}\0{item.image_path or ''}".encode("utf-8")
        digest = hashlib.sha256(seed).digest()
        values = []
        for index in range(self.dimension):
            byte = digest[index % len(digest)]
            values.append((byte / 127.5) - 1.0)
        return values

