from __future__ import annotations

import base64
import json
import mimetypes
import random
import time
from pathlib import Path
from typing import Any, Protocol

from .base import EmbeddingInput, EmbeddingVector

try:
    from requests.exceptions import ConnectionError as _RequestsConnectionError
    from requests.exceptions import Timeout as _RequestsTimeout
except ImportError:  # pragma: no cover - requests is an optional runtime dependency
    _RequestsTimeout = None
    _RequestsConnectionError = None


DASHSCOPE_MULTIMODAL_EMBEDDING_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
)
QWEN3VL_SUPPORTED_DIMENSIONS = {2560, 2048, 1536, 1024, 768, 512, 256}
DEFAULT_QUERY_INSTRUCTION = "Retrieve scientific paper passages, figures, or tables relevant to the user's query."
DEFAULT_DOCUMENT_INSTRUCTION = "Represent this scientific paper evidence for retrieval."
DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_INITIAL_SECONDS = 1.0
DEFAULT_RETRY_MAX_SECONDS = 60.0
DEFAULT_RETRY_BACKOFF_FACTOR = 2.0
DEFAULT_RETRY_JITTER = 0.1

_TRANSIENT_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 520})


class QwenResponse(Protocol):
    status_code: int
    text: str

    def json(self) -> dict[str, Any]:
        ...


class QwenHTTPClient(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        timeout: int | float | None = None,
    ) -> QwenResponse:
        ...


class QwenEmbeddingError(RuntimeError):
    pass


class Qwen3VLEmbeddingProvider:
    """DashScope HTTP provider for qwen3-vl-embedding.

    The provider sends one request per logical input. That is slower than
    large text batching, but it preserves each item's query/document role,
    instruction, and optional text+image fusion semantics during the first
    production integration.
    """

    name = "dashscope"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "qwen3-vl-embedding",
        dimension: int = 2560,
        endpoint: str = DASHSCOPE_MULTIMODAL_EMBEDDING_URL,
        client: QwenHTTPClient | None = None,
        timeout_seconds: int = 120,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        document_instruction: str = DEFAULT_DOCUMENT_INSTRUCTION,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_initial_seconds: float = DEFAULT_RETRY_INITIAL_SECONDS,
        retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
        retry_backoff_factor: float = DEFAULT_RETRY_BACKOFF_FACTOR,
        retry_jitter: float = DEFAULT_RETRY_JITTER,
    ) -> None:
        if dimension not in QWEN3VL_SUPPORTED_DIMENSIONS:
            raise ValueError(f"unsupported qwen3-vl-embedding dimension: {dimension}")
        if not api_key.strip():
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.endpoint = endpoint
        self.client = client or _load_requests_client()
        self.timeout_seconds = timeout_seconds
        self.query_instruction = query_instruction
        self.document_instruction = document_instruction
        self.max_image_bytes = max_image_bytes
        self.max_retries = max_retries
        self.retry_initial_seconds = retry_initial_seconds
        self.retry_max_seconds = retry_max_seconds
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_jitter = retry_jitter

    def _is_retryable_exception(self, exc: BaseException) -> bool:
        if isinstance(exc, (TimeoutError, OSError)):
            return True
        if _RequestsTimeout is not None and isinstance(exc, (_RequestsTimeout, _RequestsConnectionError)):
            return True
        return False

    def _sleep_backoff(self, delay: float) -> None:
        jitter = random.uniform(-self.retry_jitter, self.retry_jitter) * delay
        time.sleep(max(0.0, delay + jitter))

    def _post_with_retry(self, payload: dict[str, Any]) -> QwenResponse:
        """Call the embedding endpoint with exponential backoff.

        Retries on network/timeout errors and transient HTTP status codes. The
        final response is returned to the caller for normal error handling.
        """

        last_exception: BaseException | None = None
        delay = self.retry_initial_seconds
        for attempt in range(self.max_retries):
            try:
                response = self.client.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                last_exception = exc
                if not self._is_retryable_exception(exc) or attempt == self.max_retries - 1:
                    raise
                self._sleep_backoff(delay)
                delay = min(delay * self.retry_backoff_factor, self.retry_max_seconds)
                continue
            if response.status_code in _TRANSIENT_HTTP_STATUSES and attempt < self.max_retries - 1:
                self._sleep_backoff(delay)
                delay = min(delay * self.retry_backoff_factor, self.retry_max_seconds)
                continue
            return response
        if last_exception is not None:
            raise last_exception
        raise QwenEmbeddingError("qwen embedding request failed after retries")

    def embed(self, inputs: list[EmbeddingInput]) -> list[EmbeddingVector]:
        return [self._embed_one(item) for item in inputs]

    def _embed_one(self, item: EmbeddingInput) -> EmbeddingVector:
        payload = self._payload_for_input(item)
        response = self._post_with_retry(payload)
        body = raise_for_qwen_error(response)
        embeddings = ((body.get("output") or {}).get("embeddings") or [])
        if not embeddings:
            raise QwenEmbeddingError("qwen embedding response did not include vectors")
        vector = [float(value) for value in embeddings[0].get("embedding", [])]
        if len(vector) != self.dimension:
            raise QwenEmbeddingError(
                f"qwen embedding dimension mismatch: got {len(vector)}, expected {self.dimension}"
            )
        return EmbeddingVector(input_id=item.input_id, vector=vector)

    def _payload_for_input(self, item: EmbeddingInput) -> dict[str, Any]:
        contents: list[dict[str, str]] = []
        text = item.text.strip()
        if text:
            contents.append({"text": text})
        image_data = image_data_uri_for_input(item, max_image_bytes=self.max_image_bytes, input_id=item.input_id)
        if image_data:
            contents.append({"image": image_data})
        if not contents:
            raise QwenEmbeddingError(f"embedding input is empty: {item.input_id}")

        parameters: dict[str, Any] = {
            "dimension": self.dimension,
            "instruct": self.query_instruction if item.role == "query" else self.document_instruction,
        }
        # qwen3-vl-embedding must use fusion to turn text+image into one vector
        # for a single image_block or multimodal query.
        if len(contents) > 1:
            parameters["enable_fusion"] = True

        return {
            "model": self.model,
            "input": {"contents": contents},
            "parameters": parameters,
        }


def image_data_uri_for_input(
    item: EmbeddingInput,
    *,
    max_image_bytes: int,
    input_id: str | None = None,
    allowed_roots: list[str | Path] | None = None,
) -> str | None:
    if item.image_base64:
        if item.image_base64.startswith("data:image/"):
            return item.image_base64
        mime_type = item.image_mime_type or "image/png"
        return f"data:{mime_type};base64,{item.image_base64}"
    if not item.image_path:
        return None

    path = Path(item.image_path)
    if allowed_roots:
        resolved = path.resolve()
        roots = [Path(root).expanduser().resolve() for root in allowed_roots]
        if not any(
            _is_relative_to(resolved, root) or _is_relative_to(path, root) for root in roots
        ):
            raise QwenEmbeddingError(
                f"image path is outside allowed roots for embedding input {input_id or item.input_id}: {path}"
            )
    try:
        size = path.stat().st_size
    except FileNotFoundError as exc:
        raise QwenEmbeddingError(
            f"image file not found for embedding input {input_id or item.input_id}: {path}"
        ) from exc
    if size > max_image_bytes:
        raise QwenEmbeddingError(f"image is too large for qwen embedding: {path} ({size} bytes)")
    mime_type = item.image_mime_type or mimetypes.guess_type(path.name)[0] or "image/png"
    try:
        raw_bytes = path.read_bytes()
    except FileNotFoundError as exc:
        raise QwenEmbeddingError(
            f"image file not found for embedding input {input_id or item.input_id}: {path}"
        ) from exc
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def raise_for_qwen_error(response: QwenResponse) -> dict[str, Any]:
    if response.status_code != 200:
        # Check the status code before attempting JSON parsing so non-JSON
        # errors (e.g., HTML overload pages) are not masked by a parse error.
        try:
            body = response.json()
            message = body.get("message") or response.text[:500]
        except json.JSONDecodeError:
            message = response.text[:500]
        raise QwenEmbeddingError(f"qwen embedding request failed: HTTP {response.status_code} - {message}")
    try:
        body = response.json()
    except json.JSONDecodeError as exc:
        raise QwenEmbeddingError("qwen embedding response is not valid JSON") from exc
    code = body.get("code")
    if code not in (None, "", 0):
        raise QwenEmbeddingError(f"qwen embedding request failed: {code} - {body.get('message', 'unknown error')}")
    return body


def _load_requests_client() -> QwenHTTPClient:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
        raise QwenEmbeddingError("requests is required for Qwen3VLEmbeddingProvider") from exc
    return requests
