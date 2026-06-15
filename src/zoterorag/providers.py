from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .embeddings import Qwen3VLEmbeddingProvider
from .embeddings.qwen import DASHSCOPE_MULTIMODAL_EMBEDDING_URL, DEFAULT_DOCUMENT_INSTRUCTION, DEFAULT_QUERY_INSTRUCTION
from .extractors import ExtractorKeyPool, MinerUProvider
from .extractors.key_pool import load_dotenv_values
from .extractors.mineru import APPLY_UPLOAD_URL, BATCH_RESULT_URL


DASHSCOPE_KEY_NAMES = (
    "BAILIAN_KEY",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_KEY",
    "QWEN_API_KEY",
)
# BAILIAN_URL is intentionally excluded from embedding endpoint resolution:
# it typically points at the OpenAI-compatible chat endpoint (/compatible-mode/v1)
# which does not serve multimodal embeddings. Use the embedding-specific env
# vars below to override the default embedding URL.
DASHSCOPE_ENDPOINT_NAMES = (
    "DASHSCOPE_MULTIMODAL_EMBEDDING_URL",
    "QWEN_EMBEDDING_URL",
)

MINERU_ALLOWED_HOSTS = ("mineru.net",)
DASHSCOPE_ALLOWED_HOSTS = ("dashscope.aliyuncs.com",)


@dataclass(frozen=True)
class ProviderEnvironment:
    env_path: Path
    values: dict[str, str]

    @classmethod
    def load(cls, env_path: str | Path = ".env") -> "ProviderEnvironment":
        path = Path(env_path)
        return cls(env_path=path, values=load_dotenv_values(path))

    def first_present(self, names: tuple[str, ...]) -> tuple[str, str] | None:
        for name in names:
            value = self.values.get(name, "").strip()
            if value:
                return name, value
        return None


def provider_readiness(env_path: str | Path = ".env") -> dict[str, Any]:
    """Report external provider configuration without exposing secrets."""

    env = ProviderEnvironment.load(env_path)
    mineru_pool = ExtractorKeyPool.from_env_file(env.env_path)
    dashscope_key = env.first_present(DASHSCOPE_KEY_NAMES)
    dashscope_endpoint = env.first_present(DASHSCOPE_ENDPOINT_NAMES)
    mineru_urls = mineru_urls_from_env(env.values)
    return {
        "env_path": str(env.env_path),
        "env_exists": env.env_path.is_file(),
        "mineru": {
            "configured": mineru_pool.has_keys(),
            "key_count": len(mineru_pool.list_public_keys()),
            "keys": mineru_pool.list_public_keys(),
            "apply_upload_url_configured": bool(mineru_urls["apply_upload_url"]),
            "batch_result_url_configured": bool(mineru_urls["batch_result_url"]),
        },
        "qwen3vl_embedding": {
            "configured": dashscope_key is not None,
            "key_env_name": dashscope_key[0] if dashscope_key else None,
            "endpoint_env_name": dashscope_endpoint[0] if dashscope_endpoint else None,
            "endpoint_configured": dashscope_endpoint is not None,
        },
    }


def _validate_endpoint_url(url: str, allowed_hosts: tuple[str, ...]) -> str:
    """Require HTTPS and an allowed, non-private host for provider endpoints."""
    if not url:
        raise ValueError("endpoint URL is required")
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"endpoint URL must use https scheme: {url}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"endpoint URL has no hostname: {url}")
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        addr = None
    if addr is not None:
        if addr.is_loopback or addr.is_private or addr.is_reserved or addr.is_multicast:
            raise ValueError(f"endpoint URL must not use an IP address: {url}")
    if hostname.lower() not in {h.lower() for h in allowed_hosts}:
        raise ValueError(f"endpoint URL hostname not allowed: {hostname}")
    return url


def build_mineru_provider(env_path: str | Path = ".env", *, client: Any = None) -> MinerUProvider:
    """Build a MinerU provider; API keys are still supplied by ExtractorKeyPool."""

    env = ProviderEnvironment.load(env_path)
    urls = mineru_urls_from_env(env.values)
    apply_upload_url = urls["apply_upload_url"] or APPLY_UPLOAD_URL
    batch_result_url = urls["batch_result_url"] or BATCH_RESULT_URL
    _validate_endpoint_url(apply_upload_url, MINERU_ALLOWED_HOSTS)
    _validate_endpoint_url(batch_result_url, MINERU_ALLOWED_HOSTS)
    return MinerUProvider(
        client=client,
        apply_upload_url=apply_upload_url,
        batch_result_url=batch_result_url,
    )


def build_qwen_embedding_provider(
    profile: dict[str, Any],
    env_path: str | Path = ".env",
    *,
    client: Any = None,
) -> Qwen3VLEmbeddingProvider:
    env = ProviderEnvironment.load(env_path)
    key = env.first_present(DASHSCOPE_KEY_NAMES)
    if key is None:
        raise RuntimeError(
            "qwen3-vl embedding key is not configured; set BAILIAN_KEY or DASHSCOPE_API_KEY in the env file"
        )
    endpoint = env.first_present(DASHSCOPE_ENDPOINT_NAMES)
    endpoint_url = endpoint[1] if endpoint else DASHSCOPE_MULTIMODAL_EMBEDDING_URL
    _validate_endpoint_url(endpoint_url, DASHSCOPE_ALLOWED_HOSTS)
    embedded_profile = dict(profile.get("profile") or {})
    instruction = profile.get("instruction_template") or embedded_profile.get("instruction_template") or ""
    return Qwen3VLEmbeddingProvider(
        api_key=key[1],
        model=str(profile["model"]),
        dimension=int(profile["dimension"]),
        endpoint=endpoint_url,
        client=client,
        query_instruction=instruction or DEFAULT_QUERY_INSTRUCTION,
        document_instruction=embedded_profile.get("document_instruction", DEFAULT_DOCUMENT_INSTRUCTION),
    )


def mineru_urls_from_env(values: dict[str, str]) -> dict[str, str | None]:
    apply_upload_url = values.get("MINERU_APPLY_UPLOAD_URL", "").strip() or None
    batch_result_url = values.get("MINERU_BATCH_RESULT_URL", "").strip() or None
    generic_url = values.get("MINERU_URL", "").strip()
    if generic_url:
        if "file-urls" in generic_url:
            apply_upload_url = apply_upload_url or generic_url
        elif "extract-results" in generic_url:
            batch_result_url = batch_result_url or generic_url
        else:
            # Use MINERU_URL as the API base directly — it may already
            # include a full path prefix like /api/v4/extract/task.
            root = generic_url.rstrip("/")
            apply_upload_url = apply_upload_url or f"{root}/file-urls/batch"
            batch_result_url = batch_result_url or f"{root}/extract-results/batch/{{batch_id}}"
    return {
        "apply_upload_url": apply_upload_url,
        "batch_result_url": batch_result_url,
    }
