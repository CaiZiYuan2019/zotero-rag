from __future__ import annotations

import os
from typing import Any

from ..config import AppConfig
from .security import is_loopback_host


def validate_serve_access(config: AppConfig, *, host: str | None = None) -> dict[str, Any]:
    """Validate API bind settings before starting the server.

    Binding to a LAN-visible host is allowed only when request-level access can
    require a token. This preflight prevents accidentally exposing control
    endpoints before Uvicorn imports the optional FastAPI stack.
    """

    bind_host = host or config.server.host
    token_configured = bool(os.environ.get("ZOTERORAG_API_TOKEN"))
    loopback = is_loopback_host(bind_host)
    externally_visible = not loopback
    # Binding to an external address requires a token to be configured so that
    # request-level access control can actually enforce authentication.
    ok = loopback or token_configured
    return {
        "ok": ok,
        "host": bind_host,
        "port": config.server.port,
        "require_api_token": config.server.require_api_token,
        "token_configured": token_configured,
        "externally_visible": externally_visible,
        "error": None if ok else "refusing to bind external API without a configured ZOTERORAG_API_TOKEN",
    }


def serve_api(config_path: str, *, host: str | None = None, port: int | None = None) -> None:
    from ..config import load_config

    config = load_config(config_path)
    preflight = validate_serve_access(config, host=host)
    if not preflight["ok"]:
        raise RuntimeError(str(preflight["error"]))

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("Uvicorn is not installed. Install zotero-rag[api].") from exc

    from .app import create_app

    uvicorn.run(
        create_app(config_path),
        host=str(host or config.server.host),
        port=int(port or config.server.port),
        log_level="info",
        reload=False,
    )
