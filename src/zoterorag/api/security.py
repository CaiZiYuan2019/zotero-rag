from __future__ import annotations

import hmac
import os


LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


class AccessDenied(PermissionError):
    pass


def expected_api_token() -> str | None:
    return os.environ.get("ZOTERORAG_API_TOKEN")


def verify_api_access(
    supplied_token: str | None,
    client_host: str | None,
    require_api_token: bool = True,
) -> None:
    """Validate external API access.

    If token auth is enabled, every request must provide the configured token.
    If no token is configured, only loopback clients are accepted. This keeps a
    new install from accidentally exposing control endpoints on the LAN.
    """

    token = expected_api_token()
    if require_api_token:
        if not token:
            # Fresh local installs should be usable from the same machine, but
            # must not expose control endpoints to the LAN before a token is set.
            if client_host in LOCAL_HOSTS:
                return
            raise AccessDenied("ZOTERORAG_API_TOKEN is required for non-loopback access")
        if supplied_token and hmac.compare_digest(supplied_token, token):
            return
        raise AccessDenied("invalid or missing API token")

    if client_host not in LOCAL_HOSTS and token:
        if supplied_token and hmac.compare_digest(supplied_token, token):
            return
        raise AccessDenied("non-loopback access requires API token")
