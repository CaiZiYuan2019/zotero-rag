from __future__ import annotations

import hmac
import ipaddress
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
    is_loopback = is_loopback_host(client_host)
    if require_api_token:
        if not token:
            # Fresh local installs should be usable from the same machine, but
            # must not expose control endpoints to the LAN before a token is set.
            if is_loopback:
                return
            raise AccessDenied("ZOTERORAG_API_TOKEN is required for non-loopback access")
        if supplied_token and hmac.compare_digest(supplied_token, token):
            return
        raise AccessDenied("invalid or missing API token")

    if is_loopback:
        return
    if token:
        if supplied_token and hmac.compare_digest(supplied_token, token):
            return
        raise AccessDenied("non-loopback access requires API token")
    raise AccessDenied("non-loopback access requires configured API token")


def is_loopback_host(client_host: str | None) -> bool:
    """Return True only for local clients.

    FastAPI exposes `request.client.host` without a port, but tests and future
    adapters may pass bracketed IPv6 values. Unknown hosts are treated as
    external so startup/config mistakes fail closed.
    """

    if not client_host:
        return False
    host = client_host.strip().strip("[]").casefold()
    if host in LOCAL_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
