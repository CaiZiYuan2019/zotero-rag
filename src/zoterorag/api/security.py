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

    If a token is configured, every request must provide it. If no token is
    configured, only loopback clients are accepted. This keeps a new install
    from accidentally exposing control endpoints on the LAN.

    All failure paths raise :class:`AccessDenied` with the same message to avoid
    leaking whether a token is configured.
    """

    token = expected_api_token()
    is_loopback = is_loopback_host(client_host)
    if is_loopback and (not require_api_token or not token):
        return
    if token and supplied_token and hmac.compare_digest(supplied_token, token):
        return
    raise AccessDenied("access denied")


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
