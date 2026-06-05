from __future__ import annotations

import os
from pathlib import Path
import unittest

from zoterorag.api.server import validate_serve_access
from zoterorag.config import AppConfig, PathsConfig, ServerConfig


class ApiServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_token = os.environ.pop("ZOTERORAG_API_TOKEN", None)

    def tearDown(self) -> None:
        if self._old_token is None:
            os.environ.pop("ZOTERORAG_API_TOKEN", None)
        else:
            os.environ["ZOTERORAG_API_TOKEN"] = self._old_token

    def test_loopback_bind_is_allowed_for_local_bootstrap(self) -> None:
        config = build_config(host="127.0.0.1", require_api_token=False)
        preflight = validate_serve_access(config)

        self.assertTrue(preflight["ok"])
        self.assertFalse(preflight["externally_visible"])

    def test_external_bind_requires_token_capable_access_control(self) -> None:
        unsafe = build_config(host="0.0.0.0", require_api_token=False)
        denied = validate_serve_access(unsafe)
        self.assertFalse(denied["ok"])
        self.assertTrue(denied["externally_visible"])

        safe_required = build_config(host="0.0.0.0", require_api_token=True)
        self.assertTrue(validate_serve_access(safe_required)["ok"])

        os.environ["ZOTERORAG_API_TOKEN"] = "expected-token"
        safe_with_token = build_config(host="0.0.0.0", require_api_token=False)
        self.assertTrue(validate_serve_access(safe_with_token)["ok"])


def build_config(*, host: str, require_api_token: bool) -> AppConfig:
    return AppConfig(
        paths=PathsConfig(
            zotero_db=Path("zotero.sqlite"),
            zotero_storage=Path("storage"),
        ),
        server=ServerConfig(host=host, port=8765, require_api_token=require_api_token),
    )


if __name__ == "__main__":
    unittest.main()
