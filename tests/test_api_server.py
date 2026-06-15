from __future__ import annotations

import os
from pathlib import Path
import unittest

from zoterorag.api.app import _find_project_root, _resolve_env_path
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
        self.assertIn("configured ZOTERORAG_API_TOKEN", denied["error"])

        safe_required = build_config(host="0.0.0.0", require_api_token=True)
        self.assertFalse(validate_serve_access(safe_required)["ok"])

        os.environ["ZOTERORAG_API_TOKEN"] = "expected-token"
        safe_with_token = build_config(host="0.0.0.0", require_api_token=False)
        self.assertTrue(validate_serve_access(safe_with_token)["ok"])

        safe_with_required = build_config(host="0.0.0.0", require_api_token=True)
        self.assertTrue(validate_serve_access(safe_with_required)["ok"])


def build_config(*, host: str, require_api_token: bool) -> AppConfig:
    return AppConfig(
        paths=PathsConfig(
            zotero_db=Path("zotero.sqlite"),
            zotero_storage=Path("storage"),
        ),
        server=ServerConfig(host=host, port=8765, require_api_token=require_api_token),
    )


class EnvPathValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = _find_project_root(Path(__file__).parent.parent / "config" / "config.example.toml")

    def test_relative_env_path_inside_project_is_accepted(self) -> None:
        resolved = _resolve_env_path(".env", self.project_root)
        self.assertEqual(self.project_root / ".env", resolved)

    def test_env_path_with_dotdot_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _resolve_env_path("../secret.env", self.project_root)
        self.assertIn("..", str(ctx.exception))

    def test_absolute_env_path_outside_project_is_rejected(self) -> None:
        outside = Path(self.project_root.anchor) / "secret.env"
        with self.assertRaises(ValueError) as ctx:
            _resolve_env_path(str(outside), self.project_root)
        self.assertIn("inside project root", str(ctx.exception))

    def test_absolute_env_path_inside_project_is_accepted(self) -> None:
        inside = self.project_root / "config" / "prod.env"
        resolved = _resolve_env_path(str(inside), self.project_root)
        self.assertEqual(inside.resolve(), resolved)


if __name__ == "__main__":
    unittest.main()
