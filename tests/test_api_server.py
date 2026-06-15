from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import unittest

from tests._support import workspace_tmpdir
from zoterorag.api.app import _find_project_root, _resolve_env_path, create_app
from zoterorag.api.server import validate_serve_access
from zoterorag.config import AppConfig, PathsConfig, ServerConfig

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - optional API extra
    TestClient = None  # type: ignore[misc, assignment]


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
        inside = self.project_root / "config" / ".env.prod"
        inside.write_text("X=1", encoding="utf-8")
        try:
            resolved = _resolve_env_path(str(inside), self.project_root)
            self.assertEqual(inside.resolve(), resolved)
        finally:
            inside.unlink(missing_ok=True)

    def test_env_filename_must_be_dotenv_or_dotenv_suffix(self) -> None:
        bad = self.project_root / ".tmp-test-env-reject" / "config.toml"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("x", encoding="utf-8")
        try:
            with self.assertRaises(ValueError) as ctx:
                _resolve_env_path(str(bad), self.project_root)
            self.assertIn(".env", str(ctx.exception))
        finally:
            bad.unlink(missing_ok=True)
            bad.parent.rmdir()


@unittest.skipIf(TestClient is None, "FastAPI not installed")
class ApiProfileResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_token = os.environ.pop("ZOTERORAG_API_TOKEN", None)
        os.environ["ZOTERORAG_API_TOKEN"] = "test-token"

    def tearDown(self) -> None:
        if self._old_token is None:
            os.environ.pop("ZOTERORAG_API_TOKEN", None)
        else:
            os.environ["ZOTERORAG_API_TOKEN"] = self._old_token

    def _client(self, app) -> Any:
        return TestClient(app, headers={"X-API-Token": "test-token"})

    def _create_config(self, tmpdir: Path) -> Path:
        config_path = tmpdir / "config.toml"
        config_path.write_text(
            '[paths]\n'
            'zotero_db = "zotero.sqlite"\n'
            'zotero_storage = "storage"\n'
            'data_dir = "data"\n'
            '\n'
            '[server]\n'
            'host = "127.0.0.1"\n'
            'port = 8765\n'
            'require_api_token = false\n'
            '\n'
            '[[embedding_profiles]]\n'
            'name = "test_text"\n'
            'provider = "stub"\n'
            'model = "stub"\n'
            'dimension = 8\n'
            'modality = "text"\n'
            'backend = "sqlite-local"\n'
            'enabled = true\n'
            'default_for_text = true\n'
            '\n'
            '[[embedding_profiles]]\n'
            'name = "test_mm"\n'
            'provider = "stub"\n'
            'model = "stub"\n'
            'dimension = 8\n'
            'modality = "multimodal"\n'
            'backend = "sqlite-local"\n'
            'enabled = true\n'
            'default_for_multimodal = true\n',
            encoding="utf-8",
        )
        return config_path

    def test_unknown_profile_returns_404(self) -> None:
        with workspace_tmpdir("api-profile-") as tmpdir:
            app = create_app(self._create_config(tmpdir))
            client = self._client(app)
            response = client.get("/vectors/unknown_profile/verify")
            self.assertEqual(404, response.status_code)
            self.assertIn("unknown_profile", response.json()["detail"])

    def test_known_profile_resolves_and_verifies(self) -> None:
        with workspace_tmpdir("api-profile-") as tmpdir:
            app = create_app(self._create_config(tmpdir))
            client = self._client(app)
            response = client.get("/vectors/test_text/verify")
            self.assertEqual(200, response.status_code)
            body = response.json()
            self.assertFalse(body["ok"])
            self.assertTrue(any("vector_store" in error for error in body["errors"]))

    def test_reembed_plan_unknown_profile_returns_404(self) -> None:
        with workspace_tmpdir("api-profile-") as tmpdir:
            app = create_app(self._create_config(tmpdir))
            client = self._client(app)
            response = client.post("/reembed/plan", json={"profile_name": "unknown_profile"})
            self.assertEqual(404, response.status_code)

    def test_reembed_plan_known_profile_returns_plan(self) -> None:
        with workspace_tmpdir("api-profile-") as tmpdir:
            app = create_app(self._create_config(tmpdir))
            client = self._client(app)
            response = client.post("/reembed/plan", json={"profile_name": "test_text"})
            self.assertEqual(200, response.status_code)
            body = response.json()
            self.assertEqual("test_text", body["profile_name"])
            self.assertEqual(0, body["summary"]["document_count"])

    def test_search_text_unknown_profile_returns_404(self) -> None:
        with workspace_tmpdir("api-profile-") as tmpdir:
            app = create_app(self._create_config(tmpdir))
            client = self._client(app)
            response = client.post(
                "/search/text", json={"query": "test", "profile_name": "unknown_profile"}
            )
            self.assertEqual(404, response.status_code)

    def test_search_text_default_profile_resolves(self) -> None:
        with workspace_tmpdir("api-profile-") as tmpdir:
            app = create_app(self._create_config(tmpdir))
            client = self._client(app)
            response = client.post("/search/text", json={"query": "test"})
            self.assertNotEqual(404, response.status_code)


if __name__ == "__main__":
    unittest.main()
