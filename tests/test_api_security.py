from __future__ import annotations

import os
import unittest

from tests._support import OptionalModuleTestCase, workspace_tmpdir
from zoterorag.api.security import AccessDenied, verify_api_access


class ApiSecurityTests(OptionalModuleTestCase):
    def setUp(self) -> None:
        self._old_token = os.environ.pop("ZOTERORAG_API_TOKEN", None)

    def tearDown(self) -> None:
        if self._old_token is None:
            os.environ.pop("ZOTERORAG_API_TOKEN", None)
        else:
            os.environ["ZOTERORAG_API_TOKEN"] = self._old_token

    def test_loopback_allowed_without_token_for_local_bootstrap(self) -> None:
        verify_api_access(supplied_token=None, client_host="127.0.0.1", require_api_token=True)
        verify_api_access(supplied_token=None, client_host="::1", require_api_token=True)

    def test_non_loopback_rejected_without_configured_token(self) -> None:
        with self.assertRaises(AccessDenied):
            verify_api_access(supplied_token=None, client_host="192.168.1.50", require_api_token=True)

    def test_configured_token_must_match_when_required(self) -> None:
        os.environ["ZOTERORAG_API_TOKEN"] = "expected-token"
        with self.assertRaises(AccessDenied):
            verify_api_access(supplied_token=None, client_host="127.0.0.1", require_api_token=True)
        with self.assertRaises(AccessDenied):
            verify_api_access(supplied_token="wrong-token", client_host="127.0.0.1", require_api_token=True)

        verify_api_access(supplied_token="expected-token", client_host="192.168.1.50", require_api_token=True)

    def test_non_loopback_still_needs_token_when_auth_optional_but_token_exists(self) -> None:
        os.environ["ZOTERORAG_API_TOKEN"] = "expected-token"
        verify_api_access(supplied_token=None, client_host="127.0.0.1", require_api_token=False)
        with self.assertRaises(AccessDenied):
            verify_api_access(supplied_token=None, client_host="10.0.0.5", require_api_token=False)
        verify_api_access(supplied_token="expected-token", client_host="10.0.0.5", require_api_token=False)

    def test_documents_route_uses_api_access_control(self) -> None:
        fastapi_testclient = self.import_first_available(["fastapi.testclient"])
        from zoterorag.api.app import create_app

        with workspace_tmpdir("api-documents-") as tmpdir:
            config_path = tmpdir / "config.toml"
            config_path.write_text(
                f"""
[paths]
zotero_db = "{(tmpdir / 'zotero.sqlite').as_posix()}"
zotero_storage = "{(tmpdir / 'storage').as_posix()}"
data_dir = "{(tmpdir / 'data').as_posix()}"

[server]
require_api_token = true
""",
                encoding="utf-8",
            )
            os.environ["ZOTERORAG_API_TOKEN"] = "expected-token"
            app = create_app(config_path)
            client = fastapi_testclient.TestClient(app)

            denied = client.get("/documents")
            self.assertEqual(403, denied.status_code)

            allowed = client.get("/documents", headers={"X-API-Token": "expected-token"})
            self.assertEqual(200, allowed.status_code)
            self.assertEqual({"documents": []}, allowed.json())


if __name__ == "__main__":
    unittest.main()
