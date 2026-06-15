from __future__ import annotations

import os
import unittest

from tests._support import OptionalModuleTestCase, workspace_tmpdir
from zoterorag.api.security import AccessDenied, is_loopback_host, verify_api_access


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
        verify_api_access(supplied_token=None, client_host="::ffff:127.0.0.1", require_api_token=True)

    def test_loopback_detection_handles_local_and_unknown_hosts(self) -> None:
        self.assertTrue(is_loopback_host("localhost"))
        self.assertTrue(is_loopback_host("[::1]"))
        self.assertTrue(is_loopback_host("::ffff:127.0.0.1"))
        self.assertFalse(is_loopback_host(None))
        self.assertFalse(is_loopback_host("192.168.1.50"))
        self.assertFalse(is_loopback_host("example.invalid"))

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

    def test_non_loopback_rejected_when_auth_optional_and_no_token_configured(self) -> None:
        verify_api_access(supplied_token=None, client_host="127.0.0.1", require_api_token=False)
        with self.assertRaises(AccessDenied):
            verify_api_access(supplied_token=None, client_host="10.0.0.5", require_api_token=False)
        with self.assertRaises(AccessDenied):
            verify_api_access(supplied_token=None, client_host=None, require_api_token=False)

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

    def test_reembed_plan_route_is_read_only(self) -> None:
        fastapi_testclient = self.import_first_available(["fastapi.testclient"])
        from zoterorag.api.app import create_app

        with workspace_tmpdir("api-reembed-plan-") as tmpdir:
            config_path = tmpdir / "config.toml"
            config_path.write_text(
                f"""
[paths]
zotero_db = "{(tmpdir / 'zotero.sqlite').as_posix()}"
zotero_storage = "{(tmpdir / 'storage').as_posix()}"
data_dir = "{(tmpdir / 'data').as_posix()}"

[server]
require_api_token = true

[[embedding_profiles]]
name = "stub_text"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
enabled = true
default_for_text = true
""",
                encoding="utf-8",
            )
            os.environ["ZOTERORAG_API_TOKEN"] = "expected-token"
            app = create_app(config_path)
            client = fastapi_testclient.TestClient(app)

            planned = client.post(
                "/reembed/plan",
                headers={"X-API-Token": "expected-token"},
                json={"profile_name": "stub_text"},
            )
            self.assertEqual(200, planned.status_code)
            self.assertEqual("stub_text", planned.json()["profile_name"])

            jobs = client.get("/jobs", headers={"X-API-Token": "expected-token"})
            self.assertEqual(200, jobs.status_code)
            self.assertEqual([], jobs.json()["jobs"])

    def test_health_endpoint_requires_access_control(self) -> None:
        fastapi_testclient = self.import_first_available(["fastapi.testclient"])
        from zoterorag.api.app import create_app

        with workspace_tmpdir("api-health-") as tmpdir:
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

            denied = client.get("/health")
            self.assertEqual(403, denied.status_code)

            allowed = client.get("/health", headers={"X-API-Token": "expected-token"})
            self.assertEqual(200, allowed.status_code)
            self.assertEqual({"status": "ok"}, allowed.json())

    def test_list_limits_are_capped(self) -> None:
        fastapi_testclient = self.import_first_available(["fastapi.testclient"])
        from zoterorag.api.app import create_app

        with workspace_tmpdir("api-limits-") as tmpdir:
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

            # Very large limit is accepted but capped server-side.
            response = client.get("/documents?limit=99999999", headers={"X-API-Token": "expected-token"})
            self.assertEqual(200, response.status_code)

    def test_backup_create_rejects_outside_output_path(self) -> None:
        fastapi_testclient = self.import_first_available(["fastapi.testclient"])
        from zoterorag.api.app import create_app

        with workspace_tmpdir("api-backup-") as tmpdir:
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

            denied = client.post(
                "/backup/create",
                headers={"X-API-Token": "expected-token"},
                json={"mode": "snapshot", "out": "../escape"},
            )
            self.assertEqual(400, denied.status_code)
            self.assertEqual("invalid request", denied.json()["detail"])


class ApiControlPlaneTests(OptionalModuleTestCase):
    def setUp(self) -> None:
        self._old_token = os.environ.pop("ZOTERORAG_API_TOKEN", None)

    def tearDown(self) -> None:
        if self._old_token is None:
            os.environ.pop("ZOTERORAG_API_TOKEN", None)
        else:
            os.environ["ZOTERORAG_API_TOKEN"] = self._old_token

    def _client_for(self, tmpdir):
        fastapi_testclient = self.import_first_available(["fastapi.testclient"])
        from zoterorag.api.app import create_app

        config_path = tmpdir / "config.toml"
        config_path.write_text(
            f"""
[paths]
zotero_db = "{(tmpdir / 'zotero.sqlite').as_posix()}"
zotero_storage = "{(tmpdir / 'storage').as_posix()}"
data_dir = "{(tmpdir / 'data').as_posix()}"

[server]
require_api_token = true

[[embedding_profiles]]
name = "stub_text"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
enabled = true
default_for_text = true
""",
            encoding="utf-8",
        )
        os.environ["ZOTERORAG_API_TOKEN"] = "expected-token"
        app = create_app(config_path)
        return fastapi_testclient.TestClient(app)

    def test_status_and_progress_endpoints(self) -> None:
        with workspace_tmpdir("api-control-") as tmpdir:
            client = self._client_for(tmpdir)
            headers = {"X-API-Token": "expected-token"}

            status = client.get("/status", headers=headers)
            self.assertEqual(200, status.status_code)
            self.assertIn("runtime", status.json())

            progress = client.get("/progress", headers=headers)
            self.assertEqual(200, progress.status_code)
            self.assertIn("state", progress.json())

    def test_models_and_ingest_plan_endpoints(self) -> None:
        with workspace_tmpdir("api-control-") as tmpdir:
            client = self._client_for(tmpdir)
            headers = {"X-API-Token": "expected-token"}

            models = client.get("/models/embedding", headers=headers)
            self.assertEqual(200, models.status_code)
            self.assertIn("models", models.json())

            activate = client.post(
                "/models/embedding/activate",
                headers=headers,
                json={"profile_name": "stub_text", "mode": "text"},
            )
            self.assertEqual(200, activate.status_code)

            ingest = client.post("/ingest/start", headers=headers, json={"execute": False})
            self.assertEqual(200, ingest.status_code)
            self.assertEqual("planned", ingest.json()["job"]["status"])

            jobs = client.get("/jobs", headers=headers)
            self.assertEqual(200, jobs.status_code)
            self.assertIsInstance(jobs.json()["jobs"], list)

    def test_reembed_plan_endpoint(self) -> None:
        with workspace_tmpdir("api-control-") as tmpdir:
            client = self._client_for(tmpdir)
            headers = {"X-API-Token": "expected-token"}

            plan = client.post("/reembed/plan", headers=headers, json={"profile_name": "stub_text"})
            self.assertEqual(200, plan.status_code)
            self.assertEqual("stub_text", plan.json()["profile_name"])


if __name__ == "__main__":
    unittest.main()
