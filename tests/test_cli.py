from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import unittest

from tests._support import workspace_tmpdir
from zoterorag.cli import build_parser, main, _resolve_env_path


def run_main(argv: list[str]) -> tuple[int, dict[str, object] | list[object] | str]:
    """Run the CLI main with captured JSON stdout."""

    out = io.StringIO()
    with redirect_stdout(out):
        rc = main(argv)
    raw = out.getvalue()
    try:
        return rc, json.loads(raw)
    except json.JSONDecodeError:
        return rc, raw


def write_config(tmpdir: Path) -> Path:
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
backend = "sqlite-local"
enabled = true
default_for_text = true
default_for_multimodal = false

[[embedding_profiles]]
name = "stub_mm"
provider = "stub"
model = "stub-mm"
dimension = 8
modality = "multimodal"
backend = "sqlite-local"
enabled = true
default_for_text = false
default_for_multimodal = true
""",
        encoding="utf-8",
    )
    return config_path


class CliParserTests(unittest.TestCase):
    def test_build_parser_exposes_high_risk_commands(self) -> None:
        parser = build_parser()
        commands = {action.dest: action.choices for action in parser._actions if hasattr(action, "choices")}
        sub = commands.get("command", {})
        self.assertIn("ingest", sub)
        self.assertIn("search-vector", sub)
        self.assertIn("search", sub)
        self.assertIn("search-mm", sub)
        self.assertIn("reembed", sub)
        self.assertIn("extract", sub)
        self.assertIn("backup", sub)


class CliEnvPathValidationTests(unittest.TestCase):
    def test_resolves_relative_dotenv_inside_project_root(self) -> None:
        with workspace_tmpdir("cli-env-valid-") as tmpdir:
            env_file = tmpdir / ".env"
            env_file.write_text("X=1", encoding="utf-8")
            resolved = _resolve_env_path(".env", tmpdir)
            self.assertEqual(env_file.resolve(), resolved)

    def test_rejects_dotdot_in_env_path(self) -> None:
        with workspace_tmpdir("cli-env-dotdot-") as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                _resolve_env_path("../secret.env", tmpdir)
            self.assertIn("..", str(ctx.exception))

    def test_rejects_path_outside_project_root(self) -> None:
        with workspace_tmpdir("cli-env-outside-") as tmpdir:
            outside = tmpdir.parent / "secret.env"
            outside.write_text("X=1", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                _resolve_env_path(str(outside), tmpdir)
            self.assertIn("inside project root", str(ctx.exception))

    def test_rejects_nonexistent_env_file(self) -> None:
        with workspace_tmpdir("cli-env-missing-") as tmpdir:
            with self.assertRaises(FileNotFoundError) as ctx:
                _resolve_env_path(".env", tmpdir)
            self.assertIn(".env", str(ctx.exception))

    def test_rejects_non_env_filename(self) -> None:
        with workspace_tmpdir("cli-env-basename-") as tmpdir:
            bad = tmpdir / "config.toml"
            bad.write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                _resolve_env_path("config.toml", tmpdir)
            self.assertIn(".env", str(ctx.exception))

    def test_accepts_dotenv_suffix(self) -> None:
        with workspace_tmpdir("cli-env-suffix-") as tmpdir:
            env_file = tmpdir / ".env.prod"
            env_file.write_text("X=1", encoding="utf-8")
            resolved = _resolve_env_path(".env.prod", tmpdir)
            self.assertEqual(env_file.resolve(), resolved)

    def test_main_rejects_arbitrary_env_path(self) -> None:
        with workspace_tmpdir("cli-env-main-") as tmpdir:
            config_path = write_config(tmpdir)
            bad = tmpdir / "secret.env"
            bad.write_text("X=1", encoding="utf-8")
            outside = tmpdir.parent / "outside.env"
            outside.write_text("X=1", encoding="utf-8")

            rc, data = run_main(["--config", str(config_path), "providers", "status", "--env", str(outside)])
            self.assertEqual(1, rc)
            self.assertFalse(data["ok"])


class CliMainTests(unittest.TestCase):
    def test_init_state_creates_runtime(self) -> None:
        with workspace_tmpdir("cli-init-") as tmpdir:
            config_path = write_config(tmpdir)
            rc, data = run_main(["--config", str(config_path), "init-state"])
            self.assertEqual(0, rc)
            self.assertTrue(data["ok"])
            self.assertIn("state", data)
            self.assertIn("runtime", data)

    def test_status_reports_state_and_runtime(self) -> None:
        with workspace_tmpdir("cli-status-") as tmpdir:
            config_path = write_config(tmpdir)
            rc, data = run_main(["--config", str(config_path), "status"])
            self.assertEqual(0, rc)
            self.assertIn("state", data)
            self.assertIn("progress", data)
            self.assertIn("runtime", data)

    def test_models_list_and_activate(self) -> None:
        with workspace_tmpdir("cli-models-") as tmpdir:
            config_path = write_config(tmpdir)
            rc, data = run_main(["--config", str(config_path), "models", "list"])
            self.assertEqual(0, rc)
            self.assertIn("models", data)

            rc2, data2 = run_main(
                ["--config", str(config_path), "models", "activate", "--profile", "stub_mm", "--mode", "multimodal"]
            )
            self.assertEqual(0, rc2)
            self.assertIn("stub_mm", [m["name"] for m in data2["models"]])

    def test_vectors_list_and_verify(self) -> None:
        with workspace_tmpdir("cli-vectors-") as tmpdir:
            config_path = write_config(tmpdir)
            rc, data = run_main(["--config", str(config_path), "vectors", "list"])
            self.assertEqual(0, rc)
            self.assertIn("indexes", data)

            rc2, data2 = run_main(["--config", str(config_path), "vectors", "verify", "--profile", "stub_text"])
            self.assertEqual(1, rc2)
            self.assertFalse(data2["ok"])

    def test_ingest_start_plan_only(self) -> None:
        with workspace_tmpdir("cli-ingest-") as tmpdir:
            config_path = write_config(tmpdir)
            rc, data = run_main(["--config", str(config_path), "ingest", "start"])
            self.assertEqual(0, rc)
            self.assertEqual("planned", data["job"]["status"])
            self.assertIn("plan", data)

    def test_reembed_plan_only(self) -> None:
        with workspace_tmpdir("cli-reembed-") as tmpdir:
            config_path = write_config(tmpdir)
            rc, data = run_main(
                ["--config", str(config_path), "reembed", "--profile", "stub_text", "--from-normalized", "--plan-only"]
            )
            self.assertEqual(0, rc)
            self.assertEqual("stub_text", data["profile_name"])
            self.assertIn("documents", data)

    def test_normalize_markdown_creates_artifact(self) -> None:
        with workspace_tmpdir("cli-normalize-") as tmpdir:
            config_path = write_config(tmpdir)
            source_dir = tmpdir / "mineru"
            images_dir = source_dir / "images"
            images_dir.mkdir(parents=True)
            (images_dir / "fig.png").write_bytes(b"fake-image")
            markdown = source_dir / "full.md"
            markdown.write_text("# Paper\n\n![Figure](images/fig.png)\n", encoding="utf-8")

            rc, data = run_main(
                [
                    "--config",
                    str(config_path),
                    "normalize",
                    "markdown",
                    "--markdown",
                    str(markdown),
                    "--document-id",
                    "DOC1",
                ]
            )
            self.assertEqual(0, rc)
            self.assertIn("chunk_count", data)
            self.assertGreater(data["chunk_count"], 0)

    def test_extract_dry_run_with_invalid_options_json_fails_cleanly(self) -> None:
        with workspace_tmpdir("cli-extract-") as tmpdir:
            config_path = write_config(tmpdir)
            pdf = tmpdir / "paper.pdf"
            pdf.write_bytes(b"%PDF-1.4 fake")
            rc, data = run_main(
                [
                    "--config",
                    str(config_path),
                    "extract",
                    "dry-run",
                    "--pdf",
                    str(pdf),
                    "--options-json",
                    "not valid json",
                ]
            )
            self.assertEqual(1, rc)
            self.assertFalse(data["ok"])
            self.assertIn("invalid --options-json", data["error"])

    def test_search_alias_recognized_and_defaults_to_text(self) -> None:
        with workspace_tmpdir("cli-search-") as tmpdir:
            config_path = write_config(tmpdir)
            # Empty vector index is fine; the command should parse and execute.
            rc, data = run_main(
                ["--config", str(config_path), "search", "test query", "--embedding-provider", "stub"]
            )
            self.assertEqual(0, rc)
            self.assertIn("results", data)

    def test_search_mm_defaults_to_multimodal(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["search-mm", "test query"])
        self.assertEqual("multimodal", args.mode)
        self.assertEqual("manual", args.consumer)
        self.assertEqual("file_ref", args.image_return)


if __name__ == "__main__":
    unittest.main()
