from __future__ import annotations

import os
from pathlib import Path
import unittest

from tests._support import workspace_tmpdir
from zoterorag.config import EmbeddingProfile, ExtractConfig, IngestConfig, ServerConfig, VectorsConfig, load_config
from zoterorag.runtime import config_as_public_dict, vector_store_path_for_profile


class ConfigLoadingTests(unittest.TestCase):
    def test_default_config_path_uses_config_toml(self) -> None:
        with workspace_tmpdir("config-default-") as tmpdir:
            config_path = tmpdir / "config" / "config.toml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                f"""
[paths]
zotero_db = "{(tmpdir / 'zotero.sqlite').as_posix()}"
zotero_storage = "{(tmpdir / 'storage').as_posix()}"
data_dir = "data"

[[embedding_profiles]]
name = "stub_text"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
enabled = true
default_for_text = true
default_for_multimodal = false
""",
                encoding="utf-8",
            )
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                config = load_config()
                self.assertEqual(
                    (tmpdir / "data").resolve(),
                    config.paths.data_dir,
                )
            finally:
                os.chdir(original_cwd)

    def test_relative_data_dir_resolves_to_config_parent(self) -> None:
        with workspace_tmpdir("config-relative-") as tmpdir:
            config_dir = tmpdir / "settings"
            config_dir.mkdir()
            config_path = config_dir / "config.toml"
            config_path.write_text(
                f"""
[paths]
zotero_db = "{(tmpdir / 'zotero.sqlite').as_posix()}"
zotero_storage = "{(tmpdir / 'storage').as_posix()}"
data_dir = "../runtime_data"

[[embedding_profiles]]
name = "stub_text"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
enabled = true
default_for_text = true
default_for_multimodal = false
""",
                encoding="utf-8",
            )
            config = load_config(config_path)
            self.assertEqual(
                (tmpdir / "runtime_data").resolve(),
                config.paths.data_dir,
            )

    def test_missing_required_paths_raises(self) -> None:
        with workspace_tmpdir("config-missing-") as tmpdir:
            config_path = tmpdir / "config.toml"
            config_path.write_text(
                """
[paths]
zotero_db = "/tmp/zotero.sqlite"

[[embedding_profiles]]
name = "stub_text"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
""",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                load_config(config_path)
            self.assertIn("zotero_storage", str(ctx.exception))

    def test_environment_placeholders_in_zotero_paths_are_expanded(self) -> None:
        with workspace_tmpdir("config-env-") as tmpdir:
            config_path = tmpdir / "config.toml"
            config_path.write_text(
                """
[paths]
zotero_db = "<ZOTERO_DB_PATH>/zotero.sqlite"
zotero_storage = "$ZOTERO_STORAGE_PATH/storage"
data_dir = "data"

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
            original_db = os.environ.get("ZOTERO_DB_PATH")
            original_storage = os.environ.get("ZOTERO_STORAGE_PATH")
            try:
                os.environ["ZOTERO_DB_PATH"] = str(tmpdir / "zotero_lib")
                os.environ["ZOTERO_STORAGE_PATH"] = str(tmpdir / "zotero_lib")
                config = load_config(config_path)
                self.assertEqual(
                    Path(tmpdir / "zotero_lib" / "zotero.sqlite").resolve(),
                    config.paths.zotero_db,
                )
                self.assertEqual(
                    Path(tmpdir / "zotero_lib" / "storage").resolve(),
                    config.paths.zotero_storage,
                )
            finally:
                if original_db is None:
                    os.environ.pop("ZOTERO_DB_PATH", None)
                else:
                    os.environ["ZOTERO_DB_PATH"] = original_db
                if original_storage is None:
                    os.environ.pop("ZOTERO_STORAGE_PATH", None)
                else:
                    os.environ["ZOTERO_STORAGE_PATH"] = original_storage

    def test_unresolved_placeholders_raise_clear_error(self) -> None:
        with workspace_tmpdir("config-unresolved-") as tmpdir:
            config_path = tmpdir / "config.toml"
            config_path.write_text(
                """
[paths]
zotero_db = "<ZOTERO_DB_PATH>/zotero.sqlite"
zotero_storage = "/tmp/storage"
data_dir = "data"

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
            original = os.environ.pop("ZOTERO_DB_PATH", None)
            try:
                with self.assertRaises(ValueError) as ctx:
                    load_config(config_path)
                message = str(ctx.exception)
                self.assertIn("ZOTERO_DB_PATH", message)
                self.assertIn("unresolved placeholder", message)
            finally:
                if original is not None:
                    os.environ["ZOTERO_DB_PATH"] = original

    def test_require_api_token_rejects_non_boolean_strings(self) -> None:
        with workspace_tmpdir("config-bool-") as tmpdir:
            config_path = tmpdir / "config.toml"
            base = f"""
[paths]
zotero_db = "{(tmpdir / 'zotero.sqlite').as_posix()}"
zotero_storage = "{(tmpdir / 'storage').as_posix()}"
"""
            config_path.write_text(
                base + """
[server]
require_api_token = "yes"

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
            with self.assertRaises(ValueError) as ctx:
                load_config(config_path)
            self.assertIn("require_api_token", str(ctx.exception))

    def test_require_api_token_accepts_true_and_false_strings(self) -> None:
        with workspace_tmpdir("config-bool-") as tmpdir:
            config_path = tmpdir / "config.toml"
            base = f"""
[paths]
zotero_db = "{(tmpdir / 'zotero.sqlite').as_posix()}"
zotero_storage = "{(tmpdir / 'storage').as_posix()}"
"""
            config_path.write_text(
                base + """
[server]
require_api_token = "false"

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
            config = load_config(config_path)
            self.assertFalse(config.server.require_api_token)


class EmbeddingProfileValidationTests(unittest.TestCase):
    def _write_config(self, tmpdir, profiles_block: str, extra: str = "") -> None:
        config_path = tmpdir / "config.toml"
        config_path.write_text(
            f"""
[paths]
zotero_db = "{(tmpdir / 'zotero.sqlite').as_posix()}"
zotero_storage = "{(tmpdir / 'storage').as_posix()}"
{extra}
{profiles_block}
""",
            encoding="utf-8",
        )
        return config_path

    def test_missing_required_profile_field_raises(self) -> None:
        with workspace_tmpdir("profile-missing-") as tmpdir:
            config_path = self._write_config(
                tmpdir,
                """
[[embedding_profiles]]
name = "stub_text"
provider = "stub"
dimension = 8
modality = "text"
enabled = true
default_for_text = true
""",
            )
            with self.assertRaises(ValueError) as ctx:
                load_config(config_path)
            self.assertIn("model", str(ctx.exception))

    def test_embed_workers_defaults_to_one(self) -> None:
        with workspace_tmpdir("profile-workers-default-") as tmpdir:
            config_path = self._write_config(
                tmpdir,
                """
[[embedding_profiles]]
name = "stub_text"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
default_for_text = true
""",
            )
            config = load_config(config_path)
            self.assertEqual(1, config.embedding_profiles[0].embed_workers)

    def test_embed_workers_parsed_from_toml(self) -> None:
        with workspace_tmpdir("profile-workers-") as tmpdir:
            config_path = self._write_config(
                tmpdir,
                """
[[embedding_profiles]]
name = "stub_text"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
default_for_text = true
embed_workers = 8
rate_limit = { requests_per_second = 10 }
""",
            )
            config = load_config(config_path)
            profile = config.embedding_profiles[0]
            self.assertEqual(8, profile.embed_workers)
            self.assertEqual({"requests_per_second": 10}, profile.rate_limit)

    def test_embed_workers_invalid_values_raise(self) -> None:
        for bad in ("0", "-3", "true"):
            with workspace_tmpdir("profile-workers-bad-") as tmpdir:
                config_path = self._write_config(
                    tmpdir,
                    f"""
[[embedding_profiles]]
name = "stub_text"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
default_for_text = true
embed_workers = {bad}
""",
                )
                with self.assertRaises(ValueError) as ctx:
                    load_config(config_path)
                self.assertIn("embed_workers", str(ctx.exception))

    def test_duplicate_profile_name_raises(self) -> None:
        with workspace_tmpdir("profile-dup-") as tmpdir:
            config_path = self._write_config(
                tmpdir,
                """
[[embedding_profiles]]
name = "stub_text"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"

[[embedding_profiles]]
name = "stub_text"
provider = "other"
model = "other"
dimension = 8
modality = "text"
""",
            )
            with self.assertRaises(ValueError) as ctx:
                load_config(config_path)
            self.assertIn("duplicate", str(ctx.exception))

    def test_non_positive_dimension_raises(self) -> None:
        with workspace_tmpdir("profile-dim-") as tmpdir:
            config_path = self._write_config(
                tmpdir,
                """
[[embedding_profiles]]
name = "stub_text"
provider = "stub"
model = "stub"
dimension = 0
modality = "text"
enabled = true
default_for_text = true
""",
            )
            with self.assertRaises(ValueError) as ctx:
                load_config(config_path)
            self.assertIn("dimension", str(ctx.exception))

    def test_invalid_modality_raises(self) -> None:
        with workspace_tmpdir("profile-modality-") as tmpdir:
            config_path = self._write_config(
                tmpdir,
                """
[[embedding_profiles]]
name = "stub_text"
provider = "stub"
model = "stub"
dimension = 8
modality = "video"
enabled = true
default_for_text = true
""",
            )
            with self.assertRaises(ValueError) as ctx:
                load_config(config_path)
            self.assertIn("modality", str(ctx.exception))

    def test_multiple_defaults_for_same_modality_raises(self) -> None:
        with workspace_tmpdir("profile-defaults-") as tmpdir:
            config_path = self._write_config(
                tmpdir,
                """
[[embedding_profiles]]
name = "stub_text_a"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
enabled = true
default_for_text = true

[[embedding_profiles]]
name = "stub_text_b"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
enabled = true
default_for_text = true
""",
            )
            with self.assertRaises(ValueError) as ctx:
                load_config(config_path)
            self.assertIn("default_for_text", str(ctx.exception))

    def test_backend_defaults_to_lancedb_and_accepts_sqlite_local(self) -> None:
        with workspace_tmpdir("profile-backend-") as tmpdir:
            config_path = self._write_config(
                tmpdir,
                """
[[embedding_profiles]]
name = "lancedb_profile"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
enabled = true
default_for_text = true

[[embedding_profiles]]
name = "sqlite_profile"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
backend = "sqlite-local"
enabled = true
default_for_text = false
""",
            )
            config = load_config(config_path)
            self.assertEqual("lancedb", config.embedding_profiles[0].backend)
            self.assertEqual("sqlite-local", config.embedding_profiles[1].backend)


class VectorStorePathTests(unittest.TestCase):
    def test_lancedb_profile_uses_directory_path(self) -> None:
        profile = EmbeddingProfile(
            name="mm",
            provider="stub",
            model="stub",
            dimension=8,
            modality="multimodal",
            backend="lancedb",
        )
        path = vector_store_path_for_profile("/store", profile)
        self.assertEqual(Path("/store/mm"), path)

    def test_sqlite_profile_uses_file_path(self) -> None:
        profile = EmbeddingProfile(
            name="text",
            provider="stub",
            model="stub",
            dimension=8,
            modality="text",
            backend="sqlite-local",
        )
        path = vector_store_path_for_profile("/store", profile)
        self.assertEqual(Path("/store/text/vectors.sqlite"), path)


class ConfigPublicDictTests(unittest.TestCase):
    def test_public_dict_exposes_backend_and_image_policy(self) -> None:
        with workspace_tmpdir("config-public-") as tmpdir:
            config_path = tmpdir / "config.toml"
            config_path.write_text(
                f"""
[paths]
zotero_db = "{(tmpdir / 'zotero.sqlite').as_posix()}"
zotero_storage = "{(tmpdir / 'storage').as_posix()}"

[[embedding_profiles]]
name = "stub_text"
provider = "stub"
model = "stub"
dimension = 8
modality = "text"
backend = "sqlite-local"
image_policy = {{ max_long_edge = 1600 }}
batch_size = 32
enabled = true
default_for_text = true
""",
                encoding="utf-8",
            )
            config = load_config(config_path)
            public = config_as_public_dict(config)
            profile = public["embedding_profiles"][0]
            self.assertEqual("sqlite-local", profile["backend"])
            self.assertEqual({"max_long_edge": 1600}, profile["image_policy"])
            self.assertEqual(32, profile["batch_size"])


class ExtractConfigTests(unittest.TestCase):
    def _write_config(self, tmpdir: Path, extract_section: str) -> Path:
        config_path = tmpdir / "config.toml"
        config_path.write_text(
            f"""
[paths]
zotero_db = "{(tmpdir / 'zotero.sqlite').as_posix()}"
zotero_storage = "{(tmpdir / 'storage').as_posix()}"
data_dir = "data"

{extract_section}
""",
            encoding="utf-8",
        )
        return config_path

    def test_job_max_lifetime_defaults_and_parses(self) -> None:
        with workspace_tmpdir("extract-config-") as tmpdir:
            config = load_config(self._write_config(tmpdir, ""))
            self.assertEqual(21600, config.extract.job_max_lifetime_seconds)
        with workspace_tmpdir("extract-config-") as tmpdir:
            config = load_config(
                self._write_config(tmpdir, "[extract]\njob_max_lifetime_seconds = 7200\n")
            )
            self.assertEqual(7200, config.extract.job_max_lifetime_seconds)
            self.assertEqual(200, config.extract.max_pages_per_extraction)

    def test_job_max_lifetime_must_be_positive(self) -> None:
        with workspace_tmpdir("extract-config-") as tmpdir:
            with self.assertRaises(ValueError):
                load_config(self._write_config(tmpdir, "[extract]\njob_max_lifetime_seconds = 0\n"))
        with self.assertRaises(ValueError):
            ExtractConfig(job_max_lifetime_seconds=0)
        with self.assertRaises(ValueError):
            ExtractConfig(job_max_lifetime_seconds=True)

    def test_vectors_stale_batch_hours_defaults_and_parses(self) -> None:
        with workspace_tmpdir("vectors-config-") as tmpdir:
            config = load_config(self._write_config(tmpdir, ""))
            self.assertEqual(24.0, config.vectors.stale_embedding_batch_hours)
        with workspace_tmpdir("vectors-config-") as tmpdir:
            config = load_config(
                self._write_config(tmpdir, "[vectors]\nstale_embedding_batch_hours = 12\n")
            )
            self.assertEqual(12.0, config.vectors.stale_embedding_batch_hours)

    def test_vectors_stale_batch_hours_must_be_positive(self) -> None:
        with workspace_tmpdir("vectors-config-") as tmpdir:
            with self.assertRaises(ValueError):
                load_config(self._write_config(tmpdir, "[vectors]\nstale_embedding_batch_hours = 0\n"))
        with self.assertRaises(ValueError):
            VectorsConfig(stale_embedding_batch_hours=-1)


if __name__ == "__main__":
    unittest.main()


class ExtractPhase2ConfigTests(unittest.TestCase):
    """Phase 2 [extract] pacing/concurrency/resubmit knobs (§2.4)."""

    def _write_config(self, tmpdir: Path, extract_section: str) -> Path:
        config_path = tmpdir / "config.toml"
        config_path.write_text(
            f"""
[paths]
zotero_db = "{(tmpdir / 'zotero.sqlite').as_posix()}"
zotero_storage = "{(tmpdir / 'storage').as_posix()}"
data_dir = "data"

{extract_section}
""",
            encoding="utf-8",
        )
        return config_path

    def test_phase2_defaults_and_parsing(self) -> None:
        with workspace_tmpdir("extract-phase2-config-") as tmpdir:
            config = load_config(self._write_config(tmpdir, ""))
            self.assertEqual(1, config.extract.per_key_submit_concurrency)
            self.assertEqual(1.5, config.extract.submit_min_interval_seconds)
            self.assertEqual(2, config.extract.harvest_workers)
            self.assertEqual(600, config.extract.poll_max_per_minute)
            self.assertEqual(30.0, config.extract.poll_interval_seconds)
            self.assertEqual(3, config.extract.max_resubmit_attempts)
        with workspace_tmpdir("extract-phase2-config-") as tmpdir:
            config = load_config(
                self._write_config(
                    tmpdir,
                    "[extract]\n"
                    "per_key_submit_concurrency = 2\n"
                    "submit_min_interval_seconds = 0.5\n"
                    "harvest_workers = 4\n"
                    "poll_max_per_minute = 100\n"
                    "poll_interval_seconds = 5\n"
                    "max_resubmit_attempts = 5\n",
                )
            )
            self.assertEqual(2, config.extract.per_key_submit_concurrency)
            self.assertEqual(0.5, config.extract.submit_min_interval_seconds)
            self.assertEqual(4, config.extract.harvest_workers)
            self.assertEqual(100, config.extract.poll_max_per_minute)
            self.assertEqual(5.0, config.extract.poll_interval_seconds)
            self.assertEqual(5, config.extract.max_resubmit_attempts)

    def test_phase2_validation_rejects_bad_values(self) -> None:
        with self.assertRaises(ValueError):
            ExtractConfig(per_key_submit_concurrency=0)
        with self.assertRaises(ValueError):
            ExtractConfig(per_key_submit_concurrency=True)
        with self.assertRaises(ValueError):
            ExtractConfig(submit_min_interval_seconds=-0.1)
        with self.assertRaises(ValueError):
            ExtractConfig(submit_min_interval_seconds=True)
        with self.assertRaises(ValueError):
            ExtractConfig(harvest_workers=0)
        with self.assertRaises(ValueError):
            ExtractConfig(harvest_workers=True)
        with self.assertRaises(ValueError):
            ExtractConfig(poll_max_per_minute=-1)
        with self.assertRaises(ValueError):
            ExtractConfig(poll_interval_seconds=0)
        with self.assertRaises(ValueError):
            ExtractConfig(poll_interval_seconds=True)
        with self.assertRaises(ValueError):
            ExtractConfig(max_resubmit_attempts=-1)
        with self.assertRaises(ValueError):
            ExtractConfig(max_resubmit_attempts=True)


class IngestConfigTests(unittest.TestCase):
    """Phase 3 [ingest] batch_publish knob."""

    def _write_config(self, tmpdir: Path, ingest_section: str) -> Path:
        config_path = tmpdir / "config.toml"
        config_path.write_text(
            f"""
[paths]
zotero_db = "{(tmpdir / 'zotero.sqlite').as_posix()}"
zotero_storage = "{(tmpdir / 'storage').as_posix()}"
data_dir = "data"

{ingest_section}
""",
            encoding="utf-8",
        )
        return config_path

    def test_batch_publish_defaults_to_true(self) -> None:
        with workspace_tmpdir("ingest-config-default-") as tmpdir:
            config = load_config(self._write_config(tmpdir, ""))
            self.assertTrue(config.ingest.batch_publish)
        self.assertTrue(IngestConfig().batch_publish)

    def test_batch_publish_explicit_false(self) -> None:
        with workspace_tmpdir("ingest-config-false-") as tmpdir:
            config = load_config(self._write_config(tmpdir, "[ingest]\nbatch_publish = false\n"))
            self.assertFalse(config.ingest.batch_publish)

    def test_batch_publish_rejects_non_boolean(self) -> None:
        with workspace_tmpdir("ingest-config-bad-") as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                load_config(self._write_config(tmpdir, '[ingest]\nbatch_publish = "yes"\n'))
            self.assertIn("ingest.batch_publish", str(ctx.exception))
        with self.assertRaises(TypeError):
            IngestConfig(batch_publish=1)
