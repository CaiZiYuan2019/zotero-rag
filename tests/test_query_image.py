from __future__ import annotations

import base64
import os
import unittest

from tests._support import workspace_tmpdir
from zoterorag.search import normalize_query_image


class QueryImageTests(unittest.TestCase):
    def test_base64_query_image_is_validated_without_persisting(self) -> None:
        payload = {
            "type": "base64",
            "value": base64.b64encode(b"small image bytes").decode("ascii"),
            "mime_type": "image/png",
        }

        image = normalize_query_image(payload, max_bytes=1024)

        self.assertEqual("base64", image.kind)
        self.assertEqual("image/png", image.mime_type)
        self.assertEqual(payload["value"], image.base64_data)

    def test_file_query_image_must_stay_under_allowed_roots(self) -> None:
        with workspace_tmpdir("query-image-") as tmpdir:
            allowed = tmpdir / "data"
            outside = tmpdir / "outside"
            allowed.mkdir()
            outside.mkdir()
            inside_image = allowed / "query.png"
            outside_image = outside / "query.png"
            inside_image.write_bytes(b"image")
            outside_image.write_bytes(b"image")

            image = normalize_query_image(
                {"type": "file_path", "value": str(inside_image)},
                allowed_roots=[allowed],
            )

            self.assertEqual(str(inside_image.resolve()), image.file_path)
            with self.assertRaises(ValueError):
                normalize_query_image(
                    {"type": "file_path", "value": str(outside_image)},
                    allowed_roots=[allowed],
                )

    def test_invalid_base64_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_query_image({"type": "base64", "value": "not valid base64"})

    def test_symlink_escape_outside_allowed_roots_is_rejected(self) -> None:
        with workspace_tmpdir("query-image-symlink-") as tmpdir:
            allowed = tmpdir / "data"
            outside = tmpdir / "outside"
            allowed.mkdir()
            outside.mkdir()
            real_image = outside / "secret.png"
            real_image.write_bytes(b"image")
            symlink_image = allowed / "link.png"
            try:
                os.symlink(real_image, symlink_image)
            except OSError:  # pragma: no cover - symlinks may require privileges on Windows
                self.skipTest("unable to create symlink in test environment")

            with self.assertRaises(ValueError) as ctx:
                normalize_query_image(
                    {"type": "file_path", "value": str(symlink_image)},
                    allowed_roots=[allowed],
                )
            self.assertIn("resolves outside", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
