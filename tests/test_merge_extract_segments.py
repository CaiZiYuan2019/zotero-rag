from __future__ import annotations

from pathlib import Path
import unittest

from zoterorag.normalize.merge_extract_segments import merge_extract_segments


class MergeExtractSegmentsTest(unittest.TestCase):
    def _make_segment(
        self,
        root: Path,
        segment_index: int,
        markdown_text: str,
        image_name: str | None = None,
        image_bytes: bytes = b"fake-image",
    ) -> Path:
        root = Path(root)
        segment_dir = root / f"segment_{segment_index}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        images_dir = segment_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        if image_name is not None:
            image_path = images_dir / image_name
            image_path.write_bytes(image_bytes)
            full_text = markdown_text.replace("{image}", f"images/{image_name}")
        else:
            full_text = markdown_text

        (segment_dir / "full.md").write_text(full_text, encoding="utf-8")
        return segment_dir

    def test_merge_markdown_only(self) -> None:
        with self._temp_dir() as tmp:
            seg1 = self._make_segment(tmp, 1, "# Title\n\nFirst segment.")
            seg2 = self._make_segment(tmp, 2, "Second segment content.")

            merged_md = merge_extract_segments([seg1, seg2], tmp / "merged")

            text = merged_md.read_text(encoding="utf-8")
            self.assertIn("# Title", text)
            self.assertIn("First segment.", text)
            self.assertIn("Second segment content.", text)
            self.assertIn("## Part 2", text)

    def test_merge_rewrites_image_references(self) -> None:
        with self._temp_dir() as tmp:
            seg1 = self._make_segment(
                tmp,
                1,
                "Text one.\n\n![fig1]({image})\n",
                image_name="page_001.png",
            )
            seg2 = self._make_segment(
                tmp,
                2,
                "Text two.\n\n![fig2]({image})\n",
                image_name="page_001.png",
            )

            merged_md = merge_extract_segments([seg1, seg2], tmp / "merged")
            merged_text = merged_md.read_text(encoding="utf-8")

            self.assertIn("![fig1](images/img001.png)", merged_text)
            self.assertIn("![fig2](images/img002.png)", merged_text)
            self.assertTrue((tmp / "merged" / "images" / "img001.png").is_file())
            self.assertTrue((tmp / "merged" / "images" / "img002.png").is_file())

    def test_merge_html_image_references(self) -> None:
        with self._temp_dir() as tmp:
            seg1 = self._make_segment(
                tmp,
                1,
                'Text one.\n\n<img src="{image}" alt="fig1">\n',
                image_name="page_001.png",
            )
            seg2 = self._make_segment(
                tmp,
                2,
                'Text two.\n\n<img src="{image}" alt="fig2">\n',
                image_name="page_002.png",
            )

            merged_md = merge_extract_segments([seg1, seg2], tmp / "merged")
            merged_text = merged_md.read_text(encoding="utf-8")

            self.assertIn('<img src="images/img001.png" alt="fig1">', merged_text)
            self.assertIn('<img src="images/img002.png" alt="fig2">', merged_text)

    def test_missing_markdown_raises(self) -> None:
        with self._temp_dir() as tmp:
            empty_dir = tmp / "empty"
            empty_dir.mkdir()
            with self.assertRaises(FileNotFoundError):
                merge_extract_segments([empty_dir], tmp / "merged")

    def _temp_dir(self):
        import tempfile

        class _PathTempDir(tempfile.TemporaryDirectory):
            def __enter__(self):
                return Path(super().__enter__())

        return _PathTempDir()


if __name__ == "__main__":
    unittest.main()
