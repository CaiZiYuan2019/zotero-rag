from __future__ import annotations

import unittest
from unittest.mock import patch

from zoterorag.extractors.segments import PageSegment, compute_page_segments
from zoterorag.pdf_pages import get_pdf_page_count


class ComputePageSegmentsTest(unittest.TestCase):
    def test_fits_in_one_segment(self) -> None:
        segments = compute_page_segments(150, 200)
        self.assertEqual([PageSegment(0, "", 150)], segments)

    def test_exactly_at_limit(self) -> None:
        segments = compute_page_segments(200, 200)
        self.assertEqual([PageSegment(0, "", 200)], segments)

    def test_splits_into_two(self) -> None:
        segments = compute_page_segments(250, 200)
        self.assertEqual(
            [
                PageSegment(0, "1-200", 200),
                PageSegment(1, "201-250", 50),
            ],
            segments,
        )

    def test_splits_into_three(self) -> None:
        segments = compute_page_segments(600, 200)
        self.assertEqual(
            [
                PageSegment(0, "1-200", 200),
                PageSegment(1, "201-400", 200),
                PageSegment(2, "401-600", 200),
            ],
            segments,
        )

    def test_custom_limit(self) -> None:
        segments = compute_page_segments(75, 50)
        self.assertEqual(
            [
                PageSegment(0, "1-50", 50),
                PageSegment(1, "51-75", 25),
            ],
            segments,
        )

    def test_invalid_total_pages(self) -> None:
        with self.assertRaises(ValueError):
            compute_page_segments(0, 200)

    def test_invalid_limit(self) -> None:
        with self.assertRaises(ValueError):
            compute_page_segments(10, 0)


class GetPdfPageCountTest(unittest.TestCase):
    def test_missing_file_returns_none(self) -> None:
        self.assertIsNone(get_pdf_page_count("/nonexistent/file.pdf"))

    def test_uninstalled_pymupdf_returns_none(self) -> None:
        with patch("builtins.__import__", side_effect=ImportError("no fitz")):
            self.assertIsNone(get_pdf_page_count(__file__))


if __name__ == "__main__":
    unittest.main()
