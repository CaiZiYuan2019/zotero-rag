from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageSegment:
    """One contiguous page range to submit to a single extraction job."""

    index: int
    selected_pages: str
    page_count: int


def compute_page_segments(total_pages: int, max_pages_per_segment: int = 200) -> list[PageSegment]:
    """Split ``total_pages`` into segments of at most ``max_pages_per_segment``.

    The returned ``selected_pages`` strings use the ``start-end`` format
    understood by MinerU's ``page_ranges`` parameter. When the whole document
    fits in one segment, a single segment with an empty ``selected_pages``
    string is returned so that the caller can submit the full PDF without
    artificially constraining the page range.

    Raises:
        ValueError: If ``total_pages`` or ``max_pages_per_segment`` is not
            positive.
    """

    if total_pages <= 0:
        raise ValueError("total_pages must be positive")
    if max_pages_per_segment <= 0:
        raise ValueError("max_pages_per_segment must be positive")

    if total_pages <= max_pages_per_segment:
        return [PageSegment(0, "", total_pages)]

    segments: list[PageSegment] = []
    start = 1
    index = 0
    while start <= total_pages:
        end = min(start + max_pages_per_segment - 1, total_pages)
        segments.append(PageSegment(index, f"{start}-{end}", end - start + 1))
        start = end + 1
        index += 1
    return segments
