from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
from typing import Any

from .markdown import (
    HTML_IMAGE_RE,
    MARKDOWN_IMAGE_RE,
    collect_image_references,
    split_markdown_target,
)


def _find_source_markdown(artifact_dir: Path) -> Path | None:
    """Locate the primary markdown file inside a MinerU extract artifact."""

    candidates = list(artifact_dir.glob("*.md"))
    if not candidates:
        for child in artifact_dir.iterdir():
            if child.is_dir():
                candidates.extend(child.glob("*.md"))
    if not candidates:
        return None
    for candidate in candidates:
        if candidate.name.lower() in ("full.md", "output.md"):
            return candidate
    return max(candidates, key=lambda p: p.stat().st_size)


def _rewrite_image_references(markdown_text: str, mapping: dict[str, str]) -> str:
    """Rewrite image references using a target -> new-relative-path mapping."""

    def replace_markdown(match: re.Match[str]) -> str:
        prefix = match.group(1)
        target = match.group(2)
        suffix = match.group(3)
        leading, target_path, trailing = split_markdown_target(target)
        replacement = mapping.get(target_path)
        if replacement is None:
            return match.group(0)
        return f"{prefix}{leading}{replacement}{trailing}{suffix}"

    def replace_html(match: re.Match[str]) -> str:
        prefix = match.group(1)
        target = match.group(2)
        suffix = match.group(3)
        replacement = mapping.get(target)
        if replacement is None:
            return match.group(0)
        return f"{prefix}{replacement}{suffix}"

    text = MARKDOWN_IMAGE_RE.sub(replace_markdown, markdown_text)
    return HTML_IMAGE_RE.sub(replace_html, text)


def merge_extract_segments(
    segment_artifact_dirs: list[Path],
    output_dir: Path,
) -> Path:
    """Merge several MinerU extract artifacts into a single artifact directory.

    Each segment artifact is expected to contain one primary markdown file and
    an ``images/`` directory (or image paths relative to the markdown). The
    function concatenates the markdown files in order, renumbers images
    globally to avoid filename collisions across segments, and rewrites image
    references so they point at the merged ``images/`` directory.

    A ``## Part N`` heading is inserted between segments to preserve boundary
    information for downstream chunking.

    Args:
        segment_artifact_dirs: Ordered list of segment artifact directories.
        output_dir: Directory where ``merged_full.md`` and ``images/`` will be
            written.

    Returns:
        Path to the merged markdown file (``output_dir / "merged_full.md"``).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    global_image_index = 0
    merged_parts: list[str] = []

    for segment_index, segment_dir in enumerate(segment_artifact_dirs):
        segment_dir = Path(segment_dir)
        source_md = _find_source_markdown(segment_dir)
        if source_md is None:
            raise FileNotFoundError(f"no markdown file found in segment artifact: {segment_dir}")

        markdown_text = source_md.read_text(encoding="utf-8")
        image_refs = collect_image_references(markdown_text, source_md.parent)

        mapping: dict[str, str] = {}
        seen_targets: set[str] = set()
        for ref in image_refs:
            target = ref["target"]
            if target in seen_targets:
                continue
            seen_targets.add(target)
            source_path = ref["source_path"]
            if not source_path.is_file():
                continue

            global_image_index += 1
            suffix = source_path.suffix.lower() or ".img"
            ordered_name = f"img{global_image_index:03d}{suffix}"
            dest_path = images_dir / ordered_name
            shutil.copy2(source_path, dest_path)
            mapping[target] = f"images/{ordered_name}"

        rewritten_text = _rewrite_image_references(markdown_text, mapping)

        if segment_index > 0:
            merged_parts.append(f"\n\n## Part {segment_index + 1}\n\n")
        merged_parts.append(rewritten_text)

    merged_text = "".join(merged_parts)
    merged_md = output_dir / "merged_full.md"
    # H2: write to a sibling temp file and atomically rename it into place.
    # A crash mid-write then only leaves a .tmp file behind, so the caller's
    # exists-skip guard can never mistake a truncated merged_full.md for a
    # complete artifact. os.replace is atomic on the same directory/volume.
    tmp_md = output_dir / (merged_md.name + ".tmp")
    tmp_md.write_text(merged_text, encoding="utf-8", newline="\n")
    os.replace(tmp_md, merged_md)
    return merged_md


def segment_artifact_dirs_from_jobs(jobs: list[dict[str, Any]]) -> list[Path]:
    """Extract ordered artifact directories from a list of segment jobs."""

    dirs: list[Path] = []
    for job in jobs:
        artifact_dir = job.get("artifact_dir") or job.get("extract_dir")
        if not artifact_dir:
            raise ValueError(f"extract job {job.get('job_id')} has no artifact_dir")
        dirs.append(Path(artifact_dir))
    return dirs
