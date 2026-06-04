#!/usr/bin/env python3
"""Rename MinerU Markdown images to short, ordered names.

The script reads image references from a Markdown file in document order,
renames the files under the sibling images directory to img001.ext,
img002.ext, ..., and rewrites the Markdown references to match.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import uuid
from pathlib import Path, PurePosixPath


MARKDOWN_IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()([^)]+)(\))")
HTML_IMAGE_RE = re.compile(r"(<img\b[^>]*?\bsrc=[\"'])([^\"']+)([\"'])", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename MinerU image files according to their order in full.md."
    )
    parser.add_argument(
        "markdown",
        nargs="?",
        default="md/__mineru_20260601_081348/full.md",
        help="Markdown file to update. Default: md/__mineru_20260601_081348/full.md",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Image directory. Default: <markdown parent>/images",
    )
    parser.add_argument(
        "--prefix",
        default="img",
        help="New image filename prefix. Default: img",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Minimum numeric width. Default: 3, producing img001.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without renaming files or rewriting Markdown.",
    )
    return parser.parse_args()


def split_markdown_target(target: str) -> tuple[str, str, str]:
    """Return leading whitespace/wrapper, path, and trailing title/whitespace."""
    leading_len = len(target) - len(target.lstrip())
    leading = target[:leading_len]
    rest = target[leading_len:]

    if rest.startswith("<"):
        close_index = rest.find(">")
        if close_index != -1:
            return leading + "<", rest[1:close_index], rest[close_index:]

    match = re.match(r"([^ \t\r\n]+)(.*)", rest, re.DOTALL)
    if not match:
        return leading, rest, ""
    return leading, match.group(1), match.group(2)


def image_name_from_reference(path_text: str, images_dir_name: str) -> str | None:
    posix_path = path_text.replace("\\", "/")
    path = PurePosixPath(posix_path)
    if len(path.parts) < 2 or path.parts[0] != images_dir_name:
        return None
    return path.name


def collect_references(markdown_text: str, images_dir_name: str) -> list[str]:
    references: list[tuple[int, str]] = []

    for match in MARKDOWN_IMAGE_RE.finditer(markdown_text):
        _, path_text, _ = split_markdown_target(match.group(2))
        image_name = image_name_from_reference(path_text, images_dir_name)
        if image_name:
            references.append((match.start(), image_name))

    for match in HTML_IMAGE_RE.finditer(markdown_text):
        image_name = image_name_from_reference(match.group(2), images_dir_name)
        if image_name:
            references.append((match.start(), image_name))

    seen: set[str] = set()
    ordered_names: list[str] = []
    for _, image_name in sorted(references, key=lambda item: item[0]):
        if image_name not in seen:
            seen.add(image_name)
            ordered_names.append(image_name)
    return ordered_names


def append_unreferenced_images(ordered_names: list[str], images_dir: Path) -> list[str]:
    seen = set(ordered_names)
    all_image_names = sorted(path.name for path in images_dir.iterdir() if path.is_file())
    return ordered_names + [name for name in all_image_names if name not in seen]


def build_mapping(ordered_names: list[str], prefix: str, digits: int) -> dict[str, str]:
    width = max(digits, len(str(len(ordered_names))))
    mapping: dict[str, str] = {}
    for index, old_name in enumerate(ordered_names, start=1):
        suffix = Path(old_name).suffix.lower()
        mapping[old_name] = f"{prefix}{index:0{width}d}{suffix}"
    return mapping


def replace_markdown_references(
    markdown_text: str, mapping: dict[str, str], images_dir_name: str
) -> str:
    def replace_markdown_match(match: re.Match[str]) -> str:
        before, target, after = match.groups()
        leading, path_text, trailing = split_markdown_target(target)
        old_name = image_name_from_reference(path_text, images_dir_name)
        if not old_name or old_name not in mapping:
            return match.group(0)
        new_path = str(PurePosixPath(images_dir_name) / mapping[old_name])
        return f"{before}{leading}{new_path}{trailing}{after}"

    def replace_html_match(match: re.Match[str]) -> str:
        before, path_text, after = match.groups()
        old_name = image_name_from_reference(path_text, images_dir_name)
        if not old_name or old_name not in mapping:
            return match.group(0)
        new_path = str(PurePosixPath(images_dir_name) / mapping[old_name])
        return f"{before}{new_path}{after}"

    markdown_text = MARKDOWN_IMAGE_RE.sub(replace_markdown_match, markdown_text)
    return HTML_IMAGE_RE.sub(replace_html_match, markdown_text)


def validate_files(images_dir: Path, mapping: dict[str, str]) -> None:
    missing = [old for old in mapping if not (images_dir / old).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        extra = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        raise FileNotFoundError(f"Referenced image files are missing: {preview}{extra}")

    source_names = set(mapping)
    target_names = set(mapping.values())
    conflicts = [
        name
        for name in target_names
        if name not in source_names and (images_dir / name).exists()
    ]
    if conflicts:
        preview = ", ".join(sorted(conflicts)[:5])
        extra = "" if len(conflicts) <= 5 else f", ... ({len(conflicts)} total)"
        raise FileExistsError(f"Target image names already exist: {preview}{extra}")


def rename_files(images_dir: Path, mapping: dict[str, str]) -> int:
    temp_mapping: dict[Path, Path] = {}
    token = uuid.uuid4().hex

    for index, old_name in enumerate(mapping, start=1):
        old_path = images_dir / old_name
        # OneDrive placeholder files can reject a rename until they are hydrated.
        with old_path.open("rb") as image_file:
            image_file.read()
        temp_path = images_dir / f"__renaming_{token}_{index}{old_path.suffix}"
        os.replace(old_path, temp_path)
        temp_mapping[temp_path] = images_dir / mapping[old_name]

    renamed = 0
    try:
        for temp_path, final_path in temp_mapping.items():
            os.replace(temp_path, final_path)
            renamed += 1
    except Exception:
        for temp_path, final_path in temp_mapping.items():
            if temp_path.exists() and not final_path.exists():
                original_name = next(
                    old for old, new in mapping.items() if images_dir / new == final_path
                )
                os.replace(temp_path, images_dir / original_name)
        raise

    return renamed


def main() -> int:
    args = parse_args()
    markdown_path = Path(args.markdown).resolve()
    images_dir = (
        Path(args.images_dir).resolve()
        if args.images_dir
        else markdown_path.parent / "images"
    )

    if not markdown_path.is_file():
        print(f"Markdown file not found: {markdown_path}", file=sys.stderr)
        return 1
    if not images_dir.is_dir():
        print(f"Images directory not found: {images_dir}", file=sys.stderr)
        return 1

    images_dir_name = images_dir.name
    markdown_text = markdown_path.read_text(encoding="utf-8")
    referenced_names = collect_references(markdown_text, images_dir_name)
    ordered_names = append_unreferenced_images(referenced_names, images_dir)
    mapping = build_mapping(ordered_names, args.prefix, args.digits)

    if not mapping:
        print("No local image references found.")
        return 0

    unchanged = [old for old, new in mapping.items() if old == new]
    changed_mapping = {old: new for old, new in mapping.items() if old != new}

    validate_files(images_dir, mapping)

    if args.dry_run:
        print(f"Markdown: {markdown_path}")
        print(f"Images:   {images_dir}")
        print(f"Found {len(referenced_names)} unique image references.")
        print(f"Will rename {len(mapping)} image files.")
        for old, new in list(mapping.items())[:10]:
            print(f"{old} -> {new}")
        if len(mapping) > 10:
            print(f"... {len(mapping) - 10} more")
        return 0

    updated_text = replace_markdown_references(markdown_text, mapping, images_dir_name)
    renamed = rename_files(images_dir, changed_mapping)
    markdown_path.write_text(updated_text, encoding="utf-8", newline="")

    print(f"Updated Markdown: {markdown_path}")
    print(f"Renamed images:   {renamed}")
    print(f"Referenced images:{len(referenced_names):4d}")
    if unchanged:
        print(f"Already named:    {len(unchanged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
