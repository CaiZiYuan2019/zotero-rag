from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
from typing import Any


MARKDOWN_IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()([^)]+)(\))")
HTML_IMAGE_RE = re.compile(r"(<img\b[^>]*?\bsrc=[\"'])([^\"']+)([\"'])", re.IGNORECASE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class ImagePolicy:
    ordered_prefix: str = "img"
    ordered_digits: int = 3
    embedding_dir_name: str = "embedding_images"
    embedding_max_long_edge: int = 1600
    embedding_max_bytes: int = 5 * 1024 * 1024


@dataclass(frozen=True)
class NormalizeResult:
    document_id: str
    artifact_dir: Path
    document_md: Path
    image_manifest: Path
    chunks_path: Path
    manifest_path: Path
    chunks: list[dict[str, Any]]
    images: list[dict[str, Any]]
    manifest: dict[str, Any]

    def ledger_artifact(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "attachment_key": self.manifest.get("attachment_key"),
            "extract_job_id": self.manifest.get("extract_job_id"),
            "artifact_dir": self.artifact_dir,
            "document_md": self.document_md,
            "image_manifest": self.image_manifest,
            "chunks_path": self.chunks_path,
            "manifest_path": self.manifest_path,
            "source_markdown": self.manifest["source_markdown"],
            "status": self.manifest["status"],
            "document_hash": self.manifest["document_hash"],
            "chunk_count": len(self.chunks),
            "image_count": len(self.images),
            "payload": self.manifest,
        }


def normalize_markdown_document(
    *,
    source_markdown: str | Path,
    output_root: str | Path,
    document_id: str,
    attachment_key: str | None = None,
    extract_job_id: str | None = None,
    image_policy: ImagePolicy | None = None,
) -> NormalizeResult:
    """Normalize a MinerU-like Markdown folder into durable local artifacts.

    This function is intentionally offline. It never calls MinerU, embedding
    providers, or Zotero. It only copies local files, rewrites image references,
    and emits manifests that later pipeline stages can resume from.
    """

    policy = image_policy or ImagePolicy()
    source_path = Path(source_markdown)
    markdown_text = source_path.read_text(encoding="utf-8")
    artifact_dir = Path(output_root) / sanitize_id(document_id)
    images_dir = artifact_dir / "images"
    embedding_images_dir = artifact_dir / policy.embedding_dir_name
    images_dir.mkdir(parents=True, exist_ok=True)
    embedding_images_dir.mkdir(parents=True, exist_ok=True)

    refs = collect_image_references(markdown_text, source_path.parent)
    mapping = build_ordered_image_mapping(refs, images_dir, policy)
    normalized_text = rewrite_image_references(markdown_text, mapping)
    images = copy_images(mapping, images_dir, embedding_images_dir, policy)
    images = assign_image_runs(images, markdown_text)
    chunks = build_chunks(document_id=document_id, markdown_text=normalized_text, images=images)

    document_md = artifact_dir / "document.md"
    image_manifest = images_dir / "original_manifest.json"
    chunks_path = artifact_dir / "chunks.jsonl"
    manifest_path = artifact_dir / "document_manifest.json"
    document_md.write_text(normalized_text, encoding="utf-8", newline="\n")
    image_manifest.write_text(json.dumps(images, ensure_ascii=False, indent=2), encoding="utf-8")
    chunks_path.write_text(
        "".join(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    manifest = {
        "document_id": document_id,
        "attachment_key": attachment_key,
        "extract_job_id": extract_job_id,
        "source_markdown": str(source_path),
        "status": "normalized",
        "document_hash": sha256_text(normalized_text),
        "document_md": str(document_md),
        "image_manifest": str(image_manifest),
        "chunks_path": str(chunks_path),
        "image_count": len(images),
        "chunk_count": len(chunks),
        "image_policy": policy.__dict__,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return NormalizeResult(
        document_id=document_id,
        artifact_dir=artifact_dir,
        document_md=document_md,
        image_manifest=image_manifest,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        chunks=chunks,
        images=images,
        manifest=manifest,
    )


def collect_image_references(markdown_text: str, markdown_parent: Path) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for match in MARKDOWN_IMAGE_RE.finditer(markdown_text):
        alt_text = match.group(1).removeprefix("![").removesuffix("](")
        target = split_markdown_target(match.group(2))[1]
        refs.append(build_image_reference(match.start(), match.end(), alt_text, target, markdown_parent))
    for match in HTML_IMAGE_RE.finditer(markdown_text):
        refs.append(build_image_reference(match.start(), match.end(), "", match.group(2), markdown_parent))
    refs.sort(key=lambda item: item["position"])
    seen: set[Path] = set()
    unique = []
    for ref in refs:
        source_path = ref["source_path"]
        if source_path in seen:
            continue
        seen.add(source_path)
        unique.append(ref)
    return unique


def build_image_reference(
    position: int,
    end_position: int,
    alt_text: str,
    target: str,
    markdown_parent: Path,
) -> dict[str, Any]:
    source_path = resolve_local_image(target, markdown_parent)
    return {
        "position": position,
        "end_position": end_position,
        "alt_text": alt_text,
        "target": target,
        "source_path": source_path,
        "original_name": source_path.name,
    }


def resolve_local_image(target: str, markdown_parent: Path) -> Path:
    path_text = target.strip()
    if path_text.startswith("<") and path_text.endswith(">"):
        path_text = path_text[1:-1]
    path = Path(path_text.replace("/", "\\"))
    if not path.is_absolute():
        path = markdown_parent / path
    return path.resolve()


def build_ordered_image_mapping(
    refs: list[dict[str, Any]],
    images_dir: Path,
    policy: ImagePolicy,
) -> dict[Path, dict[str, Any]]:
    width = max(policy.ordered_digits, len(str(len(refs))))
    mapping: dict[Path, dict[str, Any]] = {}
    for index, ref in enumerate(refs, start=1):
        source = ref["source_path"]
        suffix = source.suffix.lower() or ".img"
        ordered_name = f"{policy.ordered_prefix}{index:0{width}d}{suffix}"
        mapping[source] = {
            **ref,
            "image_index": index,
            "ordered_name": ordered_name,
            "ordered_relative_path": str(PurePosixPath(images_dir.name) / ordered_name),
        }
    return mapping


def rewrite_image_references(markdown_text: str, mapping: dict[Path, dict[str, Any]]) -> str:
    by_target = {item["target"]: item["ordered_relative_path"] for item in mapping.values()}

    def replace_markdown(match: re.Match[str]) -> str:
        leading, target_path, trailing = split_markdown_target(match.group(2))
        replacement = by_target.get(target_path)
        if replacement is None:
            return match.group(0)
        return f"{match.group(1)}{leading}{replacement}{trailing}{match.group(3)}"

    def replace_html(match: re.Match[str]) -> str:
        replacement = by_target.get(match.group(2))
        if replacement is None:
            return match.group(0)
        return f"{match.group(1)}{replacement}{match.group(3)}"

    markdown_text = MARKDOWN_IMAGE_RE.sub(replace_markdown, markdown_text)
    return HTML_IMAGE_RE.sub(replace_html, markdown_text)


def copy_images(
    mapping: dict[Path, dict[str, Any]],
    images_dir: Path,
    embedding_images_dir: Path,
    policy: ImagePolicy,
) -> list[dict[str, Any]]:
    images = []
    for source, item in sorted(mapping.items(), key=lambda pair: pair[1]["image_index"]):
        if not source.is_file():
            raise FileNotFoundError(f"referenced image file not found: {source}")
        target = images_dir / item["ordered_name"]
        shutil.copy2(source, target)
        stat = target.stat()
        dimensions = read_image_dimensions(target)
        quality_flags = image_quality_flags(dimensions, stat.st_size, policy)
        embedding_image = prepare_embedding_image(
            source=target,
            embedding_images_dir=embedding_images_dir,
            ordered_name=item["ordered_name"],
            quality_flags=quality_flags,
            policy=policy,
        )
        images.append(
            {
                "image_index": item["image_index"],
                "original_name": item["original_name"],
                "original_path": str(source),
                "ordered_name": item["ordered_name"],
                "ordered_relative_path": item["ordered_relative_path"],
                "sha256": sha256_file(target),
                "size": stat.st_size,
                "width": dimensions.get("width"),
                "height": dimensions.get("height"),
                "format": dimensions.get("format"),
                "pixel_count": dimensions.get("pixel_count"),
                "image_quality_flags": quality_flags,
                "embedding_policy": {
                    "max_long_edge": policy.embedding_max_long_edge,
                    "max_bytes": policy.embedding_max_bytes,
                    **embedding_image,
                },
                "alt_text": item["alt_text"],
                "position": item["position"],
                "end_position": item["end_position"],
            }
        )
    return images


def prepare_embedding_image(
    *,
    source: Path,
    embedding_images_dir: Path,
    ordered_name: str,
    quality_flags: list[str],
    policy: ImagePolicy,
) -> dict[str, Any]:
    """Create or defer the image derivative used for multimodal embedding.

    Images already within the provider limits are copied to a separate
    `embedding_images/` tree so future resizing/compression can change only the
    embedding derivative without mutating the original evidence image.
    Oversized images are explicitly marked pending instead of silently falling
    back to the original, which could exceed qwen request limits.
    """

    if "exceeds_embedding_limits" in quality_flags:
        return {
            "status": "pending_resize",
            "embedding_relative_path": None,
            "reason": "image exceeds embedding limits; install/enable Pillow resizing before API use",
        }

    embedding_target = embedding_images_dir / ordered_name
    shutil.copy2(source, embedding_target)
    stat = embedding_target.stat()
    return {
        "status": "ready_embedding_copy",
        "embedding_relative_path": str(PurePosixPath(embedding_images_dir.name) / ordered_name),
        "embedding_sha256": sha256_file(embedding_target),
        "embedding_size": stat.st_size,
    }


def assign_image_runs(images: list[dict[str, Any]], markdown_text: str) -> list[dict[str, Any]]:
    """Group consecutive image references into stable image runs.

    MinerU can split one multi-panel figure into adjacent image files. We keep
    per-image vectors because the current qwen3-vl cloud embedding API accepts a
    single image per request, but shared run ids let retrieval merge or present
    consecutive hits together.
    """

    sorted_images = sorted(images, key=lambda item: item["position"])
    current_run = 0
    previous: dict[str, Any] | None = None
    run_members: dict[str, list[dict[str, Any]]] = {}
    for image in sorted_images:
        if previous is None or markdown_text[previous["end_position"] : image["position"]].strip():
            current_run += 1
        run_id = f"run:{current_run:05d}"
        image["image_run_id"] = run_id
        run_members.setdefault(run_id, []).append(image)
        previous = image

    for members in run_members.values():
        run_count = len(members)
        for index, image in enumerate(members, start=1):
            image["image_run_position"] = index
            image["image_run_count"] = run_count
    return images


def build_chunks(document_id: str, markdown_text: str, images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lines = markdown_text.splitlines()
    heading_path: list[str] = []
    current_text: list[str] = []
    text_chunks: list[dict[str, Any]] = []
    chunk_index = 0

    def flush_text() -> None:
        nonlocal chunk_index, current_text
        text = strip_image_paths_for_text_chunk("\n".join(line for line in current_text)).strip()
        current_text = []
        if not text:
            return
        chunk_index += 1
        text_chunks.append(
            {
                "chunk_id": f"{document_id}:text:{chunk_index:05d}",
                "document_id": document_id,
                "chunk_type": "text",
                "chunk_index": chunk_index,
                "text": text,
                "heading_path": list(heading_path),
                "metadata": {"token_estimate": estimate_tokens(text)},
            }
        )

    for line in lines:
        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush_text()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_path = heading_path[: level - 1] + [title]
            current_text.append(line)
            continue
        current_text.append(line)
        if estimate_tokens("\n".join(current_text)) >= 2200:
            flush_text()
    flush_text()

    image_chunks = []
    for image in images:
        image_chunks.append(
            {
                "chunk_id": f"{document_id}:image:{image['image_index']:05d}",
                "document_id": document_id,
                "chunk_type": "image",
                "chunk_index": len(text_chunks) + image["image_index"],
                "text": image.get("alt_text") or image["ordered_name"],
                "heading_path": [],
                "metadata": {
                    "image_index": image["image_index"],
                    "image_path": image["ordered_relative_path"],
                    "image_embedding_path": image.get("embedding_policy", {}).get("embedding_relative_path"),
                    "image_embedding_status": image.get("embedding_policy", {}).get("status"),
                    "image_run_id": f"{document_id}:{image['image_run_id']}",
                    "image_run_position": image["image_run_position"],
                    "image_run_count": image["image_run_count"],
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "pixel_count": image.get("pixel_count"),
                    "image_quality_flags": image.get("image_quality_flags", []),
                    "embedding_policy": image.get("embedding_policy", {}),
                    "image_return": "file_ref",
                },
            }
        )
    chunks = text_chunks + image_chunks
    for index, chunk in enumerate(chunks):
        chunk["prev_chunk_id"] = chunks[index - 1]["chunk_id"] if index > 0 else None
        chunk["next_chunk_id"] = chunks[index + 1]["chunk_id"] if index + 1 < len(chunks) else None
    return chunks


def split_markdown_target(target: str) -> tuple[str, str, str]:
    leading_len = len(target) - len(target.lstrip())
    leading = target[:leading_len]
    rest = target[leading_len:]
    match = re.match(r"([^ \t\r\n]+)(.*)", rest, re.DOTALL)
    if not match:
        return leading, rest, ""
    return leading, match.group(1), match.group(2)


def strip_image_paths_for_text_chunk(text: str) -> str:
    """Remove image file references from text chunks.

    `document.md` keeps renderable image links for manual use, but text chunks
    feed pure-text retrieval and LLM outputs. They may mention that an image
    exists or keep the caption, but must not carry local file paths.
    """

    def replace_markdown(match: re.Match[str]) -> str:
        alt_text = match.group(1).removeprefix("![").removesuffix("](").strip()
        return f"[Image: {alt_text}]" if alt_text else "[Image]"

    def replace_html(match: re.Match[str]) -> str:
        return "[Image]"

    text = MARKDOWN_IMAGE_RE.sub(replace_markdown, text)
    return HTML_IMAGE_RE.sub(replace_html, text)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def sanitize_id(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return cleaned[:128] if cleaned else "document"


def read_image_dimensions(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    header = file_path.read_bytes()[:64]
    if len(header) >= 24 and header.startswith(b"\x89PNG\r\n\x1a\n") and header[12:16] == b"IHDR":
        width, height = struct.unpack(">II", header[16:24])
        return image_dimension_result("png", width, height)
    if len(header) >= 10 and header.startswith((b"GIF87a", b"GIF89a")):
        width, height = struct.unpack("<HH", header[6:10])
        return image_dimension_result("gif", width, height)
    if header.startswith(b"\xff\xd8"):
        jpeg_dimensions = read_jpeg_dimensions(file_path)
        if jpeg_dimensions:
            return jpeg_dimensions
    return {"format": file_path.suffix.lower().lstrip(".") or "unknown"}


def read_jpeg_dimensions(path: Path) -> dict[str, Any] | None:
    data = path.read_bytes()
    offset = 2
    while offset + 9 < len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        offset += 2
        if marker in {0xD8, 0xD9}:
            continue
        if offset + 2 > len(data):
            return None
        segment_length = int.from_bytes(data[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(data):
            return None
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if segment_length < 7:
                return None
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            return image_dimension_result("jpeg", width, height)
        offset += segment_length
    return None


def image_dimension_result(image_format: str, width: int, height: int) -> dict[str, Any]:
    return {
        "format": image_format,
        "width": width,
        "height": height,
        "pixel_count": width * height,
    }


def image_quality_flags(dimensions: dict[str, Any], file_size: int, policy: ImagePolicy) -> list[str]:
    flags: list[str] = []
    width = dimensions.get("width")
    height = dimensions.get("height")
    if width is None or height is None:
        flags.append("unknown_dimensions")
    else:
        if min(int(width), int(height)) < 64:
            flags.append("tiny_image")
        if max(int(width), int(height)) > policy.embedding_max_long_edge:
            flags.append("exceeds_embedding_limits")
            flags.append("exceeds_embedding_long_edge")
    if file_size > policy.embedding_max_bytes:
        if "exceeds_embedding_limits" not in flags:
            flags.append("exceeds_embedding_limits")
        flags.append("exceeds_embedding_bytes")
    return flags


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
