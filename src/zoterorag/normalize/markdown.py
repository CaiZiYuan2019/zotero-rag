from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any


MARKDOWN_IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()([^)]+)(\))")
HTML_IMAGE_RE = re.compile(r"(<img\b[^>]*?\bsrc=[\"'])([^\"']+)([\"'])", re.IGNORECASE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class ImagePolicy:
    ordered_prefix: str = "img"
    ordered_digits: int = 3
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
    images_dir.mkdir(parents=True, exist_ok=True)

    refs = collect_image_references(markdown_text, source_path.parent)
    mapping = build_ordered_image_mapping(refs, images_dir, policy)
    normalized_text = rewrite_image_references(markdown_text, mapping)
    images = copy_images(mapping, images_dir, policy)
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
        refs.append(build_image_reference(match.start(), alt_text, target, markdown_parent))
    for match in HTML_IMAGE_RE.finditer(markdown_text):
        refs.append(build_image_reference(match.start(), "", match.group(2), markdown_parent))
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


def build_image_reference(position: int, alt_text: str, target: str, markdown_parent: Path) -> dict[str, Any]:
    source_path = resolve_local_image(target, markdown_parent)
    return {
        "position": position,
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


def copy_images(mapping: dict[Path, dict[str, Any]], images_dir: Path, policy: ImagePolicy) -> list[dict[str, Any]]:
    images = []
    for source, item in sorted(mapping.items(), key=lambda pair: pair[1]["image_index"]):
        if not source.is_file():
            raise FileNotFoundError(f"referenced image file not found: {source}")
        target = images_dir / item["ordered_name"]
        shutil.copy2(source, target)
        stat = target.stat()
        images.append(
            {
                "image_index": item["image_index"],
                "original_name": item["original_name"],
                "original_path": str(source),
                "ordered_name": item["ordered_name"],
                "ordered_relative_path": item["ordered_relative_path"],
                "sha256": sha256_file(target),
                "size": stat.st_size,
                "embedding_policy": {
                    "max_long_edge": policy.embedding_max_long_edge,
                    "max_bytes": policy.embedding_max_bytes,
                    "status": "not_resized_yet",
                },
                "alt_text": item["alt_text"],
                "position": item["position"],
            }
        )
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
                    "image_run_id": f"{document_id}:run:{image['image_index']:05d}",
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
