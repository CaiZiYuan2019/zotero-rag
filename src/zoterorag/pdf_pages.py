from __future__ import annotations

from pathlib import Path


def get_pdf_page_count(path: str | Path) -> int | None:
    """Return the page count of a PDF, using PyMuPDF if available.

    PyMuPDF is an optional runtime dependency. It is imported lazily so that
    environments without it can still run the rest of the pipeline. If the
    import fails, the file cannot be read, or the page count cannot be
    determined, the function returns ``None`` and callers should fall back to
    their default behavior (usually: do not split the PDF).
    """

    file_path = Path(path)
    if not file_path.is_file():
        return None
    try:
        import fitz  # type: ignore[import-not-found]

        with fitz.open(str(file_path)) as doc:
            return doc.page_count
    except Exception:
        return None
