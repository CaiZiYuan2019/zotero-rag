# Environment Requirements

This file lists packages that should be installed in the project conda
environment before running the non-stub ZoteroRAG workflows.

Target environment:

```powershell
conda activate E:\ZoteroRAG\.conda
python -m pip install --upgrade pip
```

## Required For Current Core Control Plane

These are enough for the SQLite state ledger, Zotero shadow copy, local vector
store fallback, normalize/chunk logic, CLI, and unit tests that do not use
optional API server clients.

```powershell
python -m pip install -e .
```

The current core code is intentionally mostly standard-library based. It uses
local SQLite for state and local vector tests, so no external vector service is
required for the current baseline.

## Required For MinerU And Qwen API Calls

Install before any real MinerU conversion or qwen embedding request:

```powershell
python -m pip install requests>=2.32
```

Used by:

- `zoterorag.extractors.mineru.MinerUProvider`
- `zoterorag.embeddings.qwen.Qwen3VLEmbeddingProvider`
- legacy `reference/mineru_cli.py`

Do not run real conversion/embedding in routine tests. Use fake clients or dry
runs unless intentionally doing a small manual integration check.

## Required For FastAPI Server And API Tests

Install before running `zoterorag serve` or API route tests:

```powershell
python -m pip install fastapi>=0.115 uvicorn>=0.30 httpx>=0.27
```

Notes:

- `fastapi` is needed by `src/zoterorag/api/app.py`.
- `uvicorn` is needed by `src/zoterorag/api/server.py`.
- `httpx` is needed by FastAPI/Starlette `TestClient`; without it, API route
  tests that use `fastapi.testclient` may be skipped or fail depending on the
  installed FastAPI/Starlette version.

## Required For MinerU PDF Page Counting

Install before validating page counts for real PDFs or before reusing the
legacy MinerU CLI:

```powershell
python -m pip install PyMuPDF>=1.24
```

Used by:

- `reference/mineru_cli.py` as `fitz`
- future real MinerU job preparation when selected page counts should be read
  directly from PDF files instead of supplied by the caller.

## Required For Image Processing

Install before implementing or running production image downscaling, thumbnail
generation, or image format normalization for embedding inputs:

```powershell
python -m pip install Pillow>=10.4
```

Current normalize code can read basic PNG dimensions without Pillow, but the
project plan requires embedding-image derivatives and size/resolution policies.
Those should use Pillow rather than ad hoc byte manipulation.

## Optional Vector Backend

The current implementation uses `sqlite-local` vector storage. The project plan
mentions LanceDB as the preferred future local vector backend. Install only
when the LanceDB adapter is implemented or being tested:

```powershell
python -m pip install lancedb>=0.18 pyarrow>=16
```

Do not install this just for the current SQLite vector tests.

## Optional Developer Tools

Useful during development, not required by the application runtime:

```powershell
python -m pip install pytest>=8 ruff>=0.6
```

The existing test suite is currently run with standard `unittest`, so `pytest`
is optional unless a future test workflow switches to it.

## Legacy Reference Script Dependencies

Only needed if running old scripts in `reference/` directly:

```powershell
python -m pip install customtkinter requests lancedb langchain-text-splitters PyMuPDF
```

These are not required for the new `src/zoterorag` implementation unless their
functionality is explicitly migrated.

## Suggested One-Shot Install For Next Integration Phase

For the next phase, where we will run a few MinerU conversions and then small
qwen embedding checks, this is the practical minimum:

```powershell
python -m pip install -e .
python -m pip install requests>=2.32 fastapi>=0.115 uvicorn>=0.30 httpx>=0.27 PyMuPDF>=1.24 Pillow>=10.4
```

Keep `.env` local and untracked. It should contain only secrets and endpoint
overrides such as:

```text
MINERU_KEY=...
MINERU_URL=...
BAILIAN_KEY=...
BAILIAN_URL=...
```

