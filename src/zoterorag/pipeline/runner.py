"""Cross-process runner mutual exclusion and stale-run detection (M10).

Both pipeline entry points (``start_ingest_job`` / ``start_reembed_job``),
whether reached via CLI or the API server, acquire the same runner lock
before creating and executing a job. Holding the lock for the entire run
gives an exact liveness signal: if the lock is free, any job still marked
``created``/``running`` in the ledger belongs to a dead process and is swept
to ``interrupted`` automatically; if the lock is held, a live runner exists
and the new run is refused.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from filelock import FileLock, Timeout

from ..db import JobEvent, StateLedger


RUNNER_LOCK_NAME = "ingest_runner.lock"
RUNNER_KINDS = ("ingest", "reembed")
RUNNER_ACTIVE_STATUSES = ("created", "running")


def runner_lock_path(ledger: StateLedger) -> Path:
    """Lock file colocated with the state DB, so every data directory (and
    every test tmpdir) gets its own runner lock."""
    return ledger.db_path.parent / RUNNER_LOCK_NAME


@contextmanager
def acquire_runner_lock(
    ledger: StateLedger,
    *,
    kinds: tuple[str, ...] = RUNNER_KINDS,
) -> Iterator[FileLock]:
    """Acquire the shared runner lock non-blockingly and sweep stale runs.

    Raises RuntimeError when another live runner holds the lock. Once held,
    any ``created``/``running`` job of the given kinds is necessarily a
    dead-process leftover (every executing runner holds this lock for its
    whole lifetime) and is marked ``interrupted`` with an explanatory event.
    """
    lock = FileLock(str(runner_lock_path(ledger)))
    try:
        lock.acquire(blocking=False)
    except Timeout as exc:
        raise RuntimeError(
            "another ingest/reembed runner process is active "
            f"(lock: {runner_lock_path(ledger)}). Wait for it to finish."
        ) from exc
    try:
        _interrupt_stale_active_jobs(ledger, kinds=kinds)
        yield lock
    finally:
        lock.release()


def _interrupt_stale_active_jobs(
    ledger: StateLedger, *, kinds: tuple[str, ...]
) -> list[dict[str, Any]]:
    swept: list[dict[str, Any]] = []
    while True:
        # Loop because sweeping one job updates its updated_at ordering; the
        # query always returns the newest remaining active job.
        active = ledger.find_active_job(kinds=kinds, statuses=RUNNER_ACTIVE_STATUSES)
        if active is None:
            return swept
        ledger.set_job_status(active["job_id"], "interrupted")
        ledger.add_event(
            JobEvent(
                job_id=active["job_id"],
                stage="lifecycle",
                status="interrupted",
                message=(
                    "marked interrupted by runner lock: the lock was free, so "
                    "the process running this job is dead"
                ),
                payload={"previous_status": active["status"]},
            )
        )
        swept.append(active)
