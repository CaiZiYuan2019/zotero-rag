from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from typing import Any
import uuid

from ..db import StateLedger
from ..db.state import utc_now
from .base import ExtractorProvider, StubExtractorProvider
from .cache import (
    canonical_selected_pages,
    extractor_cache_key,
    recommended_mineru_timeout_seconds,
    sha256_file,
    stable_options_hash,
)
from .deepseek import DeepSeekIncompleteResult, DeepSeekOCRError
from .mineru import MinerUAPIError, PollTimeoutError
from .recovery import classify_extract_job
from .key_pool import ApiKeyRef, ExtractorKeyPool


REUSABLE_EXTRACT_STATES = {"submitted", "running", "completed", "downloaded"}

# M2: per-job poll budgets (pages*6+30, persisted as options.timeout_seconds)
# are clamped into this band. The floor keeps jobs persisted with a tiny
# budget (e.g. an unknown page count falling back to the 1-page estimate of
# 36s) from timing out prematurely; the ceiling is the task lifetime cap so a
# single poll round never outlives the job itself.
MIN_POLL_BUDGET_SECONDS = 1800.0

# Phase 2: transient HTTP statuses whose poll-time failure keeps the job
# running and retries with exponential backoff inside harvest(). Same set as
# mineru._TRANSIENT_HTTP_STATUSES minus 429, which is graded separately
# (rate-limit cooldown + stand-down for the round).
TRANSIENT_POLL_HTTP_STATUSES = {408, 500, 502, 503, 504, 520}


@dataclass(frozen=True)
class ExtractionRequest:
    input_file: Path
    attachment_key: str | None = None
    pdf_sha256: str | None = None
    selected_pages: str = ""
    selected_page_count: int = 1
    options: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExtractionResult:
    job: dict[str, Any]
    cache_hit: bool


@dataclass(frozen=True)
class ExtractionCacheContext:
    """Everything derived from an ExtractionRequest for cache-key purposes."""

    pdf_sha256: str
    selected_pages: str
    options: dict[str, Any]
    options_hash: str
    endpoint_url: str
    cache_key: str


def extraction_cache_context(
    request: ExtractionRequest,
    provider: ExtractorProvider,
    *,
    pdf_sha256: str | None = None,
) -> ExtractionCacheContext:
    """Derive the cache context that an ``ensure_extraction`` call would use.

    H3: shared with the ingest split path so a planned page segment can be
    mapped back to its ledger job with exactly the key that submission
    persisted — the two derivations cannot drift apart.
    """

    pdf_sha256 = pdf_sha256 or request.pdf_sha256 or sha256_file(Path(request.input_file))
    selected_pages = canonical_selected_pages(request.selected_pages)
    options = dict(request.options or {})
    if "timeout_seconds" not in options:
        options["timeout_seconds"] = (
            210 if provider.name == "deepseek" else recommended_mineru_timeout_seconds(request.selected_page_count)
        )
    options_hash = stable_options_hash(options)
    endpoint_url = getattr(provider, "apply_upload_url", "") or getattr(provider, "endpoint", "")
    cache_key = extractor_cache_key(
        pdf_sha256=pdf_sha256,
        selected_pages=selected_pages,
        extractor_name=provider.name,
        extractor_version=provider.version,
        options_hash=options_hash,
        endpoint_url=endpoint_url,
    )
    return ExtractionCacheContext(
        pdf_sha256=pdf_sha256,
        selected_pages=selected_pages,
        options=options,
        options_hash=options_hash,
        endpoint_url=endpoint_url,
        cache_key=cache_key,
    )


class ExtractionManager:
    """Coordinates extraction state, cache keys, and provider calls.

    Phase 2 (MODIFICATION_PLAN_20260904 §2.2): the monolithic
    submit→poll→download flow is split into three primitives —
    ``submit_only()`` (acquire key → submit → ledger → release immediately),
    ``poll_once()`` (acquire key → one poll → ledger → release), and
    ``harvest()`` (the poll loop built on poll_once, with pacing, graded
    error classification, and M2 stand-down semantics). A key is now held
    only for the duration of a single provider call, so M keys can service
    N in-flight jobs (N >> M) from concurrent harvest threads.
    ``ensure_extraction()`` / ``resume_extraction()`` remain as thin
    wrappers with the same outward semantics as before.
    """

    def __init__(
        self,
        *,
        ledger: StateLedger,
        cache_dir: str | Path,
        provider: ExtractorProvider | None = None,
        key_pool: ExtractorKeyPool | None = None,
        failure_cooldown_seconds: float = 60.0,
        poll_interval_seconds: float = 30.0,
        poll_timeout_seconds: float = 21600.0,
        key_acquire_timeout_seconds: float = 60.0,
        job_max_lifetime_seconds: float = 21600.0,
        submit_min_interval_seconds: float = 0.0,
        poll_max_per_minute: int = 0,
        max_resubmit_attempts: int = 3,
        rate_limit_cooldown_seconds: float = 180.0,
        auth_failure_cooldown_seconds: float = 3600.0,
        resume_manager_for: Callable[[str], ExtractionManager] | None = None,
    ) -> None:
        self.ledger = ledger
        self.resume_manager_for = resume_manager_for
        self.cache_dir = Path(cache_dir)
        self.provider = provider or StubExtractorProvider()
        self.key_pool = key_pool or ExtractorKeyPool()
        self.failure_cooldown_seconds = failure_cooldown_seconds
        self.poll_interval_seconds = poll_interval_seconds
        # poll_timeout_seconds no longer drives poll deadlines directly; it is
        # the fallback budget for jobs persisted without options.timeout_seconds
        # (see _poll_budget_seconds).
        self.poll_timeout_seconds = poll_timeout_seconds
        self.key_acquire_timeout_seconds = key_acquire_timeout_seconds
        # 决策1: total task lifetime measured from submitted_at. Jobs exceeding
        # it are flipped to failed_manual_review instead of being polled or
        # resubmitted forever.
        self.job_max_lifetime_seconds = job_max_lifetime_seconds
        # Phase 2 pacing and grading knobs. The rate limiters default to off
        # (0/unlimited) so directly constructed managers behave exactly as
        # before; production values come from [extract] config via cli/app
        # wiring. The two cooldowns below are code-level knobs only (§2.2's
        # error table says "configurable"; the §2.4 config table deliberately
        # does not list them).
        self.submit_min_interval_seconds = submit_min_interval_seconds
        self.poll_max_per_minute = poll_max_per_minute
        # 决策2: cap on automatic resubmissions per job, counted in
        # payload["resubmit_attempts"] across ingest rounds.
        self.max_resubmit_attempts = max_resubmit_attempts
        self.rate_limit_cooldown_seconds = rate_limit_cooldown_seconds
        self.auth_failure_cooldown_seconds = auth_failure_cooldown_seconds
        # Submit pacer state (global minimum interval between submits).
        self._submit_pace_lock = threading.Lock()
        self._last_submit_at: float | None = None
        # Poll sliding-window state (at most poll_max_per_minute per 60s).
        self._poll_window_lock = threading.Lock()
        self._poll_calls: deque[float] = deque()

    def _acquire_key_with_wait(
        self,
        timeout_seconds: float | None = None,
        *,
        prefer_alias: str | None = None,
    ) -> ApiKeyRef | None:
        """Acquire an extractor key, waiting until one becomes available.

        Returns ``None`` immediately when the pool is empty (stub/no-key mode).
        Raises no exception here; callers decide whether a key is mandatory.
        ``prefer_alias`` (Phase 2 key affinity) tries the job's submitting key
        first, falling back to any available key — see
        ExtractorKeyPool.acquire_key for the single-account assumption.
        """
        if not self.key_pool.has_keys():
            return None
        timeout = timeout_seconds if timeout_seconds is not None else self.key_acquire_timeout_seconds
        deadline = time.monotonic() + timeout
        while True:
            api_key = self.key_pool.acquire_key(prefer_alias=prefer_alias)
            if api_key is not None:
                return api_key
            wait = min(2.0, deadline - time.monotonic())
            if wait <= 0:
                return None
            time.sleep(wait)

    def _pace_submit(self) -> None:
        """Enforce the global minimum interval between provider submits.

        The sleep happens inside the lock on purpose: pacing is a mutual-
        exclusion property, so concurrent submitters queue up here and leave
        exactly ``submit_min_interval_seconds`` apart. 0 disables pacing.
        """
        interval = float(self.submit_min_interval_seconds)
        if interval <= 0:
            return
        with self._submit_pace_lock:
            now = time.monotonic()
            if self._last_submit_at is not None:
                wait = self._last_submit_at + interval - now
                if wait > 0:
                    time.sleep(wait)
                    now = time.monotonic()
            self._last_submit_at = now

    def _poll_rate_wait(self) -> None:
        """Sliding-window cap on provider polls (poll_max_per_minute per 60s).

        0 disables the limiter. With the default poll_interval_seconds=30 the
        configured 600/min is far above anything harvest loops can reach; this
        is a protective ceiling, not a scheduler.
        """
        limit = int(self.poll_max_per_minute)
        if limit <= 0:
            return
        with self._poll_window_lock:
            while True:
                now = time.monotonic()
                while self._poll_calls and self._poll_calls[0] <= now - 60.0:
                    self._poll_calls.popleft()
                if len(self._poll_calls) < limit:
                    self._poll_calls.append(now)
                    return
                time.sleep(min(self._poll_calls[0] + 60.0 - now, 1.0))

    def _cooldown_seconds_for(self, exc: BaseException) -> float:
        """Graded key cooldown for a failed provider call (§2.2 error table).

        429 → rate_limit_cooldown_seconds (minutes); 401/403 →
        auth_failure_cooldown_seconds (long); transient 5xx/408 and
        network-level errors → 0 (the key is not at fault); everything else →
        the historical failure_cooldown_seconds.
        """
        # A provider-raised poll timeout is a local watchdog event, never the
        # key's fault — never cool the key for it (must precede the
        # MinerUAPIError branch: PollTimeoutError is a subclass).
        if isinstance(exc, PollTimeoutError):
            return 0.0
        if isinstance(exc, (MinerUAPIError, DeepSeekOCRError)):
            if exc.status_code == 429:
                return float(self.rate_limit_cooldown_seconds)
            if exc.status_code in (401, 403):
                return float(self.auth_failure_cooldown_seconds)
            if exc.status_code in TRANSIENT_POLL_HTTP_STATUSES:
                return 0.0
        if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
            return 0.0
        return float(self.failure_cooldown_seconds)

    def _all_keys_cooling(self) -> bool:
        """Whether every pooled key is currently in cooldown (pool disabled)."""
        if not self.key_pool.has_keys():
            return False
        return all(
            key["cooldown_remaining_seconds"] > 0
            for key in self.key_pool.list_public_keys()
        )

    def _poll_budget_seconds(self, options: Mapping[str, Any]) -> float:
        """Effective poll budget for one round of polling.

        M2: the per-job budget persisted as ``options["timeout_seconds"]``
        (``pages*6+30`` at submission time) is now honored, clamped into
        [MIN_POLL_BUDGET_SECONDS, job_max_lifetime_seconds]. Legacy jobs
        persisted without a budget fall back to ``self.poll_timeout_seconds``.
        """
        raw = options.get("timeout_seconds")
        try:
            per_job = float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            per_job = 0.0
        # `not (x > 0)` also rejects NaN, which would otherwise slip past the
        # clamp and turn poll_deadline into NaN — an infinite poll loop.
        if not (per_job > 0):
            per_job = float(self.poll_timeout_seconds)
        return min(max(per_job, MIN_POLL_BUDGET_SECONDS), float(self.job_max_lifetime_seconds))

    def _lifetime_exceeded(self, job: dict[str, Any]) -> bool:
        """Whether the job has outlived job_max_lifetime_seconds since submitted_at.

        Rows without a parseable submitted_at (legacy/hand-built) cannot be
        aged and are treated as not exceeded.
        """
        submitted_at = job.get("submitted_at")
        if not submitted_at:
            return False
        try:
            started = datetime.fromisoformat(str(submitted_at))
        except ValueError:
            return False
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - started).total_seconds()
        return age_seconds > float(self.job_max_lifetime_seconds)

    def _mark_job_max_lifetime(self, job: dict[str, Any]) -> None:
        """Flip a job past its lifetime cap to failed_manual_review (决策1)."""
        self.ledger.set_extract_job_state(
            job["job_id"],
            state="failed_manual_review",
            local_stage=job.get("local_stage") or "poll",
            error_code="job-max-lifetime",
            error_message=(
                f"exceeded job_max_lifetime_seconds={float(self.job_max_lifetime_seconds):.0f} "
                f"since submitted_at={job.get('submitted_at')}; manual review required"
            ),
        )

    @staticmethod
    def _job_key_alias(job: dict[str, Any]) -> str | None:
        """Alias of the key that submitted the job's current remote batch.

        payload["key_alias"] tracks the latest (re)submission; the
        api_key_alias column stays the original one from upsert time.
        """
        payload = job.get("payload") or {}
        return payload.get("key_alias") or job.get("api_key_alias")

    @staticmethod
    def _resubmit_attempts(payload: Mapping[str, Any] | None) -> int:
        """Parse payload["resubmit_attempts"] defensively (junk counts as 0)."""
        try:
            return int((payload or {}).get("resubmit_attempts", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _mark_job_max_resubmits(self, job: dict[str, Any]) -> None:
        """Flip a job that exhausted its resubmit budget to failed_manual_review (决策2)."""
        self.ledger.set_extract_job_state(
            job["job_id"],
            state="failed_manual_review",
            local_stage=job.get("local_stage") or "error",
            error_code="max-resubmit-attempts",
            error_message=(
                f"exceeded max_resubmit_attempts={self.max_resubmit_attempts}; "
                "manual review required"
            ),
        )

    def submit_only(self, request: ExtractionRequest) -> ExtractionResult:
        """Phase 2 primitive: cache check → upsert → submit → release the key.

        The key is held only for the provider.submit() call (itself a couple
        of HTTP requests) and released in ``finally``. Polling/downloading is
        harvest()'s job. A submit-stage failure keeps the historical catch-all
        semantics (failed_retryable + graded key cooldown + re-raise): no
        remote batch id survives a failed submit, so resubmission next round
        is the correct recovery.
        """
        input_file = Path(request.input_file)
        # Single source of truth for the cache-key derivation lives in
        # extraction_cache_context(); the ingest split path uses the same
        # helper to map segments back to their persisted jobs (H3).
        ctx = extraction_cache_context(request, self.provider)
        pdf_sha256 = ctx.pdf_sha256
        selected_pages = ctx.selected_pages
        options = ctx.options
        provider_options = dict(options)
        provider_options.setdefault("page_ranges", selected_pages)
        options_hash = ctx.options_hash
        cache_key = ctx.cache_key

        existing = self.ledger.get_extract_job_by_cache_key(cache_key)
        if existing is not None and existing["state"] in REUSABLE_EXTRACT_STATES:
            return ExtractionResult(job=existing, cache_hit=True)
        if existing is not None and existing["state"] == "failed_manual_review":
            raise RuntimeError(f"extract job {existing['job_id']} requires manual review: {existing.get('error_message')}")
        if existing is not None and self._lifetime_exceeded(existing):
            # 决策1: a non-reusable job past its lifetime cap must not be
            # auto-resubmitted; upsert_extract_job COALESCEs submitted_at, so
            # the clock honestly measures from the first submission.
            self._mark_job_max_lifetime(existing)
            raise RuntimeError(
                f"extract job {existing['job_id']} exceeded "
                f"job_max_lifetime_seconds={float(self.job_max_lifetime_seconds):.0f}; "
                "marked failed_manual_review"
            )
        # 决策2: the resubmit budget rides along in the payload and caps the
        # ensure-path resubmission of a failed job as well (upsert below
        # replaces the payload wholesale, so the count must be carried over
        # explicitly or the cap would reset every round).
        resubmit_attempts = 0
        if existing is not None:
            resubmit_attempts = self._resubmit_attempts(existing.get("payload"))
            if resubmit_attempts >= self.max_resubmit_attempts:
                self._mark_job_max_resubmits(existing)
                raise RuntimeError(
                    f"extract job {existing['job_id']} exceeded "
                    f"max_resubmit_attempts={self.max_resubmit_attempts}; "
                    "marked failed_manual_review"
                )

        # Pace BEFORE acquiring the key so the global submit interval does not
        # hold a key hostage while waiting (Phase 2: keys are scarce per-call
        # resources).
        self._pace_submit()
        api_key = self._acquire_key_with_wait()
        if api_key is None and self.key_pool.has_keys():
            raise RuntimeError("no extractor API key is currently available; all keys are busy or cooling down")
        api_key_alias = api_key.alias if api_key is not None else f"{self.provider.name}_stub"
        job_id = existing["job_id"] if existing is not None else str(uuid.uuid4())
        now = utc_now()
        payload: dict[str, Any] = {
            "input_file": str(input_file),
            "options": options,
            "key_alias": api_key_alias,
        }
        if existing is not None:
            # Count this resubmission at initiation time (决策2).
            payload["resubmit_attempts"] = resubmit_attempts + 1
        job = self.ledger.upsert_extract_job(
            {
                "job_id": job_id,
                "attachment_key": request.attachment_key,
                "pdf_sha256": pdf_sha256,
                "selected_pages": selected_pages,
                "cache_key": cache_key,
                "provider": self.provider.name,
                "provider_version": self.provider.version,
                "options_hash": options_hash,
                "api_key_alias": api_key_alias,
                "state": "submitted",
                "local_stage": "submitted",
                "submitted_at": now,
                "payload": payload,
            }
        )

        try:
            submitted = self.provider.submit(
                input_file,
                options_hash,
                options=provider_options,
                api_key=api_key.secret if api_key is not None else None,
            )
            self.ledger.set_extract_job_state(
                job["job_id"],
                state="running" if submitted.state == "running" else submitted.state,
                local_stage="poll",
                external_job_id=submitted.external_job_id,
                last_poll_at=utc_now(),
            )
            return ExtractionResult(
                job=self.ledger.get_extract_job(job_id=job["job_id"]) or job,
                cache_hit=False,
            )
        except Exception as exc:
            # Catch-all (submit stage): mark failed_retryable, cool the key by
            # error grade, and re-raise without crashing the caller's round.
            if api_key is not None:
                self.key_pool.mark_key_cooldown(
                    api_key.alias,
                    cooldown_seconds=self._cooldown_seconds_for(exc),
                )
            self.ledger.set_extract_job_state(
                job["job_id"],
                state="failed_retryable",
                local_stage="error",
                error_code=exc.__class__.__name__,
                error_message=str(exc),
            )
            raise
        finally:
            if api_key is not None:
                self.key_pool.release_key(api_key.alias)

    def poll_once(self, job_id: str) -> dict[str, Any] | None:
        """Phase 2 primitive: one provider poll for one job.

        Holds a key only for the single poll HTTP call (acquire → poll →
        ledger → release in ``finally``). Returns the refreshed job row, or
        ``None`` when the pool has keys but none is currently available — the
        caller (harvest) then sleeps instead of failing the job. Provider
        errors propagate after the key has been cooled down by error grade.
        """
        job = self.ledger.get_extract_job(job_id=job_id)
        if job is None:
            raise KeyError(f"extract job not found: {job_id}")
        external_job_id = job.get("external_job_id")
        if not external_job_id:
            raise RuntimeError("cannot poll extract job without external_job_id")
        options = (job.get("payload") or {}).get("options") or {}
        # Key affinity: prefer the alias that submitted the batch (MinerU
        # batches are queried through the account that created them); a busy
        # or cooling preferred key falls back to any available pooled key.
        api_key = self._acquire_key_with_wait(
            timeout_seconds=0,
            prefer_alias=self._job_key_alias(job),
        )
        if api_key is None and self.key_pool.has_keys():
            return None
        try:
            self._poll_rate_wait()
            polled = self.provider.poll(
                str(external_job_id),
                api_key=api_key.secret if api_key is not None else None,
                options=options,
            )
            self.ledger.set_extract_job_state(
                job_id,
                state=polled.state,
                local_stage="download" if polled.state == "completed" else "poll",
                external_job_id=polled.external_job_id,
                # Persisting the poll message both clears stale round errors
                # on progress (the setter overwrites unconditionally) and
                # lets harvest() report the remote failure reason verbatim.
                error_message=polled.message or None,
                last_poll_at=utc_now(),
            )
            return self.ledger.get_extract_job(job_id=job_id) or job
        except Exception as exc:
            if api_key is not None:
                cooldown = self._cooldown_seconds_for(exc)
                if cooldown > 0:
                    self.key_pool.mark_key_cooldown(api_key.alias, cooldown_seconds=cooldown)
            raise
        finally:
            if api_key is not None:
                self.key_pool.release_key(api_key.alias)

    def _stand_down(
        self,
        job: dict[str, Any],
        poll_budget: float,
        *,
        code: str = "poll-timeout",
        message: str | None = None,
    ) -> ExtractionResult:
        """M2 stand-down: end this round without failing the job.

        Keeps state=running and the external_job_id, never cools a key,
        never resubmits; the next ingest/resume round polls the same remote
        batch.
        """
        if message is None:
            message = (
                f"MinerU batch {job.get('external_job_id')} did not complete within "
                f"{poll_budget:.0f}s; will resume polling the same batch next run"
            )
        self.ledger.set_extract_job_state(
            job["job_id"],
            state="running",
            local_stage="poll",
            error_code=code,
            error_message=message,
            last_poll_at=utc_now(),
        )
        return ExtractionResult(
            job=self.ledger.get_extract_job(job_id=job["job_id"]) or job,
            cache_hit=False,
        )

    def harvest(self, job_id: str) -> ExtractionResult:
        """Phase 2 primitive: drive one submitted/running job to downloaded.

        Loop of "poll_once → ledger → release → sleep(poll_interval or
        exponential backoff)" until the remote batch completes (→ download),
        truly fails (→ failed_retryable + raise, key NOT cooled per the §2.2
        error table), or the per-job poll budget expires (→ stand-down, M2).
        Transient provider errors are classified by status_code: 429 stands
        the round down with a cooled key; 401/403 long-cools the key and
        continues on other pooled keys (all keys cooling → job fails, which
        the M9 ingest layer picks up as a blocked document); 408/5xx and
        network errors retry in-loop with exponential backoff.
        """
        job = self.ledger.get_extract_job(job_id=job_id)
        if job is None:
            raise KeyError(f"extract job not found: {job_id}")
        options = (job.get("payload") or {}).get("options") or {}
        poll_budget = self._poll_budget_seconds(options)
        poll_deadline = time.monotonic() + poll_budget
        transient_failures = 0
        first_poll = True
        while True:
            # M2 ordering (locked by fake-clock tests): the first poll runs
            # without a deadline check; every later poll is preceded by the
            # deadline check, then the interval sleep at the loop's tail.
            if not first_poll and time.monotonic() >= poll_deadline:
                return self._stand_down(job, poll_budget)
            first_poll = False
            try:
                polled_job = self.poll_once(job_id)
            except PollTimeoutError as exc:
                # Defensive: the deadline above is the normal timeout trigger,
                # but a provider-raised poll timeout gets the same stand-down
                # semantics — never a failure, never a cooldown, never a
                # resubmit.
                return self._stand_down(job, poll_budget, message=str(exc))
            except MinerUAPIError as exc:
                status = exc.status_code
                if status == 429:
                    # Quota/rate limit: the key is already cooling (poll_once);
                    # stand the round down, next round retries.
                    return self._stand_down(job, poll_budget, code="rate-limited", message=str(exc))
                if status in (401, 403):
                    if self._all_keys_cooling():
                        # Every pooled key failed auth: abort the round and
                        # fail the job (M9 surfaces it as a blocked document).
                        self.ledger.set_extract_job_state(
                            job_id,
                            state="failed_retryable",
                            local_stage="error",
                            error_code="no-keys-available",
                            error_message=str(exc),
                        )
                        raise
                    # Other keys remain: try another key on the next pass.
                    transient_failures += 1
                elif status in TRANSIENT_POLL_HTTP_STATUSES:
                    # Upstream hiccup (provider-level retries already
                    # exhausted): keep the job running, back off below.
                    transient_failures += 1
                else:
                    # Unclassified provider error: historical catch-all.
                    self.ledger.set_extract_job_state(
                        job_id,
                        state="failed_retryable",
                        local_stage="error",
                        error_code=exc.__class__.__name__,
                        error_message=str(exc),
                    )
                    raise
            except (ConnectionError, TimeoutError, OSError):
                # Network-level failure without an HTTP status: same graded
                # handling as transient 5xx — no cooldown, backoff retry.
                transient_failures += 1
            except Exception as exc:
                # Historical catch-all (e.g. plain RuntimeError from a
                # provider): failed_retryable + re-raise; poll_once already
                # applied the key cooldown.
                self.ledger.set_extract_job_state(
                    job_id,
                    state="failed_retryable",
                    local_stage="error",
                    error_code=exc.__class__.__name__,
                    error_message=str(exc),
                )
                raise
            else:
                if polled_job is not None:
                    job = polled_job
                    transient_failures = 0
                    if job["state"] == "completed":
                        return self._download_with_key(job)
                    if job["state"] == "failed":
                        # Remote batch truly failed. This raise lives outside
                        # the try/except above so the catch-all never re-cools
                        # the key for it (§2.2: remote failures do not cool
                        # keys); the job is failed_retryable and eligible for
                        # the 决策2-capped resubmit fallback.
                        self.ledger.set_extract_job_state(
                            job_id,
                            state="failed_retryable",
                            local_stage="error",
                            error_code="MinerUAPIError",
                            error_message=job.get("error_message") or "MinerU batch failed",
                        )
                        raise MinerUAPIError(
                            job.get("error_message") or "MinerU batch failed",
                            stage="extract",
                            batch_id=str(job.get("external_job_id") or ""),
                        )
                # polled_job is None: no key available right now — the tail
                # sleep below doubles as the "no key" wait.
            if transient_failures > 0:
                # Exponential backoff capped by the remaining poll budget, so
                # a flapping upstream cannot out-sleep the M2 deadline.
                backoff = self.poll_interval_seconds * (2 ** (transient_failures - 1))
                remaining = poll_deadline - time.monotonic()
                time.sleep(min(backoff, max(0.0, remaining)))
            else:
                time.sleep(self.poll_interval_seconds)

    def _download_with_key(self, job: dict[str, Any]) -> ExtractionResult:
        """Download a completed remote batch, holding a key only for the call.

        Empty pool (stub/no-key mode) runs with api_key=None directly. A
        non-empty pool whose keys are all busy stands down quietly instead of
        failing: the job stays completed/local_stage=download and the next
        round's classify routes it straight back here.
        """
        external_job_id = job.get("external_job_id")
        if not external_job_id:
            raise RuntimeError("cannot download extract job without external_job_id")
        artifact_dir = self.cache_dir / str(job["cache_key"])
        payload = dict(job.get("payload") or {})
        source_pdf_text = payload.get("source_pdf") or payload.get("input_file")
        source_pdf = Path(source_pdf_text) if source_pdf_text else None
        options = payload.get("options") or {}
        api_key = self._acquire_key_with_wait(prefer_alias=self._job_key_alias(job))
        if api_key is None and self.key_pool.has_keys():
            return ExtractionResult(
                job=self.ledger.get_extract_job(job_id=job["job_id"]) or job,
                cache_hit=False,
            )
        try:
            artifact = self.provider.download(
                str(external_job_id),
                artifact_dir,
                api_key=api_key.secret if api_key is not None else None,
                source_pdf=source_pdf,
                options=options,
            )
            payload = {**payload, "source_pdf": str(artifact.source_pdf)}
            self.ledger.set_extract_job_state(
                job["job_id"],
                state="downloaded",
                local_stage="downloaded",
                artifact_dir=artifact.artifact_dir,
                extract_dir=artifact.artifact_dir,
                manifest_path=artifact.manifest_path,
                payload=payload,
            )
            return ExtractionResult(
                job=self.ledger.get_extract_job(job_id=job["job_id"]) or job,
                cache_hit=False,
            )
        except DeepSeekIncompleteResult as exc:
            self.ledger.set_extract_job_state(
                job["job_id"], state="failed_manual_review", local_stage="error",
                error_code=type(exc).__name__, error_message=str(exc),
            )
            raise
        except Exception as exc:
            # Download-stage catch-all (same historical semantics as before).
            if api_key is not None:
                self.key_pool.mark_key_cooldown(
                    api_key.alias,
                    cooldown_seconds=self._cooldown_seconds_for(exc),
                )
            self.ledger.set_extract_job_state(
                job["job_id"],
                state="failed_retryable",
                local_stage="error",
                error_code=exc.__class__.__name__,
                error_message=str(exc),
            )
            raise
        finally:
            if api_key is not None:
                self.key_pool.release_key(api_key.alias)

    def ensure_extraction(self, request: ExtractionRequest) -> ExtractionResult:
        """Thin Phase 2 wrapper: submit_only + harvest for fresh submissions.

        Cache hits in REUSABLE_EXTRACT_STATES return untouched (driving them
        to completion is the caller's job, via resume_extraction/classify —
        e.g. ingest's _drive_extraction_to_completion). Fresh submissions
        always go through harvest(): its first poll confirms the remote batch
        state even when submit() already reported "completed" (H1).
        """
        result = self.submit_only(request)
        if result.cache_hit:
            return result
        if result.job["state"] == "downloaded":
            # Defensive: a fresh submission is never downloaded yet.
            return result
        return self.harvest(result.job["job_id"])

    def resume_extraction(self, job_id: str) -> ExtractionResult:
        """Resume one persisted extraction job from its recorded local stage.

        This method is explicit execution, not a status check. It never runs
        during diagnostics or recovery planning. For remote providers, callers
        must construct the manager with the intended provider/key pool first.
        Phase 2: key acquisition lives inside the primitives (submit_only /
        poll_once / _download_with_key), not in this dispatcher.
        """

        job = self.ledger.get_extract_job(job_id=job_id)
        if job is None:
            raise KeyError(f"extract job not found: {job_id}")
        recorded_provider = job.get("provider")
        if self.resume_manager_for and recorded_provider and recorded_provider != self.provider.name:
            return self.resume_manager_for(recorded_provider).resume_extraction(job_id)
        recovery = classify_extract_job(job)
        if recovery.action == "skip":
            return ExtractionResult(job=job, cache_hit=True)
        if recovery.action == "manual_review":
            raise RuntimeError(f"extract job {job_id} requires manual review: {recovery.reason}")
        if recovery.action in {"poll", "submit"} and self._lifetime_exceeded(job):
            # 决策1: enforce the total lifetime cap only on actions that would
            # keep remote work alive (poll/resubmit). "skip" (terminal) and
            # "download" (remote batch already completed; downloading is local
            # and free) must not be flipped to manual review.
            self._mark_job_max_lifetime(job)
            raise RuntimeError(
                f"extract job {job_id} exceeded "
                f"job_max_lifetime_seconds={float(self.job_max_lifetime_seconds):.0f}; "
                "marked failed_manual_review"
            )

        if recovery.action == "poll":
            if job.get("state") == "failed_retryable":
                return self._resume_poll_with_fallback(job)
            return self.harvest(job_id)
        if recovery.action == "download":
            return self._download_with_key(job)
        if recovery.action == "submit":
            self._resume_submit(job)
            # H1: a fresh (re)submission is always confirmed by harvest's
            # first poll before any download happens.
            return self.harvest(job_id)
        raise RuntimeError(f"unsupported extract recovery action: {recovery.action}")

    def _resume_submit(self, job: dict[str, Any]) -> dict[str, Any]:
        """Resubmit the job's PDF under a fresh remote batch (submit only).

        Phase 2: this method no longer chains into polling/downloading —
        resume_extraction hands the refreshed job to harvest() afterwards.
        Every automatic resubmission is counted in
        payload["resubmit_attempts"] (决策2) so the cap survives across
        ingest rounds.
        """
        payload = dict(job.get("payload") or {})
        # Pace before acquiring the key so the global submit interval does
        # not hold a key hostage while waiting (same as submit_only).
        self._pace_submit()
        api_key = self._acquire_key_with_wait(prefer_alias=self._job_key_alias(job))
        if api_key is None and self.key_pool.has_keys():
            # Matches the historical resume semantics: key starvation raises
            # without marking the job (the job keeps its current state).
            raise RuntimeError("no extractor API key is currently available; all keys are busy or cooling down")
        try:
            # Validation lives inside the try so a broken payload/missing
            # input file is marked failed_retryable, as the historical
            # resume_extraction catch-all used to do.
            input_file_text = payload.get("input_file")
            if not input_file_text:
                raise RuntimeError("cannot resubmit extract job without payload.input_file")
            input_file = Path(input_file_text)
            if not input_file.is_file():
                raise FileNotFoundError(f"extract input file no longer exists: {input_file}")
            options = dict(payload.get("options") or {})
            provider_options = dict(options)
            provider_options.setdefault("page_ranges", job.get("selected_pages") or "")
            submitted = self.provider.submit(
                input_file,
                str(job["options_hash"]),
                options=provider_options,
                api_key=api_key.secret if api_key is not None else None,
            )
            self.ledger.set_extract_job_state(
                job["job_id"],
                state="running" if submitted.state == "running" else submitted.state,
                local_stage="poll",
                external_job_id=submitted.external_job_id,
                last_poll_at=utc_now(),
                payload={
                    **payload,
                    "resubmit_attempts": self._resubmit_attempts(payload) + 1,
                    "key_alias": api_key.alias if api_key is not None else payload.get("key_alias"),
                },
            )
        except Exception as exc:
            # Submit-stage catch-all (same grading as submit_only).
            if api_key is not None:
                self.key_pool.mark_key_cooldown(
                    api_key.alias,
                    cooldown_seconds=self._cooldown_seconds_for(exc),
                )
            self.ledger.set_extract_job_state(
                job["job_id"],
                state="failed_retryable",
                local_stage="error",
                error_code=exc.__class__.__name__,
                error_message=str(exc),
            )
            raise
        finally:
            if api_key is not None:
                self.key_pool.release_key(api_key.alias)
        if submitted.state not in ("running", "completed"):
            self.ledger.set_extract_job_state(
                job["job_id"],
                state="failed_retryable",
                local_stage="error",
                error_code="MinerUAPIError",
                error_message="resubmitted extract job failed immediately",
            )
            raise MinerUAPIError(
                "resubmitted extract job failed immediately",
                stage="submit",
                batch_id=submitted.external_job_id,
            )
        return self.ledger.get_extract_job(job_id=job["job_id"]) or job

    def _resume_poll_with_fallback(self, job: dict[str, Any]) -> ExtractionResult:
        """Poll a previously failed job; if the remote batch is gone or still
        failed, resubmit the PDF — at most max_resubmit_attempts times (决策2).

        harvest() never raises PollTimeoutError (it stands down internally),
        so a timeout can never trigger the resubmit fallback — resubmitting a
        still-running batch would double-bill the PDF. Auth/quota errors are
        not remote-batch failures either and propagate without resubmission.
        """
        try:
            return self.harvest(job["job_id"])
        except MinerUAPIError as exc:
            if exc.status_code in (401, 403, 429):
                raise
            if self._resubmit_attempts(job.get("payload")) >= self.max_resubmit_attempts:
                self._mark_job_max_resubmits(job)
                raise RuntimeError(
                    f"extract job {job['job_id']} exceeded "
                    f"max_resubmit_attempts={self.max_resubmit_attempts}; "
                    "marked failed_manual_review"
                ) from exc
            self.ledger.set_extract_job_state(
                job["job_id"],
                state="submitted",
                local_stage="submitted",
                # NOTE: external_job_id is COALESCEd by the setter, so this
                # None does not actually clear the column; _resume_submit
                # overwrites it with the fresh batch id right after. Kept for
                # state-machine readability.
                external_job_id=None,
                error_code=None,
                error_message=None,
            )
            self._resume_submit(job)
            return self.harvest(job["job_id"])
