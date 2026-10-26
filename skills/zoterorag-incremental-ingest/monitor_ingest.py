"""Progress monitor for a running ZoteroRAG ingest job.

Usage:
    .conda/python.exe skills/zoterorag-incremental-ingest/monitor_ingest.py <job_id> [attachment_key ...]

Reads state.sqlite **read-only** and prints one line whenever the job status or
the per-document extract states change, then a final line when the job reaches a
terminal status (and exits). Designed to be driven by a Monitor/cron-style
wrapper so a long ingest can be watched without polling by hand.

Environment overrides:
    ZOTERORAG_STATE_DB               default "file:data/state/state.sqlite?mode=ro"
    ZOTERORAG_MONITOR_POLL_SECONDS   default 180

Notes:
    * attachment_key arguments are optional. Without them the monitor tracks only
      the job status.
    * Extract state is counted per attachment_key. Segment jobs created by a
      byte-identical duplicate attachment are stored under the *other* key, so a
      duplicated document can look permanently "none" — see the skill doc's
      "已知盲区" section.
    * A monitor failure must never affect the ingest run: DB read errors are
      reported as [warn] and retried, never raised.
"""

import os
import sqlite3
import sys
import time

DB = os.environ.get("ZOTERORAG_STATE_DB", "file:data/state/state.sqlite?mode=ro")
POLL_SECONDS = int(os.environ.get("ZOTERORAG_MONITOR_POLL_SECONDS", "180"))

# Extract states that mean "this document's extraction is finished".
DONE = {"downloaded", "normalized"}

# Job statuses after which the run is over and the monitor can stop.
TERMINAL = {
    "completed",
    "completed_with_errors",
    "failed",
    "cancelled",
    "interrupted",
    "blocked",
}


def read(job_id, keys):
    """Return (job_status, {attachment_key: most_advanced_extract_state})."""
    con = sqlite3.connect(DB, uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("SELECT status FROM pipeline_jobs WHERE job_id = ?", (job_id,))
    row = cur.fetchone()
    status = row["status"] if row else "MISSING"

    states = {}
    if keys:
        placeholders = ",".join("?" * len(keys))
        cur.execute(
            f"SELECT attachment_key, state FROM extract_jobs WHERE attachment_key IN ({placeholders})",
            keys,
        )
        for r in cur.fetchall():
            key, state = r["attachment_key"], r["state"]
            # A key may have several jobs (retries/segments); keep the most
            # advanced one so a stale `running` row cannot mask a `downloaded`.
            if key not in states or state in DONE:
                states[key] = state

    con.close()
    return status, states


def format_event(status, states, keys):
    if not keys:
        return "job=%s" % status
    done = sum(1 for k in keys if states.get(k) in DONE)
    detail = ", ".join("%s:%s" % (k, states.get(k, "none")) for k in keys)
    return "job=%s extract_done=%d/%d states={%s}" % (status, done, len(keys), detail)


def main(argv):
    if len(argv) < 2:
        print(
            "usage: monitor_ingest.py <job_id> [attachment_key ...]",
            file=sys.stderr,
        )
        return 2

    job_id, keys = argv[1], argv[2:]
    prev = None

    while True:
        try:
            status, states = read(job_id, keys)
        except Exception as exc:  # transient lock/IO issues must not kill the watch
            print("[warn] db read failed: %s" % exc, flush=True)
            time.sleep(POLL_SECONDS)
            continue

        signature = (status, tuple(sorted(states.items())))
        if signature != prev:
            print(format_event(status, states, keys), flush=True)
            prev = signature

        if status in TERMINAL:
            print("[terminal] ingest job %s" % status, flush=True)
            break

        time.sleep(POLL_SECONDS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
