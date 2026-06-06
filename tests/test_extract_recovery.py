from __future__ import annotations

import unittest

from tests._support import workspace_tmpdir
from zoterorag.extractors import build_extract_recovery_plan, classify_extract_job


class ExtractRecoveryTests(unittest.TestCase):
    def test_classifies_remote_resume_without_resubmit(self) -> None:
        running = classify_extract_job(base_job(state="running", local_stage="poll", external_job_id="batch-1"))
        completed = classify_extract_job(base_job(state="completed", local_stage="download", external_job_id="batch-2"))

        self.assertEqual("poll", running.action)
        self.assertTrue(running.can_resume_without_resubmit)
        self.assertEqual("download", completed.action)
        self.assertTrue(completed.can_resume_without_resubmit)

    def test_classifies_local_artifact_resume_stages(self) -> None:
        with workspace_tmpdir("extract-recovery-") as tmpdir:
            zip_path = tmpdir / "result.zip"
            extract_dir = tmpdir / "extract"
            manifest = tmpdir / "manifest.json"
            zip_path.write_bytes(b"zip")
            extract_dir.mkdir()
            manifest.write_text("{}", encoding="utf-8")

            zip_only = classify_extract_job(base_job(state="downloaded", local_stage="downloaded", zip_path=zip_path))
            extracted = classify_extract_job(
                base_job(state="downloaded", local_stage="downloaded", extract_dir=extract_dir)
            )
            ready = classify_extract_job(
                base_job(state="downloaded", local_stage="downloaded", manifest_path=manifest)
            )

            self.assertEqual("extract_zip", zip_only.action)
            self.assertTrue(zip_only.can_resume_without_resubmit)
            self.assertEqual("normalize", extracted.action)
            self.assertTrue(extracted.can_resume_without_resubmit)
            self.assertEqual("skip", ready.action)
            self.assertTrue(ready.can_resume_without_resubmit)

    def test_missing_downloaded_artifacts_require_review(self) -> None:
        item = classify_extract_job(
            base_job(
                state="downloaded",
                local_stage="downloaded",
                zip_path="missing.zip",
                manifest_path="missing.json",
            )
        )

        self.assertEqual("manual_review", item.action)
        self.assertFalse(item.can_resume_without_resubmit)
        self.assertEqual(["zip_path", "manifest_path"], item.to_dict()["missing_paths"])

    def test_plan_summarizes_actions(self) -> None:
        plan = build_extract_recovery_plan(
            [
                base_job(job_id="a", state="running", local_stage="poll", external_job_id="batch-a"),
                base_job(job_id="b", state="failed_retryable", local_stage="error"),
                base_job(job_id="c", state="failed_manual_review", local_stage="error"),
            ]
        )

        self.assertEqual(3, plan["summary"]["job_count"])
        self.assertEqual(1, plan["summary"]["resumable_without_resubmit"])
        self.assertEqual({"manual_review": 1, "poll": 1, "submit": 1}, plan["summary"]["by_action"])


def base_job(**overrides):
    job = {
        "job_id": "job-1",
        "attachment_key": "ATT1",
        "cache_key": "cache-1",
        "state": "running",
        "local_stage": "poll",
        "external_job_id": None,
    }
    job.update(overrides)
    return job


if __name__ == "__main__":
    unittest.main()
