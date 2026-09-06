from __future__ import annotations

import threading
import unittest

from zoterorag.extractors.key_pool import ApiKeyRef, ExtractorKeyPool


class KeyPoolThreadSafetyTests(unittest.TestCase):
    def test_concurrent_acquire_release_maintains_invariant(self) -> None:
        pool = ExtractorKeyPool(
            [
                ApiKeyRef(alias=f"mineru_{i}", secret=f"secret-{i}")
                for i in range(4)
            ],
            per_key_submit_concurrency=2,
        )
        errors: list[Exception] = []
        acquired: list[ApiKeyRef] = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker() -> None:
            try:
                barrier.wait(timeout=2)
                key = pool.acquire_key()
                if key is not None:
                    with lock:
                        acquired.append(key)
                    pool.release_key(key.alias)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual([], errors)
        self.assertEqual(8, len(acquired))
        public = {key["alias"]: key for key in pool.list_public_keys()}
        for key in acquired:
            self.assertEqual(0, public[key.alias]["in_flight"])

    def test_concurrent_round_robin_advances_index(self) -> None:
        pool = ExtractorKeyPool(
            [
                ApiKeyRef(alias="mineru_a", secret="secret-a"),
                ApiKeyRef(alias="mineru_b", secret="secret-b"),
            ],
            per_key_submit_concurrency=1,
        )
        aliases: list[str] = []
        lock = threading.Lock()
        barrier = threading.Barrier(4)

        def worker() -> None:
            barrier.wait(timeout=2)
            key = pool.acquire_key()
            if key is not None:
                with lock:
                    aliases.append(key.alias)
                pool.release_key(key.alias)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(["mineru_a", "mineru_b", "mineru_a", "mineru_b"], aliases)

    def test_cooldown_is_thread_safe(self) -> None:
        pool = ExtractorKeyPool(
            [ApiKeyRef(alias="mineru_a", secret="secret-a")],
            per_key_submit_concurrency=1,
        )
        errors: list[Exception] = []
        barrier = threading.Barrier(4)

        def worker() -> None:
            try:
                barrier.wait(timeout=2)
                pool.mark_key_cooldown("mineru_a", cooldown_seconds=10)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual([], errors)
        self.assertGreater(pool.cooldown_remaining("mineru_a"), 0)


if __name__ == "__main__":
    unittest.main()


# Phase 2: key affinity (prefer_alias) and from_env_file passthrough.

from pathlib import Path  # noqa: E402

from tests._support import workspace_tmpdir  # noqa: E402


class KeyPoolAffinityTests(unittest.TestCase):
    def test_from_env_file_passes_per_key_submit_concurrency(self) -> None:
        with workspace_tmpdir("keypool-env-") as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("MINERU_KEY=aaa,bbb\n", encoding="utf-8")
            pool = ExtractorKeyPool.from_env_file(env_path, per_key_submit_concurrency=3)
            self.assertEqual(3, pool.per_key_submit_concurrency)
            self.assertEqual(2, len(pool.list_public_keys()))

    def test_acquire_prefers_alias_when_available(self) -> None:
        pool = ExtractorKeyPool(
            [ApiKeyRef(alias="mineru_a", secret="s-a"), ApiKeyRef(alias="mineru_b", secret="s-b")]
        )
        key = pool.acquire_key(prefer_alias="mineru_b")
        self.assertIsNotNone(key)
        self.assertEqual("mineru_b", key.alias)

    def test_acquire_falls_back_when_preferred_busy(self) -> None:
        pool = ExtractorKeyPool(
            [ApiKeyRef(alias="mineru_a", secret="s-a"), ApiKeyRef(alias="mineru_b", secret="s-b")]
        )
        held = pool.acquire_key()
        self.assertEqual("mineru_a", held.alias)
        key = pool.acquire_key(prefer_alias="mineru_a")
        self.assertIsNotNone(key)
        self.assertEqual("mineru_b", key.alias)

    def test_acquire_falls_back_when_preferred_cooling(self) -> None:
        pool = ExtractorKeyPool(
            [ApiKeyRef(alias="mineru_a", secret="s-a"), ApiKeyRef(alias="mineru_b", secret="s-b")]
        )
        pool.mark_key_cooldown("mineru_a", cooldown_seconds=10)
        key = pool.acquire_key(prefer_alias="mineru_a")
        self.assertIsNotNone(key)
        self.assertEqual("mineru_b", key.alias)

    def test_acquire_returns_none_when_only_preferred_key_and_busy(self) -> None:
        pool = ExtractorKeyPool([ApiKeyRef(alias="mineru_a", secret="s-a")])
        held = pool.acquire_key()
        self.assertIsNotNone(held)
        self.assertIsNone(pool.acquire_key(prefer_alias="mineru_a"))
