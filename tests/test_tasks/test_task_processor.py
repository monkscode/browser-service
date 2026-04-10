"""
Unit tests for browser_service.tasks.processor — TaskProcessor.

Purpose: Verify task submission, status tracking, and listing behave correctly.
         TaskProcessor is the backbone of the async task system: if submit_task
         fails to register status, the API returns 404 for active tasks.  If
         list_tasks loses entries, the UI shows stale data.

Tests:
  - submit_task creates a status entry with 'processing' state
  - submit_task calls the executor with the given function
  - get_task_status returns the entry for known task IDs
  - get_task_status returns None for unknown task IDs
  - list_tasks returns all submitted tasks
  - get_tasks_dict returns the raw internal dict (used by workers to update)
"""

import time
import threading
import pytest
from unittest.mock import MagicMock
from concurrent.futures import ThreadPoolExecutor

from browser_service.tasks.processor import TaskProcessor


class TestTaskProcessorSubmit:
    """Tests for task submission and status registration."""

    def test_submit_creates_processing_entry(self, task_processor):
        """submit_task registers a 'processing' status immediately."""
        task_fn = MagicMock()
        task_processor.submit_task("task-001", task_fn)

        status = task_processor.get_task_status("task-001")
        assert status is not None
        assert status["status"] == "processing"
        assert "created_at" in status

    def test_submit_calls_executor(self, task_processor):
        """submit_task passes the function + args to the executor."""
        calls = []

        def track_fn(x, y):
            calls.append((x, y))

        task_processor.submit_task("task-002", track_fn, "a", "b")
        # Give executor time to pick up
        time.sleep(0.2)
        assert calls == [("a", "b")]

    def test_submit_with_kwargs(self, task_processor):
        """submit_task forwards keyword arguments to the function."""
        results = {}

        def kw_fn(key=None):
            results["key"] = key

        task_processor.submit_task("task-003", kw_fn, key="value")
        time.sleep(0.2)
        assert results.get("key") == "value"


class TestTaskProcessorQuery:
    """Tests for querying task state."""

    def test_get_task_status_known(self, task_processor):
        """get_task_status returns status dict for a submitted task."""
        task_processor.submit_task("task-010", MagicMock())
        result = task_processor.get_task_status("task-010")
        assert isinstance(result, dict)
        assert result["status"] == "processing"

    def test_get_task_status_unknown(self, task_processor):
        """get_task_status returns None for an ID that was never submitted."""
        assert task_processor.get_task_status("nonexistent") is None

    def test_list_tasks_returns_all(self, task_processor):
        """list_tasks includes every submitted task with its ID."""
        task_processor.submit_task("task-a", MagicMock())
        task_processor.submit_task("task-b", MagicMock())

        tasks = task_processor.list_tasks()
        ids = [t["task_id"] for t in tasks]
        assert "task-a" in ids
        assert "task-b" in ids
        assert len(tasks) == 2

    def test_get_tasks_dict_returns_internal(self, task_processor):
        """get_tasks_dict exposes the raw dict so workers can mutate status."""
        task_processor.submit_task("task-x", MagicMock())
        d = task_processor.get_tasks_dict()
        assert "task-x" in d
        # Workers update status directly
        d["task-x"]["status"] = "completed"
        assert task_processor.get_task_status("task-x")["status"] == "completed"


class TestCountActiveTasks:
    """Tests for count_active_tasks()."""

    def test_count_active_tasks_all_processing(self):
        """Tasks registered as 'processing' are counted as active."""
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            proc = TaskProcessor(executor)
            proc.submit_task("t1", MagicMock())
            proc.submit_task("t2", MagicMock())
            proc.submit_task("t3", MagicMock())
            assert proc.count_active_tasks() == 3
        finally:
            executor.shutdown(wait=False)

    def test_count_active_tasks_running_state(self):
        """Tasks with status 'running' are also counted."""
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            proc = TaskProcessor(executor)
            proc.submit_task("t1", MagicMock())
            proc.get_tasks_dict()["t1"]["status"] = "running"
            assert proc.count_active_tasks() == 1
        finally:
            executor.shutdown(wait=False)

    def test_count_active_tasks_zero_after_completion(self):
        """Completed tasks are not counted as active."""
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            proc = TaskProcessor(executor)
            proc.submit_task("t1", MagicMock())
            proc.submit_task("t2", MagicMock())
            d = proc.get_tasks_dict()
            d["t1"]["status"] = "completed"
            d["t2"]["status"] = "completed"
            assert proc.count_active_tasks() == 0
        finally:
            executor.shutdown(wait=False)


class TestTrySubmitTask:
    """Tests for try_submit_task() — atomic capacity-check-and-register."""

    def test_accepted_under_limit(self):
        """Returns True and registers task when below max_concurrent."""
        executor = ThreadPoolExecutor(max_workers=2)
        try:
            proc = TaskProcessor(executor)
            result = proc.try_submit_task(2, "t1", MagicMock())
            assert result is True
            assert proc.get_task_status("t1") is not None
        finally:
            executor.shutdown(wait=False)

    def test_rejected_at_limit(self):
        """Returns False without registering task when at max_concurrent."""
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            proc = TaskProcessor(executor)
            proc.try_submit_task(1, "t1", MagicMock())
            # t1 is now 'processing' — limit reached
            result = proc.try_submit_task(1, "t2", MagicMock())
            assert result is False
            assert proc.get_task_status("t2") is None
        finally:
            executor.shutdown(wait=False)

    def test_accepted_after_completion(self):
        """Returns True again once a task transitions to 'completed'."""
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            proc = TaskProcessor(executor)
            proc.try_submit_task(1, "t1", MagicMock())
            proc.get_tasks_dict()["t1"]["status"] = "completed"
            result = proc.try_submit_task(1, "t2", MagicMock())
            assert result is True
        finally:
            executor.shutdown(wait=False)

    def test_concurrent_try_submit_never_exceeds_limit(self):
        """10 threads race to submit; accepted count must not exceed max_concurrent."""
        executor = ThreadPoolExecutor(max_workers=10)
        try:
            proc = TaskProcessor(executor)
            accepted = []
            lock = threading.Lock()

            def submit(i):
                ok = proc.try_submit_task(5, f"t{i}", MagicMock())
                if ok:
                    with lock:
                        accepted.append(i)

            threads = [threading.Thread(target=submit, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(accepted) == 5, f"Expected 5 accepted, got {len(accepted)}: {accepted}"
        finally:
            executor.shutdown(wait=False)


class TestEviction:
    """Tests for lazy eviction of completed tasks (completed_task_ttl)."""

    def _make_proc(self, ttl: float) -> TaskProcessor:
        return TaskProcessor(ThreadPoolExecutor(max_workers=1), completed_task_ttl=ttl)

    def test_stale_completed_task_removed_by_count(self):
        """Completed task with past completed_at is evicted during count_active_tasks."""
        proc = self._make_proc(ttl=60)
        proc.submit_task("t1", MagicMock())
        proc.get_tasks_dict()["t1"].update({"status": "completed", "completed_at": time.time() - 120})

        proc.count_active_tasks()

        assert proc.get_task_status("t1") is None

    def test_recent_completed_task_retained(self):
        """Completed task within TTL is NOT evicted."""
        proc = self._make_proc(ttl=300)
        proc.submit_task("t1", MagicMock())
        proc.get_tasks_dict()["t1"].update({"status": "completed", "completed_at": time.time() - 10})

        proc.count_active_tasks()

        assert proc.get_task_status("t1") is not None

    def test_active_tasks_never_evicted(self):
        """Processing and running tasks are never evicted regardless of age."""
        proc = self._make_proc(ttl=1)
        proc.submit_task("t1", MagicMock())
        proc.get_tasks_dict()["t1"]["status"] = "running"
        # No completed_at — should never be touched
        proc.count_active_tasks()
        assert proc.get_task_status("t1") is not None

    def test_no_completed_at_never_evicted(self):
        """Completed task without completed_at timestamp is never evicted."""
        proc = self._make_proc(ttl=1)
        proc.submit_task("t1", MagicMock())
        proc.get_tasks_dict()["t1"]["status"] = "completed"
        # No completed_at set — float('inf') guard keeps it

        proc.count_active_tasks()

        assert proc.get_task_status("t1") is not None

    def test_ttl_zero_disables_eviction(self):
        """TTL=0 disables eviction entirely."""
        proc = self._make_proc(ttl=0)
        proc.submit_task("t1", MagicMock())
        proc.get_tasks_dict()["t1"].update({"status": "completed", "completed_at": time.time() - 9999})

        proc.count_active_tasks()

        assert proc.get_task_status("t1") is not None

    def test_eviction_frees_slot_for_try_submit(self):
        """Stale completed task is evicted inside try_submit_task, freeing its slot for counting."""
        proc = self._make_proc(ttl=60)
        # Fill to limit with one stale completed task
        proc.submit_task("old", MagicMock())
        proc.get_tasks_dict()["old"].update({"status": "completed", "completed_at": time.time() - 120})

        # try_submit with limit=1: stale task evicted first, then active=0, so accepted
        result = proc.try_submit_task(1, "new", MagicMock())
        assert result is True
        assert proc.get_task_status("old") is None

    def test_dict_shrinks_under_sustained_load(self):
        """Dict size stays bounded when completed tasks exceed TTL."""
        proc = self._make_proc(ttl=60)
        # Simulate 50 completed old tasks
        for i in range(50):
            proc.tasks[f"old-{i}"] = {"status": "completed", "completed_at": time.time() - 120}

        proc.count_active_tasks()

        assert len(proc.tasks) == 0


class TestThreadSafety:
    """Concurrency tests verifying no data corruption under parallel access."""

    def test_concurrent_submit_no_data_loss(self):
        """10 threads each submit 1 task; all 10 must appear in list_tasks()."""
        executor = ThreadPoolExecutor(max_workers=10)
        try:
            proc = TaskProcessor(executor)
            errors = []

            def submit(task_id):
                try:
                    proc.submit_task(task_id, MagicMock())
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=submit, args=(f"t{i}",)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert errors == [], f"Exceptions during concurrent submit: {errors}"
            ids = {entry["task_id"] for entry in proc.list_tasks()}
            assert ids == {f"t{i}" for i in range(10)}
        finally:
            executor.shutdown(wait=False)

    def test_concurrent_read_write_no_runtime_error(self):
        """list_tasks() called while submit_task() runs concurrently raises no errors."""
        executor = ThreadPoolExecutor(max_workers=5)
        try:
            proc = TaskProcessor(executor)
            errors = []
            stop = threading.Event()

            def writer():
                i = 0
                while not stop.is_set():
                    try:
                        proc.submit_task(f"w{i}", MagicMock())
                    except Exception as e:
                        errors.append(("write", e))
                    i += 1
                    time.sleep(0.001)

            def reader():
                while not stop.is_set():
                    try:
                        proc.list_tasks()
                    except Exception as e:
                        errors.append(("read", e))
                    time.sleep(0.001)

            w = threading.Thread(target=writer)
            r = threading.Thread(target=reader)
            w.start()
            r.start()
            time.sleep(0.5)
            stop.set()
            w.join()
            r.join()

            assert errors == [], f"Exceptions during concurrent read/write: {errors}"
        finally:
            executor.shutdown(wait=False)
