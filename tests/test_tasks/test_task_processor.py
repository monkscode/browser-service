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
