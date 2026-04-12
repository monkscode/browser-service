"""
Task processor module for managing background task execution.

This module provides the TaskProcessor class which handles:
- Task submission and execution in background threads
- Task status tracking (thread-safe for concurrent access)
- Task result management
"""

import time
import threading
from typing import Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


class TaskProcessor:
    """
    Manages background task execution and status tracking.

    Thread-safety: All reads/writes to self.tasks are protected by self._lock.
    Worker threads must update task status via update_task(), which acquires
    the lock before writing. This ensures readers (get_task_status, list_tasks,
    count_active_tasks, try_submit_task) always observe a consistent snapshot
    — dict.update() with multiple keys is not a single atomic bytecode, so
    writing outside the lock can expose partial updates to concurrent readers.

    Memory management: completed tasks are lazily evicted after completed_task_ttl
    seconds. Eviction runs inside the lock during count_active_tasks() and
    try_submit_task() — no background thread required. Tasks without a
    'completed_at' timestamp are never evicted (safe for tests that manually
    set status without the full fields).

    Metrics: _tasks_submitted is a separate cumulative counter incremented once
    per accepted submission (inside self._lock). It is never decremented, so
    tasks_submitted_count() is monotonically increasing and suitable for
    observability dashboards. tracked_task_count() returns len(self.tasks) which
    can decrease after TTL eviction and is only meaningful as a memory-usage hint.
    """

    def __init__(self, executor: ThreadPoolExecutor, completed_task_ttl: float = 300.0):
        """
        Initialize the TaskProcessor.

        Args:
            executor: ThreadPoolExecutor instance for running tasks in background
            completed_task_ttl: Seconds to retain completed tasks before eviction.
                                Set to 0 to disable eviction. Default 300s (5 min)
                                — long enough for clients to poll final results.
        """
        self.executor = executor
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.completed_task_ttl = completed_task_ttl
        self._tasks_submitted: int = 0  # cumulative, never decremented
        logger.info("TaskProcessor initialized (thread-safe)")

    def _evict_completed(self) -> None:
        """Remove completed tasks older than completed_task_ttl. Caller must hold self._lock."""
        if self.completed_task_ttl <= 0:
            return
        cutoff = time.time() - self.completed_task_ttl
        stale = [
            tid for tid, t in self.tasks.items()
            if t.get('status') == 'completed'
            and t.get('completed_at', float('inf')) < cutoff
        ]
        for tid in stale:
            del self.tasks[tid]
        if stale:
            logger.debug(f"Evicted {len(stale)} completed tasks (TTL={self.completed_task_ttl}s)")

    def submit_task(
        self,
        task_id: str,
        task_function: Callable,
        *args,
        **kwargs
    ) -> None:
        """
        Submit a task for background execution.

        Args:
            task_id: Unique identifier for the task
            task_function: The function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        """
        with self._lock:
            self.tasks[task_id] = {
                "status": "processing",
                "created_at": time.time(),
                "objective": f"Task {task_id}"
            }
            self._tasks_submitted += 1

        logger.info(f"Submitting task {task_id} for background execution")
        try:
            self.executor.submit(task_function, *args, **kwargs)
        except Exception:
            with self._lock:
                self.tasks.pop(task_id, None)
                self._tasks_submitted -= 1
            raise

    def try_submit_task(
        self,
        max_concurrent: int,
        task_id: str,
        task_function: Callable,
        *args,
        **kwargs
    ) -> bool:
        """
        Atomically check capacity and submit a task if under the limit.

        The check and registration happen inside a single lock acquisition,
        eliminating the TOCTOU race that exists when callers do
        count_active_tasks() + submit_task() as two separate calls.

        Args:
            max_concurrent: Maximum number of concurrent active tasks allowed
            task_id: Unique identifier for the task
            task_function: The function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            True if the task was accepted and submitted, False if at capacity
        """
        with self._lock:
            self._evict_completed()
            active = sum(
                1 for t in self.tasks.values()
                if t.get('status') in ['processing', 'running']
            )
            if active >= max_concurrent:
                return False
            self.tasks[task_id] = {
                "status": "processing",
                "created_at": time.time(),
                "objective": f"Task {task_id}"
            }
            self._tasks_submitted += 1

        logger.info(f"Submitting task {task_id} for background execution")
        try:
            self.executor.submit(task_function, *args, **kwargs)
        except Exception:
            with self._lock:
                self.tasks.pop(task_id, None)
                self._tasks_submitted -= 1
            raise
        return True

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific task.

        Args:
            task_id: The task identifier

        Returns:
            Task status dictionary or None if task not found
        """
        with self._lock:
            task = self.tasks.get(task_id)
            if task is None:
                return None
            # Return a shallow copy to avoid caller seeing partial updates
            return {**task}

    def list_tasks(self) -> list:
        """
        List all tasks with their status.

        Returns:
            List of task dictionaries with status information
        """
        with self._lock:
            return [
                {"task_id": task_id, **task_data}
                for task_id, task_data in self.tasks.items()
            ]

    def count_active_tasks(self) -> int:
        """
        Count currently active (processing or running) tasks.

        Also evicts stale completed tasks as a side effect (lazy cleanup).

        Returns:
            Number of active tasks
        """
        with self._lock:
            self._evict_completed()
            return sum(
                1 for t in self.tasks.values()
                if t.get('status') in ['processing', 'running']
            )

    def update_task(self, task_id: str, updates: Dict[str, Any]) -> None:
        """
        Atomically apply updates to a task record.

        Acquires the lock before writing so readers (get_task_status,
        list_tasks, count_active_tasks, try_submit_task) always observe
        a consistent snapshot.

        Args:
            task_id: The task identifier
            updates: Key/value pairs to merge into the task record
        """
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(updates)
            else:
                logger.warning(
                    f"update_task: task '{task_id}' not found (evicted or never submitted); "
                    f"update dropped: {list(updates.keys())}"
                )

    def tasks_submitted_count(self) -> int:
        """
        Return the cumulative number of tasks accepted for submission.

        This counter is monotonically increasing — it is incremented once per
        accepted submission and never decremented by TTL eviction. Use this
        for health/observability metrics that must not decrease over time.
        """
        with self._lock:
            return self._tasks_submitted

    def tracked_task_count(self) -> int:
        """
        Return the number of tasks currently held in memory (active + not-yet-evicted
        completed tasks).

        This value can decrease as TTL eviction removes stale completed records.
        Use this only as a memory-usage hint, not as a processed-task counter.
        """
        with self._lock:
            return len(self.tasks)

    def get_tasks_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the internal tasks dictionary (direct reference).

        Only use this for read-only operations (e.g. ``len()``).  Do NOT
        write to the returned dict directly — use ``update_task()`` instead
        to keep writes under the lock.

        Returns:
            The tasks dictionary (direct reference, not a copy)
        """
        return self.tasks
