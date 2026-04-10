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
    Worker threads update task status via the dict reference returned by
    get_tasks_dict(). Individual dict key writes (tasks_dict[task_id].update(...))
    are atomic under CPython's GIL and operate on different keys per worker,
    so they are safe without holding the lock. The lock protects compound
    operations like iteration (list_tasks) and count (count_active_tasks).

    Memory management: completed tasks are lazily evicted after completed_task_ttl
    seconds. Eviction runs inside the lock during count_active_tasks() and
    try_submit_task() — no background thread required. Tasks without a
    'completed_at' timestamp are never evicted (safe for tests that manually
    set status without the full fields).
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

        logger.info(f"Submitting task {task_id} for background execution")
        self.executor.submit(task_function, *args, **kwargs)

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

        logger.info(f"Submitting task {task_id} for background execution")
        self.executor.submit(task_function, *args, **kwargs)
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

    def get_tasks_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the internal tasks dictionary.

        This is used by task functions (workflow.py) to update their own
        task status. Workers write to distinct keys (their own task_id),
        so concurrent writes are safe under CPython's GIL.

        Returns:
            The tasks dictionary (direct reference, not a copy)
        """
        return self.tasks
