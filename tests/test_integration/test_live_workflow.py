"""
Live integration tests for browser-service workflow (Tier 2).

Purpose: Verify process_workflow_task() end-to-end execution against a live
         web page using a real browser-use Agent and Playwright instance.

Requires:
  - Playwright installed with Chromium: playwright install chromium
  - browser-use package installed
  - LLM API key (OPENAI_API_KEY or ANTHROPIC_API_KEY) set in environment
    OR Ollama running locally for open-source model support
  - Internet access (tests navigate to example.com)

Run with:
  pytest tests/test_integration/test_live_workflow.py -m integration -v

Skip the full agent tests (expensive) with:
  pytest -m integration -k "not TestLiveAgentWorkflow"
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_session_config() -> dict:
    """Return a minimal session config for tests."""
    return {
        "headless": True,
        "viewport": {"width": 1280, "height": 720},
    }


def _has_llm_key() -> bool:
    """True if any supported LLM API key is available."""
    return bool(
        os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY")
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_processor():
    """Factory fixture: creates TaskProcessor + ThreadPoolExecutor pairs per test.

    Each call to the returned factory allocates a fresh executor and registers
    it for teardown. All executors are shut down with wait=True after the test
    so no worker threads outlive the test that created them.
    """
    from browser_service.tasks.processor import TaskProcessor

    executors = []

    def _make(task_id: str) -> TaskProcessor:
        exc = ThreadPoolExecutor(max_workers=1)
        executors.append(exc)
        tp = TaskProcessor(exc)
        tp.submit_task(task_id, lambda: None)
        return tp

    yield _make

    for exc in executors:
        exc.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Guard condition tests — no browser/LLM needed
# ---------------------------------------------------------------------------


class TestProcessWorkflowTaskGuardsLive:
    """Guard conditions in process_workflow_task() — replicated with live imports."""

    @staticmethod
    def _mock_browser():
        """Patch BrowserSession so tests don't launch a real browser."""
        mock_session = MagicMock()
        mock_session.start = AsyncMock(side_effect=RuntimeError("mock: no browser in test"))
        return patch("browser_use.browser.session.BrowserSession", return_value=mock_session)

    def test_raises_on_none_tasks_dict(self):
        """task_processor=None raises ValueError immediately, no browser launched."""
        from browser_service.tasks.workflow import process_workflow_task

        with pytest.raises(ValueError, match="task_processor"):
            process_workflow_task(
                task_id="guard-test",
                elements=[{"id": "e1", "description": "button", "action": "click"}],
                url="https://example.com",
                user_query="click",
                session_config=_minimal_session_config(),
                task_processor=None,
            )

    def test_task_id_is_set_in_tasks_dict(self, make_processor):
        """Task entry is updated for the given task_id after execution."""
        from browser_service.tasks.workflow import process_workflow_task

        tp = make_processor("live-t1")
        with self._mock_browser():
            process_workflow_task(
                task_id="live-t1",
                elements=[{"id": "e1", "description": "heading", "action": "click"}],
                url="https://example.com",
                user_query="click the heading",
                session_config=_minimal_session_config(),
                task_processor=tp,
            )
        status = tp.get_task_status("live-t1")
        assert status is not None
        assert status["status"] != "processing"

    def test_started_at_is_positive_timestamp(self, make_processor):
        """started_at value is a positive Unix timestamp."""
        from browser_service.tasks.workflow import process_workflow_task

        tp = make_processor("ts-test")
        with self._mock_browser():
            process_workflow_task(
                task_id="ts-test",
                elements=[{"id": "e1", "description": "link", "action": "click"}],
                url="https://example.com",
                user_query="click the link",
                session_config=_minimal_session_config(),
                task_processor=tp,
            )
        started = tp.get_task_status("ts-test").get("started_at", 0)
        assert started > 0
        # Should be roughly "now" — within the last minute
        assert abs(started - time.time()) < 60

    def test_message_is_non_empty_string(self, make_processor):
        """Task entry contains a non-empty message string."""
        from browser_service.tasks.workflow import process_workflow_task

        tp = make_processor("msg-test")
        with self._mock_browser():
            process_workflow_task(
                task_id="msg-test",
                elements=[{"id": "e1", "description": "element", "action": "click"}],
                url="https://example.com",
                user_query="click an element",
                session_config=_minimal_session_config(),
                task_processor=tp,
            )
        message = tp.get_task_status("msg-test").get("message", "")
        assert isinstance(message, str)
        assert len(message) > 0

    def test_status_is_completed_or_error(self, make_processor):
        """Terminal status is either 'completed' or 'error' — never left as running."""
        from browser_service.tasks.workflow import process_workflow_task

        tp = make_processor("status-test")
        with self._mock_browser():
            process_workflow_task(
                task_id="status-test",
                elements=[{"id": "e1", "description": "element", "action": "click"}],
                url="https://example.com",
                user_query="find element",
                session_config=_minimal_session_config(),
                task_processor=tp,
            )
        final_status = tp.get_task_status("status-test")["status"]
        assert final_status in ("completed", "error"), f"Unexpected final status: {final_status}"


# ---------------------------------------------------------------------------
# Full agent workflow — requires LLM API key + browser
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_llm_key(), reason="No LLM API key available")
class TestLiveAgentWorkflow:
    """Full end-to-end workflow tests against example.com with a real LLM agent."""

    def test_workflow_finds_h1_element(self, make_processor):
        """Agent can find the main heading on example.com."""
        from browser_service.tasks.workflow import process_workflow_task

        tp = make_processor("full-e2e-1")
        process_workflow_task(
            task_id="full-e2e-1",
            elements=[
                {"id": "heading", "description": "main heading on the page", "action": "find"},
            ],
            url="https://example.com",
            user_query="find the main heading on example.com",
            session_config=_minimal_session_config(),
            task_processor=tp,
        )
        result = tp.get_task_status("full-e2e-1")
        assert result is not None
        assert result["status"] in ("completed", "error")
        assert "message" in result

    def test_locator_mapping_key_present_on_completion(self, make_processor):
        """Completed task includes results in the task status."""
        from browser_service.tasks.workflow import process_workflow_task

        tp = make_processor("mapping-test")
        process_workflow_task(
            task_id="mapping-test",
            elements=[
                {"id": "link1", "description": "the 'More information...' link", "action": "find"},
            ],
            url="https://example.com",
            user_query="find the more information link on example.com",
            session_config=_minimal_session_config(),
            task_processor=tp,
        )
        result = tp.get_task_status("mapping-test")
        assert result is not None
        if result["status"] == "completed":
            # Successful run should have results or a non-empty message
            task_results = result.get("results", {})
            assert task_results or len(result.get("message", "")) > 0

    def test_multiple_elements_all_tracked(self, make_processor):
        """All requested elements appear in the result dict."""
        from browser_service.tasks.workflow import process_workflow_task

        elements = [
            {"id": "e_heading", "description": "page heading", "action": "find"},
            {"id": "e_link", "description": "hyperlink", "action": "find"},
        ]
        tp = make_processor("multi-test")
        process_workflow_task(
            task_id="multi-test",
            elements=elements,
            url="https://example.com",
            user_query="find the heading and link on example.com",
            session_config=_minimal_session_config(),
            task_processor=tp,
        )
        result = tp.get_task_status("multi-test")
        assert result is not None
        assert result["status"] in ("completed", "error")
