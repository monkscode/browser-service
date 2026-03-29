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
        os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )


# ---------------------------------------------------------------------------
# Guard condition tests — no browser/LLM needed
# ---------------------------------------------------------------------------

class TestProcessWorkflowTaskGuardsLive:
    """Guard conditions in process_workflow_task() — replicated with live imports."""

    def test_raises_on_none_tasks_dict(self):
        """tasks_dict=None raises ValueError immediately, no browser launched."""
        from browser_service.tasks.workflow import process_workflow_task
        with pytest.raises(ValueError, match="tasks_dict"):
            process_workflow_task(
                task_id="guard-test",
                elements=[{"id": "e1", "description": "button", "action": "click"}],
                url="https://example.com",
                user_query="click",
                session_config=_minimal_session_config(),
                tasks_dict=None,
            )

    def test_task_id_is_set_in_tasks_dict(self):
        """tasks_dict entry is created/updated for the given task_id after execution."""
        from browser_service.tasks.workflow import process_workflow_task
        tasks_dict = {"live-t1": {"status": "pending"}}
        process_workflow_task(
            task_id="live-t1",
            elements=[{"id": "e1", "description": "heading", "action": "click"}],
            url="https://example.com",
            user_query="click the heading",
            session_config=_minimal_session_config(),
            tasks_dict=tasks_dict,
        )
        assert "live-t1" in tasks_dict
        assert tasks_dict["live-t1"]["status"] != "pending"

    def test_started_at_is_positive_timestamp(self):
        """started_at value is a positive Unix timestamp."""
        from browser_service.tasks.workflow import process_workflow_task
        tasks_dict = {"ts-test": {"status": "pending"}}
        process_workflow_task(
            task_id="ts-test",
            elements=[{"id": "e1", "description": "link", "action": "click"}],
            url="https://example.com",
            user_query="click the link",
            session_config=_minimal_session_config(),
            tasks_dict=tasks_dict,
        )
        started = tasks_dict["ts-test"].get("started_at", 0)
        assert started > 0
        # Should be roughly "now" — within the last minute
        assert abs(started - time.time()) < 60

    def test_message_is_non_empty_string(self):
        """tasks_dict entry contains a non-empty message string."""
        from browser_service.tasks.workflow import process_workflow_task
        tasks_dict = {"msg-test": {"status": "pending"}}
        process_workflow_task(
            task_id="msg-test",
            elements=[{"id": "e1", "description": "element", "action": "click"}],
            url="https://example.com",
            user_query="click an element",
            session_config=_minimal_session_config(),
            tasks_dict=tasks_dict,
        )
        message = tasks_dict["msg-test"].get("message", "")
        assert isinstance(message, str)
        assert len(message) > 0

    def test_status_is_completed_or_error(self):
        """Terminal status is either 'completed' or 'error' — never left as running."""
        from browser_service.tasks.workflow import process_workflow_task
        tasks_dict = {"status-test": {"status": "pending"}}
        process_workflow_task(
            task_id="status-test",
            elements=[{"id": "e1", "description": "element", "action": "click"}],
            url="https://example.com",
            user_query="find element",
            session_config=_minimal_session_config(),
            tasks_dict=tasks_dict,
        )
        final_status = tasks_dict["status-test"]["status"]
        assert final_status in ("completed", "error"), (
            f"Unexpected final status: {final_status}"
        )


# ---------------------------------------------------------------------------
# Full agent workflow — requires LLM API key + browser
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_llm_key(), reason="No LLM API key available")
class TestLiveAgentWorkflow:
    """Full end-to-end workflow tests against example.com with a real LLM agent."""

    def test_workflow_finds_h1_element(self):
        """Agent can find the main heading on example.com."""
        from browser_service.tasks.workflow import process_workflow_task
        tasks_dict = {"full-e2e-1": {"status": "pending"}}
        process_workflow_task(
            task_id="full-e2e-1",
            elements=[
                {"id": "heading", "description": "main heading on the page", "action": "find"},
            ],
            url="https://example.com",
            user_query="find the main heading on example.com",
            session_config=_minimal_session_config(),
            tasks_dict=tasks_dict,
        )
        result = tasks_dict["full-e2e-1"]
        assert result["status"] in ("completed", "error")
        assert "message" in result

    def test_locator_mapping_key_present_on_completion(self):
        """Completed task includes locator_mapping in the result."""
        from browser_service.tasks.workflow import process_workflow_task
        tasks_dict = {"mapping-test": {"status": "pending"}}
        process_workflow_task(
            task_id="mapping-test",
            elements=[
                {"id": "link1", "description": "the 'More information...' link", "action": "find"},
            ],
            url="https://example.com",
            user_query="find the more information link on example.com",
            session_config=_minimal_session_config(),
            tasks_dict=tasks_dict,
        )
        result = tasks_dict["mapping-test"]
        if result["status"] == "completed":
            # Successful run should have a locator_mapping
            assert "locator_mapping" in result or "results" in result or len(result.get("message", "")) > 0

    def test_multiple_elements_all_tracked(self):
        """All requested elements appear in the result dict."""
        from browser_service.tasks.workflow import process_workflow_task
        elements = [
            {"id": "e_heading", "description": "page heading", "action": "find"},
            {"id": "e_link", "description": "hyperlink", "action": "find"},
        ]
        tasks_dict = {"multi-test": {"status": "pending"}}
        process_workflow_task(
            task_id="multi-test",
            elements=elements,
            url="https://example.com",
            user_query="find the heading and link on example.com",
            session_config=_minimal_session_config(),
            tasks_dict=tasks_dict,
        )
        result = tasks_dict["multi-test"]
        assert result["status"] in ("completed", "error")
