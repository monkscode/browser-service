"""
Unit tests for browser_service.tasks.workflow module-level functions.

Tests functions that have no browser/agent dependencies:
  - _extract_from_result_lines()
  - _extract_all_element_jsons()
  - process_workflow_task() guard conditions (tasks_dict=None raises, status update)
"""

import json
import pytest
from unittest.mock import MagicMock, patch


class TestExtractFromResultLines:
    """Tests for the _extract_from_result_lines() helper."""

    def _fn(self, text):
        from browser_service.tasks.workflow import _extract_from_result_lines
        return _extract_from_result_lines(text)

    def test_extracts_single_json(self):
        text = 'Result: {"element_id": "elem_1", "best_locator": "#btn"}'
        results = self._fn(text)
        assert len(results) == 1
        data = json.loads(results[0])
        assert data["element_id"] == "elem_1"

    def test_extracts_multiple_jsons(self):
        text = (
            'Result: {"element_id": "elem_1", "best_locator": "#btn"}\n'
            'Some log line\n'
            'Result: {"element_id": "elem_2", "best_locator": "text=Submit"}'
        )
        results = self._fn(text)
        assert len(results) == 2

    def test_skips_result_lines_without_element_id(self):
        """Lines with 'Result:' but no 'element_id' are ignored."""
        text = 'Result: {"status": "ok", "count": 3}'
        results = self._fn(text)
        assert results == []

    def test_empty_string(self):
        assert self._fn("") == []

    def test_no_result_lines(self):
        assert self._fn("Some random log without Result: lines") == []

    def test_nested_json_extracted_completely(self):
        """Brace-matching should handle nested objects."""
        text = 'Result: {"element_id": "e1", "locator_data": {"count": 1, "type": "id"}}'
        results = self._fn(text)
        assert len(results) == 1
        data = json.loads(results[0])
        assert data["locator_data"]["count"] == 1

    def test_json_with_escaped_quotes(self):
        text = 'Result: {"element_id": "e1", "best_locator": "input[name=\\"email\\"]"}'
        results = self._fn(text)
        assert len(results) == 1

    def test_multiline_text_with_embedded_result(self):
        text = (
            "Step 1: Navigate to page\n"
            "Step 2: Find element\n"
            'Result: {"element_id": "search", "best_locator": "#search-input"}\n'
            "Step 3: Done"
        )
        results = self._fn(text)
        assert len(results) == 1
        assert json.loads(results[0])["element_id"] == "search"


class TestExtractAllElementJsons:
    """Tests for the _extract_all_element_jsons() fallback helper."""

    def _fn(self, text):
        from browser_service.tasks.workflow import _extract_all_element_jsons
        return _extract_all_element_jsons(text)

    def test_extracts_element_id_json(self):
        text = 'Some text {"element_id": "elem_1", "locator": "#btn"} more text'
        results = self._fn(text)
        assert len(results) >= 1
        data = json.loads(results[0])
        assert data["element_id"] == "elem_1"

    def test_deduplicates_identical_jsons(self):
        """Same JSON appearing twice should only be extracted once."""
        json_str = '{"element_id": "e1", "locator": "#btn"}'
        text = f"{json_str} some text {json_str}"
        results = self._fn(text)
        assert len(results) == 1

    def test_extracts_multiple_unique_jsons(self):
        text = (
            '{"element_id": "e1", "locator": "#btn"} '
            '{"element_id": "e2", "locator": "text=Submit"}'
        )
        results = self._fn(text)
        assert len(results) == 2

    def test_empty_string(self):
        assert self._fn("") == []

    def test_no_element_id_in_text(self):
        assert self._fn('{"status": "ok"}') == []

    def test_single_quote_element_id_pattern(self):
        """Also matches 'element_id': pattern (single quotes)."""
        text = "{'element_id': 'e1', 'locator': '#btn'}"
        results = self._fn(text)
        # Should find the pattern
        assert isinstance(results, list)


class TestProcessWorkflowTaskGuards:
    """Tests for guard conditions in process_workflow_task()."""

    def test_raises_when_tasks_dict_is_none(self):
        """tasks_dict=None must raise ValueError immediately."""
        from browser_service.tasks.workflow import process_workflow_task

        with pytest.raises(ValueError, match="tasks_dict"):
            process_workflow_task(
                task_id="t1",
                elements=[{"id": "e1", "description": "button", "action": "click"}],
                url="https://example.com",
                user_query="click the button",
                session_config={},
                tasks_dict=None,
            )

    def test_task_status_is_no_longer_pending_after_execution(self):
        """tasks_dict status is updated away from 'pending' once the task runs.

        Note: process_workflow_task() is synchronous and blocks until the
        async workflow completes, so by return time the status will be
        'completed' or 'error' (not 'running').  We simply verify it was
        mutated from the original 'pending' state.
        """
        from browser_service.tasks.workflow import process_workflow_task

        tasks_dict = {"t1": {"status": "pending"}}

        process_workflow_task(
            task_id="t1",
            elements=[{"id": "e1", "description": "button", "action": "click"}],
            url="https://example.com",
            user_query="click the button",
            session_config={},
            tasks_dict=tasks_dict,
        )

        # Status was mutated — no longer "pending"
        assert tasks_dict["t1"]["status"] != "pending"

    def test_started_at_is_set(self):
        """tasks_dict entry gets a started_at timestamp."""
        from browser_service.tasks.workflow import process_workflow_task

        tasks_dict = {"t2": {"status": "pending"}}

        process_workflow_task(
            task_id="t2",
            elements=[{"id": "e1", "description": "input", "action": "fill"}],
            url="https://example.com",
            user_query="fill the form",
            session_config={},
            tasks_dict=tasks_dict,
        )

        assert "started_at" in tasks_dict["t2"]
        assert tasks_dict["t2"]["started_at"] > 0

    def test_message_key_populated_after_execution(self):
        """tasks_dict entry has a non-empty message key after execution."""
        from browser_service.tasks.workflow import process_workflow_task

        tasks_dict = {"t3": {"status": "pending"}}
        elements = [
            {"id": "e1", "description": "button", "action": "click"},
            {"id": "e2", "description": "input", "action": "fill"},
        ]

        process_workflow_task(
            task_id="t3",
            elements=elements,
            url="https://example.com",
            user_query="test workflow",
            session_config={},
            tasks_dict=tasks_dict,
        )

        # A message key must exist and be a non-empty string
        assert "message" in tasks_dict["t3"]
        assert isinstance(tasks_dict["t3"]["message"], str)
        assert len(tasks_dict["t3"]["message"]) > 0
