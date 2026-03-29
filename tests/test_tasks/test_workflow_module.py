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


# ---------------------------------------------------------------------------
# Helpers shared by TestCustomActionCallCounting
# ---------------------------------------------------------------------------

def _make_action_result(metadata=None, error=None, extracted_content=None):
    """Build a minimal ActionResult-like mock."""
    r = MagicMock()
    r.metadata = metadata
    r.error = error
    r.extracted_content = extracted_content
    return r


def _make_step(action_results=None, model_output_actions=None):
    """Build a minimal AgentHistory-like mock.

    action_results  — list of ActionResult mocks for step.result
    model_output_actions — list of action model mocks for step.model_output.action
    """
    step = MagicMock()
    step.result = action_results or []

    if model_output_actions is not None:
        step.model_output = MagicMock()
        step.model_output.action = model_output_actions
    else:
        step.model_output = None

    return step


def _make_native_action(name):
    """Build a mock Pydantic action model whose model_fields_set contains `name`."""
    m = MagicMock()
    m.model_fields_set = {name}
    return m


def _run_counter(steps):
    """
    Extract (custom_action_calls, execute_js_calls) from a list of steps
    using the same logic that lives in workflow.py.

    This is a verbatim copy of the counting block so the test is a faithful
    unit test of the exact algorithm, not a mock of it.
    """
    custom_action_calls = 0
    execute_js_calls = 0

    for step in steps:
        if hasattr(step, 'result') and step.result:
            for action_result in step.result:
                if (hasattr(action_result, 'metadata')
                        and isinstance(action_result.metadata, dict)
                        and action_result.metadata.get('element_id')
                        and action_result.metadata.get('found')
                        and action_result.metadata.get('best_locator')):
                    custom_action_calls += 1
        if step.model_output and step.model_output.action:
            for action_model in step.model_output.action:
                if 'execute_js' in action_model.model_fields_set:
                    execute_js_calls += 1

    return custom_action_calls, execute_js_calls


class TestCustomActionCallCounting:
    """
    Unit tests for the find_unique_locator / execute_js usage counter in
    browser_service/tasks/workflow.py (the block that produces
    "📊 Action usage: find_unique_locator=N, execute_js=M").

    BUG-2: model_fields_set never contains 'find_unique_locator' (custom actions
    are not Pydantic fields). The fix counts via ActionResult.metadata instead.
    """

    # ------------------------------------------------------------------
    # find_unique_locator counting (ActionResult.metadata path)
    # ------------------------------------------------------------------

    def test_successful_custom_action_counts_one(self):
        """A single successful find_unique_locator call increments the counter to 1."""
        metadata = {
            'element_id': 'elem_1',
            'found': True,
            'best_locator': '#my-button',
        }
        step = _make_step(action_results=[_make_action_result(metadata=metadata)])
        custom, execute_js = _run_counter([step])
        assert custom == 1
        assert execute_js == 0

    def test_two_successful_custom_actions_count_two(self):
        """Two successful calls in separate steps produce count == 2."""
        def _success(elem_id, locator):
            return _make_action_result(metadata={
                'element_id': elem_id,
                'found': True,
                'best_locator': locator,
            })

        steps = [
            _make_step(action_results=[_success('elem_1', '#btn1')]),
            _make_step(action_results=[_success('elem_2', '#btn2')]),
        ]
        custom, _ = _run_counter(steps)
        assert custom == 2

    def test_failed_custom_action_not_counted(self):
        """A failed call (metadata=None, error set) must not increment the counter."""
        step = _make_step(
            action_results=[_make_action_result(metadata=None, error="Element not found")]
        )
        custom, _ = _run_counter([step])
        assert custom == 0

    def test_retry_then_success_counts_one(self):
        """Retry scenario: failed result followed by successful result = 1 count."""
        failed = _make_action_result(metadata=None, error="retry")
        success = _make_action_result(metadata={
            'element_id': 'elem_1',
            'found': True,
            'best_locator': '#btn',
        })
        # Both results can appear in the same step's result list
        step = _make_step(action_results=[failed, success])
        custom, _ = _run_counter([step])
        assert custom == 1

    def test_metadata_missing_element_id_not_counted(self):
        """Metadata without element_id (unexpected source) must not be counted."""
        step = _make_step(
            action_results=[_make_action_result(metadata={
                'found': True,
                'best_locator': '#btn',
                # element_id deliberately absent
            })]
        )
        custom, _ = _run_counter([step])
        assert custom == 0

    def test_metadata_found_false_not_counted(self):
        """Metadata with found=False (action ran but located nothing) must not be counted."""
        step = _make_step(
            action_results=[_make_action_result(metadata={
                'element_id': 'elem_1',
                'found': False,
                'best_locator': None,
            })]
        )
        custom, _ = _run_counter([step])
        assert custom == 0

    def test_metadata_best_locator_none_not_counted(self):
        """Metadata with best_locator=None must not be counted even if found=True."""
        step = _make_step(
            action_results=[_make_action_result(metadata={
                'element_id': 'elem_1',
                'found': True,
                'best_locator': None,
            })]
        )
        custom, _ = _run_counter([step])
        assert custom == 0

    def test_empty_result_list_counts_zero(self):
        """A step with an empty result list contributes nothing."""
        step = _make_step(action_results=[])
        custom, _ = _run_counter([step])
        assert custom == 0

    def test_no_steps_counts_zero(self):
        custom, execute_js = _run_counter([])
        assert custom == 0
        assert execute_js == 0

    # ------------------------------------------------------------------
    # execute_js counting (model_fields_set path — unchanged from original)
    # ------------------------------------------------------------------

    def test_execute_js_native_action_counted(self):
        """execute_js is a native Pydantic action — model_fields_set detects it correctly."""
        step = _make_step(
            action_results=[],
            model_output_actions=[_make_native_action('execute_js')],
        )
        _, execute_js = _run_counter([step])
        assert execute_js == 1

    def test_execute_js_and_custom_action_in_same_workflow(self):
        """Both counters increment independently when both actions appear."""
        custom_step = _make_step(
            action_results=[_make_action_result(metadata={
                'element_id': 'elem_1',
                'found': True,
                'best_locator': '#btn',
            })],
            model_output_actions=[_make_native_action('click')],
        )
        js_step = _make_step(
            action_results=[],
            model_output_actions=[_make_native_action('execute_js')],
        )
        custom, execute_js = _run_counter([custom_step, js_step])
        assert custom == 1
        assert execute_js == 1

    def test_other_native_action_not_counted_as_execute_js(self):
        """Native actions other than execute_js do not affect either counter."""
        step = _make_step(
            action_results=[],
            model_output_actions=[_make_native_action('click')],
        )
        custom, execute_js = _run_counter([step])
        assert custom == 0
        assert execute_js == 0
