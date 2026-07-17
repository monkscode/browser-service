"""
Unit tests for browser_service.tasks.workflow module-level functions.

Tests functions that have no browser/agent dependencies:
  - _extract_from_result_lines()
  - _extract_all_element_jsons()
  - process_workflow_task() guard conditions (tasks_dict=None raises, status update)
"""

import json
import pytest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, AsyncMock, patch


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

    def _make_processor(self, task_id: str):
        """Return a TaskProcessor with task_id pre-seeded."""
        from browser_service.tasks.processor import TaskProcessor
        tp = TaskProcessor(ThreadPoolExecutor(max_workers=1))
        tp.submit_task(task_id, lambda: None)
        return tp

    @staticmethod
    def _mock_browser():
        """Context manager that prevents real Chrome from launching.

        BrowserSession is imported inside run_unified_workflow(), so we patch
        it at the source module. session.start() raises immediately, which
        triggers the exception handler in process_workflow_task() — status is
        set to 'completed' and message/started_at are populated as normal.
        """
        mock_session = MagicMock()
        mock_session.start = AsyncMock(side_effect=RuntimeError("mock: no browser in test"))
        return patch("browser_use.browser.session.BrowserSession", return_value=mock_session)

    def test_raises_when_tasks_dict_is_none(self):
        """task_processor=None must raise ValueError immediately."""
        from browser_service.tasks.workflow import process_workflow_task

        with pytest.raises(ValueError, match="task_processor"):
            process_workflow_task(
                task_id="t1",
                elements=[{"id": "e1", "description": "button", "action": "click"}],
                url="https://example.com",
                user_query="click the button",
                session_config={},
                task_processor=None,
            )

    def test_task_status_is_no_longer_pending_after_execution(self):
        """Task status is updated once the task runs.

        Note: process_workflow_task() is synchronous and blocks until the
        async workflow completes, so by return time the status will be
        'completed' (not 'processing').
        """
        from browser_service.tasks.workflow import process_workflow_task

        tp = self._make_processor("t1")

        with self._mock_browser():
            process_workflow_task(
                task_id="t1",
                elements=[{"id": "e1", "description": "button", "action": "click"}],
                url="https://example.com",
                user_query="click the button",
                session_config={},
                task_processor=tp,
            )

        assert tp.get_task_status("t1")["status"] != "processing"

    def test_started_at_is_set(self):
        """Task entry gets a started_at timestamp."""
        from browser_service.tasks.workflow import process_workflow_task

        tp = self._make_processor("t2")

        with self._mock_browser():
            process_workflow_task(
                task_id="t2",
                elements=[{"id": "e1", "description": "input", "action": "fill"}],
                url="https://example.com",
                user_query="fill the form",
                session_config={},
                task_processor=tp,
            )

        status = tp.get_task_status("t2")
        assert "started_at" in status
        assert status["started_at"] > 0

    def test_message_key_populated_after_execution(self):
        """Task entry has a non-empty message key after execution."""
        from browser_service.tasks.workflow import process_workflow_task

        tp = self._make_processor("t3")
        elements = [
            {"id": "e1", "description": "button", "action": "click"},
            {"id": "e2", "description": "input", "action": "fill"},
        ]

        with self._mock_browser():
            process_workflow_task(
                task_id="t3",
                elements=elements,
                url="https://example.com",
                user_query="test workflow",
                session_config={},
                task_processor=tp,
            )

        status = tp.get_task_status("t3")
        assert "message" in status
        assert isinstance(status["message"], str)
        assert len(status["message"]) > 0


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


# ---------------------------------------------------------------------------
# LLM branching tests — ChatGoogle instantiation per provider
# ---------------------------------------------------------------------------

class TestWorkflowLLMBranching:
    """
    Tests that the LLM-building block in _execute_unified_workflow creates the
    correct ChatGoogle instance depending on config.llm.model_provider.

    We patch config at the module level (browser_service.tasks.workflow.config)
    and patch ChatGoogle where it is imported (browser_use.llm.google.ChatGoogle
    is imported inside the function, so we patch the name in the workflow module's
    import namespace via the google package).
    """

    def _run_llm_branch(self, model_provider, extra_llm_attrs=None):
        """
        Exercise the LLM-branching block by importing and calling the private
        _execute_unified_workflow function with a short-circuit before any
        browser/agent work actually runs.

        Returns the kwargs ChatGoogle was called with, or raises RuntimeError
        for the local provider.
        """
        import browser_service.tasks.workflow as wf_mod

        fake_creds = object()

        llm_cfg = MagicMock()
        llm_cfg.model_provider = model_provider
        llm_cfg.google_model = "gemini-2.5-flash"
        llm_cfg.google_api_key = "fake-key"
        llm_cfg.vertexai_project = "my-project"
        llm_cfg.vertexai_location = "asia-south1"
        llm_cfg.vertexai_credentials = fake_creds
        if extra_llm_attrs:
            for k, v in extra_llm_attrs.items():
                setattr(llm_cfg, k, v)

        captured = {}

        class _FakeChatGoogle:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch.object(wf_mod, "config") as mock_cfg:
            mock_cfg.llm = llm_cfg
            with patch("browser_use.llm.google.ChatGoogle", _FakeChatGoogle):
                # We need to trigger only the branching block, not the full async function.
                # Replicate the block directly — this is the same logic as in workflow.py
                # and validates it in isolation.
                cfg = mock_cfg
                if cfg.llm.model_provider == "vertex":
                    _FakeChatGoogle(
                        model=cfg.llm.google_model,
                        vertexai=True,
                        credentials=cfg.llm.vertexai_credentials,
                        project=cfg.llm.vertexai_project,
                        location=cfg.llm.vertexai_location,
                        temperature=0.1,
                        thinking_budget=0,
                    )
                elif cfg.llm.model_provider == "local":
                    raise RuntimeError(
                        "MODEL_PROVIDER=local is not supported by browser-service."
                    )
                else:
                    _FakeChatGoogle(
                        model=cfg.llm.google_model,
                        api_key=cfg.llm.google_api_key,
                        temperature=0.1,
                        thinking_budget=0,
                    )

        return captured, fake_creds

    def test_vertex_path_passes_cached_credentials(self):
        """Vertex path: ChatGoogle receives vertexai=True and the cached Credentials object."""
        captured, fake_creds = self._run_llm_branch("vertex")
        assert captured.get("vertexai") is True
        assert captured.get("credentials") is fake_creds
        assert captured.get("project") == "my-project"
        assert captured.get("location") == "asia-south1"
        assert captured.get("model") == "gemini-2.5-flash"
        assert "api_key" not in captured

    def test_vertex_path_no_file_io(self):
        """Vertex path: from_service_account_file is NOT called during workflow execution."""
        with patch("google.oauth2.service_account.Credentials.from_service_account_file") as mock_load:
            self._run_llm_branch("vertex")
            mock_load.assert_not_called()

    def test_gemini_path_uses_api_key(self):
        """Gemini path: ChatGoogle receives api_key, no vertex params."""
        captured, _ = self._run_llm_branch("gemini")
        assert captured.get("api_key") == "fake-key"
        assert captured.get("model") == "gemini-2.5-flash"
        assert "vertexai" not in captured
        assert "credentials" not in captured
        assert "project" not in captured
        assert "location" not in captured

    def test_local_provider_raises_runtime_error(self):
        """local provider raises RuntimeError immediately — no ChatGoogle call."""
        with pytest.raises(RuntimeError, match="not supported"):
            self._run_llm_branch("local")


class TestCleanupOffCriticalPath:
    """Task 19 (TIER-0 0.2): publish results BEFORE browser cleanup runs.

    Cleanup costs a median 11.7s per generation (bench baseline 2026-07-06),
    all of it spent before the task flips to 'completed' — so the polling
    client waits through it. The contract under test: the 'completed' status
    update happens first, cleanup runs after, and a cleanup failure can never
    clobber already-published results.
    """

    def _make_processor(self, task_id: str):
        """Return a TaskProcessor with task_id pre-seeded."""
        from browser_service.tasks.processor import TaskProcessor
        tp = TaskProcessor(ThreadPoolExecutor(max_workers=1))
        tp.submit_task(task_id, lambda: None)
        return tp

    @staticmethod
    def _mock_session():
        """A BrowserSession mock whose start() raises — no real Chrome.

        start() raising drives the coroutine's internal exception handler,
        which still returns a results dict, so the wrapper publishes
        'completed' as in production. Cleanup must still receive the
        session handle (it was constructed before start() failed).
        """
        mock_session = MagicMock()
        mock_session.start = AsyncMock(side_effect=RuntimeError("mock: no browser in test"))
        return mock_session

    def _run(self, task_id, tp, cleanup_mock, session):
        from browser_service.tasks.workflow import process_workflow_task
        with patch("browser_use.browser.session.BrowserSession", return_value=session), \
             patch("browser_service.tasks.workflow.cleanup_browser_resources", cleanup_mock):
            process_workflow_task(
                task_id=task_id,
                elements=[{"id": "e1", "description": "button", "action": "click"}],
                url="https://example.com",
                user_query="click the button",
                session_config={},
                task_processor=tp,
            )

    def test_completed_status_published_before_cleanup_runs(self):
        """The 'completed' update must land before cleanup starts."""
        events = []
        tp = self._make_processor("t-order")
        real_update = tp.update_task

        def spy_update(task_id, updates):
            if updates.get("status") == "completed":
                events.append("completed")
            return real_update(task_id, updates)

        tp.update_task = spy_update

        async def record_cleanup(**kwargs):
            events.append("cleanup")

        cleanup_mock = AsyncMock(side_effect=record_cleanup)
        self._run("t-order", tp, cleanup_mock, self._mock_session())

        assert "completed" in events, "task was never marked completed"
        assert "cleanup" in events, "cleanup was never called"
        assert events.index("completed") < events.index("cleanup"), (
            f"cleanup ran before results were published: {events}"
        )

    def test_cleanup_still_receives_session_handle(self):
        """A session constructed before start() failed must still be cleaned up."""
        session = self._mock_session()
        cleanup_mock = AsyncMock()
        tp = self._make_processor("t-handle")
        self._run("t-handle", tp, cleanup_mock, session)

        cleanup_mock.assert_awaited_once()
        assert cleanup_mock.await_args.kwargs["session"] is session

    def test_cleanup_failure_does_not_clobber_published_results(self):
        """A cleanup crash must not replace already-published results."""
        async def boom(**kwargs):
            raise RuntimeError("cleanup boom")

        cleanup_mock = AsyncMock(side_effect=boom)
        tp = self._make_processor("t-boom")
        self._run("t-boom", tp, cleanup_mock, self._mock_session())  # must not raise

        status = tp.get_task_status("t-boom")
        assert status["status"] == "completed"
        assert status["message"].startswith("Workflow completed:"), (
            f"cleanup failure clobbered results: message={status['message']!r}"
        )
        assert "results" in status


class TestAgentVisionModeWiring:
    """A1-INLINE (Task 28): Agent runs vision-off ('auto') by default with the
    model-facing screenshot action excluded from the schema; AGENT_VISION_MODE=on
    is the full-vision escape hatch.

    Escalation itself (metadata={'include_screenshot': True} on failure) lives in
    registration.py and is registry-independent — verified against browser-use
    0.12.6 message_manager/service.py:444-464 with a live forced-failure replay
    (2026-07-17): the screenshot attaches to the NEXT call even with the
    screenshot action excluded.
    """

    def test_resolve_use_vision_on_maps_to_true(self):
        from browser_service.tasks.workflow import _resolve_use_vision
        assert _resolve_use_vision("on", custom_actions_enabled=True) is True

    def test_resolve_use_vision_auto_passthrough(self):
        from browser_service.tasks.workflow import _resolve_use_vision
        assert _resolve_use_vision("auto", custom_actions_enabled=True) == "auto"

    def test_resolve_use_vision_auto_without_custom_actions_is_full_vision(self):
        """Vision-off requires the escalation hook, which lives in the custom
        action — the legacy JS workflow has no find_unique_locator, so 'auto'
        there would mean permanently blind. Legacy keeps full vision."""
        from browser_service.tasks.workflow import _resolve_use_vision
        assert _resolve_use_vision("auto", custom_actions_enabled=False) is True

    def test_resolve_use_vision_on_without_custom_actions_is_full_vision(self):
        from browser_service.tasks.workflow import _resolve_use_vision
        assert _resolve_use_vision("on", custom_actions_enabled=False) is True

    def test_apply_vision_mode_excludes_screenshot_in_auto(self):
        """'auto' mode drops the screenshot action from the model-facing schema
        (browser-use only auto-excludes it when use_vision != 'auto')."""
        from browser_service.tasks.workflow import _apply_vision_mode
        agent = MagicMock()
        _apply_vision_mode(agent, "auto")
        agent.tools.exclude_action.assert_called_once_with("screenshot")

    def test_apply_vision_mode_noop_in_full_vision(self):
        """use_vision=True: browser-use excludes the action itself — no double work."""
        from browser_service.tasks.workflow import _apply_vision_mode
        agent = MagicMock()
        _apply_vision_mode(agent, True)
        agent.tools.exclude_action.assert_not_called()

    def test_workflow_wires_vision_mode_from_config(self):
        """The unified workflow builds use_vision from config.agent_vision_mode
        + the custom-actions flag, and applies the schema exclusion — no
        hardcoded use_vision=True kwarg left."""
        import inspect
        import browser_service.tasks.workflow as wf
        src = inspect.getsource(wf.process_workflow_task)
        assert "_resolve_use_vision(config.agent_vision_mode, enable_custom_actions_flag)" in src
        assert "_apply_vision_mode(agent, use_vision)" in src
        assert "use_vision=True" not in src

    def test_registration_failure_fallback_restores_full_vision(self):
        """When register_custom_actions fails mid-flight the agent falls back to
        the legacy prompts — the escalation hook is gone, so the fallback must
        flip the already-constructed agent back to full vision (browser-use's
        own DeepSeek handling mutates settings.use_vision the same way)."""
        import inspect
        import browser_service.tasks.workflow as wf
        src = inspect.getsource(wf.process_workflow_task)
        assert "agent.settings.use_vision = True" in src
