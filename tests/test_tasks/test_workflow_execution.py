"""
Unit tests for browser_service.tasks.workflow — process_workflow_task end to end.

Purpose: process_workflow_task is the whole browser-service pipeline in one
         function: open a session, build the prompt, run the browser-use agent,
         pull locator results out of custom-action metadata, re-rank them,
         validate, and report metrics. It is ~1800 lines and was the single
         largest untested surface in the package — the existing workflow tests
         cover only the module-level JSON helpers and the task_processor guard.

Harness: run_unified_workflow is nested inside process_workflow_task and reaches
         for browser-use at call time, so nothing can be imported and tested in
         isolation. workflow_harness substitutes every external edge — the
         browser session, the LLM, the Agent, custom-action registration, client
         config, prompt builders, cleanup and metrics — and hands back the knobs.
         Tests then drive real code paths through it rather than asserting on
         mocks: the agent history fed in is shaped exactly like browser-use's
         AgentHistoryList, so the extraction and re-ranking logic under test is
         the production logic.

Tests:
  - Task status transitions to running then completed, with the result attached
  - A missing task_processor is rejected before any browser work starts
  - Locator results are extracted from custom-action metadata
  - An element the agent never reports is backfilled as failed, success False
  - A found=False payload is counted as a failure rather than dropped
  - The engine's own failure reason survives into the emitted result
  - A found payload with no locator is a failure on every extraction path
  - A success wins over a failure for the same element, in either order
  - A locator for an unrequested id does not stand in for a missing one
  - Re-ranking promotes the stable id locator over a volatile xpath
  - max_steps scales with element count
  - Vision mode: 'auto' excludes the screenshot action, 'on' does not
  - Failed custom-action registration falls back to legacy prompts with vision
  - Token usage is read from agent history, and from the fallback service
  - Provider selection: gemini, vertex, and the unsupported local provider
  - Agent failure is reported as a failed task rather than raised
  - Cleanup runs after the task is marked complete, never before
  - Metrics are recorded for a root workflow and skipped for a child
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# --------------------------------------------------------------------------
# Doubles shaped like the browser-use objects the function actually reads
# --------------------------------------------------------------------------


class FakeUsage:
    """Stands in for browser-use's UsageSummary."""

    def __init__(self, prompt=100, completion=50, total=150, cached=10, cost=0.0021):
        self.total_prompt_tokens = prompt
        self.total_completion_tokens = completion
        self.total_tokens = total
        self.total_prompt_cached_tokens = cached
        self.total_cost = cost


class FakeActionResult:
    """One ActionResult; custom actions publish their locator payload in metadata."""

    def __init__(self, metadata=None):
        self.metadata = metadata


class FakeModelOutput:
    def __init__(self, actions):
        self.action = actions


class FakeStep:
    """One AgentHistory entry."""

    def __init__(self, results=None, actions=None):
        self.result = list(results or [])
        self.model_output = FakeModelOutput(actions) if actions is not None else None


class FakeHistory:
    """Stands in for AgentHistoryList."""

    def __init__(self, steps=None, usage=None):
        self.history = list(steps or [])
        self.usage = usage


class FakeTools:
    def __init__(self):
        self.excluded = []

    def exclude_action(self, name):
        self.excluded.append(name)


class FakeAgent:
    """Stands in for browser_use.Agent.

    A real class rather than a MagicMock because the function under test makes
    several hasattr() decisions — on `tools`, `_pw_teardown` and
    `token_cost_service` — that a MagicMock would answer True to unconditionally,
    silently routing tests down branches they did not intend to exercise.
    """

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.task = kwargs.get("task")
        self.override_system_message = kwargs.get("override_system_message")
        self.settings = MagicMock()
        self.settings.use_vision = kwargs.get("use_vision")
        self.tools = FakeTools()
        self.run_calls = []
        self.result = FakeHistory()
        self.run_error = None

    async def run(self, max_steps=None):
        self.run_calls.append(max_steps)
        if self.run_error:
            raise self.run_error
        return self.result


class FakeSession:
    """Stands in for browser_use's BrowserSession."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.started = False
        self.llm_screenshot_size = None
        self.start_error = None

    async def start(self):
        if self.start_error:
            raise self.start_error
        self.started = True


class FakeClientConfig:
    name = "Default"
    minimum_wait_page_load_time = 0.5
    wait_for_network_idle_page_load_time = 1.0
    wait_between_actions = 0.5
    system_prompt_additions = []


class Harness:
    """Handles a test uses to steer a run and inspect what happened."""

    def __init__(self):
        self.agents = []
        self.sessions = []
        self.register_result = True
        self.cleanup = AsyncMock()
        self.record_metrics = MagicMock()
        self.task_processor = MagicMock()
        self.updates = []

        def _update(task_id, payload):
            # Snapshot rather than store by reference: the caller reuses dicts,
            # and the ordering assertions below depend on point-in-time values.
            self.updates.append((task_id, dict(payload)))

        self.task_processor.update_task.side_effect = _update

    @property
    def agent(self):
        return self.agents[-1]

    @property
    def session(self):
        return self.sessions[-1]

    def statuses(self):
        return [payload.get("status") for _, payload in self.updates]


def loc(locator, type_, unique=True, valid=True, **extra):
    """One candidate locator.

    unique and valid are not decoration: the re-ranker drops any candidate
    missing either flag before scoring, so a fixture without them re-ranks an
    empty list and silently keeps the original winner.
    """
    return {"locator": locator, "type": type_, "unique": unique, "valid": valid, **extra}


def locator_metadata(element_id, best_locator, all_locators=None, found=True):
    """A custom-action metadata payload as registration.py publishes it."""
    return {
        "element_id": element_id,
        "found": found,
        "best_locator": best_locator,
        "all_locators": all_locators if all_locators is not None else [loc(best_locator, "id")],
    }


@pytest.fixture
def workflow_harness():
    """Substitute every external edge of run_unified_workflow.

    Yields a Harness. Defaults describe the common case — gemini provider,
    custom actions registering successfully, vision mode 'auto' — and each test
    overrides only the axis it is about.

    browser_use.llm.google and browser_use.browser.session are replaced with
    stub modules in sys.modules rather than patched in place. Importing the real
    browser_use.llm.google pulls google.genai -> mcp.types, which under coverage
    tracing dies with KeyError: 'pydantic.root_model' — pydantic's
    _get_caller_frame_info(depth=3) misreads the stack once the tracer adds
    frames. The rest of the suite never imports it, so this file would otherwise
    be the one place the whole run breaks under --cov. run_unified_workflow
    imports both lazily from inside the function, so a stub is all it ever sees;
    nothing here needs the real classes, since both are substituted anyway.
    """
    from browser_service.tasks import workflow as wf

    harness = Harness()

    def make_agent(**kwargs):
        agent = FakeAgent(**kwargs)
        harness.agents.append(agent)
        return agent

    def make_session(**kwargs):
        session = FakeSession(**kwargs)
        harness.sessions.append(session)
        return session

    session_stub = types.ModuleType("browser_use.browser.session")
    session_stub.BrowserSession = MagicMock(side_effect=make_session)
    google_stub = types.ModuleType("browser_use.llm.google")
    google_stub.ChatGoogle = MagicMock()
    harness.browser_session_cls = session_stub.BrowserSession
    harness.chat_google = google_stub.ChatGoogle

    fake_config = MagicMock()
    fake_config.headless = True
    fake_config.enable_custom_actions = True
    fake_config.agent_vision_mode = "auto"
    fake_config.llm.model_provider = "gemini"
    fake_config.llm.google_model = "gemini-2.5-flash"
    fake_config.llm.google_api_key = "test-key"

    with (
        patch.dict(
            sys.modules,
            {
                "browser_use.browser.session": session_stub,
                "browser_use.llm.google": google_stub,
            },
        ),
        patch.object(wf, "Agent", side_effect=make_agent),
        patch.object(wf, "config", fake_config),
        patch.object(
            wf, "register_custom_actions", side_effect=lambda *a, **k: harness.register_result
        ),
        patch.object(wf, "capture_session_pid", return_value=4242),
        patch.object(wf, "cleanup_browser_resources", harness.cleanup),
        patch.object(wf, "get_client_config", return_value=FakeClientConfig()),
        patch.object(wf, "build_workflow_prompt", return_value="WORKFLOW PROMPT"),
        patch.object(wf, "build_system_prompt", return_value="SYSTEM PROMPT"),
        patch.object(wf, "record_workflow_metrics", harness.record_metrics),
        patch.object(wf, "_nl_settings", None),
    ):
        harness.config = fake_config
        yield harness


def run_workflow(harness, elements=None, **kwargs):
    """Invoke process_workflow_task with harness defaults."""
    from browser_service.tasks.workflow import process_workflow_task

    elements = elements if elements is not None else [{"id": "elem_1", "description": "search box"}]
    params = {
        "task_id": "task-1",
        "elements": elements,
        "url": "https://example.test/search",
        "user_query": "find the search box",
        "session_config": {},
        "task_processor": harness.task_processor,
    }
    params.update(kwargs)
    process_workflow_task(**params)
    return harness


def arm(harness, steps, usage=None):
    """Pre-load the history the fake agent will return.

    The Agent is constructed inside the function under test, so its result has
    to be installed via the factory rather than up front.
    """
    from browser_service.tasks import workflow as wf

    original = wf.Agent.side_effect

    def make_and_arm(**kwargs):
        agent = original(**kwargs)
        agent.result = FakeHistory(steps, usage)
        return agent

    wf.Agent.side_effect = make_and_arm


class TestTaskLifecycle:
    """Tests for status transitions and the guard on task_processor."""

    def test_requires_task_processor(self):
        """Without a processor there is nothing to report to — fail before opening a browser."""
        from browser_service.tasks.workflow import process_workflow_task

        with pytest.raises(ValueError, match="task_processor"):
            process_workflow_task(
                task_id="t",
                elements=[],
                url="https://example.test",
                user_query="q",
                session_config={},
                task_processor=None,
            )

    def test_marks_running_then_completed(self, workflow_harness):
        """The task goes running before the browser opens and completed after results land."""
        arm(workflow_harness, [FakeStep([FakeActionResult(locator_metadata("elem_1", "id=q"))])])

        run_workflow(workflow_harness)

        assert workflow_harness.statuses() == ["running", "completed"]

    def test_completed_update_carries_results(self, workflow_harness):
        """The completed payload contains the results the poller will read."""
        arm(workflow_harness, [FakeStep([FakeActionResult(locator_metadata("elem_1", "id=q"))])])

        run_workflow(workflow_harness)

        _, completed = workflow_harness.updates[-1]
        assert completed["results"]["summary"]["successful"] == 1
        assert completed["results"]["results"][0]["best_locator"] == "id=q"


class TestResultExtraction:
    """Tests for pulling locator payloads out of custom-action metadata."""

    def test_extracts_locator_from_metadata(self, workflow_harness):
        """The primary path reads ActionResult.metadata, not the agent's text output."""
        arm(
            workflow_harness,
            [FakeStep([FakeActionResult(locator_metadata("elem_1", "id=search"))])],
        )

        run_workflow(workflow_harness)

        results = workflow_harness.updates[-1][1]["results"]
        assert results["success"] is True
        assert results["results"][0]["element_id"] == "elem_1"

    def test_all_requested_elements_found_reports_success(self, workflow_harness):
        """The success flag still means what it says when nothing is missing."""
        arm(
            workflow_harness,
            [
                FakeStep(
                    [
                        FakeActionResult(locator_metadata("elem_1", "id=search")),
                        FakeActionResult(locator_metadata("elem_2", "id=submit")),
                    ]
                )
            ],
        )

        run_workflow(
            workflow_harness,
            elements=[
                {"id": "elem_1", "description": "search box"},
                {"id": "elem_2", "description": "submit button"},
            ],
        )

        results = workflow_harness.updates[-1][1]["results"]
        assert results["success"] is True
        assert results["summary"]["successful"] == 2
        assert results["summary"]["failed"] == 0

    def test_silently_dropped_element_reports_failure(self, workflow_harness):
        """An element the agent never reported is a missing locator, not a success.

        When the agent never calls find_unique_locator for elem_2 it emits no
        metadata for it, so extraction produces no entry. The element is backfilled
        as found=False and `success` is measured against the requested element ids,
        so half the locators missing reports success False — matching the summary,
        which already carried the truth (total_elements 2, successful 1).
        """
        arm(
            workflow_harness,
            [FakeStep([FakeActionResult(locator_metadata("elem_1", "id=search"))])],
        )

        run_workflow(
            workflow_harness,
            elements=[
                {"id": "elem_1", "description": "search box"},
                {"id": "elem_2", "description": "submit button"},
            ],
        )

        results = workflow_harness.updates[-1][1]["results"]
        assert results["summary"]["total_elements"] == 2
        assert results["summary"]["successful"] == 1
        assert results["summary"]["success_rate"] == 0.5
        assert results["summary"]["failed"] == 1
        assert results["success"] is False
        dropped = [r for r in results["results"] if r["element_id"] == "elem_2"]
        assert dropped and dropped[0]["found"] is False

    def test_not_found_payload_reason_is_preserved(self, workflow_harness):
        """The engine's own diagnosis survives into the emitted result.

        The rejected payload is the only place the reason exists — the backfill
        would otherwise report a generic 'no result' for an element the agent
        diagnosed precisely.
        """
        rejected = {
            "element_id": "elem_2",
            "found": False,
            "error": "Semantic mismatch: expected 'Submit'",
            "error_type": "SemanticMismatch",
            "semantic_match": False,
        }
        arm(
            workflow_harness,
            [
                FakeStep(
                    [
                        FakeActionResult(locator_metadata("elem_1", "id=search")),
                        FakeActionResult(rejected),
                    ]
                )
            ],
        )

        run_workflow(
            workflow_harness,
            elements=[
                {"id": "elem_1", "description": "search box"},
                {"id": "elem_2", "description": "submit button"},
            ],
        )

        results = workflow_harness.updates[-1][1]["results"]
        entry = next(r for r in results["results"] if r["element_id"] == "elem_2")
        assert entry["found"] is False
        assert entry["error"] == "Semantic mismatch: expected 'Submit'"
        assert entry["error_type"] == "SemanticMismatch"
        assert entry["semantic_match"] is False
        # The request's own description stands; the payload cannot rewrite it.
        assert entry["description"] == "submit button"

    def test_failed_element_contributes_no_approach_metrics(self, workflow_harness):
        """element_approach_metrics stays what it has always been: located elements.

        The consuming chart buckets each entry by fallback_depth to show which
        strategy tier resolved it. A failure stamps depth 7 without having
        resolved anything, so it must not enter that series.
        """
        arm(
            workflow_harness,
            [
                FakeStep(
                    [
                        FakeActionResult(
                            {
                                "element_id": "elem_1",
                                "found": False,
                                "error": "coordinates landed on BODY",
                                "approach_metrics": {
                                    "locator_approach": "coordinate_fallback",
                                    "fallback_depth": 7,
                                    "success": False,
                                },
                            }
                        )
                    ]
                )
            ],
        )

        run_workflow(workflow_harness)

        results = workflow_harness.updates[-1][1]["results"]
        assert results["results"][0]["error"] == "coordinates landed on BODY"
        assert "approach_metrics" not in results["results"][0]
        assert results["summary"]["element_approach_metrics"] == []

    def test_unreported_element_gets_the_generic_reason(self, workflow_harness):
        """No payload means no diagnosis to preserve — say exactly that."""
        arm(
            workflow_harness,
            [FakeStep([FakeActionResult(locator_metadata("elem_1", "id=search"))])],
        )

        run_workflow(
            workflow_harness,
            elements=[
                {"id": "elem_1", "description": "search box"},
                {"id": "elem_2", "description": "submit button"},
            ],
        )

        results = workflow_harness.updates[-1][1]["results"]
        entry = next(r for r in results["results"] if r["element_id"] == "elem_2")
        assert entry["error"] == "No locator reported by the agent"

    def test_found_payload_without_locator_is_a_failure(self, workflow_harness):
        """found=True with no best_locator is rejected — and must not count as found.

        The extractor's gate requires both, so such a payload is dropped. It has
        to come back as failed, never as a found entry that carries no locator.
        """
        arm(
            workflow_harness,
            [
                FakeStep(
                    [
                        FakeActionResult(
                            {"element_id": "elem_1", "found": True, "error": "locator went missing"}
                        )
                    ]
                )
            ],
        )

        run_workflow(workflow_harness)

        results = workflow_harness.updates[-1][1]["results"]
        assert results["success"] is False
        assert results["summary"]["successful"] == 0
        assert results["results"][0]["found"] is False
        assert results["results"][0]["error"] == "locator went missing"

    def test_later_success_wins_over_an_earlier_failure(self, workflow_harness):
        """A retry that succeeds must not be overwritten by the failure before it."""
        arm(
            workflow_harness,
            [
                FakeStep(
                    [FakeActionResult({"element_id": "elem_1", "found": False, "error": "no dice"})]
                ),
                FakeStep([FakeActionResult(locator_metadata("elem_1", "id=search"))]),
            ],
        )

        run_workflow(workflow_harness)

        results = workflow_harness.updates[-1][1]["results"]
        assert results["success"] is True
        assert len(results["results"]) == 1
        assert results["results"][0]["found"] is True
        assert results["results"][0]["best_locator"] == "id=search"
        assert "error" not in results["results"][0]

    def test_failure_after_a_success_does_not_undo_it(self, workflow_harness):
        """The reverse order: a late failure must not displace a locator already found."""
        arm(
            workflow_harness,
            [
                FakeStep([FakeActionResult(locator_metadata("elem_1", "id=search"))]),
                FakeStep(
                    [FakeActionResult({"element_id": "elem_1", "found": False, "error": "no dice"})]
                ),
            ],
        )

        run_workflow(workflow_harness)

        results = workflow_harness.updates[-1][1]["results"]
        assert results["success"] is True
        assert len(results["results"]) == 1
        assert results["results"][0]["best_locator"] == "id=search"

    def test_found_false_payload_is_counted_as_failed(self, workflow_harness):
        """An explicit not-found report becomes a failed result, not a dropped one.

        The extractor only records metadata with found truthy, so an element the
        agent explicitly reports as not found leaves results_list untouched. The
        backfill puts it back as found=False, so it is counted as failed and the
        workflow reports success False.
        """
        arm(
            workflow_harness,
            [
                FakeStep(
                    [
                        FakeActionResult(locator_metadata("elem_1", "id=search")),
                        FakeActionResult(locator_metadata("elem_2", "id=submit", found=False)),
                    ]
                )
            ],
        )

        run_workflow(
            workflow_harness,
            elements=[
                {"id": "elem_1", "description": "search box"},
                {"id": "elem_2", "description": "submit button"},
            ],
        )

        results = workflow_harness.updates[-1][1]["results"]
        assert [r["element_id"] for r in results["results"]] == ["elem_1", "elem_2"]
        assert results["summary"]["failed"] == 1
        assert results["success"] is False

    def test_unrequested_element_does_not_cover_a_missing_one(self, workflow_harness):
        """success is measured per requested id, not by counting found entries.

        An agent that reports a locator for an id nobody asked for must not let
        that entry stand in for a requested element it never found — a count
        comparison would call this run a success.
        """
        arm(
            workflow_harness,
            [
                FakeStep(
                    [
                        FakeActionResult(locator_metadata("elem_1", "id=search")),
                        FakeActionResult(locator_metadata("elem_99", "id=stray")),
                    ]
                )
            ],
        )

        run_workflow(
            workflow_harness,
            elements=[
                {"id": "elem_1", "description": "search box"},
                {"id": "elem_2", "description": "submit button"},
            ],
        )

        results = workflow_harness.updates[-1][1]["results"]
        assert results["success"] is False

    def test_no_results_at_all_is_a_failure(self, workflow_harness):
        """Zero extracted results: every requested element comes back as failed."""
        arm(workflow_harness, [FakeStep([FakeActionResult(None)])])

        run_workflow(
            workflow_harness,
            elements=[
                {"id": "elem_1", "description": "search box"},
                {"id": "elem_2", "description": "submit button"},
            ],
        )

        results = workflow_harness.updates[-1][1]["results"]
        assert results["success"] is False
        assert [r["element_id"] for r in results["results"]] == ["elem_1", "elem_2"]
        assert all(r["found"] is False for r in results["results"])
        assert results["summary"]["failed"] == 2

    def test_ignores_action_results_without_metadata(self, workflow_harness):
        """Native browser-use actions carry no metadata and must not become results."""
        arm(
            workflow_harness,
            [
                FakeStep([FakeActionResult(None)]),
                FakeStep([FakeActionResult(locator_metadata("elem_1", "id=q"))]),
            ],
        )

        run_workflow(workflow_harness)

        assert len(workflow_harness.updates[-1][1]["results"]["results"]) == 1


class TestReranking:
    """Tests for the post-process re-rank that decides the emitted best_locator."""

    def test_promotes_id_over_xpath(self, workflow_harness):
        """A volatile xpath must not stay the winner when a stable id is available."""
        arm(
            workflow_harness,
            [
                FakeStep(
                    [
                        FakeActionResult(
                            locator_metadata(
                                "elem_1",
                                "xpath=//div[3]/input",
                                all_locators=[
                                    loc("xpath=//div[3]/input", "xpath"),
                                    loc("id=search", "id"),
                                ],
                            )
                        )
                    ]
                )
            ],
        )

        run_workflow(workflow_harness)

        result = workflow_harness.updates[-1][1]["results"]["results"][0]
        assert result["best_locator"] == "id=search"
        assert result["all_locators"][0]["locator"] == "id=search"


class TestAgentConfiguration:
    """Tests for how the Agent and session are constructed."""

    def test_max_steps_scales_with_elements(self, workflow_harness):
        """Budget is navigate + 3/element + done + buffer; too low and the agent truncates."""
        arm(workflow_harness, [])

        run_workflow(
            workflow_harness,
            elements=[{"id": f"elem_{i}", "description": "x"} for i in range(3)],
        )

        assert workflow_harness.agent.run_calls == [1 + 3 * 3 + 1 + 8]

    def test_viewport_is_pinned(self, workflow_harness):
        """Vision coordinates only match Playwright's if the viewport is fixed."""
        arm(workflow_harness, [])

        run_workflow(workflow_harness)

        kwargs = workflow_harness.session.init_kwargs
        assert kwargs["viewport"] == {"width": 1920, "height": 1080}
        assert kwargs["no_viewport"] is False

    def test_auto_vision_excludes_screenshot_action(self, workflow_harness):
        """In 'auto' the screenshot action is dropped so its schema tokens are not paid."""
        workflow_harness.config.agent_vision_mode = "auto"
        arm(workflow_harness, [])

        run_workflow(workflow_harness)

        assert workflow_harness.agent.init_kwargs["use_vision"] == "auto"
        assert workflow_harness.agent.tools.excluded == ["screenshot"]

    def test_vision_on_keeps_screenshot_action(self, workflow_harness):
        """The 'on' escape hatch keeps full vision and the action in the schema."""
        workflow_harness.config.agent_vision_mode = "on"
        arm(workflow_harness, [])

        run_workflow(workflow_harness)

        assert workflow_harness.agent.init_kwargs["use_vision"] is True
        assert workflow_harness.agent.tools.excluded == []


class TestCustomActionRegistrationFallback:
    """Tests for the legacy fallback when registration fails."""

    def test_failed_registration_restores_full_vision(self, workflow_harness):
        """The escalation hook dies with registration, so 'auto' would leave the agent blind."""
        workflow_harness.register_result = False
        arm(workflow_harness, [])

        run_workflow(workflow_harness)

        assert workflow_harness.agent.settings.use_vision is True

    def test_failed_registration_rebuilds_prompts(self, workflow_harness):
        """Prompts are rebuilt for legacy mode rather than left describing the custom action."""
        from browser_service.tasks import workflow as wf

        workflow_harness.register_result = False
        arm(workflow_harness, [])

        with patch.object(wf, "build_workflow_prompt", return_value="LEGACY PROMPT") as build:
            run_workflow(workflow_harness)

        assert workflow_harness.agent.task == "LEGACY PROMPT"
        assert build.call_args.kwargs["include_custom_action"] is False

    def test_disabled_custom_actions_skips_registration(self, workflow_harness):
        """With the flag off, registration is never attempted."""
        from browser_service.tasks import workflow as wf

        arm(workflow_harness, [])

        with patch.object(wf, "register_custom_actions") as register:
            run_workflow(workflow_harness, enable_custom_actions=False)

        register.assert_not_called()


class TestTokenAccounting:
    """Tests for token usage, which drives cost reporting."""

    def test_reads_usage_from_history(self, workflow_harness):
        """Token counts come from AgentHistoryList.usage when present."""
        arm(
            workflow_harness,
            [FakeStep([FakeActionResult(locator_metadata("elem_1", "id=q"))])],
            usage=FakeUsage(prompt=1200, completion=340, total=1540, cached=64, cost=0.0042),
        )

        run_workflow(workflow_harness)

        summary = workflow_harness.updates[-1][1]["results"]["summary"]
        assert summary["input_tokens"] == 1200
        assert summary["output_tokens"] == 340
        assert summary["total_tokens"] == 1540
        assert summary["cached_tokens"] == 64
        assert summary["actual_cost"] == pytest.approx(0.0042)

    def test_missing_usage_reports_zeroes(self, workflow_harness):
        """No usage data must report zero cost, not crash or invent a number."""
        arm(workflow_harness, [FakeStep([FakeActionResult(locator_metadata("elem_1", "id=q"))])])

        run_workflow(workflow_harness)

        summary = workflow_harness.updates[-1][1]["results"]["summary"]
        assert summary["total_tokens"] == 0
        assert summary["actual_cost"] == 0.0

    def test_llm_call_count_is_history_length(self, workflow_harness):
        """One history entry is one LLM round trip."""
        arm(
            workflow_harness,
            [
                FakeStep([FakeActionResult(None)]),
                FakeStep([FakeActionResult(None)]),
                FakeStep([FakeActionResult(locator_metadata("elem_1", "id=q"))]),
            ],
        )

        run_workflow(workflow_harness)

        assert workflow_harness.updates[-1][1]["results"]["summary"]["total_llm_calls"] == 3


class TestProviderSelection:
    """Tests for LLM provider wiring."""

    def test_gemini_provider_uses_api_key(self, workflow_harness):
        """The Gemini Developer API path passes an api_key, not credentials."""
        arm(workflow_harness, [])

        run_workflow(workflow_harness)

        chat = workflow_harness.chat_google
        assert chat.call_args.kwargs["api_key"] == "test-key"
        assert "vertexai" not in chat.call_args.kwargs

    def test_vertex_provider_uses_credentials(self, workflow_harness):
        """Vertex uses pre-loaded credentials — no file I/O inside the workflow."""
        workflow_harness.config.llm.model_provider = "vertex"
        workflow_harness.config.llm.vertexai_credentials = "CREDS"
        workflow_harness.config.llm.vertexai_project = "proj"
        workflow_harness.config.llm.vertexai_location = "us-central1"
        arm(workflow_harness, [])

        run_workflow(workflow_harness)

        chat = workflow_harness.chat_google
        assert chat.call_args.kwargs["vertexai"] is True
        assert chat.call_args.kwargs["credentials"] == "CREDS"
        assert chat.call_args.kwargs["project"] == "proj"

    def test_local_provider_is_rejected(self, workflow_harness):
        """browser-service needs a Google vision model; local must fail loudly, not silently."""
        workflow_harness.config.llm.model_provider = "local"
        arm(workflow_harness, [])

        run_workflow(workflow_harness)

        results = workflow_harness.updates[-1][1]["results"]
        assert results["success"] is False
        assert "not supported" in results["error"]


class TestFailureHandling:
    """Tests for the error paths."""

    def test_agent_failure_becomes_failed_result(self, workflow_harness):
        """An agent crash is reported through the task, not raised at the caller."""
        from browser_service.tasks import workflow as wf

        original = wf.Agent.side_effect

        def make_broken(**kwargs):
            agent = original(**kwargs)
            agent.run_error = RuntimeError("model refused")
            return agent

        wf.Agent.side_effect = make_broken

        run_workflow(workflow_harness)

        results = workflow_harness.updates[-1][1]["results"]
        assert results["success"] is False
        assert "model refused" in results["error"]
        assert results["summary"]["failed"] == 1

    def test_session_start_failure_is_reported(self, workflow_harness):
        """A browser that will not launch fails the task cleanly."""
        workflow_harness.browser_session_cls.side_effect = RuntimeError("chrome missing")

        run_workflow(workflow_harness)

        results = workflow_harness.updates[-1][1]["results"]
        assert results["success"] is False
        assert "chrome missing" in results["error"]

    def test_failed_session_is_still_handed_to_cleanup(self, workflow_harness):
        """A session whose start() failed must still be cleaned up, not leaked."""
        from browser_service.tasks import workflow as wf

        def make_failing(**kwargs):
            session = FakeSession(**kwargs)
            session.start_error = RuntimeError("start timed out")
            workflow_harness.sessions.append(session)
            return session

        workflow_harness.browser_session_cls.side_effect = make_failing

        run_workflow(workflow_harness)

        assert workflow_harness.updates[-1][1]["results"]["success"] is False
        assert workflow_harness.cleanup.await_args.kwargs["session"] is workflow_harness.session


class TestCleanupOrdering:
    """Tests for the Task 19 guarantee that cleanup runs after completion is published."""

    def test_cleanup_runs_after_completed_status(self, workflow_harness):
        """Cleanup costs seconds; the poller must see 'completed' before it starts."""
        order = []

        def record_update(task_id, payload):
            order.append(f"status:{payload.get('status')}")
            workflow_harness.updates.append((task_id, dict(payload)))

        workflow_harness.task_processor.update_task.side_effect = record_update
        workflow_harness.cleanup.side_effect = lambda *a, **k: order.append("cleanup")

        arm(workflow_harness, [FakeStep([FakeActionResult(locator_metadata("elem_1", "id=q"))])])
        run_workflow(workflow_harness)

        assert order.index("status:completed") < order.index("cleanup")

    def test_cleanup_receives_captured_pid(self, workflow_harness):
        """The PID captured while Chrome was alive is what cleanup must be given."""
        arm(workflow_harness, [])

        run_workflow(workflow_harness)

        assert workflow_harness.cleanup.await_args.kwargs["browser_pid"] == 4242


class TestMetricsRecording:
    """Tests for workflow metrics reporting."""

    def test_records_metrics_for_root_workflow(self, workflow_harness):
        """A root workflow reports its own metrics."""
        from browser_service.tasks import workflow as wf

        settings = MagicMock()
        settings.TRACK_LLM_COSTS = True
        settings.APP_PORT = 5000
        arm(workflow_harness, [FakeStep([FakeActionResult(locator_metadata("elem_1", "id=q"))])])

        with patch.object(wf, "_nl_settings", settings):
            run_workflow(workflow_harness)

        assert workflow_harness.record_metrics.call_args.kwargs["workflow_id"] == "task-1"

    def test_child_workflow_defers_to_parent(self, workflow_harness):
        """A child must not double-count; the parent emits unified metrics."""
        from browser_service.tasks import workflow as wf

        settings = MagicMock()
        settings.TRACK_LLM_COSTS = True
        arm(workflow_harness, [FakeStep([FakeActionResult(locator_metadata("elem_1", "id=q"))])])

        with patch.object(wf, "_nl_settings", settings):
            run_workflow(workflow_harness, parent_workflow_id="parent-9")

        workflow_harness.record_metrics.assert_not_called()

    def test_metrics_failure_does_not_fail_workflow(self, workflow_harness):
        """Reporting is best-effort — a metrics outage must not fail a good run."""
        from browser_service.tasks import workflow as wf

        settings = MagicMock()
        settings.TRACK_LLM_COSTS = True
        settings.APP_PORT = 5000
        workflow_harness.record_metrics.side_effect = RuntimeError("metrics endpoint down")
        arm(workflow_harness, [FakeStep([FakeActionResult(locator_metadata("elem_1", "id=q"))])])

        with patch.object(wf, "_nl_settings", settings):
            run_workflow(workflow_harness)

        assert workflow_harness.statuses() == ["running", "completed"]
