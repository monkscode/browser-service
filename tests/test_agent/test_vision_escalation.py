"""
Unit tests for the A1-INLINE vision escalation hook (Task 28) in
browser_service.agent.registration — the registered find_unique_locator action.

Purpose: With use_vision='auto' the agent runs vision-off; browser-use (0.13.7,
         message_manager/service.py:448-469) attaches the current screenshot to the
         NEXT LLM call when an ActionResult carries metadata={'include_screenshot': True},
         and on a failure the handler also keeps vision on for the rest of the run
         (_keep_vision_on — TestStickyVision, TestTheRun4Shape).
         These tests pin the trigger contract:
           - BOTH validation-failure ActionResults carry the escalation metadata
           - failure messages tell the model to re-examine the attached screenshot
           - the terminal dead-browser path does NOT escalate (run is over)
           - the success path metadata (the result dict) is not polluted
           - BOTH failure ActionResults also carry the engine's failure payload,
             which is how workflow.py learns why a locator is missing. metadata
             is observability-only in browser-use (ActionResult.metadata is read
             for include_screenshot and nothing else), so this costs no tokens.

The registered handler is exercised through the real browser-use Tools registry
(register_custom_actions on a fake agent, then registry.execute_action) with the
locator engine and Playwright connection mocked — no browser, no LLM.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_service.agent.registration import register_custom_actions


class FakeBrowserSession:
    """Minimal browser_session stand-in: CDP URL present, empty selector map."""

    cdp_url = "ws://127.0.0.1:9222/devtools/browser/00000000-0000-0000-0000-000000000000"
    cdp_client = None
    is_cdp_connected = True
    _original_viewport_size = (1920, 1080)
    agent_focus_target_id = None

    async def get_selector_map(self):
        return {}

    async def get_current_page_url(self):
        return "https://example.com/products"


class FakeDeadBrowserSession:
    """No CDP URL anywhere — _ensure_playwright must raise the terminal error."""

    cdp_url = None
    cdp_client = None
    is_cdp_connected = False
    agent_focus_target_id = None

    async def get_selector_map(self):
        return {}

    async def get_current_page_url(self):
        return "https://example.com/products"


class FakeAgent:
    tools = None


def _fake_playwright():
    """A playwright.async_api.async_playwright replacement that hands back a
    fake CDP-connected browser with one non-blank page."""
    fake_page = AsyncMock()
    fake_page.url = "https://example.com/products"
    # F1: a snapshot that could not be read keeps the read-change step inert here.
    fake_page.content = AsyncMock(return_value=None)
    ctx = MagicMock()
    ctx.pages = [fake_page]
    fake_browser = MagicMock()
    fake_browser.contexts = [ctx]
    fake_instance = MagicMock()
    fake_instance.chromium.connect_over_cdp = AsyncMock(return_value=fake_browser)
    starter = MagicMock()
    starter.start = AsyncMock(return_value=fake_instance)
    return MagicMock(return_value=starter)


ELEMENTS = [{"id": "elem_1", "action": "get_text", "value": ""}]


async def _invoke(engine_result, element_index, browser_session=None):
    """Register the custom action on a fresh fake agent with the engine mocked,
    then execute find_unique_locator through the real registry."""
    agent = FakeAgent()
    session = browser_session if browser_session is not None else FakeBrowserSession()
    params = {
        "x": 500,
        "y": 300,
        "element_id": "elem_1",
        "element_description": "submit button",
    }
    if element_index is not None:
        params["element_index"] = element_index

    with (
        patch(
            "browser_service.agent.actions.find_unique_locator_action",
            new=AsyncMock(return_value=engine_result),
        ),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
    ):
        assert register_custom_actions(agent, elements=ELEMENTS) is True
        return await agent.tools.registry.execute_action(
            "find_unique_locator", params, browser_session=session
        )


ENGINE_FAILURE = {"found": False, "error": "no unique locator found"}


class TestEscalationOnValidationFailure:
    """Both validation-failure ActionResults must request the screenshot."""

    @pytest.mark.asyncio
    async def test_retry_without_index_carries_include_screenshot(self):
        """element_index=None + engine failure → retry ActionResult escalates."""
        result = await _invoke(ENGINE_FAILURE, element_index=None)
        assert result.error is not None
        assert result.is_done is not True
        assert result.metadata is not None
        assert result.metadata.get("include_screenshot") is True

    @pytest.mark.asyncio
    async def test_retry_without_index_message_mentions_screenshot(self):
        """The retry message tells the model to re-examine the attached screenshot."""
        result = await _invoke(ENGINE_FAILURE, element_index=None)
        assert "screenshot" in result.error.lower()

    @pytest.mark.asyncio
    async def test_failure_with_index_carries_include_screenshot(self):
        """element_index provided + engine failure → failure ActionResult escalates."""
        result = await _invoke(ENGINE_FAILURE, element_index=42)
        assert result.error is not None
        assert result.is_done is not True
        assert result.metadata is not None
        assert result.metadata.get("include_screenshot") is True

    @pytest.mark.asyncio
    async def test_failure_with_index_message_mentions_screenshot(self):
        result = await _invoke(ENGINE_FAILURE, element_index=42)
        assert "screenshot" in result.error.lower()


ENGINE_FAILURE_DETAILED = {
    "element_id": "elem_1",
    "description": "submit button",
    "found": False,
    "error": "Semantic mismatch: expected 'Submit' but none of the 3 unique locators matched",
    "error_type": "SemanticMismatch",
    "semantic_match": False,
    "candidates_found": 3,
}


class TestFailurePayloadIsPublished:
    """The engine's reason must reach metadata, not just the agent-facing prose.

    workflow.py drops rejected payloads out of its results list, so metadata is
    the only channel that can tell it why an element has no locator. Without
    this the reason exists solely in the log.
    """

    @pytest.mark.asyncio
    async def test_failure_with_index_carries_engine_payload(self):
        result = await _invoke(ENGINE_FAILURE_DETAILED, element_index=42)

        assert result.metadata["element_id"] == "elem_1"
        assert result.metadata["found"] is False
        assert result.metadata["error"] == ENGINE_FAILURE_DETAILED["error"]
        assert result.metadata["error_type"] == "SemanticMismatch"
        # The escalation flag rides alongside it, not instead of it.
        assert result.metadata["include_screenshot"] is True

    @pytest.mark.asyncio
    async def test_retry_without_index_carries_engine_payload(self):
        result = await _invoke(ENGINE_FAILURE_DETAILED, element_index=None)

        assert result.metadata["element_id"] == "elem_1"
        assert result.metadata["found"] is False
        assert result.metadata["error"] == ENGINE_FAILURE_DETAILED["error"]
        assert result.metadata["include_screenshot"] is True


class TestNoEscalationOnOtherPaths:
    """Terminal and success paths must NOT request a screenshot."""

    @pytest.mark.asyncio
    async def test_dead_browser_terminal_path_does_not_escalate(self):
        """_PlaywrightConnectionError → is_done=True, no screenshot request
        (the run is over; attaching an image to a dead run is wasted tokens)."""
        result = await _invoke(
            ENGINE_FAILURE, element_index=None, browser_session=FakeDeadBrowserSession()
        )
        assert result.is_done is True
        assert not (result.metadata or {}).get("include_screenshot")

    @pytest.mark.asyncio
    async def test_success_path_metadata_not_polluted(self):
        """Success keeps metadata=result-dict; it must not gain include_screenshot."""
        engine_success = {
            "found": True,
            "best_locator": "id=submit",
            "validated": True,
            "count": 1,
            "validation_method": "playwright",
        }
        result = await _invoke(engine_success, element_index=None)
        assert result.error is None
        assert result.metadata is not None
        assert "include_screenshot" not in result.metadata


class TestSchemaContract:
    """The model-facing action schema must match the vision-off prompt contract."""

    def test_element_index_schema_description_says_required(self):
        """The Field description the LLM sees calls element_index REQUIRED —
        the old 'HIGHLY RECOMMENDED' soft wording contradicted the prompt
        contract (templates.py: 'element_index (int, required)')."""
        agent = FakeAgent()
        assert register_custom_actions(agent, elements=ELEMENTS) is True
        field = agent.tools.registry.registry.actions[
            "find_unique_locator"
        ].param_model.model_fields["element_index"]
        assert "REQUIRED" in field.description
        assert "HIGHLY RECOMMENDED" not in field.description


class FakeAgentWithSettings:
    """A fake agent that carries browser-use's per-run settings (use_vision)."""

    tools = None

    def __init__(self, use_vision):
        self.settings = SimpleNamespace(use_vision=use_vision)


async def _invoke_many(agent, engine_results, element_index=42, browser_session=None):
    """Register once on ``agent``, then run find_unique_locator once per engine result."""
    session = browser_session if browser_session is not None else FakeBrowserSession()
    params = {"x": 500, "y": 300, "element_id": "elem_1", "element_description": "submit button"}
    if element_index is not None:
        params["element_index"] = element_index
    engine = AsyncMock(side_effect=list(engine_results))
    with (
        patch("browser_service.agent.actions.find_unique_locator_action", new=engine),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
    ):
        assert register_custom_actions(agent, elements=ELEMENTS) is True
        return [
            await agent.tools.registry.execute_action(
                "find_unique_locator", params, browser_session=session
            )
            for _ in engine_results
        ]


ENGINE_SUCCESS = {
    "found": True,
    "best_locator": "id=submit",
    "validated": True,
    "count": 1,
    "validation_method": "playwright",
}
STICKY_SIGNAL = "vision-sticky-on"


class TestStickyVision:
    """After the first failed locator step of a run, the run keeps the screenshot.

    browser-use attaches an include_screenshot image to the ONE next call only; the agent
    can spend that call on another action and then point blind again (the bs 1.0.50 deploy
    probe, run 4). Flipping the run's settings.use_vision from 'auto' to True keeps every
    later call sighted (browser-use reads it per step, agent/service.py:1138, 0.13.7).
    """

    @pytest.mark.asyncio
    async def test_failure_with_index_turns_vision_on(self):
        agent = FakeAgentWithSettings("auto")
        await _invoke_many(agent, [ENGINE_FAILURE], element_index=42)
        assert agent.settings.use_vision is True

    @pytest.mark.asyncio
    async def test_retry_without_index_turns_vision_on(self):
        agent = FakeAgentWithSettings("auto")
        await _invoke_many(agent, [ENGINE_FAILURE], element_index=None)
        assert agent.settings.use_vision is True

    @pytest.mark.asyncio
    async def test_success_leaves_vision_auto(self):
        agent = FakeAgentWithSettings("auto")
        (result,) = await _invoke_many(agent, [ENGINE_SUCCESS], element_index=None)
        assert result.error is None  # the success path, not the generic exception path
        assert agent.settings.use_vision == "auto"

    @pytest.mark.asyncio
    async def test_dead_browser_leaves_vision_auto(self):
        agent = FakeAgentWithSettings("auto")
        (result,) = await _invoke_many(
            agent, [ENGINE_FAILURE], element_index=None, browser_session=FakeDeadBrowserSession()
        )
        assert result.is_done is True  # the terminal dead-browser path
        assert agent.settings.use_vision == "auto"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("explicit", [True, False])
    async def test_explicit_off_and_on_are_left_alone(self, explicit, caplog):
        agent = FakeAgentWithSettings(explicit)
        with caplog.at_level("INFO", logger="browser_service.agent.registration"):
            await _invoke_many(agent, [ENGINE_FAILURE], element_index=42)
        assert agent.settings.use_vision is explicit
        assert STICKY_SIGNAL not in caplog.text

    @pytest.mark.asyncio
    async def test_signal_logged_once_per_run(self, caplog):
        agent = FakeAgentWithSettings("auto")
        with caplog.at_level("INFO", logger="browser_service.agent.registration"):
            await _invoke_many(agent, [ENGINE_FAILURE, ENGINE_FAILURE], element_index=42)
        assert caplog.text.count(STICKY_SIGNAL) == 1

    @pytest.mark.asyncio
    async def test_agent_without_settings_does_not_crash(self):
        """Today's FakeAgent has no settings: the failure result is unchanged."""
        result = await _invoke(ENGINE_FAILURE, element_index=42)
        assert result.metadata.get("include_screenshot") is True
        assert result.is_done is not True

    def test_returns_true_only_when_it_flips(self):
        from browser_service.agent.registration import _keep_vision_on

        agent = FakeAgentWithSettings("auto")
        assert _keep_vision_on(agent) is True
        assert _keep_vision_on(agent) is False  # already True: no second flip, no second signal
        assert _keep_vision_on(FakeAgent()) is False


AFTER_FIND_ELEMENTS = "Found 14 elements matching '*'."


def _next_call_has_image(use_vision, last_result, tmp_path) -> bool:
    """Build the next LLM call with browser-use's OWN message builder (0.13.7, no browser, no LLM)
    and report whether the page screenshot is in it."""
    from browser_use.agent.message_manager.service import MessageManager
    from browser_use.browser.views import BrowserStateSummary
    from browser_use.dom.views import SerializedDOMState
    from browser_use.filesystem.file_system import FileSystem
    from browser_use.llm.messages import ContentPartImageParam, SystemMessage

    manager = MessageManager(
        task="t", system_message=SystemMessage(content="s"), file_system=FileSystem(tmp_path)
    )
    state = BrowserStateSummary(
        dom_state=SerializedDOMState(_root=None, selector_map={}),
        url="http://127.0.0.1/login.html",
        title="t",
        tabs=[],
        screenshot="iVBORw0KGgo=",
    )
    manager.create_state_messages(
        browser_state_summary=state, result=last_result, use_vision=use_vision
    )
    content = manager.get_messages()[-1].content
    return isinstance(content, list) and any(isinstance(p, ContentPartImageParam) for p in content)


class TestTheRun4Shape:
    """The bs 1.0.50 deploy probe, run 4: a failed locator step, then an action whose result asks
    for no screenshot (find_elements), then the re-point. Only a sticky flip puts the page in that
    third call; on 'auto' it goes out blind (run 4 pointed at (640, 485) for a message at y 155)."""

    @pytest.mark.asyncio
    async def test_after_a_failure_and_another_action_the_call_still_has_the_page(self, tmp_path):
        from browser_use.agent.views import ActionResult

        agent = FakeAgentWithSettings("auto")
        await _invoke_many(agent, [ENGINE_FAILURE], element_index=None)
        after = [ActionResult(extracted_content=AFTER_FIND_ELEMENTS)]
        assert _next_call_has_image(agent.settings.use_vision, after, tmp_path) is True

    def test_on_auto_that_call_is_blind(self, tmp_path):
        """Known negative: today's behaviour for run 4's step 6."""
        from browser_use.agent.views import ActionResult

        after = [ActionResult(extracted_content=AFTER_FIND_ELEMENTS)]
        assert _next_call_has_image("auto", after, tmp_path) is False

    def test_on_auto_the_one_shot_still_attaches(self, tmp_path):
        """Known positive for the detector: the call right after a failure carries the page."""
        from browser_use.agent.views import ActionResult

        failed = [ActionResult(error="x", metadata={"include_screenshot": True})]
        assert _next_call_has_image("auto", failed, tmp_path) is True
