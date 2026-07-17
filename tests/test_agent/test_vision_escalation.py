"""
Unit tests for the A1-INLINE vision escalation hook (Task 28) in
browser_service.agent.registration — the registered find_unique_locator action.

Purpose: With use_vision='auto' the agent runs vision-off; browser-use 0.12.6
         attaches the current screenshot to the NEXT LLM call only when an
         ActionResult carries metadata={'include_screenshot': True}
         (message_manager/service.py:444-464, verified live 2026-07-17).
         These tests pin the trigger contract:
           - BOTH validation-failure ActionResults carry the escalation metadata
           - failure messages tell the model to re-examine the attached screenshot
           - the terminal dead-browser path does NOT escalate (run is over)
           - the success path metadata (the result dict) is not polluted

The registered handler is exercised through the real browser-use Tools registry
(register_custom_actions on a fake agent, then registry.execute_action) with the
locator engine and Playwright connection mocked — no browser, no LLM.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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

    with patch(
        "browser_service.agent.actions.find_unique_locator_action",
        new=AsyncMock(return_value=engine_result),
    ), patch("playwright.async_api.async_playwright", new=_fake_playwright()):
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
        field = (
            agent.tools.registry.registry.actions["find_unique_locator"]
            .param_model.model_fields["element_index"]
        )
        assert "REQUIRED" in field.description
        assert "HIGHLY RECOMMENDED" not in field.description
