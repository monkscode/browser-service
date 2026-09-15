import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_service.agent.registration import register_custom_actions
from browser_service.config import config


class FakeBrowserSession:
    cdp_url = "ws://127.0.0.1:9222/devtools/browser/00000000-0000-0000-0000-000000000000"
    cdp_client = None
    is_cdp_connected = True
    _original_viewport_size = (1920, 1080)
    agent_focus_target_id = None

    async def get_selector_map(self):
        return {}

    async def get_current_page_url(self):
        return "https://sujal.astppbilling.org/"


class FakeAgent:
    tools = None


def _fake_playwright():
    fake_page = AsyncMock()
    fake_page.url = "https://sujal.astppbilling.org/"
    ctx = MagicMock()
    ctx.pages = [fake_page]
    fake_browser = MagicMock()
    fake_browser.contexts = [ctx]
    fake_instance = MagicMock()
    fake_instance.chromium.connect_over_cdp = AsyncMock(return_value=fake_browser)
    starter = MagicMock()
    starter.start = AsyncMock(return_value=fake_instance)
    return MagicMock(return_value=starter)


ELEMENTS = [{"id": "elem_3", "action": "get_text", "value": ""}]
PARAMS = {
    "x": 640,
    "y": 400,
    "element_id": "elem_3",
    "element_description": "Sign In button in the login form",
    "element_index": 12,
}

STRONG_A = {
    "found": True,
    "best_locator": "id=save_button",
    "validated": True,
    "count": 1,
    "unique": True,
    "valid": True,
    "validation_method": "playwright",
}
STRONG_B = {**STRONG_A, "best_locator": 'text="Sign In"', "semantic_match": True}


async def _run(results: list, flag: bool):
    agent = FakeAgent()
    session = FakeBrowserSession()
    engine = AsyncMock(side_effect=[dict(r) for r in results])
    with (
        patch.object(config.locator, "enable_action_fit", flag),
        patch("browser_service.agent.actions.find_unique_locator_action", new=engine),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
    ):
        assert register_custom_actions(agent, elements=ELEMENTS) is True
        out = []
        for _ in results:
            out.append(
                await agent.tools.registry.execute_action(
                    "find_unique_locator", dict(PARAMS), browser_session=session
                )
            )
    return engine, out


async def test_flag_on_passes_the_step_action():
    engine, _ = await _run([STRONG_A], flag=True)
    assert engine.await_args.kwargs["action"] == "get_text"


async def test_flag_off_passes_no_action():
    engine, _ = await _run([STRONG_A], flag=False)
    assert engine.await_args.kwargs["action"] is None


async def test_misfit_result_never_replaces_a_strong_one():
    engine, out = await _run([STRONG_A, {**STRONG_B, "action_fit": "misfit"}], flag=True)
    assert "already validated" in out[1].extracted_content
    assert "id=save_button" in out[1].extracted_content
