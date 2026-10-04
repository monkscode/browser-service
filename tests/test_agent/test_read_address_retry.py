"""#31: an unconfirmed read address gets ONE retry with its reason, then a not-found."""

from unittest.mock import AsyncMock, patch

import pytest

from browser_service.agent.registration import register_custom_actions
from tests.test_agent.test_vision_escalation import FakeAgent, FakeBrowserSession, _fake_playwright

ELEMENTS = [
    {"id": "elem_1", "action": "get_text", "value": ""},
    {"id": "elem_2", "action": "get_text", "value": ""},
]
UNCONFIRMED = {
    "found": False,
    "error": "the read locator would carry the value being read; x",
    "read_address_unconfirmed": "the screen point shows 'name (a to z)', not 'products'",
}
PARAMS = {"x": 500, "y": 300, "element_description": "the Products heading", "element_index": 7}


async def _run(calls):
    agent = FakeAgent()
    with (
        patch(
            "browser_service.agent.actions.find_unique_locator_action",
            new=AsyncMock(return_value=UNCONFIRMED),
        ),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
    ):
        assert register_custom_actions(agent, elements=ELEMENTS) is True
        return [
            await agent.tools.registry.execute_action(
                "find_unique_locator",
                {**PARAMS, "element_id": eid},
                browser_session=FakeBrowserSession(),
            )
            for eid in calls
        ]


@pytest.mark.asyncio
async def test_first_unconfirmed_asks_again_with_the_reason_and_a_screenshot():
    (first,) = await _run(["elem_1"])
    assert "name (a to z)" in first.error
    assert "point at the element itself" in first.error.lower()
    assert first.metadata.get("include_screenshot") is True
    assert first.is_done is not True


@pytest.mark.asyncio
async def test_second_unconfirmed_is_not_found_and_says_do_not_retry():
    _, second = await _run(["elem_1", "elem_1"])
    assert "NOT FOUND" in second.error
    assert "do not retry" in second.error.lower()
    assert not second.metadata.get("include_screenshot")


@pytest.mark.asyncio
async def test_the_counter_is_per_element():
    first, other = await _run(["elem_1", "elem_2"])
    assert "point at the element itself" in other.error.lower()
