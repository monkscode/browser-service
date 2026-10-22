"""
#62 step 1: log NLRF's own step description beside the agent's pick.

Every check of the chosen element compares the agent's own signals with each
other, so an agent that picks the wrong element and describes THAT element
passes them all. Logging NLRF's description (and the action) next to the
agent's `Element:` line makes such picks countable from the log.

The step's VALUE is never logged here (owner, 2026-10-22): for a login it is
the password, and NLRF #131 is removing those lines, not adding one.

Harness: the real registered handler through browser-use's Tools registry,
engine + Playwright faked, _do_interaction faked at its seam
(test_interaction_memory_line.py).
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from browser_service.agent.registration import register_custom_actions
from tests.test_agent.test_interaction_memory_line import (
    STRONG,
    FakeAgent,
    FakeBrowserSession,
    _fake_playwright,
)

LOGGER = "browser_service.agent.registration"
MARKER = "Zq62ValueMarker"
PASSWORD = {
    "id": "elem_2",
    "description": "password input field in the login form",
    "action": "input",
    "value": MARKER,
}


async def _call(elements, element_id, caplog):
    """Register once, call find_unique_locator once; return the log lines and
    the value the interaction seam received."""
    received = []

    async def _fake_interaction(
        browser_session,
        active_page,
        locator_str,
        element_id,
        element_index,
        element_specs,
        performed_actions,
        **kwargs,
    ):
        spec = element_specs.get(element_id, {})
        received.append(spec.get("value", ""))
        return "", "not_applicable", spec.get("action", "get_text"), spec.get("value", "")

    agent = FakeAgent()
    with (
        caplog.at_level(logging.INFO, logger=LOGGER),
        patch(
            "browser_service.agent.actions.find_unique_locator_action",
            new=AsyncMock(return_value=dict(STRONG)),
        ),
        patch("browser_service.agent.registration._do_interaction", new=_fake_interaction),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
    ):
        assert register_custom_actions(agent, elements=elements) is True
        await agent.tools.registry.execute_action(
            "find_unique_locator",
            {
                "x": 640,
                "y": 400,
                "element_id": element_id,
                "element_description": "the sidebar filter box",
                "element_index": 12,
            },
            browser_session=FakeBrowserSession(),
        )
    lines = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    return lines, received


@pytest.mark.asyncio
async def test_step_line_carries_nlrf_description_and_action(caplog):
    lines, _ = await _call([PASSWORD], "elem_2", caplog)
    step = lines.index("   Step (NLRF): password input field in the login form (action: input)")
    assert lines[step - 1] == "   Element: elem_2 - the sidebar filter box"


@pytest.mark.asyncio
async def test_step_value_never_logged(caplog):
    lines, received = await _call([PASSWORD], "elem_2", caplog)
    # The value did reach the handler, so its absence below is not vacuous.
    assert received == [MARKER]
    assert not [line for line in lines if MARKER in line]


@pytest.mark.asyncio
async def test_unknown_element_id(caplog):
    lines, _ = await _call([PASSWORD], "elem_9", caplog)
    assert "   Step (NLRF): not in the request" in lines


@pytest.mark.asyncio
async def test_element_without_description(caplog):
    lines, _ = await _call([{"id": "elem_1", "action": "click"}], "elem_1", caplog)
    assert "   Step (NLRF): (no description) (action: click)" in lines
