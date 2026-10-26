"""
A failed secret step's note names ``<NAME>`` and never asks for the placeholder
to be typed (NLRF #131 Part 2, D18; the D19 pin on browser-use's sensitive fields).

When ``_do_interaction`` reports ``auto_failed`` for a step whose value is
``{{NAME}}``, the old note said ``input(text="{{NAME}}")``. The note is
``extracted_content``, which browser-use 0.13.7 drops while ``long_term_memory``
is set (always, here) -- today it shows only in browser-use's Final Result log
line; this keeps it name-safe if an upgrade ever shows it to the agent.
Made-up values only.
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
    _params,
)

V = "x-made-up-1"
MOVE_ON = "proceed to the next element"


async def _failed_note(action, value, secrets, index=12, caplog=None):
    async def _fake(*args, **kwargs):
        return "", "auto_failed", action, value

    agent = FakeAgent()
    params = _params("elem_1")
    if index is None:
        del params["element_index"]
    elements = [{"id": "elem_1", "action": action, "value": value}]
    engine = AsyncMock(return_value=dict(STRONG))
    with (
        patch("browser_service.agent.actions.find_unique_locator_action", new=engine),
        patch("browser_service.agent.registration._do_interaction", new=_fake),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
    ):
        assert register_custom_actions(agent, elements=elements, secrets=secrets) is True
        result = await agent.tools.registry.execute_action(
            "find_unique_locator", params, browser_session=FakeBrowserSession()
        )
    return result.extracted_content


@pytest.mark.parametrize("action", ["input", "type", "select"])
@pytest.mark.parametrize("index", [12, None])
async def test_secret_failure_says_move_on(action, index, caplog):
    caplog.set_level(logging.DEBUG)
    note = await _failed_note(action, "{{PW}}", {"PW": V}, index)
    assert "<PW>" in note
    assert MOVE_ON in note
    for banned in ("input(", "select_dropdown(", "{{", "FALLBACK REQUIRED", "Action required", V):
        assert banned not in note
    assert V not in caplog.text


async def test_unknown_name_keeps_todays_note():
    note = await _failed_note("input", "{{UNKNOWN}}", {"PW": V})
    assert 'FALLBACK REQUIRED: Call input(index=12, text="{{UNKNOWN}}").' in note


async def test_plain_value_note_is_byte_equal():
    note = await _failed_note("input", "plain", {"PW": V})
    assert (
        "\n⚠️ Automated input failed — all strategies exhausted. "
        'FALLBACK REQUIRED: Call input(index=12, text="plain").'
    ) in note


def test_browser_use_still_has_the_sensitive_typing_fields():
    """A browser-use bump that renames them must fail CI (the other tests mock the event)."""
    from browser_use.browser.events import TypeTextEvent

    assert {"is_sensitive", "sensitive_key_name"} <= set(TypeTextEvent.model_fields)
