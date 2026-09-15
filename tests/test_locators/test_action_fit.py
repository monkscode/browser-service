"""
E2a action-fitness rules, pure (no browser). The facts dicts are what
FIT_FACTS_JS returns; test_action_fit_facts.py proves the JS produces them
on real Chromium.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from browser_service.locators.action_fit import action_misfit, check_action_fit


@pytest.mark.parametrize(
    "action, facts, expected_text, want",
    [
        # fill — Playwright fill() semantics
        ("input", {"tag": "input", "type": ""}, None, ""),
        ("input", {"tag": "input", "type": "search"}, None, ""),
        ("input", {"tag": "textarea"}, None, ""),
        ("input", {"tag": "div", "contentEditable": True}, None, ""),
        ("input", {"tag": "label"}, None, ""),  # Playwright fills the label's control
        ("input", {"tag": "div"}, None, "cannot be filled"),  # flipkart u02
        ("input", {"tag": "iframe"}, None, "cannot be filled"),  # mce_0_ifr, 21 corpus fails
        ("input", {"tag": "input", "type": "checkbox"}, None, "cannot be filled"),
        ("input", {"tag": "input", "type": "text", "disabled": True}, None, "disabled"),
        ("input", {"tag": "input", "type": "text", "readOnly": True}, None, "read-only"),
        ("type", {"tag": "span"}, None, "cannot be filled"),
        # click / select — Playwright "enabled"
        ("click", {"tag": "div"}, None, ""),
        ("click", {"tag": "button", "disabled": True}, None, "disabled"),
        ("click", {"tag": "div", "ariaDisabled": True}, None, "aria-disabled"),
        ("select", {"tag": "select", "disabled": True}, None, "disabled"),
        # read — narrow on purpose (get_text is nlrf's locate-only catch-all)
        (
            "get_text",
            {"tag": "input", "value": "", "placeholder": "What needs to be done?"},
            "buy milk",
            "empty",
        ),  # todomvc u10
        ("get_text", {"tag": "input", "value": "NewYork"}, "NewYork", ""),  # parabank u09
        ("get_text", {"tag": "input", "value": ""}, None, ""),
        ("get_text", {"tag": "input", "value": "", "placeholder": "Search"}, "Search", ""),
        ("get_text", {"tag": "input", "value": "", "labelText": "Email *"}, "email *", ""),
        ("get_text", {"tag": "div"}, "anything", ""),
        ("get_text", {"tag": "input", "type": "file", "value": ""}, None, ""),
        # no rule / no evidence
        ("hover", {"tag": "div"}, None, ""),
        (None, {"tag": "div"}, None, ""),
        ("input", None, None, ""),
        ("input", {"tag": ""}, None, ""),
    ],
)
def test_action_misfit(action, facts, expected_text, want):
    reason = action_misfit(action, facts, expected_text)
    if want:
        assert want in reason
    else:
        assert reason == ""


async def test_no_action_reads_nothing():
    root = MagicMock()
    assert await check_action_fit(root, "#x", None) == ""
    root.locator.assert_not_called()


async def test_unreadable_element_is_unknown_not_misfit():
    loc = MagicMock()
    loc.evaluate = AsyncMock(side_effect=TimeoutError("detached"))
    root = MagicMock()
    root.locator.return_value = loc
    assert await check_action_fit(root, "#x", "input") == ""


async def test_non_dict_facts_are_unknown():
    loc = MagicMock()
    loc.evaluate = AsyncMock(return_value="ref: <Node>")
    root = MagicMock()
    root.locator.return_value = loc
    assert await check_action_fit(root, "#x", "input") == ""
