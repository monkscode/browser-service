"""
An org secret is typed into its own field and never shown (NLRF #131 Part 2).

NLRF sends ``secrets: {NAME: value}`` beside the steps; a step whose value is
exactly ``{{NAME}}`` keeps that text, so the agent's task text shows only the
name. ``_do_interaction`` swaps the name for the value only for the typing call:
browser-use's ``TypeTextEvent`` is marked sensitive (its log shows ``<NAME>``),
the field is read once, and when it does not hold the value the existing
Playwright ``fill()`` tier types it. Every note, log line and the returned value
show the name — never the value, never the field's read-back.

Made-up values only. Every test also asserts the value is in no log line and in
no note.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_service.agent.registration import (
    _do_interaction,
    register_custom_actions,
    secret_name,
)

V = "Zq131-made-up-7"
V2 = "a'b\"c\\d{{x}}%"


class FakeEvent:
    """An awaitable that also exposes event_result, like browser-use's event handles."""

    def __init__(self, error=None, on_await=None):
        self._error = error
        self._on_await = on_await

    def __await__(self):
        async def _run():
            if self._on_await:
                await self._on_await()

        return _run().__await__()

    async def event_result(self, raise_if_any=False, raise_if_none=False):
        if self._error:
            raise self._error
        return None


@pytest.fixture
def type_event():
    """browser-use's event models replaced by plain constructors (they validate `node`)."""
    import browser_use.browser.events as events

    with (
        patch.object(events, "TypeTextEvent", MagicMock()) as typed,
        patch.object(events, "ClickElementEvent", MagicMock()),
        patch.object(events, "SelectDropdownOptionEvent", MagicMock()) as selected,
    ):
        typed.selected = selected
        yield typed


@pytest.fixture(autouse=True)
def all_logs(caplog):
    caplog.set_level(logging.DEBUG)
    return caplog


def _page(read_values):
    """A Playwright page whose step locator reads `read_values` in turn (an Exception raises)."""
    page = MagicMock()
    page.wait_for_load_state = AsyncMock()
    loc = MagicMock()
    loc.fill = AsyncMock()
    loc.evaluate = AsyncMock()
    loc.select_option = AsyncMock()
    loc.input_value = AsyncMock(side_effect=list(read_values))
    page.locator.return_value = loc
    frame = MagicMock()
    frame.locator.return_value = loc
    page.frame_locator.return_value = frame
    return page, loc


def _session(event):
    session = MagicMock()
    session.get_element_by_index = AsyncMock(return_value=MagicMock())
    session.event_bus.dispatch = MagicMock(return_value=event)
    return session


async def _interact(value, secrets, page, event=None, action="input", **kwargs):
    performed: set = set()
    result = await _do_interaction(
        _session(event or FakeEvent()),
        page,
        kwargs.pop("locator_str", "#pw"),
        "elem_1",
        6,
        {"elem_1": {"action": action, "value": value}},
        performed,
        secrets=secrets,
        **kwargs,
    )
    return (*result, performed)


def _never_shown(caplog, note, value=V):
    assert value not in caplog.text
    assert value not in note


def test_secret_name_rule_matches_nlrf():
    assert secret_name("{{PW}}", {"PW": V}) == "PW"
    assert secret_name("{{ pw }}", {"PW": V}) == "PW"
    assert secret_name("user {{PW}}", {"PW": V}) is None
    assert secret_name("{{OTHER}}", {"PW": V}) is None
    assert secret_name("{{PW}}", {}) is None


async def test_secret_typed_through_sensitive_event(type_event, all_logs):
    page, loc = _page([V])

    note, status, action, value, performed = await _interact("{{PW}}", {"PW": V}, page)

    kwargs = type_event.call_args.kwargs
    assert kwargs["text"] == V
    assert kwargs["is_sensitive"] is True
    assert kwargs["sensitive_key_name"] == "PW"
    assert (status, action, value) == ("auto_ok", "input", "{{PW}}")
    assert "<PW>" in note
    assert performed == {"elem_1"}
    loc.fill.assert_not_awaited()
    _never_shown(all_logs, note)


async def test_plain_value_typed_as_today(type_event, all_logs):
    page, loc = _page([])

    note, status, _, value, _ = await _interact("plain", None, page)

    kwargs = type_event.call_args.kwargs
    assert kwargs["text"] == "plain"
    assert not kwargs.get("is_sensitive")
    assert note == "\n✅ AUTO-ACTION COMPLETE: Typed 'plain'"
    assert (status, value) == ("auto_ok", "plain")
    loc.input_value.assert_not_awaited()  # the field check is for secrets only


async def test_unknown_name_typed_literally(type_event, all_logs):
    page, loc = _page([])

    note, status, _, value, _ = await _interact("{{OTHER}}", {"PW": V}, page)

    kwargs = type_event.call_args.kwargs
    assert kwargs["text"] == "{{OTHER}}"
    assert not kwargs.get("is_sensitive")
    assert value == "{{OTHER}}"
    _never_shown(all_logs, note)


async def test_field_wrong_after_event_falls_through_to_fill(type_event, all_logs):
    page, loc = _page(["John" + V, V])  # the event appended to a pre-filled value

    note, status, _, value, performed = await _interact("{{PW}}", {"PW": V}, page)

    loc.fill.assert_awaited_once()
    assert loc.fill.await_args.args[0] == V
    assert (status, value) == ("auto_ok", "{{PW}}")
    assert "<PW>" in note
    assert performed == {"elem_1"}
    _never_shown(all_logs, note)


async def test_field_right_after_event_no_fill(type_event, all_logs):
    page, loc = _page([V])

    note, status, _, _, _ = await _interact("{{PW}}", {"PW": V}, page)

    loc.fill.assert_not_awaited()
    assert note == "\n✅ AUTO-ACTION COMPLETE: Typed '<PW>'"
    assert status == "auto_ok"


async def test_unreadable_field_keeps_event_result(type_event, all_logs):
    page, loc = _page([Exception(f"cannot read {V}")])

    note, status, _, _, performed = await _interact("{{PW}}", {"PW": V}, page)

    loc.fill.assert_not_awaited()
    assert status == "auto_ok"
    assert performed == {"elem_1"}
    _never_shown(all_logs, note)


async def test_reformatting_field_filled_once(type_event, all_logs):
    page, loc = _page(["(123)", "(123)"])  # a mask: the field never holds the value

    note, status, _, _, performed = await _interact("{{PW}}", {"PW": V}, page)

    loc.fill.assert_awaited_once()
    assert status == "auto_partial"
    assert performed == {"elem_1"}
    assert "(123)" not in note
    _never_shown(all_logs, note)


async def test_check_reads_through_the_frame(type_event, all_logs):
    page, loc = _page([V])

    await _interact(
        "{{PW}}",
        {"PW": V},
        page,
        locator_str="iframe#f >>> #pw",
        iframe_context="iframe#f",
    )

    page.frame_locator.assert_called_with("iframe#f")
    page.frame_locator.return_value.locator.assert_called_with("#pw")
    loc.input_value.assert_awaited_once()


async def test_fallback_failure_never_prints(type_event, all_logs):
    page, loc = _page([])
    loc.fill.side_effect = Exception(f"bad {V}")
    loc.evaluate.side_effect = Exception(f"bad {V}")

    note, status, _, value, performed = await _interact(
        "{{PW}}", {"PW": V}, page, event=FakeEvent(error=Exception(f"bad {V}"))
    )

    assert status == "auto_failed"
    assert "Exception" in note
    assert value == "{{PW}}"
    assert performed == set()
    _never_shown(all_logs, note)


async def test_flatpickr_secret_never_logged(type_event, all_logs):
    page, loc = _page([V])
    loc.evaluate.return_value = "no_fp"

    note, status, _, _, _ = await _interact(
        "{{PW}}", {"PW": V}, page, datepicker_framework="flatpickr"
    )

    assert "flatpickr.tier0_js_failed" in all_logs.text
    assert "<PW>" in all_logs.text
    assert loc.evaluate.await_args.args[1] == V  # the widget still gets the value
    assert status == "auto_ok"
    _never_shown(all_logs, note)


async def test_odd_characters_typed_exactly_and_never_printed(type_event, all_logs):
    page, loc = _page(["", V2])  # wrong after the event, right after fill

    note, status, _, _, _ = await _interact("{{ODD}}", {"ODD": V2}, page)

    assert type_event.call_args.kwargs["text"] == V2
    assert loc.fill.await_args.args[0] == V2
    assert status == "auto_ok"
    _never_shown(all_logs, note, V2)


async def test_secret_select_skips_the_event_path(type_event, all_logs):
    """browser-use's select event logs the option text with no sensitive switch."""
    page, loc = _page([])

    note, status, _, value, _ = await _interact("{{PW}}", {"PW": V}, page, action="select")

    type_event.selected.assert_not_called()
    assert loc.select_option.await_args.kwargs["label"] == V
    assert (status, value) == ("auto_ok", "{{PW}}")
    assert "<PW>" in note
    _never_shown(all_logs, note)


async def test_register_custom_actions_passes_secrets():
    from tests.test_agent.test_interaction_frame_wiring import (
        CHECKBOX,
        FakeAgent,
        FakeBrowserSession,
        _fake_playwright,
    )

    seen: dict = {}

    async def _recording_interaction(*args, **kwargs):
        seen.update(kwargs, specs=args[5])
        return "", "auto_ok", "input", "{{PW}}"

    engine = AsyncMock(
        return_value={
            "found": True,
            "best_locator": "#pw",
            "validated": True,
            "count": 1,
            "unique": True,
            "valid": True,
            "validation_method": "playwright",
        }
    )
    agent = FakeAgent()
    with (
        patch("browser_service.agent.actions.find_unique_locator_action", new=engine),
        patch("browser_service.agent.registration._do_interaction", new=_recording_interaction),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
    ):
        elements = [{"id": "elem_1", "action": "input", "value": "{{PW}}"}]
        assert register_custom_actions(agent, elements=elements, secrets={"PW": V}) is True
        await agent.tools.registry.execute_action(
            "find_unique_locator",
            {
                "x": 70,
                "y": 230,
                "element_id": "elem_1",
                "element_description": "the password field",
                "element_index": 12,
            },
            browser_session=FakeBrowserSession({12: CHECKBOX}),
        )

    assert seen["secrets"] == {"PW": V}
    assert seen["specs"]["elem_1"]["value"] == "{{PW}}"  # the spec keeps the name


MARK_SCRIPT = "el => el.setAttribute('nlrf-secret', '')"


async def test_secret_field_marked_before_typing(type_event, all_logs):
    """D8: the step's field is marked BEFORE the sensitive typing, so no later read shows it."""
    page, loc = _page([V])
    order: list = []

    async def _marked(script, **kwargs):
        order.append(("mark", script))

    loc.evaluate.side_effect = _marked
    type_event.side_effect = lambda **kw: order.append(("type", None)) or MagicMock()

    # `_session` hands back a plain event; the dispatch itself is what we order against.
    await _interact("{{PW}}", {"PW": V}, page)

    assert order[0] == ("mark", MARK_SCRIPT)
    assert ("type", None) in order
    assert order.index(("mark", MARK_SCRIPT)) < order.index(("type", None))


async def test_plain_value_never_marks(type_event, all_logs):
    page, loc = _page([])

    await _interact("plain", None, page)

    loc.evaluate.assert_not_awaited()


async def test_mark_failure_never_prints(type_event, all_logs):
    page, loc = _page([V])
    loc.evaluate.side_effect = Exception(f"bad {V}")

    note, status, _, _, performed = await _interact("{{PW}}", {"PW": V}, page)

    assert type_event.call_args.kwargs["text"] == V  # typing still happened
    assert status == "auto_ok"
    assert performed == {"elem_1"}
    assert "Exception" in all_logs.text
    _never_shown(all_logs, note)


@pytest.mark.integration
async def test_secret_field_carries_the_mark_on_a_real_page(type_event, all_logs):
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.set_content('<input id="u"><input id="v">')
            await _interact("{{PW}}", {"PW": V}, page, locator_str="#u")
            assert await page.locator("#u").get_attribute("nlrf-secret") == ""
            await _interact("plain", None, page, locator_str="#v")
            assert await page.locator("#v").get_attribute("nlrf-secret") is None
        finally:
            await browser.close()


@pytest.mark.integration
async def test_prefilled_field_corrected_on_a_real_page(type_event, all_logs):
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            # A fake event that leaves the field as it is: the check sees "John", fill() types V.
            page = await browser.new_page()
            await page.set_content('<input id="u" value="John">')
            note, status, _, _, _ = await _interact("{{PW}}", {"PW": V}, page, locator_str="#u")
            assert await page.locator("#u").input_value() == V
            assert status == "auto_ok"
            assert "<PW>" in note
            _never_shown(all_logs, note)

            # Negative: an event that leaves V in the field — no fill, the event's own note.
            page2 = await browser.new_page()
            await page2.set_content('<input id="u" value="John">')

            async def _types_it():
                await page2.locator("#u").fill(V)

            note2, status2, _, _, _ = await _interact(
                "{{PW}}", {"PW": V}, page2, event=FakeEvent(on_await=_types_it), locator_str="#u"
            )
            assert note2 == "\n✅ AUTO-ACTION COMPLETE: Typed '<PW>'"
            assert status2 == "auto_ok"
            assert await page2.locator("#u").input_value() == V
            _never_shown(all_logs, note2)
        finally:
            await browser.close()
