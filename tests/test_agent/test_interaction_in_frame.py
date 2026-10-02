"""
The Playwright interaction fallback on real Chromium (browser-service #36).

After a locator is found the service performs the step's action on the page.
When browser-use's own event cannot do it, ``_do_interaction_playwright`` runs
a chain of Playwright tiers. Until now that chain had no test on a real
browser (test_interaction_reliability.py mocks the page), so this file pins
every action on a main-page control first — the behaviour that must not
change when the chain learns about frames.

Then the same actions with the control inside an iframe. A locator found in a
frame is returned as ``iframe[id="x"] >>> <locator>`` — Browser Library's
frame piercing, which Playwright itself reads as "a child of the iframe
element". The chain handed that string to ``page.locator`` and so could never
act in a frame; its Tom Select tier ran on the PAGE's document and set a
same-id widget in the main page instead. The step's frame now reaches the
chain, which resolves through ``page.frame_locator(frame)`` — the mechanism
the validator uses, and the one Browser Library turns ``>>>`` into.

The fixtures: ``interaction_frame_inner.html`` holds one control per action —
two checkboxes, a text input, a button, a native <select>, a flatpickr input,
a Tom Select and a custom click-to-open dropdown.
``interaction_frame_outer.html`` hosts it in ``iframe#app-frame`` next to
main-page twins of the same selectors, so an action that lost its frame is
seen on the twin.
"""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.registration import _do_interaction, _do_interaction_playwright

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "test_locators" / "locator_fixtures"

# What the fixture's controls hold, read inside the document that owns them.
STATE_JS = """(_root) => ({
    remember: document.querySelector('[name=remember]').checked,
    newsletter: document.querySelector('[name=newsletter]').checked,
    email: document.getElementById('email').value,
    clicked: document.getElementById('save').dataset.clicked || null,
    plan: document.getElementById('plan').value,
    date: document.getElementById('event_date').value,
    pricelist: document.getElementById('pricelist_id').value,
    colour: document.getElementById('colour-trigger').dataset.picked || null,
})"""

UNTOUCHED = {
    "remember": False,
    "newsletter": True,
    "email": "",
    "clicked": None,
    "plan": "f",
    "date": "",
    "pricelist": "1",
    "colour": None,
}

TOM_SELECT = "id=pricelist_id-ts-control"

# One row per tier of the chain: the inner locator, the step, the widget
# flags the locator result would carry, and the ONE change the step must make.
ACTIONS = [
    pytest.param('[name="remember"]', "check", "", {}, {"remember": True}, id="check"),
    pytest.param('[name="newsletter"]', "uncheck", "", {}, {"newsletter": False}, id="uncheck"),
    pytest.param("id=save", "click", "", {}, {"clicked": "yes"}, id="click"),
    pytest.param("id=email", "input", "a@b.co", {}, {"email": "a@b.co"}, id="input"),
    pytest.param("id=plan", "select", "Pro", {}, {"plan": "p"}, id="native-select"),
    pytest.param(
        TOM_SELECT,
        "select",
        "Gold",
        {"dropdown_framework": "tom-select", "select_id": "pricelist_id"},
        {"pricelist": "2"},
        id="tom-select-by-select-id",
    ),
    pytest.param(
        TOM_SELECT,
        "select",
        "Gold",
        {"dropdown_framework": "tom-select"},
        {"pricelist": "2"},
        id="tom-select-by-locator",
    ),
    pytest.param(
        "id=event_date",
        "input",
        "2026-07-01",
        {"datepicker_framework": "flatpickr"},
        {"date": "2026-07-01"},
        id="flatpickr",
    ),
    pytest.param(
        "id=colour-trigger", "select", "Green", {}, {"colour": "Green"}, id="click-to-open"
    ),
]


def _file_url(name: str) -> str:
    return (FIXTURES_DIR / name).resolve().as_uri()


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page_obj = await ctx.new_page()
        try:
            yield page_obj
        finally:
            await browser.close()


async def _changed(root) -> dict:
    """The controls of ``root``'s document that no longer hold their initial value."""
    now = await root.locator(":root").evaluate(STATE_JS)
    return {k: v for k, v in now.items() if UNTOUCHED[k] != v}


async def _act(page, locator: str, action: str, value: str, **kwargs):
    performed: set = set()
    note, status = await _do_interaction_playwright(
        page, locator, action, value, "elem_1", performed, **kwargs
    )
    return note, status, performed


# ─────────────────────────────────────────────────────────────────────────────
# Pins — the chain on a MAIN-PAGE control. Green before #36 and after it.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("locator, action, value, flags, change", ACTIONS)
async def test_main_page_action_is_performed(page, locator, action, value, flags, change):
    await page.goto(_file_url("interaction_frame_inner.html"), wait_until="load")

    note, status, performed = await _act(page, locator, action, value, **flags)

    assert status == "auto_ok", note
    assert await _changed(page) == change
    assert performed == {"elem_1"}


async def test_main_page_tom_select_by_select_id_does_not_need_the_locator(page):
    """The select-id tier looks the <select> up in the document itself: it
    works even when the locator names an element that is no Tom Select."""
    await page.goto(_file_url("interaction_frame_inner.html"), wait_until="load")

    note, status, _ = await _act(
        page,
        "id=email",
        "select",
        "Gold",
        dropdown_framework="tom-select",
        select_id="pricelist_id",
    )

    assert status == "auto_ok", note
    assert await _changed(page) == {"pricelist": "2"}


async def test_main_page_tom_select_falls_back_to_the_locator_when_the_id_is_wrong(page):
    """A select id that names nothing must not end the step: the tier that
    walks from the located control to its <select> still selects."""
    await page.goto(_file_url("interaction_frame_inner.html"), wait_until="load")

    note, status, _ = await _act(
        page, TOM_SELECT, "select", "Gold", dropdown_framework="tom-select", select_id="no_such_id"
    )

    assert status == "auto_ok", note
    assert await _changed(page) == {"pricelist": "2"}


# ─────────────────────────────────────────────────────────────────────────────
# The same chain with the control inside an iframe (#36).
# ─────────────────────────────────────────────────────────────────────────────

FRAME = 'iframe[id="app-frame"]'

# The outer page's twins of the frame's controls.
TWIN_STATE_JS = """() => ({
    remember: document.querySelector('[name=remember]').checked,
    email: document.getElementById('email').value,
    clicked: document.getElementById('save').dataset.clicked || null,
    pricelist: document.getElementById('pricelist_id').value,
})"""

TWINS_UNTOUCHED = {"remember": False, "email": "", "clicked": None, "pricelist": "1"}


async def _open_host(page):
    await page.goto(_file_url("interaction_frame_outer.html"), wait_until="load")
    return page.frame_locator(FRAME)


@pytest.mark.parametrize("locator, action, value, flags, change", ACTIONS)
async def test_in_frame_action_is_performed_in_the_frame(
    page, locator, action, value, flags, change
):
    frame = await _open_host(page)

    note, status, performed = await _act(
        page, f"{FRAME} >>> {locator}", action, value, iframe_context=FRAME, **flags
    )

    assert status == "auto_ok", note
    assert await _changed(frame) == change
    assert await page.evaluate(TWIN_STATE_JS) == TWINS_UNTOUCHED
    assert performed == {"elem_1"}


@pytest.mark.parametrize(
    "frame_selector",
    [
        'iframe[id="app-frame"]',
        'iframe[title="Application"]',
        "iframe.app-shell",
        "iframe >> nth=1",
    ],
)
async def test_every_frame_selector_form_the_detector_emits_is_entered(page, frame_selector):
    frame = await _open_host(page)

    note, status, _ = await _act(
        page, f'{frame_selector} >>> [name="remember"]', "check", "", iframe_context=frame_selector
    )

    assert status == "auto_ok", note
    assert await _changed(frame) == {"remember": True}
    assert await page.evaluate(TWIN_STATE_JS) == TWINS_UNTOUCHED


async def test_the_frame_argument_decides_where_the_step_acts_not_the_string(page):
    """A locator that reaches the chain without its hop is still resolved in
    the step's frame — never at page level, where it names the twin."""
    frame = await _open_host(page)

    note, status, _ = await _act(page, '[name="remember"]', "check", "", iframe_context=FRAME)

    assert status == "auto_ok", note
    assert await _changed(frame) == {"remember": True}
    assert await page.evaluate(TWIN_STATE_JS) == TWINS_UNTOUCHED


@pytest.mark.parametrize(
    "frame_selector",
    [
        pytest.param('iframe[id="gone"]', id="no-such-frame"),
        pytest.param("iframe", id="two-frames-match"),
    ],
)
async def test_a_frame_that_cannot_be_entered_fails_and_touches_nothing(page, frame_selector):
    """The inner locator alone names the main-page twin: when the frame cannot
    be entered the step is reported as not performed, not tried at page level."""
    frame = await _open_host(page)

    note, status, performed = await _act(
        page, f'{frame_selector} >>> [name="remember"]', "check", "", iframe_context=frame_selector
    )

    assert status == "auto_failed", note
    assert await _changed(frame) == {}
    assert await page.evaluate(TWIN_STATE_JS) == TWINS_UNTOUCHED
    assert performed == set()


@pytest.mark.parametrize(
    "locator, action, change",
    [
        # check skips browser-use's event path by design; a click with no
        # element index has no event path to take.
        pytest.param('[name="remember"]', "check", {"remember": True}, id="check"),
        pytest.param("id=save", "click", {"clicked": "yes"}, id="click-without-an-index"),
    ],
)
async def test_the_orchestrator_hands_the_frame_to_the_chain(page, locator, action, change):
    frame = await _open_host(page)
    performed: set = set()

    note, status, done_action, _ = await _do_interaction(
        None,
        page,
        f"{FRAME} >>> {locator}",
        "elem_1",
        None,
        {"elem_1": {"action": action, "value": ""}},
        performed,
        iframe_context=FRAME,
    )

    assert (status, done_action) == ("auto_ok", action), note
    assert await _changed(frame) == change
    assert await page.evaluate(TWIN_STATE_JS) == TWINS_UNTOUCHED


# ─────────────────────────────────────────────────────────────────────────────
# The option list rendered away from its trigger (the Select2 shape): only the
# document-wide search of the click-to-open tier can find the option.
# ─────────────────────────────────────────────────────────────────────────────

DETACH_MENU_JS = "(_root) => { document.body.appendChild(document.getElementById('colour-menu')); }"


async def test_main_page_option_list_outside_the_triggers_parent_is_found(page):
    await page.goto(_file_url("interaction_frame_inner.html"), wait_until="load")
    await page.locator(":root").evaluate(DETACH_MENU_JS)

    note, status, _ = await _act(page, "id=colour-trigger", "select", "Green")

    assert status == "auto_ok", note
    assert await _changed(page) == {"colour": "Green"}


async def test_in_frame_option_list_outside_the_triggers_parent_is_found_in_the_frame(page):
    """The document-wide search is the FRAME's document: the main page holds
    no dropdown at all, so a page-level search finds nothing."""
    frame = await _open_host(page)
    await frame.locator(":root").evaluate(DETACH_MENU_JS)

    note, status, _ = await _act(
        page, f"{FRAME} >>> id=colour-trigger", "select", "Green", iframe_context=FRAME
    )

    assert status == "auto_ok", note
    assert await _changed(frame) == {"colour": "Green"}
