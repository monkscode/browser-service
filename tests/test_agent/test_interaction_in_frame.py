"""
The Playwright interaction fallback on real Chromium (browser-service #36).

After a locator is found the service performs the step's action on the page.
When browser-use's own event cannot do it, ``_do_interaction_playwright`` runs
a chain of Playwright tiers. Until now that chain had no test on a real
browser (test_interaction_reliability.py mocks the page), so this file pins
every action on a main-page control first — the behaviour that must not
change when the chain learns about frames.

The fixture: ``interaction_frame_inner.html`` holds one control per action —
two checkboxes, a text input, a button, a native <select>, a flatpickr input,
a Tom Select and a custom click-to-open dropdown.
"""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.registration import _do_interaction_playwright

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
    pytest.param("id=colour-trigger", "select", "Green", {}, {"colour": "Green"}, id="click-to-open"),
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
        page, "id=email", "select", "Gold", dropdown_framework="tom-select", select_id="pricelist_id"
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
