"""
The click-to-open tier waits for the dropdown THIS click opened (browser-service #40).

For a custom dropdown, ``_do_interaction_playwright`` clicks the control, waits
for an option container, then clicks the option. The wait used to take the
FIRST container of the document — a hidden menu of another dropdown, a navbar
menu, a listbox that was already showing — so the second dropdown of a page
timed out, and a visible container elsewhere could be mistaken for the one the
click opened. The option search also looked inside hidden containers.

Now the wait is for a container the click made visible: one beside the control
when its parent holds any (``nth(visible_in_parent)``), else anywhere (a portal
at the end of <body>). Options are searched only in visible containers.

Every page is ``click_to_open_dropdowns.html?shape=<key>`` — a fresh document
per test. An option click records ``"<dropdown>:<option>"`` in
``window.__picks``, so a test sees WHICH dropdown was acted on. ``None`` means
the honest answer is "not performed": nothing is picked and the step fails.
``click_to_open_dropdowns_frame.html`` hosts the same page in ``iframe#app-frame``;
the host's own ``window.__picks`` must stay empty.
"""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.registration import _do_interaction_playwright

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "test_locators" / "locator_fixtures"

FRAME = 'iframe[id="app-frame"]'
TOM_SELECT = {"dropdown_framework": "tom-select"}

# (shape, trigger, value, flags, expected pick — None = honest auto_failed)
MAIN_SHAPES = [
    pytest.param("one", "#size-trigger", "Large", {}, "size:Large", id="one"),
    pytest.param("two", "#colour-trigger", "Green", {}, "colour:Green", id="two-select-first"),
    pytest.param("two", "#size-trigger", "Large", {}, "size:Large", id="two-select-second"),
    pytest.param("portal-same", "#to-trigger", "Bet", {}, "to:Beta", id="portal-same-partial"),
    pytest.param("portal-same", "#to-trigger", "Beta", {}, "to:Beta", id="portal-same"),
    pytest.param("shared-parent", "#size-trigger", "Beta", {}, "size:Beta", id="shared-parent"),
    pytest.param("tom-select", "#ts2-ts-control", "Beta", TOM_SELECT, "ts2:Beta", id="tom-select"),
    pytest.param(
        "navbar-select2", "#country-trigger", "India", {}, "country:India", id="navbar-select2"
    ),
    pytest.param(
        "created-in-parent", "#size-trigger", "Large", {}, "size:Large", id="created-in-parent"
    ),
    pytest.param("delayed", "#size-trigger", "Large", {}, "size:Large", id="delayed"),
    pytest.param(
        "unrelated-visible-before",
        "#size-trigger",
        "Large",
        {},
        "size:Large",
        id="unrelated-visible-before",
    ),
    pytest.param(
        "nothing-opens-listbox-after",
        "#size-trigger",
        "Large",
        {},
        None,
        id="nothing-opens-listbox-after",
    ),
    pytest.param(
        "nothing-opens-listbox-before",
        "#size-trigger",
        "Large",
        {},
        None,
        id="nothing-opens-listbox-before",
    ),
    pytest.param(
        "portal-nothing-opens-listbox-before",
        "#size-trigger",
        "Large",
        {},
        None,
        id="portal-nothing-opens-listbox-before",
    ),
    pytest.param("opens-other-near", "#size-trigger", "Large", {}, None, id="opens-other-near"),
    pytest.param(
        "wide-parent-nothing-opens",
        "#size-trigger",
        "Large",
        {},
        None,
        id="wide-parent-nothing-opens",
    ),
    # Pins an ACCEPTED COST of the design (an honest failure where main selects), not the ideal.
    pytest.param(
        "always-visible-listbox", "#lb-menu", "Large", {}, None, id="always-visible-listbox"
    ),
]

FRAME_SHAPES = [
    pytest.param("two", "#size-trigger", "Large", {}, "size:Large", id="two-select-second"),
    pytest.param("portal-same", "#to-trigger", "Beta", {}, "to:Beta", id="portal-same"),
    pytest.param("shared-parent", "#size-trigger", "Beta", {}, "size:Beta", id="shared-parent"),
    pytest.param("tom-select", "#ts2-ts-control", "Beta", TOM_SELECT, "ts2:Beta", id="tom-select"),
    pytest.param(
        "nothing-opens-listbox-after",
        "#size-trigger",
        "Large",
        {},
        None,
        id="nothing-opens-listbox-after",
    ),
    pytest.param(
        "portal-nothing-opens-listbox-before",
        "#size-trigger",
        "Large",
        {},
        None,
        id="portal-nothing-opens-listbox-before",
    ),
]


def _file_url(name: str, shape: str) -> str:
    return (FIXTURES_DIR / name).resolve().as_uri() + f"?shape={shape}"


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


async def _select(page, root, locator: str, trigger: str, value: str, **kwargs):
    await root.locator(trigger).wait_for(state="attached")
    performed: set = set()
    note, status = await _do_interaction_playwright(
        page, locator, "select", value, "elem_1", performed, **kwargs
    )
    return note, status, performed


def _assert_outcome(picks, note, status, performed, value, expected):
    assert picks == ([expected] if expected else []), note
    assert status == ("auto_ok" if expected else "auto_failed"), note
    # performed_actions is added to only on success.
    assert ("elem_1" in performed) == bool(expected), note
    if expected:
        # Which loop matched: exact when the value is the whole option text, else partial.
        loop = "exact" if value == expected.split(":")[1] else "partial"
        assert f"(click-to-open, {loop}," in note, note


@pytest.mark.parametrize("shape, trigger, value, flags, expected", MAIN_SHAPES)
async def test_main_page_select_acts_on_the_dropdown_the_click_opened(
    page, shape, trigger, value, flags, expected
):
    await page.goto(_file_url("click_to_open_dropdowns.html", shape), wait_until="load")

    note, status, performed = await _select(page, page, trigger, trigger, value, **flags)

    picks = await page.locator(":root").evaluate("() => window.__picks")
    _assert_outcome(picks, note, status, performed, value, expected)


@pytest.mark.parametrize("shape, trigger, value, flags, expected", FRAME_SHAPES)
async def test_in_frame_select_acts_on_the_dropdown_the_click_opened(
    page, shape, trigger, value, flags, expected
):
    await page.goto(_file_url("click_to_open_dropdowns_frame.html", shape), wait_until="load")
    frame = page.frame_locator(FRAME)

    note, status, performed = await _select(
        page, frame, f"{FRAME} >>> {trigger}", trigger, value, iframe_context=FRAME, **flags
    )

    picks = await frame.locator(":root").evaluate("() => window.__picks")
    _assert_outcome(picks, note, status, performed, value, expected)
    assert await page.evaluate("() => window.__picks") == [], note
