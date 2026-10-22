"""
The input fallback leaves the page the generated test will be on (browser-service #38).

When browser-use's own typing event does not run, ``_do_interaction_playwright``
types in tiers. Its second tier called ``loc.triple_click()``, which Playwright's
Python ``Locator`` does not have, so it never ran; a ``fill()`` that the page
reformatted (a mask, ``maxlength``, an OTP box) then fell to the JavaScript tier,
which wrote the raw value over the page's own result and fired an extra
``input`` and ``change`` event.

The generated test types with Browser Library's ``Fill Text``, which is
``locator.fill`` — the same call as the first tier (``Type Text`` clears with
``locator.fill`` too, then types keys). So the rule pinned here:
after the fallback, the page is exactly what ``locator.fill`` leaves on a fresh
copy of it (values, events, what became visible). Where ``fill`` cannot type at
all (a key-driven ``<div>``), the JavaScript tier still runs as before: the
test's own ``Fill Text`` fails on that line too, and sending clicks and keys
there was measured to tick checkboxes, press buttons and fire page shortcuts.
"""

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.registration import _do_interaction, _do_interaction_playwright
from browser_service.prompts.templates import SEQUENTIAL_PROCESSING_RULES

pytestmark = pytest.mark.integration

# Counts events at the document, so an extra input / change event shows.
EVENTS = """<script>window.ev = {input: 0, change: 0, keydown: 0, click: 0};
for (const k in window.ev) document.addEventListener(k, () => window.ev[k]++, true);</script>"""

MASK = """const m = document.getElementById('t');
m.addEventListener('input', () => { const d = m.value.replace(/\\D/g, '').slice(0, 7);
  m.value = d.length > 3 ? d.slice(0, 3) + '-' + d.slice(3) : d; });"""

# (id, html, value) — a page whose own code may reformat what fill() types.
SHAPES = [
    ("maxlength", '<input id="t" maxlength="3">', "hello"),
    ("maxlength_twin", '<input id="t" maxlength="10">', "hello"),
    (
        "mask_and_change",
        f'<input id="t"><p id="dep" hidden>shown on change</p><script>{MASK}'
        "m.addEventListener('change', () => document.getElementById('dep').hidden = false);</script>",
        "5551234",
    ),
    ("mask_events", f'<input id="t"><script>{MASK}</script>', "5551234"),
    (
        "otp_boxes",
        '<input id="t" maxlength="1"><input id="o2" maxlength="1"><input id="o3" maxlength="1">'
        "<script>for (const b of document.querySelectorAll('input')) b.addEventListener('input', () => {"
        " if (b.value.length === 1 && b.nextElementSibling) b.nextElementSibling.focus(); });</script>",
        "123",
    ),
    ("plain_twin", '<input id="t">', "hello"),
    # input_value() raises on a contenteditable editor: fill()'s text is kept, never a JS write
    (
        "contenteditable",
        '<div id="t" contenteditable="true" style="width:200px;height:30px"></div>',
        "hello",
    ),
]

# What a document holds after typing: every control's value, what is visible, the events fired.
STATE_JS = """() => ({
    values: [...document.querySelectorAll('input, textarea')].map(e => e.value),
    texts: [...document.querySelectorAll('[contenteditable]')].map(e => e.textContent),
    dep_visible: !!document.getElementById('dep') && !document.getElementById('dep').hidden,
    ev: window.ev,
})"""


@pytest.fixture
async def browser():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True)
        try:
            yield b
        finally:
            await b.close()


async def _fresh(browser, html: str):
    page = await browser.new_page()
    await page.set_content(EVENTS + html)
    return page


async def _framed(browser, html: str):
    doc = (EVENTS + html).replace("&", "&amp;").replace('"', "&quot;")
    page = await browser.new_page()
    await page.set_content(f'<input id="t"><iframe id="f" srcdoc="{doc}"></iframe>')
    await page.wait_for_function(
        "document.getElementById('f').contentDocument.getElementById('t') !== null"
    )
    return page


@pytest.mark.parametrize("shape_id, html, value", SHAPES, ids=[s[0] for s in SHAPES])
async def test_the_fallback_leaves_the_page_fill_leaves(browser, shape_id, html, value):
    reference = await _fresh(browser, html)
    await reference.locator("#t").fill(value)
    expected = await reference.evaluate(STATE_JS)

    page = await _fresh(browser, html)
    performed: set = set()
    note, status = await _do_interaction_playwright(page, "#t", "input", value, "elem_1", performed)

    assert await page.evaluate(STATE_JS) == expected
    assert status in ("auto_ok", "auto_partial")
    assert "elem_1" in performed
    assert "JS" not in note  # fill() answered; the JavaScript tier did not write over it


async def test_a_value_the_page_shortened_is_named_in_the_note(browser):
    page = await _fresh(browser, '<input id="t" maxlength="3">')
    note, status = await _do_interaction_playwright(page, "#t", "input", "hello", "elem_1", set())

    assert status == "auto_partial"
    assert "field shows 'hel'" in note


async def test_the_fallback_leaves_the_frame_fill_leaves(browser):
    reference = await _framed(browser, '<input id="t" maxlength="3">')
    await reference.frame_locator('iframe[id="f"]').locator("#t").fill("hello")
    expected = await reference.frames[1].evaluate(STATE_JS)

    page = await _framed(browser, '<input id="t" maxlength="3">')
    await _do_interaction_playwright(
        page,
        'iframe[id="f"] >>> #t',
        "input",
        "hello",
        "elem_1",
        set(),
        iframe_context='iframe[id="f"]',
    )

    assert await page.frames[1].evaluate(STATE_JS) == expected
    assert await page.locator("#t").input_value() == ""  # the main-page twin is untouched


async def test_the_orchestrator_with_no_element_index_keeps_fill_result(browser):
    page = await _fresh(browser, '<input id="t" maxlength="3">')
    performed: set = set()
    _note, status, action, _value = await _do_interaction(
        None,
        page,
        "#t",
        "elem_1",
        None,
        {"elem_1": {"action": "input", "value": "hello"}},
        performed,
    )

    assert (status, action) == ("auto_partial", "input")
    assert await page.locator("#t").input_value() == "hel"


async def test_a_key_driven_element_gets_no_click_and_no_key(browser):
    """fill() refuses a focusable <div> that types on keydown, and so does the
    test's Fill Text / Type Text. The JavaScript tier answers as before; nothing
    clicks it or sends it keys (a key tier would also hit checkboxes, buttons and
    page shortcuts when the step points at the wrong element)."""
    page = await _fresh(
        browser,
        # a size, so a click could land on it (an empty div has none, and a click there would time out)
        '<div id="t" tabindex="0" style="width:200px;height:30px"></div><script>const d = document.getElementById("t");'
        'd.addEventListener("keydown", e => { if (e.key.length === 1) d.textContent += e.key; });</script>',
    )
    note, status = await _do_interaction_playwright(page, "#t", "input", "hello", "elem_1", set())

    assert status == "auto_partial"
    assert "(JS, unverified)" in note
    assert await page.locator("#t").text_content() == ""
    ev = await page.evaluate("() => window.ev")
    assert (ev["click"], ev["keydown"]) == (0, 0)


async def test_a_readonly_field_is_set_without_a_click(browser):
    """A date field that opens its calendar on click stays closed: fill() waits
    for an editable field, then the JavaScript tier sets the value."""
    page = await _fresh(
        browser,
        '<input id="t" readonly><div id="cal" hidden>calendar</div>'
        '<script>document.getElementById("t").addEventListener("click",'
        ' () => document.getElementById("cal").hidden = false);</script>',
    )
    _note, status = await _do_interaction_playwright(
        page, "#t", "input", "2026-07-01", "elem_1", set()
    )

    assert status == "auto_partial"
    assert await page.locator("#t").input_value() == "2026-07-01"
    assert await page.locator("#cal").is_hidden()


def test_the_agent_is_told_the_typing_tiers_that_exist():
    assert (
        "input/type: auto-filled using layered strategies (fill → JS)"
        in SEQUENTIAL_PROCESSING_RULES
    )
