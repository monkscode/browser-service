"""
Task 5 (analysis doc A1/C7) — classifier-gate the TEXT-FIRST checkbox
detour.

The legacy detour in ``_find_element_by_expected_text`` routed nearly
every short-text element into checkbox hunting (``len(text) < 30``) and
returned whatever the label/adjacency/nth strategies found, stamped
``validated: true`` with no verification. These tests pin the fixed
behavior:

  1. Finder level — ``find_checkbox_or_radio_by_label`` prefers an exact
     label match over substring, and no longer maps a trailing digit to
     the Nth checkbox on the page.
  2. Gate level — the detour runs only on real evidence (description
     keywords or vision hint) corroborated by the DOM probe at the click
     coordinates; never on text length alone. Inside iframes the probe
     cannot see into the frame, so evidence alone gates.

Same conventions as test_locator_fixtures.py: real headless Chromium
against static fixtures in ``locator_fixtures/``, marked integration.
"""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.handlers.checkbox import (
    find_checkbox_or_radio_by_label,
)
from browser_service.locators.smart_locator import (
    _find_element_by_expected_text,
)

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent / "locator_fixtures"


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


async def _center(page, selector: str) -> tuple:
    box = await page.locator(selector).bounding_box()
    assert box is not None, f"no bounding box for {selector}"
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def _is_checkbox_result(result) -> bool:
    return bool(result) and result.get("element_type") in ("checkbox", "radio")


# ======================================================================
# Finder level — find_checkbox_or_radio_by_label
# ======================================================================


async def test_finder_prefers_exact_label_over_substring(page):
    """'Option 1' must resolve to the 'Option 1' checkbox even when
    'Option 10' appears first in the DOM (has-text is substring)."""
    await page.set_content("""
        <label><input type="checkbox" id="cb-opt10"> Option 10</label>
        <label><input type="checkbox" id="cb-opt1"> Option 1</label>
    """)
    result = await find_checkbox_or_radio_by_label(page, "Option 1")
    assert result is not None
    assert await page.locator(result["locator"]).get_attribute("id") == "cb-opt1"


async def test_finder_substring_still_matches_decorated_labels(page):
    """No exact label exists — substring fallback must still find
    'Option 1 (recommended)' for 'Option 1'."""
    await page.set_content("""
        <label><input type="checkbox" id="cb-opt1"> Option 1 (recommended)</label>
    """)
    result = await find_checkbox_or_radio_by_label(page, "Option 1")
    assert result is not None
    assert await page.locator(result["locator"]).get_attribute("id") == "cb-opt1"


async def test_finder_trailing_number_strategy_removed(page):
    """A trailing digit in the label must NOT map to the Nth checkbox on
    the page. No label matches 'Option 2' here -> finder returns None."""
    await page.set_content("""
        <input type="checkbox" id="cb-a">
        <input type="checkbox" id="cb-b">
        <input type="checkbox" id="cb-c">
    """)
    result = await find_checkbox_or_radio_by_label(page, "Option 2")
    assert result is None, f"trailing-number strategy returned {result}"


# ======================================================================
# Gate level — the TEXT-FIRST detour in _find_element_by_expected_text
# ======================================================================


async def test_pagination_step_never_returns_checkbox(page):
    """The reproduced wrong-element bug: 'Page 2' on a page with five
    filter checkboxes must never resolve to a checkbox."""
    await page.goto(_file_url("checkbox_gate_page.html"),
                    wait_until="domcontentloaded")
    x, y = await _center(page, 'a:text-is("Page 2")')
    result = await _find_element_by_expected_text(
        page, "Page 2", "pagination control for page 2", x, y,
        vision_type_hint=None, probe_page=page, iframe_context=None,
    )
    assert not _is_checkbox_result(result), f"got checkbox: {result}"
    # The standard text search should find the pagination link itself.
    assert result is not None
    tag = await page.locator(result["locator"]).evaluate("e => e.tagName")
    assert tag == "A"


async def test_short_text_alone_does_not_enter_checkbox_hunt(page):
    """Text length must not trigger the detour. 'Sign Up' sits right
    after a checkbox (the adjacency strategy would grab it)."""
    await page.goto(_file_url("checkbox_gate_page.html"),
                    wait_until="domcontentloaded")
    x, y = await _center(page, "#signup-text")
    result = await _find_element_by_expected_text(
        page, "Sign Up", "the Sign Up element", x, y,
        vision_type_hint=None, probe_page=page, iframe_context=None,
    )
    assert not _is_checkbox_result(result), f"got checkbox: {result}"


async def test_select_the_phrase_does_not_enter_checkbox_hunt(page):
    """'select the …' is dropdown phrasing, not checkbox evidence."""
    await page.goto(_file_url("checkbox_gate_page.html"),
                    wait_until="domcontentloaded")
    x, y = await _center(page, "#signup-text")
    result = await _find_element_by_expected_text(
        page, "Sign Up", "select the promo option", x, y,
        vision_type_hint=None, probe_page=page, iframe_context=None,
    )
    assert not _is_checkbox_result(result), f"got checkbox: {result}"


async def test_check_the_keyword_without_structure_is_probe_rejected(page):
    """'check the …' can mean 'verify'. Coordinates on a plain paragraph
    with no checkbox structure nearby -> probe rejects, detour skipped,
    trailing '2' must not map to the 2nd checkbox on the page."""
    await page.goto(_file_url("checkbox_gate_page.html"),
                    wait_until="domcontentloaded")
    x, y = await _center(page, "#account-note")
    result = await _find_element_by_expected_text(
        page, "Account 2", "check the account 2 status", x, y,
        vision_type_hint=None, probe_page=page, iframe_context=None,
    )
    assert not _is_checkbox_result(result), f"got checkbox: {result}"


async def test_genuine_checkbox_keyword_and_probe_detour_works(page):
    """Evidence + probe agreement: a real checkbox step must still get
    the input-targeting locator (clicking bare label text may not
    toggle)."""
    await page.goto(_file_url("checkbox_gate_page.html"),
                    wait_until="domcontentloaded")
    x, y = await _center(page, "#remember-label")
    result = await _find_element_by_expected_text(
        page, "Remember me", "check the Remember me checkbox", x, y,
        vision_type_hint=None, probe_page=page, iframe_context=None,
    )
    assert _is_checkbox_result(result), f"expected checkbox, got: {result}"
    name = await page.locator(result["locator"]).get_attribute("name")
    assert name == "remember"


async def test_vision_hint_checkbox_enters_detour(page):
    """No keywords in the description — the vision hint alone (plus
    probe agreement) must open the gate."""
    await page.goto(_file_url("checkbox_gate_page.html"),
                    wait_until="domcontentloaded")
    x, y = await _center(page, "#remember-label")
    result = await _find_element_by_expected_text(
        page, "Remember me", "the Remember me element", x, y,
        vision_type_hint="checkbox", probe_page=page, iframe_context=None,
    )
    assert _is_checkbox_result(result), f"expected checkbox, got: {result}"


async def test_iframe_checkbox_keyword_gate_skips_probe(page):
    """Inside an iframe the probe cannot see the frame's DOM — evidence
    alone must gate, and the detour must still find the checkbox via the
    frame-scoped search context."""
    await page.goto(_file_url("checkbox_gate_iframe_outer.html"),
                    wait_until="domcontentloaded")
    frame = page.frame_locator("#prefs-frame")
    result = await _find_element_by_expected_text(
        frame, "Remember me", "check the Remember me checkbox", 100, 100,
        vision_type_hint=None, probe_page=page,
        iframe_context='iframe[id="prefs-frame"]',
    )
    assert _is_checkbox_result(result), f"expected checkbox, got: {result}"
