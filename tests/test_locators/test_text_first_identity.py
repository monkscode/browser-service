"""
Task 8 (analysis doc A5 + A6) — identity-check lucky singletons; report
description-derived locators honestly.

A5: ``_find_element_by_expected_text`` accepted a ``count == 1`` text match
without ever checking it is the element the vision model saw at (x, y) —
coordinates were only consulted for ``count > 1`` disambiguation. A hidden
node with the same text (icon button rendered without text content, its
label duplicated in a collapsed menu) was returned as a priority-0,
``semantic_match=True`` locator. Pinned here:

  1. A singleton that is hidden or nowhere near the coordinates is
     rejected and the NEXT selector in the list gets its chance — in the
     icon-button case ``role=button[name=…]`` finds the right element.
  2. A singleton at the coordinates is accepted exactly as before, and a
     singleton with no coordinates available is accepted unchecked
     (nothing to check against).

A6: STEP 2's semantic check is gated on ``if expected_text:`` — absent,
a locator derived from description keywords (substring matchers like
``[title*="…"]``) was returned stamped ``semantic_match=True``. Acceptance
is unchanged (owner decision 2026-07-06: honest-reporting minimum, no
surface synthesis), but it now reports ``semantic_match=False`` with
``validation_method="description_derived"``.

Same conventions as test_form_control_semantics.py: real headless Chromium
against static fixtures in ``locator_fixtures/``, marked integration.
"""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.smart_locator import (
    _find_element_by_expected_text,
    find_unique_locator_at_coordinates,
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


async def _center(page, locator: str) -> tuple:
    box = await page.locator(locator).bounding_box()
    return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


# ======================================================================
# A5 — TEXT-FIRST singleton identity check
# ======================================================================


async def test_hidden_singleton_rejected_next_selector_wins(page):
    """text="Search" uniquely matches a HIDDEN menu span, not the icon
    button vision saw at the coords. The hidden match must be rejected so
    role=button[name="Search"] (later in the same list) finds the real
    element."""
    await page.goto(_file_url("text_first_identity.html"))
    x, y = await _center(page, 'button[aria-label="Search"]')
    result = await _find_element_by_expected_text(
        page, "Search", "Search button", x, y, probe_page=page
    )
    assert result is not None
    assert result["locator"] == 'role=button[name="Search"]'


async def test_far_singleton_rejected(page):
    """A unique visible text match hundreds of px from the coords is not
    the element vision saw; with no other selector matching, TEXT-FIRST
    returns None and the pipeline's later steps take over."""
    await page.goto(_file_url("text_first_identity.html"))
    x, y = await _center(page, "button:has-text('Submit')")
    result = await _find_element_by_expected_text(
        page, "Promotions", "Promotions banner", x, y, probe_page=page
    )
    assert result is None


async def test_singleton_at_coords_accepted(page):
    """The mainline case (375 of 375 historical singleton accepts had
    coords available): a unique text match AT the coordinates is accepted
    exactly as before."""
    await page.goto(_file_url("text_first_identity.html"))
    x, y = await _center(page, "button:has-text('Submit')")
    result = await _find_element_by_expected_text(
        page, "Submit", "Submit button", x, y, probe_page=page
    )
    assert result is not None
    assert result["locator"] == 'text="Submit"'


async def test_singleton_without_coords_accepted_unchecked(page):
    """No coordinates -> nothing to check identity against -> singleton
    accepted as before."""
    await page.goto(_file_url("text_first_identity.html"))
    result = await _find_element_by_expected_text(
        page, "Promotions", "Promotions banner", None, None, probe_page=page
    )
    assert result is not None
    assert result["locator"] == 'text="Promotions"'


# ======================================================================
# A6 — description-derived acceptance reported honestly
# ======================================================================


async def test_description_derived_reported_unverified(page):
    """No expected_text: STEP 2 derives [title*="Search"] from the
    description and returns the settings gear. Acceptance unchanged, but
    the result must say UNVERIFIED instead of semantic_match=True."""
    await page.goto(_file_url("description_derived.html"))
    x, y = await _center(page, 'a[title="Search settings"]')
    result = await find_unique_locator_at_coordinates(
        page,
        x,
        y,
        "el1",
        "Search field",
        expected_text=None,
    )
    assert result["found"] is True
    assert result["best_locator"] == '[title*="Search"]'
    assert result["semantic_match"] is False
    assert result["all_locators"][0]["validation_method"] == "description_derived"
    assert result["validation_summary"]["validation_method"] == "description_derived"
