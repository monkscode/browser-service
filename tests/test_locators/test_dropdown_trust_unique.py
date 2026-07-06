"""
Task 7 (analysis doc A3) — no dropdown acceptance on a double validation
failure.

In the STEP-0 candidate loop, a dropdown-ish element whose unique candidate
fails the semantic text check gets a coordinate check as a second opinion.
When the coordinates ALSO mismatch, the code used to accept the locator
anyway ("trusting browser-use vision") and stamp ``semantic_match=True`` /
``validation_method="trust_unique"`` — two independent wrong-element
signals overridden and reported as validated. Pinned here:

  1. Double failure → the candidate is rejected; with every candidate
     pointing at the same wrong element, STEP 0 returns None and the
     pipeline falls through to its later strategies.
  2. The coordinate-PASS path (the only one observed in 10 weeks of logs)
     still accepts, reported as ``validation_method="coordinates"``.

Same conventions as test_form_control_semantics.py: real headless Chromium
against a static fixture in ``locator_fixtures/``, marked integration.
"""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.smart_locator import (
    _generate_locators_from_element_data,
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
        await page_obj.goto(_file_url("dropdown_trust_unique.html"))
        try:
            yield page_obj
        finally:
            await browser.close()


def _combobox_element_data(el_id: str) -> dict:
    """element_data as browser-use vision extracts it for a custom
    (non-form-control) combobox div."""
    return {
        "tagName": "div",
        "id": el_id,
        "name": "",
        "className": "",
        "ariaLabel": "",
        "placeholder": "",
        "title": "",
        "role": "combobox",
        "dataTestId": "",
        "textContent": "Choose an option",
        "xpath": "html/body/div[2]/div[2]",
    }


async def _center(page, locator: str) -> tuple:
    box = await page.locator(locator).bounding_box()
    return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


async def test_double_failure_rejects_wrong_combobox(page):
    """The doc's example: vision hands over the Currency combobox for the
    step "Country dropdown"; its coordinates point at the Country combobox.
    Semantic check fails AND coordinate check fails — the wrong element
    must not come back stamped validated. Every STEP-0 candidate resolves
    to the same wrong element, so the whole step returns None."""
    result = await _generate_locators_from_element_data(
        page,
        _combobox_element_data("currency"),
        "el1",
        "Country dropdown",
        expected_text="Country",
        confirmed_coords=await _center(page, "#country"),
        page=page,
    )
    assert result is None


async def test_coordinate_pass_still_accepts(page):
    """The legitimate path (all 6 real fires in 10 weeks of logs): text
    check fails because the value text lives outside the element, but the
    coordinates confirm it is the element vision saw — still accepted."""
    result = await _generate_locators_from_element_data(
        page,
        _combobox_element_data("country"),
        "el2",
        "Country dropdown",
        expected_text="Country",
        confirmed_coords=await _center(page, "#country"),
        page=page,
    )
    assert result is not None and result["found"] is True
    assert result["best_locator"] == "#country"
    assert result["semantic_match"] is True
    assert result["validation_summary"]["validation_method"] == "coordinates"
