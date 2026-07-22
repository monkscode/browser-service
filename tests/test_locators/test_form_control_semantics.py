"""
Task 6 (analysis doc A2) — label-aware semantic validation for form controls.

Two fixes are pinned here:

  1. ``validate_semantic_match``'s locator path resolves associated label
     text (``<label for=>``, wrapping ``<label>``, ``aria-labelledby``)
     into the haystack, so a correctly labeled input with no placeholder
     passes on its own merits instead of needing the structural carve-out.
  2. The STEP-0 form-control carve-out still accepts a unique structural
     locator when the semantic check fails, but reports honestly:
     ``semantic_match=False`` with ``validation_method=
     "form_control_structural"`` — an unverified acceptance is no longer
     indistinguishable from a verified one.

The carve-out itself is intentionally NOT narrowed (owner decision
2026-07-04: defer until the distinguishing logs show a real wrong-field
acceptance).

Same conventions as test_checkbox_gate.py: real headless Chromium against
a static fixture in ``locator_fixtures/``, marked integration.
"""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.smart_locator import (
    _generate_locators_from_element_data,
    validate_semantic_match,
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
        await page_obj.goto(_file_url("form_control_semantics.html"))
        try:
            yield page_obj
        finally:
            await browser.close()


def _element_data(el_id: str, placeholder: str = "") -> dict:
    """element_data as browser-use vision extracts it for an <input>."""
    return {
        "tagName": "input",
        "id": el_id,
        "name": "",
        "className": "",
        "ariaLabel": "",
        "placeholder": placeholder,
        "title": "",
        "role": "",
        "dataTestId": "",
        "textContent": "",
    }


# ======================================================================
# validate_semantic_match (locator path) — label text is semantic surface
# ======================================================================


async def test_label_for_text_matches_input(page):
    """<label for="email">Email address</label> must let #email pass for
    expected_text "Email address" even with no placeholder."""
    ok, _ = await validate_semantic_match(None, "Email address", page=page, locator="#email")
    assert ok


async def test_aria_labelledby_text_matches_input(page):
    ok, _ = await validate_semantic_match(None, "Phone number", page=page, locator="#phone")
    assert ok


async def test_wrapping_label_text_matches_input(page):
    ok, _ = await validate_semantic_match(None, "Nickname", page=page, locator="#nickname")
    assert ok


async def test_wrong_field_still_fails(page):
    """The doc's example: expected "Enter username" against the password
    field. Placeholder AND label both say Password — must stay a mismatch."""
    ok, _ = await validate_semantic_match(None, "Enter username", page=page, locator="#password")
    assert not ok


async def test_bare_input_still_fails(page):
    """No label/placeholder/aria/value: nothing to compare, check fails
    (the carve-out, not the check, handles this case)."""
    ok, _ = await validate_semantic_match(None, "First name", page=page, locator="#first_name")
    assert not ok


# ======================================================================
# STEP-0 carve-out — acceptance unchanged, reporting honest
# ======================================================================


async def test_carveout_acceptance_reports_semantic_match_false(page):
    """Wrong-field acceptance (coords on #password, step says username)
    must no longer be stamped semantic_match=True."""
    result = await _generate_locators_from_element_data(
        page,
        _element_data("password", placeholder="Password"),
        "el1",
        "username field",
        expected_text="Enter username",
        page=page,
    )
    assert result is not None and result["found"] is True
    assert result["best_locator"] == "#password"
    assert result["semantic_match"] is False
    assert result["all_locators"][0]["semantic_match"] is False
    assert result["validation_summary"]["validation_method"] == "form_control_structural"


async def test_bare_input_rescue_survives_reported_honestly(page):
    """The legitimate rescue: unique id on a surface-less input is still
    accepted, but as an unverified acceptance."""
    result = await _generate_locators_from_element_data(
        page,
        _element_data("first_name"),
        "el2",
        "first name field",
        expected_text="First name",
        page=page,
    )
    assert result is not None and result["found"] is True
    assert result["best_locator"] == "#first_name"
    assert result["semantic_match"] is False
    assert result["validation_summary"]["validation_method"] == "form_control_structural"


async def test_labeled_input_passes_without_carveout(page):
    """With label resolution, #email passes the semantic check outright —
    validated for real, no carve-out involved."""
    result = await _generate_locators_from_element_data(
        page,
        _element_data("email"),
        "el3",
        "email field",
        expected_text="Email address",
        page=page,
    )
    assert result is not None and result["found"] is True
    assert result["best_locator"] == "#email"
    assert result["semantic_match"] is True
    assert result["validation_summary"]["validation_method"] == "text"
