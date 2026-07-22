"""
Guards the element_info payload contract consumed by nlrf's identify agent.

nlrf's Task G state-verification assembler reads three element_info keys:
  - className       — the class list observed on the element at locate time
                      (captured AFTER the preceding workflow steps ran, so for
                      "click Save, verify the field shows an error" it is the
                      field in its error state; the error-family marker is
                      picked from this evidence)
  - ariaInvalid     — the aria-invalid attribute value, for sites that mark
                      invalid fields via ARIA instead of a CSS class
  - parentClassName — the parent's class list, for Bootstrap-3-style sites
                      that mark the wrapper div instead of the field

The coordinate-path result build is too entangled with live Playwright
objects to unit-test directly, so source-level guards keep its contract
keys from being silently dropped in a refactor. The STEP-0 accept is
testable behaviorally and gets a real-page test below: it is the dominant
accept path for well-attributed fields, and starving it silently disables
the downstream marker scan (same failure d3f6859 fixed on the candidate
path).
"""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

import browser_service.locators.smart_locator as sl_mod
from browser_service.locators.smart_locator import (
    _generate_locators_from_element_data,
)

SRC = Path(sl_mod.__file__).read_text(encoding="utf-8")

FIXTURES_DIR = Path(__file__).parent / "locator_fixtures"


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page_obj = await ctx.new_page()
        await page_obj.goto((FIXTURES_DIR / "stability_ids.html").resolve().as_uri())
        try:
            yield page_obj
        finally:
            await browser.close()


class TestElementInfoPayloadContract:
    def test_class_name_in_payload(self):
        assert '"className": element_data["className"]' in SRC

    def test_aria_invalid_in_payload(self):
        assert '"ariaInvalid": element_data.get("ariaInvalid", "")' in SRC

    def test_parent_class_in_payload(self):
        """Bootstrap-3-style sites mark errors on the PARENT div — nlrf's
        marker scan reads element_info.parentClassName to assert
        `${locator} >> xpath=..` instead of falling to the placeholder."""
        assert '"parentClassName": element_data.get("parentClassName", "")' in SRC


@pytest.mark.integration
class TestStep0ElementInfoPayload:
    """The STEP-0 (element-data) accept must carry the same DOM evidence
    the candidate path carries — element_info is copied from element_data,
    so the accept that BUILT its locator from element_data has no excuse
    to drop the keys nlrf reads."""

    async def test_step0_accept_carries_marker_evidence(self, page):
        result = await _generate_locators_from_element_data(
            search_context=page,
            element_data={
                "tagName": "input",
                "id": "save_button",
                "name": "save",
                "className": "form-control has-error",
                "ariaInvalid": "true",
                "parentClassName": "form-group has-error",
                "ariaLabel": "",
                "placeholder": "",
                "title": "",
                "role": "",
                "dataTestId": "",
                "type": "text",
                "xpath": "",
                "textContent": "",
            },
            element_id="elem_g7",
            element_description="save field",
            page=page,
        )
        assert result is not None and result["found"] is True
        info = result["element_info"]
        assert info.get("className") == "form-control has-error"
        assert info.get("ariaInvalid") == "true"
        assert info.get("parentClassName") == "form-group has-error"
