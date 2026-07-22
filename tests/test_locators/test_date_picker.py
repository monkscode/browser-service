"""
Task D (G4) — date-picker element type: flatpickr detection + widget
metadata so the Assembler can emit the setDate idiom.

Real sites (ASTPP customerReport) render date filters as READONLY
flatpickr inputs (`#customer_cdr_from_date`, class `flatpickr-input`,
`el._flatpickr` instance). `Fill Text` on a readonly input waits for
editability forever — the generated test dies with a timeout at that
step, every run. The only robust path is the widget's own API:
`el._flatpickr.setDate(value, true)` — verified live on the ASTPP site
(2026-07-08), including a date-only value against the site's
'Y-m-d H:i:S' datetime format (flatpickr's parser is lenient).

Pinned here:
  1. Classifier: Tier-0 `flatpickr-input` class rule on inputs;
     "date-picker" vision hint maps; phrase-safe description keywords
     vote (never bare "date" — substring-matches update/validate).
  2. DOM probe: nearest-container date-input scan (mirrors file-upload's
     dedicated scan) — anchors the input from a wrap-mode calendar
     button, rejects containers with no date input, never scans
     body-wide; framework verdict flatpickr vs native.
  3. Handler + STEP-0 dispatch: commits ONLY on flatpickr (native
     input[type=date] falls through to the generic path unchanged);
     payload carries element_type='date-picker', top-level
     datepicker_framework='flatpickr' (the dropdown_framework/select_id
     pipe precedent), element_info.readonly; dropdown_framework must
     NOT leak the flatpickr name.
  4. Plumbing: "date-picker" in the custom action's element_type enum,
     "flatpickr" in framework_hint, system prompt lists in sync
     (including the Task E 'file-upload' gap).

Integration tests use real headless Chromium against
locator_fixtures/date_picker.html (same conventions as
test_file_upload.py).
"""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.classifier import (
    classify_element_type,
    map_vision_hint,
)

FIXTURES_DIR = Path(__file__).parent / "locator_fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]


# ======================================================================
# Pure classifier tests (no browser)
# ======================================================================

class TestClassifierDatePicker:

    def test_vision_hint_maps(self):
        assert map_vision_hint("date-picker") == "date-picker"

    def test_flatpickr_class_is_tier0(self):
        """ASTPP shape: input.flatpickr-input is HTML-deterministic —
        classified date-picker/flatpickr at high confidence in Tier 0."""
        info = classify_element_type(
            {"tagName": "input", "type": "text",
             "className": "text field form-control flatpickr-input",
             "id": "customer_cdr_from_date"},
            "From Date filter",
        )
        assert info.primary_type == "date-picker"
        assert info.framework == "flatpickr"
        assert info.confidence == "high"
        assert "tier:0" in info.signals

    def test_tier0_input_date_still_wins(self):
        """Existing Tier-0 rule pinned: input[type=date] is native."""
        info = classify_element_type(
            {"tagName": "input", "type": "date", "id": "dob"},
            "date of birth field",
        )
        assert info.primary_type == "date-picker"
        assert info.framework == "native"
        assert info.confidence == "high"

    def test_flatpickr_class_on_non_input_no_tier0(self):
        """The class rule is input-scoped — a div carrying the class
        (calendar overlay chrome) must not classify date-picker in
        Tier 0."""
        info = classify_element_type(
            {"tagName": "div", "className": "flatpickr-input"},
            "some wrapper",
        )
        assert "tier:0" not in info.signals or info.primary_type != "date-picker"

    def test_hint_plus_desc_keyword_votes_high(self):
        info = classify_element_type(
            {"tagName": "button", "className": "calendar-toggle",
             "type": "button"},
            "calendar button for the from date filter",
            vision_type_hint="date-picker",
        )
        assert info.primary_type == "date-picker"
        assert info.confidence == "high"

    def test_hint_alone_votes_medium(self):
        info = classify_element_type(
            {"tagName": "button", "className": "btn", "type": "button"},
            "the second control in the panel",
            vision_type_hint="date-picker",
        )
        assert info.primary_type == "date-picker"
        assert info.confidence == "medium"

    @pytest.mark.parametrize("desc", [
        "from date filter for the CDR report",
        "date picker for the start date",
        "calendar field to choose the end date",
        "due date input on the invoice form",
    ])
    def test_desc_keywords_vote(self, desc):
        info = classify_element_type(
            {"tagName": "input", "type": "text", "className": "form-control"},
            desc,
        )
        assert info.primary_type == "date-picker"

    @pytest.mark.parametrize("desc", [
        "update the customer record button",
        "validate the form submit button",
        "candidate name input field",
    ])
    def test_bare_date_substring_does_not_vote(self, desc):
        """'date' hides inside update/validate/candidate — substring
        matching on the bare word would misroute half the app."""
        info = classify_element_type(
            {"tagName": "input", "type": "text", "className": ""},
            desc,
        )
        assert info.primary_type != "date-picker"


# ======================================================================
# Integration: probe + handler + STEP-0 dispatch on a real page
# ======================================================================

@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page_obj = await ctx.new_page()
        await page_obj.goto(
            (FIXTURES_DIR / "date_picker.html").resolve().as_uri()
        )
        try:
            yield page_obj
        finally:
            await browser.close()


async def _center(page, selector: str) -> tuple:
    box = await page.locator(selector).bounding_box()
    assert box, f"{selector} has no bounding box"
    return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


@pytest.mark.integration
class TestDatePickerProbe:

    async def _probe(self, page, coords):
        from browser_service.locators.dom_probe import probe_specialized_type
        return await probe_specialized_type(
            page=page, suspected_type="date-picker", coords=coords,
        )

    async def test_flatpickr_input_confirms_itself(self, page):
        coords = await _center(page, "#customer_cdr_from_date")
        result = await self._probe(page, coords)
        assert result["confirmed"] is True
        assert result["framework"] == "flatpickr"
        anchor_id = await page.locator(
            f"xpath={result['anchor_xpath']}"
        ).get_attribute("id")
        assert anchor_id == "customer_cdr_from_date"

    async def test_calendar_button_anchors_sibling_input(self, page):
        """wrap-mode shape: vision clicks the calendar toggle BUTTON —
        the container scan must anchor the id-less sibling input."""
        coords = await _center(page, ".calendar-toggle")
        result = await self._probe(page, coords)
        assert result["confirmed"] is True
        assert result["framework"] == "flatpickr"
        anchor_name = await page.locator(
            f"xpath={result['anchor_xpath']}"
        ).get_attribute("name")
        assert anchor_name == "event_date"

    async def test_native_date_input_confirms_as_native(self, page):
        coords = await _center(page, "#dob")
        result = await self._probe(page, coords)
        assert result["confirmed"] is True
        assert result["framework"] == "native"

    async def test_plain_text_input_rejects(self, page):
        """The toolbar search box's container has no date input; the
        scan must NOT escape to body (three flatpickr inputs exist
        page-wide)."""
        coords = await _center(page, "#search_box")
        result = await self._probe(page, coords)
        assert result["confirmed"] is False


@pytest.mark.integration
class TestDatePickerDispatch:
    """STEP-0 end-to-end: payload carries the Assembler contract."""

    async def _dispatch(self, page, element_data, coords, description,
                        vision_type_hint="date-picker"):
        from browser_service.locators.smart_locator import (
            _generate_locators_from_element_data,
        )
        return await _generate_locators_from_element_data(
            page, element_data, "elem_1", description,
            expected_text=None,
            confirmed_coords=coords,
            vision_type_hint=vision_type_hint,
            page=page,
        )

    async def test_astpp_readonly_input_payload(self, page):
        coords = await _center(page, "#customer_cdr_from_date")
        element_data = {
            "tagName": "input", "id": "customer_cdr_from_date",
            "name": "callstart[]",
            "className": "text field form-control flatpickr-input",
            "textContent": "", "ariaLabel": "", "placeholder": "",
            "title": "", "role": "", "dataTestId": "", "type": "text",
            "xpath": "",
            "coordinates": {"x": coords[0], "y": coords[1]},
            "parentId": "filter-panel", "parentClass": "",
        }
        result = await self._dispatch(
            page, element_data, coords, "From Date filter for the CDR report",
            vision_type_hint=None,
        )
        assert result is not None and result["found"]
        assert result["best_locator"] == "id=customer_cdr_from_date"
        assert result["element_type"] == "date-picker"
        assert result["datepicker_framework"] == "flatpickr"
        # flatpickr must NOT leak into the dropdown routing field.
        assert result.get("dropdown_framework", "") == ""
        assert result["element_info"]["readonly"] is True
        assert result["stability"] == "stable"

    async def test_calendar_button_resolves_input_by_name(self, page):
        """Button clicked, input has no id; page has THREE flatpickr
        inputs so the sole-input fallback is ambiguous — the scoped
        name candidate must win."""
        coords = await _center(page, ".calendar-toggle")
        element_data = {
            "tagName": "button", "id": "", "name": "",
            "className": "calendar-toggle", "textContent": "",
            "ariaLabel": "", "placeholder": "", "title": "", "role": "",
            "dataTestId": "", "type": "button", "xpath": "",
            "coordinates": {"x": coords[0], "y": coords[1]},
            "parentId": "", "parentClass": "flatpickr-wrap",
        }
        result = await self._dispatch(
            page, element_data, coords, "calendar button for the event date",
        )
        assert result is not None and result["found"]
        assert result["best_locator"] == 'input.flatpickr-input[name="event_date"]'
        assert result["element_type"] == "date-picker"
        assert result["datepicker_framework"] == "flatpickr"

    async def test_native_date_input_falls_through_to_generic(self, page):
        """input[type=date]: Fill Text already works — the handler must
        NOT commit; the generic path keeps emitting the plain id."""
        coords = await _center(page, "#dob")
        element_data = {
            "tagName": "input", "id": "dob", "name": "dob",
            "className": "", "textContent": "", "ariaLabel": "",
            "placeholder": "", "title": "", "role": "", "dataTestId": "",
            "type": "date", "xpath": "",
            "coordinates": {"x": coords[0], "y": coords[1]},
            "parentId": "profile", "parentClass": "",
        }
        result = await self._dispatch(
            page, element_data, coords, "date of birth field",
            vision_type_hint=None,
        )
        assert result is not None and result["found"]
        assert result["best_locator"] == "#dob"  # generic id path, unchanged
        assert result.get("datepicker_framework") in (None, "")


# ======================================================================
# Plumbing tripwires
# ======================================================================

class TestPlumbing:

    def test_element_type_enum_offers_date_picker(self):
        src = (REPO_ROOT / "browser_service" / "agent"
               / "registration.py").read_text(encoding="utf-8")
        assert '"date-picker"' in src

    def test_framework_hint_enum_offers_flatpickr(self):
        src = (REPO_ROOT / "browser_service" / "agent"
               / "registration.py").read_text(encoding="utf-8")
        assert '"flatpickr"' in src

    def test_system_prompt_lists_in_sync_with_schema(self):
        """The vision LLM reads the system prompt's element_type list —
        it must offer date-picker AND file-upload (the latter was missed
        when Task E extended the schema enum)."""
        src = (REPO_ROOT / "browser_service" / "prompts"
               / "system.py").read_text(encoding="utf-8")
        assert "date-picker" in src
        assert "file-upload" in src
        assert "flatpickr" in src
