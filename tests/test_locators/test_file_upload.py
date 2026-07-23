"""
Task E (G5) — file-upload element type: return the file INPUT, not the
styled button.

Real sites hide `<input type="file">` behind a styled button (ASTPP:
`#customer_import_mapper` is display:none, the visible control is an
unlabelled Browse button). Vision clicks the button; today the engine
emits a weak locator for the button and the Assembler generates Click —
which opens the browser's NATIVE file dialog and hangs the test forever.
The only robust upload path is `Upload File By Selector` aimed at the
input itself (hidden inputs are legal targets).

Pinned here:
  1. Classifier: "file-upload" vision hint maps; description keywords
     (upload / choose file / browse / attach) vote.
  2. DOM probe: nearest-container `input[type=file]` scan — finds the
     sibling input the generic walk misses, rejects controls with no
     file input in their container (toolbar Import link), never scans
     body-wide.
  3. Handler + STEP-0 dispatch: returns the INPUT's locator (id → name →
     unique input[type=file] → anchor xpath), element_type='file-upload',
     element_info.hidden_input for the Assembler.
  4. Plumbing: "file-upload" available in the custom action's
     element_type enum.

Integration tests use real headless Chromium against
locator_fixtures/file_upload.html (same conventions as
test_collection_scoping.py).
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


class TestClassifierFileUpload:
    def test_vision_hint_maps(self):
        assert map_vision_hint("file-upload") == "file-upload"

    def test_hint_plus_desc_keyword_votes_high(self):
        info = classify_element_type(
            {"tagName": "button", "className": "btn browse-customer", "type": "button"},
            "Browse button to upload the customer CSV file",
            vision_type_hint="file-upload",
        )
        assert info.primary_type == "file-upload"
        assert info.confidence == "high"

    def test_hint_alone_votes_medium(self):
        info = classify_element_type(
            {"tagName": "button", "className": "btn", "type": "button"},
            "the second button in the form",
            vision_type_hint="file-upload",
        )
        assert info.primary_type == "file-upload"
        assert info.confidence == "medium"

    @pytest.mark.parametrize(
        "desc",
        [
            "upload the ratedeck csv",
            "choose file control for the import",
            "attach file button",
        ],
    )
    def test_desc_keywords_vote(self, desc):
        info = classify_element_type(
            {"tagName": "button", "className": "btn", "type": "button"},
            desc,
        )
        assert info.primary_type == "file-upload"

    def test_plain_import_button_does_not_vote(self):
        """ASTPP list toolbar has an Import BUTTON that just navigates —
        'import' alone must not start a file-upload hunt."""
        info = classify_element_type(
            {"tagName": "a", "id": "import", "className": ""},
            "Import button in the toolbar",
        )
        assert info.primary_type != "file-upload"

    def test_tier0_input_file_still_wins(self):
        """Existing Tier-0 rule pinned: a real file input is deterministic."""
        info = classify_element_type(
            {"tagName": "input", "type": "file", "id": "avatar_file"},
            "avatar upload field",
        )
        assert info.primary_type == "file-upload"
        assert info.confidence == "high"


# ======================================================================
# Integration: probe + handler + STEP-0 dispatch on a real page
# ======================================================================

pytestmark_integration = pytest.mark.integration


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page_obj = await ctx.new_page()
        await page_obj.goto((FIXTURES_DIR / "file_upload.html").resolve().as_uri())
        try:
            yield page_obj
        finally:
            await browser.close()


async def _center(page, selector: str) -> tuple:
    box = await page.locator(selector).bounding_box()
    assert box, f"{selector} has no bounding box"
    return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


@pytest.mark.integration
class TestFileUploadProbe:
    async def _probe(self, page, coords):
        from browser_service.locators.dom_probe import probe_specialized_type

        return await probe_specialized_type(
            page=page,
            suspected_type="file-upload",
            coords=coords,
        )

    async def test_button_with_sibling_hidden_input_confirms(self, page):
        """The ASTPP shape: input is the button's SIBLING — the generic
        ancestor-sibling walk misses it; the container scan must find it."""
        coords = await _center(page, ".browse-customer")
        result = await self._probe(page, coords)
        assert result["confirmed"] is True
        assert result["anchor_tag"] == "input"
        # Anchor must be THE sibling input, not the one in the next section
        anchor_id = await page.locator(f"xpath={result['anchor_xpath']}").get_attribute("id")
        assert anchor_id == "customer_import_mapper"

    async def test_nearest_container_wins_over_neighbor_section(self, page):
        """Clicking the ratedeck button must anchor the ratedeck input,
        not the customer one in the adjacent section."""
        coords = await _center(page, ".browse-ratedeck")
        result = await self._probe(page, coords)
        assert result["confirmed"] is True
        anchor_name = await page.locator(f"xpath={result['anchor_xpath']}").get_attribute("name")
        assert anchor_name == "ratedeck_csv"

    async def test_toolbar_link_with_no_nearby_input_rejects(self, page):
        """The Import link's container has no file input; the scan must
        NOT escape to body (three file inputs exist page-wide)."""
        coords = await _center(page, "#import")
        result = await self._probe(page, coords)
        assert result["confirmed"] is False

    async def test_direct_file_input_confirms_itself(self, page):
        coords = await _center(page, "#avatar_file")
        result = await self._probe(page, coords)
        assert result["confirmed"] is True
        anchor_id = await page.locator(f"xpath={result['anchor_xpath']}").get_attribute("id")
        assert anchor_id == "avatar_file"


@pytest.mark.integration
class TestFileUploadDispatch:
    """STEP-0 end-to-end: element_data of the BUTTON + file-upload hint
    → payload carries the INPUT's locator and the Assembler contract."""

    @staticmethod
    def _button_element_data(class_name, coords):
        return {
            "tagName": "button",
            "id": "",
            "name": "",
            "className": class_name,
            "textContent": "Browse",
            "ariaLabel": "",
            "placeholder": "",
            "title": "",
            "role": "",
            "dataTestId": "",
            "type": "button",
            "xpath": "",
            "coordinates": {"x": coords[0], "y": coords[1]},
            "parentId": "",
            "parentClass": "section",
        }

    async def _dispatch(self, page, element_data, coords, description):
        from browser_service.locators.smart_locator import (
            _generate_locators_from_element_data,
        )

        return await _generate_locators_from_element_data(
            page,
            element_data,
            "elem_1",
            description,
            expected_text=None,
            confirmed_coords=coords,
            vision_type_hint="file-upload",
            page=page,
        )

    async def test_hidden_input_with_id(self, page):
        coords = await _center(page, ".browse-customer")
        result = await self._dispatch(
            page,
            self._button_element_data("btn browse-customer", coords),
            coords,
            "Browse button to upload the customer CSV file",
        )
        assert result is not None and result["found"]
        assert result["best_locator"] == "id=customer_import_mapper"
        assert result["element_type"] == "file-upload"
        assert result["element_info"]["hidden_input"] is True
        assert result["stability"] == "stable"

    async def test_hidden_input_name_only(self, page):
        """Input has no id and the page has THREE file inputs — the bare
        input[type=file] fallback is ambiguous; name anchoring must win."""
        coords = await _center(page, ".browse-ratedeck")
        result = await self._dispatch(
            page,
            self._button_element_data("btn browse-ratedeck", coords),
            coords,
            "Choose file button to upload the ratedeck csv",
        )
        assert result is not None and result["found"]
        assert result["best_locator"] == 'input[type="file"][name="ratedeck_csv"]'
        assert result["element_type"] == "file-upload"
        assert result["element_info"]["hidden_input"] is True

    async def test_visible_input_direct(self, page):
        coords = await _center(page, "#avatar_file")
        element_data = {
            "tagName": "input",
            "id": "avatar_file",
            "name": "",
            "className": "",
            "textContent": "",
            "ariaLabel": "",
            "placeholder": "",
            "title": "",
            "role": "",
            "dataTestId": "",
            "type": "file",
            "xpath": "",
            "coordinates": {"x": coords[0], "y": coords[1]},
            "parentId": "avatar-upload",
            "parentClass": "section",
        }
        result = await self._dispatch(
            page,
            element_data,
            coords,
            "avatar upload field",
        )
        assert result is not None and result["found"]
        assert result["best_locator"] == "id=avatar_file"
        assert result["element_type"] == "file-upload"
        assert result["element_info"]["hidden_input"] is False

    async def test_no_nearby_input_falls_through_to_generic(self, page):
        """Toolbar Import link with a (mistaken) file-upload hint: probe
        rejects, generic path still returns the link's own id."""
        coords = await _center(page, "#import")
        element_data = {
            "tagName": "a",
            "id": "import",
            "name": "",
            "className": "",
            "textContent": "Import",
            "ariaLabel": "",
            "placeholder": "",
            "title": "",
            "role": "",
            "dataTestId": "",
            "type": "",
            "xpath": "",
            "coordinates": {"x": coords[0], "y": coords[1]},
            "parentId": "toolbar",
            "parentClass": "section",
        }
        result = await self._dispatch(
            page,
            element_data,
            coords,
            "Import button in the toolbar",
        )
        assert result is not None and result["found"]
        assert result["best_locator"] == "#import"  # generic id path, unchanged
        assert result["element_type"] != "file-upload"


# ======================================================================
# Plumbing tripwire
# ======================================================================


class TestPlumbing:
    def test_element_type_enum_offers_file_upload(self):
        src = (REPO_ROOT / "browser_service" / "agent" / "registration.py").read_text(
            encoding="utf-8"
        )
        assert '"file-upload"' in src
