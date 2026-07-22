"""
Tests for the E1 candidate-path element_info enrichment (Option B).

Purpose: the agent-candidate fast path in actions.py returned
         element_info={} on every accept — element_type / dropdown_framework /
         datepicker_framework / element_classes / aria_invalid / parent_classes
         all vanished whenever the vision agent's proposed locator validated.
         Downstream (nlrf) composer triggers starved and widget idioms were
         lost: readonly flatpickr degraded to Fill Text (the G4 failure).

Option B contract (owner-approved 2026-07-15):
  - element_info copied from the already-extracted element_data
  - classifier stamp gated to Tier-0 DOM-evidence verdicts with attribute
    evidence beyond the bare tag name (type= / className: / role= signals);
    bare-tagName verdicts (select/tr/li) and Tier-1 verdicts stay unstamped —
    full-path parity: nlrf's tagName fallback routes those
  - select_id only when the element is itself a <select> with an id
  - date-picker stamps set dropdown_framework='' (Task D guard)
  - vision hints never feed the stamp (no DOM-probe corroboration on this path)
  - no page round-trip

Tests:
  element_info copy: full dict / None element_data / partial element_data
  Tier-0 stamps: flatpickr, file-upload, tom-select class, role=checkbox
  Skips: bare <select> (but select_id set), <tr>, vision hints, Tier-1
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_playwright_page():
    """Mock Playwright page where the candidate locator validates unique."""
    page = AsyncMock()
    page.url = "https://example.com/app"
    page.title = AsyncMock(return_value="App Page")

    locator_mock = MagicMock()
    locator_mock.count = AsyncMock(return_value=1)
    locator_mock.first = MagicMock()
    locator_mock.first.text_content = AsyncMock(return_value="Submit")
    page.locator = MagicMock(return_value=locator_mock)

    return page


async def _accept_candidate(page, element_data, candidate="id=field-1", **kwargs):
    """Run the candidate fast path to an accept and return the result."""
    from browser_service.agent.actions import find_unique_locator_action

    result = await find_unique_locator_action(
        page=page,
        x=100, y=200,
        element_id="elem_1",
        element_description=kwargs.pop("element_description", "form field"),
        candidate_locator=candidate,
        element_data=element_data,
        **kwargs,
    )
    # Guard: these tests only mean anything on the fast-path accept.
    assert result["found"] is True
    assert result["all_locators"][0]["type"] == "candidate"
    return result


class TestElementInfoCopy:
    """element_info must carry the element_data evidence nlrf's merge reads."""

    @pytest.mark.asyncio
    async def test_element_info_copied_from_element_data(self, mock_playwright_page):
        """tagName/className/ariaInvalid/parentClassName/id all flow through."""
        result = await _accept_candidate(
            mock_playwright_page,
            element_data={
                "tagName": "input",
                "type": "text",
                "id": "email",
                "className": "form-control invalid",
                "ariaInvalid": "true",
                "parentClassName": "form-group has-error",
                "textContent": "",
            },
        )
        info = result["element_info"]
        assert info["tagName"] == "input"
        assert info["className"] == "form-control invalid"
        assert info["ariaInvalid"] == "true"
        assert info["parentClassName"] == "form-group has-error"
        assert info["id"] == "email"

    @pytest.mark.asyncio
    async def test_element_info_empty_without_element_data(self, mock_playwright_page):
        """No element_data → element_info stays {} (nothing to copy)."""
        result = await _accept_candidate(mock_playwright_page, element_data=None)
        assert result["element_info"] == {}

    @pytest.mark.asyncio
    async def test_partial_element_data_tolerated(self, mock_playwright_page):
        """element_data producers vary — missing keys must not raise."""
        result = await _accept_candidate(
            mock_playwright_page, element_data={"tagName": "input"}
        )
        assert result["element_info"]["tagName"] == "input"


class TestTier0ClassifierStamp:
    """Tier-0 verdicts with attribute evidence stamp routing fields."""

    @pytest.mark.asyncio
    async def test_flatpickr_stamps_datepicker_and_clears_dropdown(
        self, mock_playwright_page
    ):
        """The G4 case: readonly flatpickr input must route to the setDate
        idiom, and dropdown_framework must be explicit-empty (Task D guard)."""
        result = await _accept_candidate(
            mock_playwright_page,
            element_data={
                "tagName": "input",
                "type": "text",
                "id": "customer_cdr_from_date",
                "className": "form-control input flatpickr-input",
            },
        )
        assert result["element_type"] == "date-picker"
        assert result["datepicker_framework"] == "flatpickr"
        assert result["dropdown_framework"] == ""

    @pytest.mark.asyncio
    async def test_file_input_stamps_file_upload(self, mock_playwright_page):
        """input[type=file] → file-upload; no dropdown_framework leakage."""
        result = await _accept_candidate(
            mock_playwright_page,
            element_data={
                "tagName": "input",
                "type": "file",
                "id": "customer_import_mapper",
            },
        )
        assert result["element_type"] == "file-upload"
        assert not result.get("dropdown_framework")

    @pytest.mark.asyncio
    async def test_tom_select_class_stamps_dropdown_framework(
        self, mock_playwright_page
    ):
        """Framework className evidence → dropdown/tom-select; the control div
        is not a <select>, so select_id stays unset."""
        result = await _accept_candidate(
            mock_playwright_page,
            element_data={
                "tagName": "div",
                "className": "ts-control",
                "id": "",
            },
        )
        assert result["element_type"] == "dropdown"
        assert result["dropdown_framework"] == "tom-select"
        assert result.get("select_id") is None

    @pytest.mark.asyncio
    async def test_role_evidence_stamps_checkbox(self, mock_playwright_page):
        """role=checkbox is attribute evidence — custom checkbox stamps."""
        result = await _accept_candidate(
            mock_playwright_page,
            element_data={"tagName": "div", "role": "checkbox"},
        )
        assert result["element_type"] == "checkbox"

    @pytest.mark.asyncio
    async def test_role_combobox_not_stamped(self, mock_playwright_page):
        """role=combobox is on the approved skip list: a dropdown/combobox-input
        stamp matches no composer TYPE rule, while the tagName fallback routes
        TYPE 2/3 correctly — same full-path parity as the bare-select skip."""
        result = await _accept_candidate(
            mock_playwright_page,
            element_data={"tagName": "div", "role": "combobox"},
        )
        assert "element_type" not in result
        assert not result.get("dropdown_framework")


class TestStampSkips:
    """Bare-tagName and Tier-1 verdicts must NOT stamp (full-path parity)."""

    @pytest.mark.asyncio
    async def test_bare_select_not_stamped_but_select_id_set(
        self, mock_playwright_page
    ):
        """<select id=...>: tagName flows via element_info (nlrf's fallback
        routes TYPE 1), no element_type stamp — but select_id is the element's
        own id, per the Option B clause."""
        result = await _accept_candidate(
            mock_playwright_page,
            element_data={"tagName": "select", "id": "country", "className": ""},
        )
        assert "element_type" not in result
        assert not result.get("dropdown_framework")
        assert result["select_id"] == "country"
        assert result["element_info"]["tagName"] == "select"

    @pytest.mark.asyncio
    async def test_select_without_id_gets_no_select_id(self, mock_playwright_page):
        result = await _accept_candidate(
            mock_playwright_page,
            element_data={"tagName": "select", "id": ""},
        )
        assert result.get("select_id") is None

    @pytest.mark.asyncio
    async def test_table_row_not_stamped(self, mock_playwright_page):
        """<tr> → collection/table-row is bare-tagName evidence; stamping it
        put 'table-row' into dropdown_framework and fired the DROPDOWN block
        on collection steps (the q04 breadth case). Must stay unstamped."""
        result = await _accept_candidate(
            mock_playwright_page,
            element_data={"tagName": "tr", "className": "data-row"},
        )
        assert "element_type" not in result
        assert not result.get("dropdown_framework")

    @pytest.mark.asyncio
    async def test_vision_hints_never_stamp(self, mock_playwright_page):
        """Vision hints are one source of truth; the full path corroborates
        them with a DOM probe before committing. The candidate path has no
        probe — hints must not produce a stamp."""
        result = await _accept_candidate(
            mock_playwright_page,
            element_data={"tagName": "div", "className": "field-widget"},
            vision_type_hint="date-picker",
            vision_framework_hint="flatpickr",
        )
        assert "element_type" not in result
        assert not result.get("datepicker_framework")
        assert not result.get("dropdown_framework")
