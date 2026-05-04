"""
Unit tests for browser_service.locators.handlers.dropdown.

Pure-function coverage:
  - _normalize_label_or_description (Section 6 normalization rules)
  - _xpath_string_literal (XPath quoting)
  - _is_real_select_id / _select_id_from_input_id (Tom Select id helpers)

Async coverage with mocks:
  - find_locator framework dispatch (tom-select / native / others)
  - Tom Select Strategy 1a/1b id-anchored locators
  - Always-fallback (returns None) when no strategy succeeds
  - Result dict shape (dropdown_framework, classifier_confidence, etc.)
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from browser_service.locators.classifier import ElementTypeInfo
from browser_service.locators.handlers.dropdown import (
    _is_real_select_id,
    _normalize_label_or_description,
    _select_id_from_input_id,
    _xpath_string_literal,
    find_locator,
)


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------


class TestNormalizeLabelOrDescription:
    """Section 6: required-marker, UI-suffix, whitespace normalization."""

    def test_strips_trailing_asterisk(self):
        assert _normalize_label_or_description("Rate Group *") == "Rate Group"

    def test_strips_asterisk_with_no_space(self):
        assert _normalize_label_or_description("Currency*") == "Currency"

    def test_strips_html_required_artefact(self):
        assert (
            _normalize_label_or_description("Role<span> *</span>") == "Role"
        )

    def test_strips_dropdown_suffix(self):
        assert (
            _normalize_label_or_description("Rate Group dropdown")
            == "Rate Group"
        )

    def test_strips_field_suffix(self):
        assert (
            _normalize_label_or_description("Email field") == "Email"
        )

    def test_strips_input_suffix(self):
        assert _normalize_label_or_description("Search input") == "Search"

    def test_strips_select_suffix(self):
        assert _normalize_label_or_description("Country select") == "Country"

    def test_strips_picker_suffix(self):
        assert _normalize_label_or_description("Date picker") == "Date"

    def test_strips_chooser_suffix(self):
        assert _normalize_label_or_description("File chooser") == "File"

    def test_suffix_match_is_case_insensitive(self):
        assert (
            _normalize_label_or_description("Rate Group DROPDOWN")
            == "Rate Group"
        )

    def test_suffix_only_input_returns_empty(self):
        """A description that is just 'dropdown' has no useful label."""
        assert _normalize_label_or_description("dropdown") == ""

    def test_does_not_strip_substring_match(self):
        """'subselect' ends with 'select' but isn't a UI suffix word."""
        assert (
            _normalize_label_or_description("subselect") == "subselect"
        )

    def test_collapses_internal_whitespace(self):
        assert (
            _normalize_label_or_description("Rate    Group  *  ")
            == "Rate Group"
        )

    def test_combined_required_and_ui_suffix(self):
        """Description rendered with both: 'Rate Group * dropdown'."""
        # Required-marker stripper only matches at end; here 'dropdown'
        # is at end. UI-suffix stripper runs after, so '*' survives.
        # Verify this real edge case behaves as documented.
        result = _normalize_label_or_description("Rate Group * dropdown")
        # First pass: trailing-* doesn't fire (text ends with 'dropdown').
        # Suffix pass strips ' dropdown' → 'Rate Group *'. Whitespace
        # collapse → 'Rate Group *'. We accept this as documented behavior;
        # in practice descriptions don't contain '*' (Step Planner strips).
        assert "Rate Group" in result

    def test_empty_input(self):
        assert _normalize_label_or_description("") == ""

    def test_none_input(self):
        assert _normalize_label_or_description(None) == ""

    def test_whitespace_only_input(self):
        assert _normalize_label_or_description("   ") == ""


class TestXpathStringLiteral:
    def test_no_quotes_uses_single(self):
        assert _xpath_string_literal("Rate Group") == "'Rate Group'"

    def test_string_with_single_quote_uses_double(self):
        assert _xpath_string_literal("Driver's licence") == '"Driver\'s licence"'

    def test_string_with_double_quote_uses_single(self):
        assert _xpath_string_literal('Click "OK"') == "'Click \"OK\"'"

    def test_string_with_both_quote_kinds_uses_concat(self):
        result = _xpath_string_literal("It's \"complex\"")
        assert result.startswith("concat(")
        assert "It" in result
        assert "complex" in result


class TestIsRealSelectId:
    def test_real_id(self):
        assert _is_real_select_id("permission_id") is True

    def test_auto_generated_is_not_real(self):
        assert _is_real_select_id("tomselect-6") is False

    def test_auto_generated_with_index(self):
        assert _is_real_select_id("tomselect-11") is False

    def test_none_is_not_real(self):
        assert _is_real_select_id(None) is False

    def test_empty_is_not_real(self):
        assert _is_real_select_id("") is False


class TestSelectIdFromInputId:
    def test_strips_ts_control_suffix(self):
        assert (
            _select_id_from_input_id("permission_id-ts-control")
            == "permission_id"
        )

    def test_returns_none_when_suffix_missing(self):
        assert _select_id_from_input_id("permission_id") is None

    def test_returns_none_for_empty(self):
        assert _select_id_from_input_id("") is None

    def test_returns_none_for_none(self):
        assert _select_id_from_input_id(None) is None

    def test_handles_auto_generated_input_id(self):
        # 'tomselect-6-ts-control' → 'tomselect-6'. The caller is
        # responsible for filtering via _is_real_select_id.
        assert (
            _select_id_from_input_id("tomselect-6-ts-control")
            == "tomselect-6"
        )


# ----------------------------------------------------------------------
# find_locator framework dispatch
# ----------------------------------------------------------------------


def _make_search_context(count_value: int = 0, evaluate_value=None):
    """
    Build a mock search_context that supports both:
      - search_context.locator(s).count() → coroutine returning count_value
      - search_context.evaluate(js, args)  → coroutine returning evaluate_value
    """
    mock_locator = MagicMock()
    mock_locator.count = AsyncMock(return_value=count_value)

    ctx = MagicMock()
    ctx.locator = MagicMock(return_value=mock_locator)
    ctx.evaluate = AsyncMock(return_value=evaluate_value)
    return ctx


def _info(framework: str = "", confidence: str = "high") -> ElementTypeInfo:
    return ElementTypeInfo(
        primary_type="dropdown",
        framework=framework,
        confidence=confidence,
        signals=["test:signal"],
    )


@pytest.mark.asyncio
async def test_native_framework_falls_through_to_none():
    result = await find_locator(
        page=None,
        element_data={"tagName": "select", "id": "country"},
        type_info=_info(framework="native"),
        element_id="elem_1",
        element_description="Country dropdown",
        expected_text=None,
        search_context=_make_search_context(),
        iframe_context=None,
        confirmed_coords=None,
    )
    assert result is None


@pytest.mark.asyncio
async def test_combobox_input_framework_falls_through():
    result = await find_locator(
        page=None,
        element_data={"role": "combobox"},
        type_info=_info(framework="combobox-input"),
        element_id="elem_1",
        element_description="Search box",
        expected_text=None,
        search_context=_make_search_context(),
        iframe_context=None,
        confirmed_coords=None,
    )
    assert result is None


@pytest.mark.asyncio
async def test_unknown_framework_falls_through():
    result = await find_locator(
        page=None,
        element_data={},
        type_info=_info(framework="select2"),
        element_id="elem_1",
        element_description="Country dropdown",
        expected_text=None,
        search_context=_make_search_context(),
        iframe_context=None,
        confirmed_coords=None,
    )
    assert result is None


@pytest.mark.asyncio
async def test_tom_select_no_signals_returns_none():
    """No coords, no element_data id, no description → all strategies fail."""
    result = await find_locator(
        page=None,
        element_data={"tagName": "div", "className": "ts-wrapper"},
        type_info=_info(framework="tom-select"),
        element_id="elem_1",
        element_description="",
        expected_text=None,
        search_context=_make_search_context(count_value=0),
        iframe_context=None,
        confirmed_coords=None,
    )
    assert result is None


# ----------------------------------------------------------------------
# Tom Select Strategy 1a (id-anchored input)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strategy_1a_uses_input_id_from_element_data():
    """
    element_data carries id='permission_id-ts-control' → derive select_id
    'permission_id' (real) → Strategy 1a uses css=#permission_id-ts-control.
    """
    ctx = _make_search_context(count_value=1)
    result = await find_locator(
        page=None,
        element_data={
            "tagName": "input",
            "id": "permission_id-ts-control",
            "role": "combobox",
        },
        type_info=_info(framework="tom-select"),
        element_id="elem_2",
        element_description="Role *",
        expected_text=None,
        search_context=ctx,
        iframe_context=None,
        confirmed_coords=None,
    )
    assert result is not None
    assert result["best_locator"] == "css=#permission_id-ts-control"
    assert result["element_type"] == "dropdown"
    assert result["dropdown_framework"] == "tom-select"
    assert result["select_id"] == "permission_id"
    assert result["unique"] is True
    assert result["classifier_confidence"] == "high"
    assert result["classifier_signals"] == ["test:signal"]
    assert "Strategy 1a" in result["all_locators"][0]["strategy"]


@pytest.mark.asyncio
async def test_strategy_1a_skipped_when_input_id_is_auto_generated():
    """tomselect-N is unstable across renders — must skip 1a/1b for these."""
    ctx = _make_search_context(count_value=1)
    result = await find_locator(
        page=None,
        element_data={"id": "tomselect-6-ts-control"},
        type_info=_info(framework="tom-select"),
        element_id="elem_3",
        element_description="",  # no label → 2/3 also skipped
        expected_text=None,
        search_context=ctx,
        iframe_context=None,
        confirmed_coords=None,  # no Strategy 4 either
    )
    # No strategy can run → falls through.
    assert result is None


@pytest.mark.asyncio
async def test_strategy_1a_via_coord_probe():
    """
    Coord probe returns real select_id and input_id → Strategy 1a uses
    the input id directly.
    """
    ctx = _make_search_context(
        count_value=1,
        evaluate_value={
            "selectId": "currency_id",
            "inputId": "currency_id-ts-control",
        },
    )
    result = await find_locator(
        page=None,
        element_data={"tagName": "div", "className": "ts-wrapper"},
        type_info=_info(framework="tom-select"),
        element_id="elem_4",
        element_description="Currency *",
        expected_text=None,
        search_context=ctx,
        iframe_context=None,
        confirmed_coords=(420, 380),
    )
    assert result is not None
    assert result["best_locator"] == "css=#currency_id-ts-control"
    assert result["select_id"] == "currency_id"


# ----------------------------------------------------------------------
# Tom Select fall-through to label-based strategies
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strategy_2_used_when_no_real_select_id():
    """
    Coord probe returns auto-gen select_id ('tomselect-6') → Strategies 1a/1b
    skipped → Strategy 2 (label-text traversal) fires using normalized label.
    """
    # First call: 1a count check would not run (no real input). Strategy 2
    # XPath gets count=1.
    locator_calls = {"count": 0}

    def counting_locator(selector):
        locator_calls["last"] = selector
        locator_calls["count"] += 1
        m = MagicMock()
        m.count = AsyncMock(return_value=1)
        return m

    ctx = MagicMock()
    ctx.locator = counting_locator
    ctx.evaluate = AsyncMock(return_value={
        "selectId": "tomselect-6",  # auto-generated
        "inputId": "tomselect-6-ts-control",
    })

    result = await find_locator(
        page=None,
        element_data={"tagName": "div", "className": "ts-wrapper"},
        type_info=_info(framework="tom-select"),
        element_id="elem_5",
        element_description="Timezone *",
        expected_text=None,
        search_context=ctx,
        iframe_context=None,
        confirmed_coords=(100, 200),
    )

    assert result is not None
    last_locator = locator_calls["last"]
    assert last_locator.startswith("xpath=")
    assert "//label[normalize-space()='Timezone']" in last_locator
    assert "ts-wrapper" in last_locator
    assert "ts-control" in last_locator
    assert "Strategy 2" in result["all_locators"][0]["strategy"]


# ----------------------------------------------------------------------
# Tom Select Strategy 4 (coord-based DOM walk)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strategy_4_uses_wrapper_index_xpath():
    """
    No id, no usable label → Strategy 4 evaluates wrapperIndex from coords
    and emits a positional XPath.
    """
    locator_calls = {"selectors": []}

    def counting_locator(selector):
        locator_calls["selectors"].append(selector)
        m = MagicMock()
        m.count = AsyncMock(return_value=1)
        return m

    # evaluate call order after offscreen-coords fix:
    #   1. _resolve: coord probe → no ids
    #   2. _resolve Source 3: viewport height check
    #   3. _resolve Source 3: scroll-then-probe (coords.y > vh=0) → no ids
    #   4. Strategy 4: wrapperIndex=4
    ctx = MagicMock()
    ctx.locator = counting_locator
    ctx.evaluate = AsyncMock(side_effect=[
        {"selectId": None, "inputId": None},  # _resolve coord probe
        {"h": 1080},                           # viewport height check
        {"wrapperIndex": 4},                    # strategy 4 probe
    ])

    result = await find_locator(
        page=None,
        element_data={"tagName": "div", "className": "ts-wrapper"},
        type_info=_info(framework="tom-select"),
        element_id="elem_6",
        element_description="",  # no label → 2/3 skipped
        expected_text=None,
        search_context=ctx,
        iframe_context=None,
        confirmed_coords=(50, 50),
    )

    assert result is not None
    selector = locator_calls["selectors"][-1]
    assert selector.startswith("xpath=")
    # 0-based wrapperIndex 4 → 1-based XPath [5].
    assert "[5]" in selector
    assert "Strategy 4" in result["all_locators"][0]["strategy"]


# ----------------------------------------------------------------------
# Always-fallback contract — never raises, returns None on any error
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_count_check_exception_returns_none():
    """If Playwright raises, the strategy returns None and the next runs."""
    failing_locator = MagicMock()
    failing_locator.count = AsyncMock(side_effect=RuntimeError("boom"))

    ctx = MagicMock()
    ctx.locator = MagicMock(return_value=failing_locator)
    ctx.evaluate = AsyncMock(return_value=None)  # no coord probe data

    result = await find_locator(
        page=None,
        element_data={"id": "permission_id-ts-control"},
        type_info=_info(framework="tom-select"),
        element_id="elem_7",
        element_description="Role",
        expected_text=None,
        search_context=ctx,
        iframe_context=None,
        confirmed_coords=None,
    )
    # All strategies error out → None (never raises).
    assert result is None


@pytest.mark.asyncio
async def test_evaluate_exception_does_not_raise():
    """JS probe failure must be swallowed and produce None overall."""
    ctx = MagicMock()
    ctx.locator = MagicMock(
        return_value=MagicMock(count=AsyncMock(return_value=0))
    )
    ctx.evaluate = AsyncMock(side_effect=RuntimeError("page closed"))

    result = await find_locator(
        page=None,
        element_data={},
        type_info=_info(framework="tom-select"),
        element_id="elem_8",
        element_description="",
        expected_text=None,
        search_context=ctx,
        iframe_context=None,
        confirmed_coords=(10, 10),
    )
    assert result is None
