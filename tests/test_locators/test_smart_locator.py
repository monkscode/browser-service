"""
Unit tests for browser_service.locators.smart_locator.

Tests deterministic helper functions that have no Playwright dependency:
  - _escape_css_selector()
  - is_dropdown_element()
  - Priority constants are ordered correctly
"""

import pytest

from browser_service.locators.smart_locator import (
    DROPDOWN_CSS_PATTERNS,
    DROPDOWN_KEYWORDS,
    PRIORITY_ARIA_LABEL,
    PRIORITY_CSS_CLASS,
    PRIORITY_ID,
    PRIORITY_NAME,
    PRIORITY_ROLE,
    PRIORITY_TEST_ID,
    PRIORITY_TEXT,
    PRIORITY_XPATH_TEXT,
    _escape_css_selector,
    _is_collection_element,
    is_dropdown_element,
)


class TestEscapeCssSelector:
    """Tests for _escape_css_selector()."""

    def test_alphanumeric_unchanged(self):
        assert _escape_css_selector("submitButton") == "submitButton"

    def test_hyphen_preserved(self):
        assert _escape_css_selector("my-button") == "my-button"

    def test_underscore_preserved(self):
        assert _escape_css_selector("my_button") == "my_button"

    def test_colon_escaped(self):
        result = _escape_css_selector("btn:primary")
        assert "\\:" in result

    def test_dot_escaped(self):
        result = _escape_css_selector("btn.active")
        assert "\\." in result

    def test_bracket_escaped(self):
        result = _escape_css_selector("btn[0]")
        assert "\\[" in result

    def test_empty_string_returns_empty(self):
        assert _escape_css_selector("") == ""

    def test_whitespace_only_returns_empty(self):
        assert _escape_css_selector("   ") == ""

    def test_none_returns_empty(self):
        assert _escape_css_selector(None) == ""

    def test_multi_word_class_returns_empty(self):
        """Classes with spaces (multi-word) cannot be used in simple CSS — return empty."""
        assert _escape_css_selector("foo bar") == ""

    def test_numbers_are_kept(self):
        result = _escape_css_selector("btn123")
        assert result == "btn123"

    def test_parenthesis_escaped(self):
        result = _escape_css_selector("fn()")
        assert "\\(" in result


class TestIsDropdownElement:
    """Tests for is_dropdown_element()."""

    # --- Description-based detection ---

    def test_description_dropdown_keyword(self):
        assert is_dropdown_element({}, "Select a country dropdown") is True

    def test_description_select_keyword(self):
        assert is_dropdown_element({}, "Select payment method") is True

    def test_description_combobox_keyword(self):
        assert is_dropdown_element({}, "Open the combobox for cities") is True

    def test_description_picker_keyword(self):
        assert is_dropdown_element({}, "Date picker field") is True

    def test_description_no_dropdown_keyword(self):
        """A description with no dropdown keyword must classify as False.

        `isinstance(result, bool)` passed for True as well, so the negative case
        — the one that stops non-dropdowns entering the dropdown handler — was
        never actually checked.
        """
        assert is_dropdown_element({}, "Click the submit button") is False

    # --- ARIA role detection ---

    def test_combobox_role(self):
        assert is_dropdown_element({"role": "combobox"}) is True

    def test_listbox_role(self):
        assert is_dropdown_element({"role": "listbox"}) is True

    def test_button_role_not_dropdown(self):
        result = is_dropdown_element({"role": "button"})
        assert result is False

    def test_role_case_insensitive(self):
        assert is_dropdown_element({"role": "COMBOBOX"}) is True

    # --- Native select tag ---

    def test_select_tag_is_dropdown(self):
        assert is_dropdown_element({"tagName": "select"}) is True

    def test_input_tag_not_dropdown(self):
        result = is_dropdown_element({"tagName": "input"})
        assert result is False

    # --- CSS class patterns ---

    def test_kendo_multiselect_class(self):
        assert is_dropdown_element({"className": "k-multiselect some-class"}) is True

    def test_kendo_dropdown_class(self):
        assert is_dropdown_element({"className": "k-dropdown"}) is True

    def test_material_ui_select_class(self):
        assert is_dropdown_element({"className": "MuiSelect-root"}) is True

    def test_react_select_class(self):
        assert is_dropdown_element({"className": "react-select__control"}) is True

    def test_generic_button_class_not_dropdown(self):
        result = is_dropdown_element({"className": "btn btn-primary"})
        assert result is False

    # --- Empty / None element_data ---

    def test_none_element_data_with_description(self):
        assert is_dropdown_element(None, "Select from dropdown") is True

    def test_none_element_data_no_description(self):
        result = is_dropdown_element(None)
        assert result is False

    def test_empty_dict_no_description(self):
        result = is_dropdown_element({})
        assert result is False


class TestPriorityOrdering:
    """Priority constants must be ordered correctly (lower = higher priority)."""

    def test_id_beats_test_id(self):
        assert PRIORITY_ID < PRIORITY_TEST_ID

    def test_test_id_beats_name(self):
        assert PRIORITY_TEST_ID < PRIORITY_NAME

    def test_name_beats_aria_label(self):
        assert PRIORITY_NAME < PRIORITY_ARIA_LABEL

    def test_aria_label_beats_text(self):
        assert PRIORITY_ARIA_LABEL < PRIORITY_TEXT

    def test_text_beats_role(self):
        assert PRIORITY_TEXT < PRIORITY_ROLE

    def test_role_beats_css_class(self):
        assert PRIORITY_ROLE < PRIORITY_CSS_CLASS

    def test_css_class_beats_xpath_text(self):
        assert PRIORITY_CSS_CLASS < PRIORITY_XPATH_TEXT

    def test_all_priorities_are_non_negative(self):
        priorities = [
            PRIORITY_ID,
            PRIORITY_TEST_ID,
            PRIORITY_NAME,
            PRIORITY_ARIA_LABEL,
            PRIORITY_TEXT,
            PRIORITY_ROLE,
            PRIORITY_CSS_CLASS,
        ]
        assert all(p >= 0 for p in priorities)

    def test_all_priorities_are_unique(self):
        priorities = [
            PRIORITY_ID,
            PRIORITY_TEST_ID,
            PRIORITY_NAME,
            PRIORITY_ARIA_LABEL,
            PRIORITY_TEXT,
            PRIORITY_ROLE,
            PRIORITY_CSS_CLASS,
        ]
        assert len(set(priorities)) == len(priorities)


class TestDropdownConstants:
    """Verify DROPDOWN_KEYWORDS and DROPDOWN_CSS_PATTERNS are well-formed."""

    def test_dropdown_keywords_not_empty(self):
        assert len(DROPDOWN_KEYWORDS) > 0

    def test_dropdown_keywords_all_lowercase(self):
        for kw in DROPDOWN_KEYWORDS:
            assert kw == kw.lower(), f"Keyword '{kw}' should be lowercase"

    def test_dropdown_keywords_contains_select(self):
        assert "select" in DROPDOWN_KEYWORDS

    def test_dropdown_keywords_contains_combobox(self):
        assert "combobox" in DROPDOWN_KEYWORDS

    def test_dropdown_css_patterns_not_empty(self):
        assert len(DROPDOWN_CSS_PATTERNS) > 0

    def test_dropdown_css_patterns_are_strings(self):
        for pattern in DROPDOWN_CSS_PATTERNS:
            assert isinstance(pattern, str)


class TestIsCollectionElement:
    """Tests for _is_collection_element() — Method 3 per-token + nav-prefix exclusion (Bug 2)."""

    # --- Method 1: Description keywords ---

    def test_description_rows_keyword(self):
        assert _is_collection_element({}, "all visible rows") is True

    def test_description_items_keyword(self):
        assert _is_collection_element({}, "list of items") is True

    def test_description_no_keyword(self):
        assert _is_collection_element({}, "submit button") is False

    # --- Method 2: HTML collection tags ---

    def test_tr_tag_is_collection(self):
        assert _is_collection_element({"tagName": "tr"}, "row") is True

    def test_li_tag_is_collection(self):
        assert _is_collection_element({"tagName": "li"}, "menu entry") is True

    def test_div_tag_alone_not_collection(self):
        assert _is_collection_element({"tagName": "div", "className": ""}, "panel") is False

    # --- Method 3: Class-token matching (Bug 2 — the regression we're guarding against) ---

    def test_nav_item_is_NOT_collection(self):
        """Bug 2 regression: 'nav-item' must not match the 'item' pattern."""
        assert (
            _is_collection_element(
                {"tagName": "a", "className": "nav-item"}, "Customer option in Create menu"
            )
            is False
        )

    def test_nav_item_with_extra_classes_is_NOT_collection(self):
        """Real ASTPP markup typically has additional bootstrap classes alongside nav-item."""
        assert (
            _is_collection_element(
                {"tagName": "a", "className": "nav-item dropdown-item active"}, "Customer"
            )
            is False
        )

    def test_menu_item_is_NOT_collection(self):
        """tagName=a so Method 2 (li/tr/option) doesn't fire — isolates Method 3 nav-prefix exclusion."""
        assert (
            _is_collection_element({"tagName": "a", "className": "menu-item"}, "settings link")
            is False
        )

    def test_breadcrumb_item_is_NOT_collection(self):
        """tagName left as 'span' so Method 2 doesn't fire — isolates Method 3 behavior."""
        assert (
            _is_collection_element(
                {"tagName": "span", "className": "breadcrumb-item"}, "Home crumb"
            )
            is False
        )

    def test_pagination_item_is_NOT_collection(self):
        assert (
            _is_collection_element({"tagName": "span", "className": "pagination-item"}, "page 2")
            is False
        )

    def test_dropdown_item_is_NOT_collection(self):
        """dropdown-* items are menu chrome, not data collections."""
        assert (
            _is_collection_element({"tagName": "a", "className": "dropdown-item"}, "Customer")
            is False
        )

    # Positive cases — real collection class tokens still match

    def test_bare_row_class_is_collection(self):
        """Bootstrap '.row' on a non-tr element — Method 3 should still match."""
        assert _is_collection_element({"tagName": "div", "className": "row"}, "data row") is True

    def test_card_class_is_collection(self):
        assert (
            _is_collection_element({"tagName": "div", "className": "card mb-3"}, "product card")
            is True
        )

    def test_grid_item_class_is_collection(self):
        """Multi-word token 'grid-item' is in the pattern set verbatim."""
        assert _is_collection_element({"tagName": "div", "className": "grid-item"}, "tile") is True

    def test_substring_only_match_is_NOT_collection(self):
        """'rowboat' contains 'row' as substring but is not the 'row' token."""
        assert (
            _is_collection_element({"tagName": "div", "className": "rowboat"}, "decorative")
            is False
        )

    def test_empty_class_and_tag_not_collection(self):
        assert _is_collection_element({"tagName": "div", "className": ""}, "header") is False


class TestDropdownOverridesCollection:
    """
    Bug 3 guard: when both dropdown and collection signals fire on the same
    element, dropdown must win. The guard added in
    _generate_locators_from_element_data uses these two helpers as inputs;
    these tests prove the helper inputs produce the right verdict for the
    real Tom Select / collection-class overlap case.
    """

    def test_tom_select_wrapper_is_both_dropdown_and_collection(self):
        """
        Pre-fix Tom Select scenario — element_data carries 'form-group row'
        because browser-use coords landed on the outer Bootstrap row.
        _is_collection_element fires (legitimately, on the 'row' token), and
        is_dropdown_element ALSO fires (description contains 'dropdown').
        The guard in smart_locator.py uses is_dropdown_element to override.
        """
        element_data = {"tagName": "div", "className": "form-group row"}
        description = "Rate Group dropdown"
        assert _is_collection_element(element_data, description) is True
        assert is_dropdown_element(element_data, description) is True

    def test_tom_select_wrapper_with_ts_class_is_dropdown(self):
        """When the wrapper class itself is .ts-wrapper, dropdown still wins."""
        element_data = {"tagName": "div", "className": "ts-wrapper"}
        description = "Country dropdown"
        assert is_dropdown_element(element_data, description) is True

    def test_pure_table_row_is_collection_not_dropdown(self):
        """Negative control: a real table row with no dropdown signals stays a collection."""
        element_data = {"tagName": "tr", "className": "data-row"}
        description = "first data row"
        assert _is_collection_element(element_data, description) is True
        assert is_dropdown_element(element_data, description) is False

    def test_combobox_role_overrides_collection_class(self):
        """role=combobox is a high-confidence dropdown signal even on a div with collection-like classes."""
        element_data = {"tagName": "div", "className": "row form-group", "role": "combobox"}
        description = "country selector"
        assert is_dropdown_element(element_data, description) is True
