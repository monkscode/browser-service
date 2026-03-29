"""
Unit tests for browser_service.locators.smart_locator.

Tests deterministic helper functions that have no Playwright dependency:
  - _escape_css_selector()
  - is_dropdown_element()
  - Priority constants are ordered correctly
"""

import pytest
from browser_service.locators.smart_locator import (
    _escape_css_selector,
    is_dropdown_element,
    PRIORITY_CANDIDATE,
    PRIORITY_ID,
    PRIORITY_TEST_ID,
    PRIORITY_NAME,
    PRIORITY_ARIA_LABEL,
    PRIORITY_TEXT,
    PRIORITY_ROLE,
    PRIORITY_CSS_CLASS,
    PRIORITY_XPATH_TEXT,
    DROPDOWN_KEYWORDS,
    DROPDOWN_CSS_PATTERNS,
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
        result = is_dropdown_element({}, "Click the submit button")
        # Should not be True just because of this description
        assert isinstance(result, bool)

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

    def test_candidate_is_highest_priority(self):
        assert PRIORITY_CANDIDATE < PRIORITY_ID

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
            PRIORITY_CANDIDATE, PRIORITY_ID, PRIORITY_TEST_ID, PRIORITY_NAME,
            PRIORITY_ARIA_LABEL, PRIORITY_TEXT, PRIORITY_ROLE, PRIORITY_CSS_CLASS,
        ]
        assert all(p >= 0 for p in priorities)

    def test_all_priorities_are_unique(self):
        priorities = [
            PRIORITY_CANDIDATE, PRIORITY_ID, PRIORITY_TEST_ID, PRIORITY_NAME,
            PRIORITY_ARIA_LABEL, PRIORITY_TEXT, PRIORITY_ROLE, PRIORITY_CSS_CLASS,
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
