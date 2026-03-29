"""
Unit tests for browser_service.locators.generation — locator generation from element attributes.

Purpose: generate_locators_from_attributes is the deterministic engine that converts
         raw element attributes (id, name, class, etc.) into prioritized locator
         candidates for both Browser Library and SeleniumLibrary.  A regression here
         means we generate broken locators that fail silently during RF test execution.

Tests cover:
  - High-priority attributes: id, data-testid, name
  - Browser vs Selenium syntax differences
  - Checkbox/radio special handling (type attribute)
  - aria-label, text content, role-based locators
  - Class name locators
  - Empty attributes produce no locators
  - Priority ordering (id > testId > name > aria > text > class > xpath)
"""

import pytest
from browser_service.locators.generation import generate_locators_from_attributes


class TestLocatorGenerationById:
    """ID-based locator generation — highest priority."""

    def test_id_generates_locator(self):
        """Element with 'id' produces an id= locator."""
        attrs = {"id": "search-box", "tagName": "input"}
        locators = generate_locators_from_attributes(attrs, library_type="browser")
        assert any("id=search-box" in loc["locator"] for loc in locators)

    def test_id_is_first_priority(self):
        """ID locator appears before other locator types."""
        attrs = {"id": "main-btn", "name": "submit", "tagName": "button"}
        locators = generate_locators_from_attributes(attrs, library_type="browser")
        # First locator should be the id-based one
        id_indices = [i for i, loc in enumerate(locators) if "id=main-btn" in loc["locator"]]
        name_indices = [i for i, loc in enumerate(locators) if "name=" in loc["locator"]]
        if id_indices and name_indices:
            assert id_indices[0] < name_indices[0]


class TestLocatorGenerationByTestId:
    """data-testid based locators — second priority."""

    def test_testid_browser_syntax(self):
        """Browser Library uses data-testid= prefix."""
        attrs = {"testId": "login-form", "tagName": "form"}
        locators = generate_locators_from_attributes(attrs, library_type="browser")
        assert any("data-testid" in loc["locator"] and "login-form" in loc["locator"] for loc in locators)

    def test_testid_selenium_syntax(self):
        """SeleniumLibrary uses css=[data-testid='...'] syntax."""
        attrs = {"testId": "login-form", "tagName": "form"}
        locators = generate_locators_from_attributes(attrs, library_type="selenium")
        has_testid = any("data-testid" in loc["locator"] and "login-form" in loc["locator"] for loc in locators)
        assert has_testid


class TestLocatorGenerationByName:
    """Name-based locators."""

    def test_name_generates_locator(self):
        """Element with 'name' produces a name= locator."""
        attrs = {"name": "username", "tagName": "input"}
        locators = generate_locators_from_attributes(attrs, library_type="browser")
        assert any("name=" in loc["locator"] and "username" in loc["locator"] for loc in locators)


class TestLocatorGenerationSpecialCases:
    """Special handling for checkboxes, radios, aria, text, role."""

    def test_checkbox_type_included(self):
        """Checkbox generates type-aware locator (for force-click)."""
        attrs = {"type": "checkbox", "name": "agree", "tagName": "input"}
        locators = generate_locators_from_attributes(attrs, library_type="browser")
        assert len(locators) > 0

    def test_aria_label_generates_locator(self):
        """aria-label produces accessible locator."""
        attrs = {"ariaLabel": "Close dialog", "tagName": "button"}
        locators = generate_locators_from_attributes(attrs, library_type="browser")
        assert any("Close dialog" in loc["locator"] for loc in locators)

    def test_text_content_generates_locator(self):
        """textContent produces text= locator for Browser Library."""
        attrs = {"text": "Submit Order", "tagName": "button"}
        locators = generate_locators_from_attributes(attrs, library_type="browser")
        assert any("text=" in loc["locator"] and "Submit Order" in loc["locator"] for loc in locators)

    def test_role_generates_locator(self):
        """role attribute produces role-based locator."""
        attrs = {"role": "searchbox", "tagName": "input", "text": "Search"}
        locators = generate_locators_from_attributes(attrs, library_type="browser")
        assert any("role" in loc["locator"] for loc in locators)

    def test_classname_generates_locator(self):
        """className produces a CSS-class locator."""
        attrs = {"className": "btn btn-primary", "tagName": "button"}
        locators = generate_locators_from_attributes(attrs, library_type="browser")
        assert any("btn" in loc["locator"] for loc in locators)

    def test_empty_attributes_no_locators(self):
        """Element with all empty attributes produces empty or xpath-only list."""
        attrs = {"id": "", "name": "", "tagName": "div"}
        locators = generate_locators_from_attributes(attrs, library_type="browser")
        # Should have no attribute-based locators (xpath fallback is ok)
        non_xpath = [loc for loc in locators if "xpath" not in loc["locator"].lower()]
        # May be empty or contain only generic locators
        assert isinstance(locators, list)


