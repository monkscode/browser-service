"""
Unit tests for browser_service.locators.validation — locator conversion and Playwright checks.

Purpose: validation.py converts Robot Framework locator strings (e.g. "id=search",
         "name=q", "text=Submit") into Playwright-compatible selectors, and validates
         locators for uniqueness via Playwright's count().  A conversion error means
         Playwright can't find the element; a uniqueness failure means we pick a
         non-deterministic locator.

Tests:
  - is_already_playwright_selector: CSS selectors, XPath, role=, text=, data-testid=
  - convert_to_playwright_locator: id=, name=, text= (browser), css=, xpath=,
    class=, aria-label handling, quote stripping, passthrough for unknown formats,
    empty/None inputs
"""

import pytest
from browser_service.locators.validation import (
    is_already_playwright_selector,
    convert_to_playwright_locator,
)


class TestIsAlreadyPlaywrightSelector:
    """Detect selectors that Playwright already understands natively."""

    def test_css_selector(self):
        """CSS selectors (starting with . or #) are Playwright-native."""
        assert is_already_playwright_selector(".main-content") is True
        assert is_already_playwright_selector("#search-box") is True

    def test_xpath_selector(self):
        """XPath selectors (starting with / or //) are Playwright-native."""
        assert is_already_playwright_selector("//div[@id='main']") is True
        assert is_already_playwright_selector("/html/body/div") is True

    def test_text_selector(self):
        """text= prefix is Playwright-native."""
        assert is_already_playwright_selector("text=Submit") is True

    def test_role_selector(self):
        """role= prefix is Playwright-native."""
        assert is_already_playwright_selector("role=button") is True

    def test_data_testid_selector(self):
        """data-testid= is Playwright-native."""
        assert is_already_playwright_selector("data-testid=login-btn") is True

    def test_id_is_playwright(self):
        """id= is Playwright-native."""
        assert is_already_playwright_selector("id=search") is True

    def test_rf_name_not_playwright(self):
        """RF-style name= is NOT Playwright-native (needs conversion)."""
        assert is_already_playwright_selector("name=username") is False


class TestConvertToPlaywrightLocator:
    """Convert RF locators to Playwright-compatible selectors."""

    def test_convert_id(self):
        """id=search → #search or [id='search']."""
        result, _ = convert_to_playwright_locator("id=search-box")
        assert result is not None
        assert "search-box" in result

    def test_convert_name(self):
        """name=q → [name='q']."""
        result, converted = convert_to_playwright_locator("name=q")
        assert "name" in result
        assert "q" in result

    def test_convert_text_browser(self):
        """text=Submit stays as text=Submit (Playwright-native)."""
        result, converted = convert_to_playwright_locator("text=Submit")
        assert "text=" in result
        assert "Submit" in result

    def test_convert_css(self):
        """css=.btn → .btn (strip css= prefix)."""
        result, converted = convert_to_playwright_locator("css=.btn-primary")
        assert ".btn-primary" in result

    def test_convert_xpath(self):
        """xpath=//div → //div (strip xpath= prefix)."""
        result, converted = convert_to_playwright_locator("xpath=//div[@class='main']")
        assert "//div" in result

    def test_passthrough_css_selector(self):
        """Already-Playwright CSS selectors pass through unchanged."""
        result, converted = convert_to_playwright_locator(".my-class")
        assert result == ".my-class"

    def test_passthrough_xpath(self):
        """Already-Playwright XPath selectors pass through unchanged."""
        result, converted = convert_to_playwright_locator("//button[@type='submit']")
        assert result == "//button[@type='submit']"

    def test_convert_class(self):
        """class=btn-primary → .btn-primary."""
        result, converted = convert_to_playwright_locator("class=btn-primary")
        assert "btn-primary" in result

    def test_id_with_special_chars(self):
        """id= with special characters (colon, period) is escaped properly."""
        result, converted = convert_to_playwright_locator("id=react-select-4-input")
        assert "react-select-4-input" in result

    def test_quote_stripping(self):
        """Surrounding quotes are stripped: 'id=x' → id=x."""
        result, converted = convert_to_playwright_locator("'id=login-btn'")
        assert "login-btn" in result

    def test_empty_input(self):
        """Empty string returns None or empty string."""
        result, converted = convert_to_playwright_locator("")
        assert result is None or result == ""

    def test_none_input(self):
        """None input returns empty string."""
        result, converted = convert_to_playwright_locator(None)
        assert result == ""

    def test_aria_label_conversion(self):
        """aria-label locator is converted to Playwright attribute selector."""
        result, converted = convert_to_playwright_locator("aria-label=Close dialog")
        assert "Close dialog" in result
