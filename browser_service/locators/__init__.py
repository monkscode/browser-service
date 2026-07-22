"""
Locator Management Module

This module provides locator validation functionality
for browser automation. It uses Playwright's built-in methods for fast, reliable
locator validation without requiring large JavaScript code generation.

Key Components:
- validation: Validate locators using Playwright API (uniqueness checking)
- conversion: Convert various locator formats to Playwright-compatible format
- smart_locator: Advanced locator finding with 21+ strategies (core IP)

Priority Order (highest to lowest):
1. id - Most stable, fastest
2. data-testid - Designed for testing
3. name - Semantic, stable
4. aria-label - Accessibility, semantic
5. text - Content-based (can be fragile)
6. role - Playwright-specific, semantic
7. css-class - Lower priority, can change
"""

from .smart_locator import (
    candidate_targets_row_anchor,
    correct_expected_text_for_row_anchor,
    find_unique_locator_at_coordinates,
    validate_semantic_match,
)
from .validation import (
    PLAYWRIGHT_NATIVE_ENGINES,
    convert_to_playwright_locator,
    is_already_playwright_selector,
    validate_locator_playwright,
)

__all__ = [
    "validate_locator_playwright",
    "convert_to_playwright_locator",
    "is_already_playwright_selector",
    "PLAYWRIGHT_NATIVE_ENGINES",
    "candidate_targets_row_anchor",
    "correct_expected_text_for_row_anchor",
    "find_unique_locator_at_coordinates",
    "validate_semantic_match",
]
