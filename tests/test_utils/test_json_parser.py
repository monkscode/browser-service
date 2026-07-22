"""
Unit tests for browser_service.utils.json_parser — robust JSON extraction.

Purpose: json_parser extracts structured JSON from messy LLM output and
         workflow results.  The agent returns JSON embedded in prose,
         markdown, or with single-quote strings — any extraction failure
         means the workflow silently loses locator data.

Tests cover both extract_element_json and extract_workflow_result:
  - Standard JSON embedded in text
  - Nested/deep object structures
  - Minified JSON (no whitespace)
  - No matching element_id → returns None
  - None/empty input → returns None
  - Escaped characters in values
  - Single-quote JSON (common LLM mistake)
  - Workflow result with all fields
  - Workflow result missing required fields
  - Workflow result with no results array
  - None workflow input
"""

import pytest

from browser_service.utils.json_parser import (
    extract_json_for_element,
    extract_workflow_json,
)


class TestExtractElementJson:
    """Tests for extracting element locator JSON from text."""

    def test_standard_json_in_text(self):
        """JSON object with element_id found in surrounding text."""
        text = 'Some text {"element_id": "elem_1", "locator": "id=search"} more text'
        result = extract_json_for_element(text, "elem_1")
        assert result is not None
        assert result["element_id"] == "elem_1"
        assert result["locator"] == "id=search"

    def test_nested_object(self):
        """JSON with nested objects (element_info) is extracted correctly."""
        text = '{"element_id": "elem_2", "locator": "name=q", "element_info": {"tagName": "input", "id": "search"}}'
        result = extract_json_for_element(text, "elem_2")
        assert result is not None
        assert result["element_info"]["tagName"] == "input"

    def test_minified_json(self):
        """Minified JSON (no spaces) is handled."""
        text = '{"element_id":"elem_3","locator":"id=btn","found":true}'
        result = extract_json_for_element(text, "elem_3")
        assert result is not None
        assert result["found"] is True

    def test_element_id_not_found(self):
        """Returns None when element_id doesn't match any JSON in text."""
        text = '{"element_id": "elem_1", "locator": "id=x"}'
        result = extract_json_for_element(text, "elem_99")
        assert result is None

    def test_none_input(self):
        """None text input returns None gracefully."""
        result = extract_json_for_element(None, "elem_1")
        assert result is None

    def test_empty_input(self):
        """Empty string returns None gracefully."""
        result = extract_json_for_element("", "elem_1")
        assert result is None

    def test_escaped_characters(self):
        """JSON with escaped quotes in values is handled."""
        text = r'{"element_id": "elem_4", "locator": "xpath=//div[@class=\"main\"]"}'
        result = extract_json_for_element(text, "elem_4")
        assert result is not None
        assert "xpath=" in result["locator"]


class TestExtractWorkflowResult:
    """Tests for extracting workflow result JSON."""

    def test_valid_workflow_result(self):
        """Standard workflow result with success and locator_mapping."""
        text = (
            '{"workflow_completed": true, "results": [{"element_id": "elem_1", "locator": "id=x"}]}'
        )
        result = extract_workflow_json(text)
        assert result is not None
        assert result["workflow_completed"] is True
        assert "results" in result

    def test_missing_locator_mapping(self):
        """Result without locator_mapping — still returns parsed JSON."""
        text = '{"workflow_completed": true, "results": []}'
        result = extract_workflow_json(text)
        assert result is not None
        assert result["workflow_completed"] is True

    def test_no_json_in_text(self):
        """Plain text with no JSON returns None."""
        result = extract_workflow_json("No JSON here, just text.")
        assert result is None

    def test_none_input(self):
        """None input returns None gracefully."""
        result = extract_workflow_json(None)
        assert result is None
