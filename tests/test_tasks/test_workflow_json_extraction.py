"""
Mocked integration tests for workflow result JSON extraction.

Purpose: After the browser-use agent finishes, the result text contains JSON
         snippets with element locators.  extract_element_json_from_result and
         extract_all_element_jsons parse these for the final response.
         A regression here means we silently lose locator data.

Tests:
  extract_element_json_from_result: valid/no-element-id/nested/empty
  extract_all_element_jsons: found/dedup/empty-input
"""

import pytest
from browser_service.utils.json_parser import (
    extract_json_for_element,
    extract_workflow_json,
)


class TestExtractFromResultLines:
    """Tests for extracting element JSON from result text lines."""

    def test_valid_element_in_result(self):
        """Element JSON is correctly extracted from agent result text."""
        result_text = """
        I found the element. Here is the locator information:
        {"element_id": "elem_1", "locator": "id=search-box", "found": true, "element_info": {"tagName": "input"}}
        Task completed successfully.
        """
        result = extract_json_for_element(result_text, "elem_1")
        assert result is not None
        assert result["locator"] == "id=search-box"
        assert result["found"] is True

    def test_element_id_not_in_result(self):
        """Returns None when the requested element_id is not in text."""
        result_text = '{"element_id": "elem_1", "locator": "id=x"}'
        result = extract_json_for_element(result_text, "elem_5")
        assert result is None

    def test_multiple_elements_finds_right_one(self):
        """When text has multiple elements, extracts the correct one."""
        result_text = """
        {"element_id": "elem_1", "locator": "id=search"}
        {"element_id": "elem_2", "locator": "name=submit"}
        """
        result1 = extract_json_for_element(result_text, "elem_1")
        result2 = extract_json_for_element(result_text, "elem_2")
        assert result1 is not None
        assert result1["locator"] == "id=search"
        assert result2 is not None
        assert result2["locator"] == "name=submit"

    def test_empty_result_text(self):
        """Empty result text returns None."""
        assert extract_json_for_element("", "elem_1") is None
        assert extract_json_for_element(None, "elem_1") is None


class TestExtractWorkflowResult:
    """Tests for extracting complete workflow result JSON."""

    def test_workflow_with_success_and_mapping(self):
        """Full workflow result with locator_mapping."""
        text = '{"workflow_completed": true, "results": [{"element_id": "elem_1", "locator": "id=search"}]}'
        result = extract_workflow_json(text)
        assert result is not None
        assert result["workflow_completed"] is True

    def test_workflow_no_json(self):
        """Plain text without JSON returns None."""
        assert extract_workflow_json("Agent completed with no results") is None

    def test_workflow_none_input(self):
        """None input handled gracefully."""
        assert extract_workflow_json(None) is None
