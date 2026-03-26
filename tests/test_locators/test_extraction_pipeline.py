"""
Unit tests for browser_service.locators.extraction.

Tests extract_and_validate_locators() with mocked Playwright pages.
Also tests extract_element_attributes() error path and return contract.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page(evaluate_return=None, evaluate_side_effect=None):
    """Build a mock Playwright page for extract_element_attributes()."""
    page = MagicMock()
    if evaluate_side_effect:
        page.evaluate = AsyncMock(side_effect=evaluate_side_effect)
    else:
        page.evaluate = AsyncMock(return_value=evaluate_return)
    return page


SAMPLE_ATTRS = {
    "id": "submit-btn",
    "name": None,
    "testId": None,
    "ariaLabel": "Submit form",
    "role": "button",
    "title": None,
    "placeholder": None,
    "type": "submit",
    "tagName": "button",
    "className": "btn btn-primary",
    "text": "Submit",
    "href": None,
    "visible": True,
    "boundingBox": {"x": 100, "y": 200, "width": 80, "height": 30},
}


class TestExtractElementAttributes:
    """Tests for extract_element_attributes() — mocks page.evaluate."""

    @pytest.mark.asyncio
    async def test_returns_dict_when_element_found(self):
        from browser_service.locators.extraction import extract_element_attributes
        page = _make_page(evaluate_return=SAMPLE_ATTRS)
        result = await extract_element_attributes(page, {"x": 100, "y": 200})
        assert isinstance(result, dict)
        assert result["tagName"] == "button"

    @pytest.mark.asyncio
    async def test_returns_none_when_element_not_found(self):
        """JS returns null → Python returns None."""
        from browser_service.locators.extraction import extract_element_attributes
        page = _make_page(evaluate_return=None)
        result = await extract_element_attributes(page, {"x": 0, "y": 0})
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        """Any exception from page.evaluate is caught and returns None."""
        from browser_service.locators.extraction import extract_element_attributes
        page = _make_page(evaluate_side_effect=Exception("JS error"))
        result = await extract_element_attributes(page, {"x": 50, "y": 50})
        assert result is None

    @pytest.mark.asyncio
    async def test_passes_coords_to_evaluate(self):
        from browser_service.locators.extraction import extract_element_attributes
        page = _make_page(evaluate_return=SAMPLE_ATTRS)
        coords = {"x": 150.5, "y": 300.0}
        await extract_element_attributes(page, coords)
        # evaluate should have been called with the coords
        page.evaluate.assert_called_once()
        call_args = page.evaluate.call_args
        # Second argument (after the JS string) should be the coords
        assert call_args.args[1] == coords or coords in call_args.args


class TestExtractAndValidateLocatorsPipeline:
    """Tests for extract_and_validate_locators() — full pipeline with mocked deps."""

    @pytest.mark.asyncio
    @patch("browser_service.locators.extraction.validate_locator_playwright")
    @patch("browser_service.locators.extraction.generate_locators_from_attributes")
    @patch("browser_service.locators.extraction.extract_element_attributes")
    async def test_element_not_found_returns_found_false(
        self, mock_extract, mock_gen, mock_validate
    ):
        """When element attributes are None, result has found=False."""
        from browser_service.locators.extraction import extract_and_validate_locators
        mock_extract.return_value = None
        page = MagicMock()
        result = await extract_and_validate_locators(page, "Submit button", {"x": 10, "y": 20})
        assert result["found"] is False
        assert "error" in result

    @pytest.mark.asyncio
    @patch("browser_service.locators.extraction.validate_locator_playwright")
    @patch("browser_service.locators.extraction.generate_locators_from_attributes")
    @patch("browser_service.locators.extraction.extract_element_attributes")
    async def test_no_locators_generated_returns_found_false(
        self, mock_extract, mock_gen, mock_validate
    ):
        """When locator generation produces empty list, result has found=False."""
        from browser_service.locators.extraction import extract_and_validate_locators
        mock_extract.return_value = SAMPLE_ATTRS
        mock_gen.return_value = []
        page = MagicMock()
        result = await extract_and_validate_locators(page, "Submit button", {"x": 10, "y": 20})
        assert result["found"] is False
        assert "element_info" in result

    @pytest.mark.asyncio
    @patch("browser_service.locators.extraction.validate_locator_playwright")
    @patch("browser_service.locators.extraction.generate_locators_from_attributes")
    @patch("browser_service.locators.extraction.extract_element_attributes")
    async def test_unique_locator_sets_found_true(
        self, mock_extract, mock_gen, mock_validate
    ):
        """When a unique+correct locator is found, result has found=True and best_locator set."""
        from browser_service.locators.extraction import extract_and_validate_locators

        mock_extract.return_value = SAMPLE_ATTRS
        mock_gen.return_value = [
            {"locator": "#submit-btn", "type": "id", "priority": 1}
        ]
        mock_validate.return_value = {
            "validated": True, "valid": True, "unique": True,
            "correct_element": True, "count": 1,
        }

        page = MagicMock()
        result = await extract_and_validate_locators(page, "Submit button", {"x": 10, "y": 20})

        assert result["found"] is True
        assert result["best_locator"] == "#submit-btn"
        assert result["unique"] is True
        assert result["valid"] is True

    @pytest.mark.asyncio
    @patch("browser_service.locators.extraction.validate_locator_playwright")
    @patch("browser_service.locators.extraction.generate_locators_from_attributes")
    @patch("browser_service.locators.extraction.extract_element_attributes")
    async def test_prefers_unique_over_non_unique(
        self, mock_extract, mock_gen, mock_validate
    ):
        """Unique+correct locator beats valid-but-non-unique locator."""
        from browser_service.locators.extraction import extract_and_validate_locators

        mock_extract.return_value = SAMPLE_ATTRS
        mock_gen.return_value = [
            {"locator": ".btn-primary", "type": "css_class", "priority": 10},
            {"locator": "#submit-btn", "type": "id", "priority": 1},
        ]

        def validate_side_effect(page, locator, coords):
            if locator == "#submit-btn":
                return {"validated": True, "valid": True, "unique": True, "correct_element": True, "count": 1}
            else:
                return {"validated": True, "valid": False, "unique": False, "correct_element": True, "count": 5}

        mock_validate.side_effect = validate_side_effect
        page = MagicMock()
        result = await extract_and_validate_locators(page, "Submit button", {"x": 10, "y": 20})

        assert result["best_locator"] == "#submit-btn"

    @pytest.mark.asyncio
    @patch("browser_service.locators.extraction.validate_locator_playwright")
    @patch("browser_service.locators.extraction.generate_locators_from_attributes")
    @patch("browser_service.locators.extraction.extract_element_attributes")
    async def test_result_has_validation_summary(
        self, mock_extract, mock_gen, mock_validate
    ):
        """Result always contains validation_summary with counts."""
        from browser_service.locators.extraction import extract_and_validate_locators

        mock_extract.return_value = SAMPLE_ATTRS
        mock_gen.return_value = [
            {"locator": "#btn", "type": "id", "priority": 1}
        ]
        mock_validate.return_value = {
            "validated": True, "valid": True, "unique": True,
            "correct_element": True, "count": 1,
        }

        page = MagicMock()
        result = await extract_and_validate_locators(page, "Button", {"x": 10, "y": 20})

        assert "validation_summary" in result
        summary = result["validation_summary"]
        assert "total_generated" in summary
        assert "unique" in summary
        assert summary["validation_method"] == "playwright"

    @pytest.mark.asyncio
    @patch("browser_service.locators.extraction.validate_locator_playwright")
    @patch("browser_service.locators.extraction.generate_locators_from_attributes")
    @patch("browser_service.locators.extraction.extract_element_attributes")
    async def test_no_valid_locators_found_false(
        self, mock_extract, mock_gen, mock_validate
    ):
        """All locators fail validation → found=False, count=0."""
        from browser_service.locators.extraction import extract_and_validate_locators

        mock_extract.return_value = SAMPLE_ATTRS
        mock_gen.return_value = [
            {"locator": ".multi-match", "type": "css_class", "priority": 10}
        ]
        mock_validate.return_value = {
            "validated": True, "valid": False, "unique": False,
            "correct_element": False, "count": 5,
        }

        page = MagicMock()
        result = await extract_and_validate_locators(page, "Button", {"x": 10, "y": 20})

        assert result["found"] is False
        assert result["count"] == 0
        assert result["unique"] is False

    @pytest.mark.asyncio
    @patch("browser_service.locators.extraction.validate_locator_playwright")
    @patch("browser_service.locators.extraction.generate_locators_from_attributes")
    @patch("browser_service.locators.extraction.extract_element_attributes")
    async def test_element_info_always_present_when_attrs_found(
        self, mock_extract, mock_gen, mock_validate
    ):
        """element_info is present in result when attributes were extracted."""
        from browser_service.locators.extraction import extract_and_validate_locators

        mock_extract.return_value = SAMPLE_ATTRS
        mock_gen.return_value = []
        page = MagicMock()
        result = await extract_and_validate_locators(page, "Button", {"x": 10, "y": 20})

        assert "element_info" in result
        assert result["element_info"]["tagName"] == "button"

    @pytest.mark.asyncio
    @patch("browser_service.locators.extraction.validate_locator_playwright")
    @patch("browser_service.locators.extraction.generate_locators_from_attributes")
    @patch("browser_service.locators.extraction.extract_element_attributes")
    async def test_best_locator_uses_priority_order(
        self, mock_extract, mock_gen, mock_validate
    ):
        """When multiple unique locators exist, lowest priority number wins."""
        from browser_service.locators.extraction import extract_and_validate_locators

        mock_extract.return_value = SAMPLE_ATTRS
        mock_gen.return_value = [
            {"locator": "text=Submit", "type": "text", "priority": 6},
            {"locator": "#submit-btn", "type": "id", "priority": 1},
        ]
        mock_validate.return_value = {
            "validated": True, "valid": True, "unique": True,
            "correct_element": True, "count": 1,
        }

        page = MagicMock()
        result = await extract_and_validate_locators(page, "Button", {"x": 10, "y": 20})

        # ID (priority=1) should beat text (priority=6)
        assert result["best_locator"] == "#submit-btn"
