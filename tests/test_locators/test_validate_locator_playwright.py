"""Unit tests for validate_locator_playwright's count/visibility/coord-match flow.

Driven by a page stub, no browser: a unique locator whose element's bounding
box contains the expected coordinates validates as correct; a locator matching
nothing reports count 0.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from browser_service.locators.validation import validate_locator_playwright


class _ValPage:
    def __init__(self, count, element_info=None, visible=True):
        self._count = count
        self._info = element_info
        self._visible = visible

    def locator(self, _selector):
        node = MagicMock()
        node.count = AsyncMock(return_value=self._count)
        first = MagicMock()
        first.evaluate = AsyncMock(return_value=self._info)
        first.is_visible = AsyncMock(return_value=self._visible)
        node.first = first
        return node


_ELEMENT_INFO = {
    "tag": "button",
    "id": "submit",
    "className": "btn",
    "text": "Go",
    "visible": True,
    "boundingBox": {"x": 100, "y": 200, "width": 50, "height": 20},
}


class TestValidateLocatorPlaywright:
    @pytest.mark.asyncio
    async def test_unique_locator_with_coords_inside_bbox_is_correct(self):
        page = _ValPage(count=1, element_info=_ELEMENT_INFO)
        result = await validate_locator_playwright(
            page, "id=submit", expected_coords={"x": 120, "y": 210}
        )
        assert result["valid"] is True
        assert result["unique"] is True
        assert result["count"] == 1
        assert result["correct_element"] is True
        assert result["bounding_box"] == _ELEMENT_INFO["boundingBox"]

    @pytest.mark.asyncio
    async def test_coords_outside_bbox_flags_wrong_element(self):
        page = _ValPage(count=1, element_info=_ELEMENT_INFO)
        result = await validate_locator_playwright(
            page, "id=submit", expected_coords={"x": 999, "y": 999}
        )
        assert result["valid"] is True  # still unique
        assert result["correct_element"] is False  # but not at the expected point

    @pytest.mark.asyncio
    async def test_no_match_reports_count_zero(self):
        page = _ValPage(count=0)
        result = await validate_locator_playwright(page, "id=missing")
        assert result["valid"] is False
        assert result["count"] == 0
