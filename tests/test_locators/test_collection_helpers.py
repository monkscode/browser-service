"""Unit tests for the pure/deterministic collection helpers.

Covers the classification and selector-derivation logic in
browser_service.locators.handlers.collection that runs before any live
page work: _is_collection_element, _extract_collection_class, and the
tag/class strategies of _find_collection_locator (driven by a counting
page stub, no browser).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from browser_service.locators.handlers.collection import (
    _extract_collection_class,
    _find_collection_by_text_traversal,
    _find_collection_locator,
    _is_collection_element,
    find_locator,
)


class _CountPage:
    """A page stub whose locator(sel).count() returns a per-selector count."""

    def __init__(self, counts: dict):
        self._counts = counts

    def locator(self, selector):
        m = MagicMock()
        m.count = AsyncMock(return_value=self._counts.get(selector, 0))
        return m


class TestIsCollectionElement:
    def test_description_keyword_detects(self):
        assert _is_collection_element({}, "get all visible rows") is True

    def test_standard_collection_tag_detects(self):
        assert _is_collection_element({"tagName": "TR"}, "a cell") is True

    def test_class_pattern_detects(self):
        assert _is_collection_element({"className": "product record"}, "x") is True

    def test_nav_prefixed_class_is_not_a_collection(self):
        # nav-item must not read as the 'item' collection pattern
        assert _is_collection_element({"className": "nav-item"}, "menu link") is False

    def test_plain_element_is_not_a_collection(self):
        assert _is_collection_element({"tagName": "button", "className": "btn"}, "submit") is False


class TestExtractCollectionClass:
    def test_semantic_pattern_class_wins(self):
        assert _extract_collection_class({"className": "product-row highlighted"}) == "product-row"

    def test_empty_class_returns_none(self):
        assert _extract_collection_class({"className": ""}) is None

    def test_utility_only_classes_return_none(self):
        # short + letter-number + single-letter-hyphen utilities are all skipped
        assert _extract_collection_class({"className": "mt-4 px-12 d-flex p-2"}) is None

    def test_meaningful_component_class_is_returned(self):
        # long, non-utility, no collection keyword → component-like, returned
        assert _extract_collection_class({"className": "productContainer"}) == "productContainer"


class TestFindCollectionLocator:
    @pytest.mark.asyncio
    async def test_table_row_strategy(self):
        page = _CountPage({"tbody tr": 5})
        result = await _find_collection_locator(page, {"tagName": "tr"}, "row")
        assert result == "tbody tr"

    @pytest.mark.asyncio
    async def test_list_item_strategy(self):
        page = _CountPage({"ul li": 3})
        result = await _find_collection_locator(page, {"tagName": "li"}, "item")
        assert result == "ul li"

    @pytest.mark.asyncio
    async def test_class_based_strategy(self):
        page = _CountPage({".product-row": 8})
        result = await _find_collection_locator(page, {"tagName": "div"}, "product-row")
        assert result == ".product-row"

    @pytest.mark.asyncio
    async def test_no_multiple_matches_returns_none(self):
        page = _CountPage({})  # every selector counts 0
        result = await _find_collection_locator(page, {"tagName": "div"}, "row")
        assert result is None


class _TravLocator:
    def __init__(self, count, row_info=None):
        self._count = count
        self._row_info = row_info

    @property
    def first(self):
        return self

    async def count(self):
        return self._count

    async def evaluate(self, _js):
        return self._row_info


class _TraversalPage:
    """page.locator(text=…) is the beacon; any other selector is the validation query."""

    def __init__(self, beacon_count, row_info, validate_count):
        self._beacon = beacon_count
        self._row_info = row_info
        self._validate = validate_count

    def locator(self, selector):
        if selector.startswith("text="):
            return _TravLocator(self._beacon, self._row_info)
        return _TravLocator(self._validate)


class TestFindCollectionByTextTraversal:
    @pytest.mark.asyncio
    async def test_class_anchored_locator_built_and_validated(self):
        row_info = {
            "tag": "tr",
            "className": "product-row",
            "parentAnchor": "tbody",
            "role": "",
            "siblingCount": 5,
        }
        page = _TraversalPage(beacon_count=1, row_info=row_info, validate_count=5)
        result = await _find_collection_by_text_traversal(page, "Cierra")
        assert result["locator"] == "tbody > tr.product-row"
        assert result["count"] == 5
        assert result["row_class"] == "product-row"

    @pytest.mark.asyncio
    async def test_bare_tr_without_class_or_anchor(self):
        row_info = {"tag": "tr", "className": "", "parentAnchor": "", "role": "", "siblingCount": 4}
        page = _TraversalPage(beacon_count=1, row_info=row_info, validate_count=4)
        result = await _find_collection_by_text_traversal(page, "Cierra")
        assert result["locator"] == "tbody tr"

    @pytest.mark.asyncio
    async def test_role_row_without_class(self):
        row_info = {
            "tag": "div",
            "className": "",
            "parentAnchor": "",
            "role": "row",
            "siblingCount": 6,
        }
        page = _TraversalPage(beacon_count=1, row_info=row_info, validate_count=6)
        result = await _find_collection_by_text_traversal(page, "Cierra")
        assert result["locator"] == '[role="row"]'

    @pytest.mark.asyncio
    async def test_beacon_not_found_returns_none(self):
        page = _TraversalPage(beacon_count=0, row_info=None, validate_count=0)
        assert await _find_collection_by_text_traversal(page, "Nobody") is None

    @pytest.mark.asyncio
    async def test_short_text_is_skipped(self):
        assert await _find_collection_by_text_traversal(_TraversalPage(1, {}, 1), "a") is None


class TestFindLocatorEntryPoint:
    @pytest.mark.asyncio
    async def test_text_traversal_result_is_built_into_collection_result(self):
        """find_locator's primary path returns a built collection result when the
        text-traversal beacon resolves a multi-row locator."""
        type_info = SimpleNamespace(confidence=0.9, signals=["collection"], framework="native")
        row_info = {
            "tag": "tr",
            "className": "product-row",
            "parentAnchor": "tbody",
            "role": "",
            "siblingCount": 5,
        }
        search_context = _TraversalPage(beacon_count=1, row_info=row_info, validate_count=5)

        result = await find_locator(
            page=None,
            element_data={"tagName": "tr", "className": "product-row"},
            type_info=type_info,
            element_id="elem_1",
            element_description="first product row",
            expected_text="Cierra",
            search_context=search_context,
            iframe_context=None,
            confirmed_coords=None,
        )

        assert result is not None
        assert result["best_locator"] == "tbody > tr.product-row"
        assert result["count"] == 5
