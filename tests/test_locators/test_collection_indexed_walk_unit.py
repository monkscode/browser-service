"""bs #33 — unit tests (no browser) for the walk that starts from the indexed element.

`_find_collection_from_indexed_element` locates the indexed element by its stored xpath and walks up from it,
the element itself included; anything but exactly one match means no walk. `find_locator` uses it only when the
request has NO expected text and the class route found nothing.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from browser_service.locators.handlers.collection import (
    _find_collection_from_indexed_element,
    find_locator,
)

ROW_INFO = {"tag": "li", "className": None, "parentAnchor": "ol", "siblingCount": 20, "role": ""}


class _Start:
    """The locator for the indexed element: a count, and the walk's answer."""

    def __init__(self, count, row_info=None, raise_on_count=False):
        self._count = count
        self._row_info = row_info
        self._raise = raise_on_count
        self.evaluate_args = []

    async def count(self):
        if self._raise:
            raise RuntimeError("page closed")
        return self._count

    async def evaluate(self, _js, *args):
        self.evaluate_args.append(args)
        return self._row_info


class _Counted:
    def __init__(self, count):
        self._count = count

    async def count(self):
        return self._count


class _Page:
    def __init__(self, start, validate_count=0):
        self.start = start
        self.selectors = []
        self._validate = validate_count

    def locator(self, selector):
        self.selectors.append(selector)
        if selector.startswith("xpath="):
            return self.start
        return _Counted(self._validate)


class _CountingContext:
    """A search context whose `locator(sel).count()` returns a fixed number, or raises."""

    def __init__(self, count=1, raises=False):
        self._count = count
        self._raises = raises

    def locator(self, _selector):
        return self

    async def count(self):
        if self._raises:
            raise RuntimeError("page closed")
        return self._count


def _type_info():
    return SimpleNamespace(confidence="low", signals=["desc:collection(+1)"], framework="list-item")


class TestFindCollectionFromIndexedElement:
    @pytest.mark.asyncio
    async def test_no_xpath_means_no_walk(self):
        page = _Page(_Start(1, ROW_INFO), validate_count=20)
        assert await _find_collection_from_indexed_element(page, {"tagName": "a"}) is None
        assert page.selectors == []

    @pytest.mark.asyncio
    async def test_stored_xpath_is_made_absolute(self):
        page = _Page(_Start(1, ROW_INFO), validate_count=20)
        await _find_collection_from_indexed_element(page, {"xpath": "html/body/ol/li[1]/a"})
        assert page.selectors[0] == "xpath=/html/body/ol/li[1]/a"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("matches", [0, 2])
    async def test_anything_but_one_match_means_no_walk(self, matches):
        start = _Start(matches, ROW_INFO)
        page = _Page(start, validate_count=20)
        assert (
            await _find_collection_from_indexed_element(page, {"xpath": "html/body/ol/li[1]/a"})
            is None
        )
        assert start.evaluate_args == []

    @pytest.mark.asyncio
    async def test_an_error_means_no_walk_and_no_exception(self):
        page = _Page(_Start(1, ROW_INFO, raise_on_count=True), validate_count=20)
        assert (
            await _find_collection_from_indexed_element(page, {"xpath": "html/body/ol/li[1]/a"})
            is None
        )

    @pytest.mark.asyncio
    async def test_walk_includes_the_indexed_element_and_names_its_source(self):
        start = _Start(1, ROW_INFO)
        page = _Page(start, validate_count=20)
        result = await _find_collection_from_indexed_element(
            page, {"xpath": "html/body/ol/li[1]/a"}
        )
        assert result == {
            "locator": "ol > li",
            "count": 20,
            "row_class": None,
            "tag": "li",
            "source": "indexed_element_walk",
        }
        assert start.evaluate_args == [(True,)]


class TestFindLocatorNoTextFallback:
    @staticmethod
    async def _call(element_data, expected_text=None, search_context=None, is_collection=True):
        return await find_locator(
            page=None,
            element_data=element_data,
            type_info=_type_info(),
            element_id="elem_2",
            element_description="all book product items in the main product grid",
            expected_text=expected_text,
            search_context=search_context or object(),
            iframe_context=None,
            confirmed_coords=None,
            is_collection=is_collection,
        )

    @pytest.mark.asyncio
    async def test_no_text_and_no_class_returns_the_walked_collection(self):
        walked = {
            "locator": "ol > li",
            "count": 20,
            "row_class": None,
            "tag": "li",
            "source": "indexed_element_walk",
        }
        with patch(
            "browser_service.locators.handlers.collection._find_collection_from_indexed_element",
            new=AsyncMock(return_value=walked),
        ):
            result = await self._call(
                {"tagName": "a", "className": "", "xpath": "html/body/ol/li[1]/a"}
            )
        assert result["best_locator"] == "ol > li"
        assert result["count"] == 20
        assert result["element_type"] == "collection"
        assert result["unique"] is False
        assert result["element_info"]["source"] == "indexed_element_walk"

    @pytest.mark.asyncio
    async def test_a_walk_that_finds_one_item_returns_none(self):
        walked = {
            "locator": "ol > li",
            "count": 1,
            "row_class": None,
            "tag": "li",
            "source": "indexed_element_walk",
        }
        with patch(
            "browser_service.locators.handlers.collection._find_collection_from_indexed_element",
            new=AsyncMock(return_value=walked),
        ):
            assert await self._call({"tagName": "a", "className": "", "xpath": "x"}) is None

    @pytest.mark.asyncio
    async def test_a_class_without_a_many_match_locator_falls_back_to_the_walk(self):
        walked = {
            "locator": "ol > li",
            "count": 20,
            "row_class": None,
            "tag": "li",
            "source": "indexed_element_walk",
        }
        with (
            patch(
                "browser_service.locators.handlers.collection._find_collection_locator",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "browser_service.locators.handlers.collection._find_collection_from_indexed_element",
                new=AsyncMock(return_value=walked),
            ) as walk,
        ):
            result = await self._call({"tagName": "a", "className": "product-row", "xpath": "x"})
        walk.assert_awaited_once()
        assert result["best_locator"] == "ol > li"

    @pytest.mark.asyncio
    async def test_a_text_hint_never_reaches_the_walk(self):
        with (
            patch(
                "browser_service.locators.handlers.collection._find_collection_by_text_traversal",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "browser_service.locators.handlers.collection._find_collection_from_indexed_element",
                new=AsyncMock(),
            ) as walk,
        ):
            result = await self._call(
                {"tagName": "a", "className": "", "xpath": "x"}, expected_text="A Light"
            )
        walk.assert_not_awaited()
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("flag", [None, False])
    async def test_an_unflagged_request_never_reaches_the_walk(self, flag):
        with patch(
            "browser_service.locators.handlers.collection._find_collection_from_indexed_element",
            new=AsyncMock(),
        ) as walk:
            result = await self._call(
                {"tagName": "a", "className": "", "xpath": "x"}, is_collection=flag
            )
        walk.assert_not_awaited()
        assert result is None

    @pytest.mark.asyncio
    async def test_a_single_match_class_locator_falls_back_to_the_walk(self):
        walked = {
            "locator": "ol > li",
            "count": 20,
            "row_class": None,
            "tag": "li",
            "source": "indexed_element_walk",
        }
        with (
            patch(
                "browser_service.locators.handlers.collection._find_collection_locator",
                new=AsyncMock(return_value=".product-row"),
            ),
            patch(
                "browser_service.locators.handlers.collection._find_collection_from_indexed_element",
                new=AsyncMock(return_value=walked),
            ) as walk,
        ):
            result = await self._call(
                {"tagName": "a", "className": "product-row", "xpath": "x"},
                search_context=_CountingContext(1),
            )
        walk.assert_awaited_once()
        assert result["best_locator"] == "ol > li"

    @pytest.mark.asyncio
    async def test_a_count_error_never_walks(self):
        with (
            patch(
                "browser_service.locators.handlers.collection._find_collection_locator",
                new=AsyncMock(return_value=".product-row"),
            ),
            patch(
                "browser_service.locators.handlers.collection._find_collection_from_indexed_element",
                new=AsyncMock(),
            ) as walk,
        ):
            result = await self._call(
                {"tagName": "a", "className": "product-row", "xpath": "x"},
                search_context=_CountingContext(raises=True),
            )
        walk.assert_not_awaited()
        assert result is None
