"""A text-first locator may never answer a collection request.

``_find_element_by_expected_text`` matches ONE element by its own text and
returns ``count: 1, unique: True``. It therefore cannot represent a collection
— it pins a single item. When the collection path gives up, the fallback chain
used to reach it anyway: bench q10 ("get the titles of all books on the first
page, and verify there are 20 books") came back as
``[title="A Light in the Attic"]``, one match, asserted against twenty.

Corpus evidence (element_approach_metrics joined to test_status across every
baseline CSV — 193 collection elements over 176 workflow ids):

    strategy        n    pass rate
    element_data   67       94.0%
    accessibility  90       93.3%
    collection     26       80.8%
    text_first     10       40.0%

Skipping text-first strands nothing: semantic, the accessibility API and the
21 coordinate strategies all still run below it, and only after all of those
does the function report found: false.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestTextFirstIsForbiddenOnCollections:
    COLLECTION_DESC = "all book titles on the first page"
    SINGLE_DESC = "the search button in the header"

    async def _run(
        self,
        *,
        is_collection,
        description,
        expected_text="A Light in the Attic",
        row_anchor_text=None,
    ):
        from browser_service.locators import smart_locator

        with (
            patch.object(
                smart_locator, "_find_collection_by_text_traversal", AsyncMock(return_value=None)
            ),
            patch.object(
                smart_locator, "_find_element_by_expected_text", AsyncMock(return_value=None)
            ) as text_first,
            patch.object(
                smart_locator, "_find_element_by_description", AsyncMock(return_value=None)
            ) as semantic,
            patch.object(
                smart_locator, "_find_element_via_accessibility", AsyncMock(return_value=None)
            ),
        ):
            page = MagicMock()
            page.evaluate = AsyncMock(return_value=None)
            await smart_locator.find_unique_locator_at_coordinates(
                page=page,
                x=100,
                y=200,
                element_id="elem_1",
                element_description=description,
                expected_text=expected_text,
                element_data=None,
                search_context=page,
                is_collection=is_collection,
                row_anchor_text=row_anchor_text,
            )
            return text_first, semantic

    @pytest.mark.asyncio
    async def test_explicit_collection_flag_skips_text_first(self):
        text_first, _ = await self._run(is_collection=True, description=self.COLLECTION_DESC)
        assert text_first.await_count == 0

    @pytest.mark.asyncio
    async def test_keyword_detected_collection_skips_text_first(self):
        # The description alone is enough — _should_treat_as_collection's
        # keyword fallback classifies this without the explicit flag.
        text_first, _ = await self._run(is_collection=None, description=self.COLLECTION_DESC)
        assert text_first.await_count == 0

    @pytest.mark.asyncio
    async def test_collection_request_still_reaches_semantic(self):
        _, semantic = await self._run(is_collection=True, description=self.COLLECTION_DESC)
        assert semantic.await_count == 1

    @pytest.mark.asyncio
    async def test_single_element_request_still_uses_text_first(self):
        text_first, _ = await self._run(is_collection=False, description=self.SINGLE_DESC)
        assert text_first.await_count == 1

    @pytest.mark.asyncio
    async def test_unflagged_single_element_request_still_uses_text_first(self):
        text_first, _ = await self._run(is_collection=None, description=self.SINGLE_DESC)
        assert text_first.await_count == 1

    @pytest.mark.asyncio
    async def test_row_anchor_keeps_text_first_despite_the_collection_flag(self):
        # Standing decision (2026-07-16, owner-approved): a row_anchor_text
        # names ONE row and wins over an explicit is_collection=True. That
        # request wants a single element, so text-first stays available to it.
        text_first, _ = await self._run(
            is_collection=True,
            description="the edit link in the row containing Smith",
            expected_text="Smith",
            row_anchor_text="Smith",
        )
        assert text_first.await_count == 1
