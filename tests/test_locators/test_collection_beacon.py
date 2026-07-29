"""The collection beacon: resolving expected_text that is not a text node.

``_find_collection_by_text_traversal`` starts from a beacon — the element
carrying expected_text — and walks up to the repeating container. books.toscrape
renders ``<a title="A Light in the Attic">A Light in the ...</a>``: the full
title lives in an attribute and the visible text node is truncated. When
browser-use reports the truncated form the beacon matches and the run passes;
when it reports the full title neither the exact nor the substring lookup
matches, the collection path gives up, and the chain answers a 20-item request
with a one-item locator.

Bench q10, same query, three repeats: rep1 and rep3 produced ``ol > li`` (20
matches, passed); rep2 produced ``[title="A Light in the Attic"]`` (1 match,
``'1 == 20' should be true``). In the captured browser-service log the beacon
lookup failed 28 of the 30 times the collection path was entered.
"""

import pytest

from browser_service.locators.handlers.collection import (
    _beacon_prefixes,
    _find_collection_by_text_traversal,
)

BOOKS_FULL = "A Light in the Attic"
BOOKS_RENDERED = "A Light in the ..."
GRID_ROW = "Cierra Vega 39 cierra@example.com 10000 Insurance"


class _Loc:
    def __init__(self, count: int, row_info):
        self._count = count
        self._row_info = row_info

    @property
    def first(self):
        return self

    async def count(self):
        return self._count

    async def evaluate(self, _js):
        return self._row_info


class _NeedlePage:
    """Beacon lookups resolve only for needles the page actually contains.

    ``resolvable`` is the text the DOM really carries, so an exact
    ``text="..."`` query matches on equality and a bare ``text=...`` query
    matches on containment — Playwright's two modes.
    """

    def __init__(self, resolvable: list[str], row_info: dict | None, validate_count: int):
        self.resolvable = resolvable
        self.row_info = row_info
        self.validate_count = validate_count
        self.tried: list[str] = []

    def locator(self, selector: str):
        if not selector.startswith("text="):
            return _Loc(self.validate_count, None)
        needle = selector[len("text="):]
        if needle.startswith('"') and needle.endswith('"'):
            needle = needle[1:-1]
            self.tried.append(needle)
            hit = needle in self.resolvable
        else:
            self.tried.append(needle)
            hit = any(needle.lower() in r.lower() for r in self.resolvable)
        return _Loc(1 if hit else 0, self.row_info)


BOOK_ROW_INFO = {
    "tag": "li",
    "className": None,
    "parentAnchor": "ol",
    "role": "",
    "siblingCount": 20,
}


class TestBeaconPrefixes:
    def test_drops_one_word_at_a_time_longest_first(self):
        assert _beacon_prefixes(BOOKS_FULL)[0] == "A Light in the"

    def test_excludes_the_full_text(self):
        assert BOOKS_FULL not in _beacon_prefixes(BOOKS_FULL)

    def test_stops_before_a_single_word(self):
        # One word is not specific enough to anchor a collection, and the only
        # corpus case it would unlock ('Cierra') is blocked further down the
        # traversal regardless: every ReactTable class matches the
        # isLayoutClass `^[a-z]{1,2}-` rule, so the row container never carries
        # a meaningful class. Verified on a live demoqa-shaped DOM.
        assert all(len(p.split()) >= 2 for p in _beacon_prefixes(GRID_ROW))
        assert "Cierra" not in _beacon_prefixes(GRID_ROW)

    def test_stops_on_short_prefixes(self):
        assert all(len(p) >= 8 for p in _beacon_prefixes(BOOKS_FULL))

    def test_single_word_text_has_no_prefixes(self):
        assert _beacon_prefixes("Nobody") == []

    def test_empty_text_has_no_prefixes(self):
        assert _beacon_prefixes("") == []

    def test_collapses_runs_of_whitespace(self):
        assert _beacon_prefixes("A  Light\tin   the Attic")[0] == "A Light in the"


class TestBeaconRetry:
    @pytest.mark.asyncio
    async def test_full_title_is_rescued_when_the_page_renders_it_truncated(self):
        page = _NeedlePage([BOOKS_RENDERED], BOOK_ROW_INFO, 20)
        result = await _find_collection_by_text_traversal(page, BOOKS_FULL)
        assert result is not None, "the production q10 failure must now resolve"
        assert result["locator"] == "ol > li"
        assert result["count"] == 20

    @pytest.mark.asyncio
    async def test_the_longest_matching_prefix_wins(self):
        page = _NeedlePage([BOOKS_RENDERED], BOOK_ROW_INFO, 20)
        await _find_collection_by_text_traversal(page, BOOKS_FULL)
        assert page.tried[-1] == "A Light in the"

    @pytest.mark.asyncio
    async def test_the_reported_text_is_still_tried_first(self):
        page = _NeedlePage([BOOKS_RENDERED], BOOK_ROW_INFO, 20)
        await _find_collection_by_text_traversal(page, BOOKS_FULL)
        assert page.tried[0] == BOOKS_FULL

    @pytest.mark.asyncio
    async def test_an_exact_match_never_reaches_the_prefix_ladder(self):
        # The retry is strictly additive to the failure path: a beacon that
        # resolves today must resolve identically, with no extra lookups.
        page = _NeedlePage([BOOKS_FULL], BOOK_ROW_INFO, 20)
        result = await _find_collection_by_text_traversal(page, BOOKS_FULL)
        assert result["count"] == 20
        assert page.tried == [BOOKS_FULL]

    @pytest.mark.asyncio
    async def test_the_truncated_form_still_works_unchanged(self):
        page = _NeedlePage([BOOKS_RENDERED], BOOK_ROW_INFO, 20)
        result = await _find_collection_by_text_traversal(page, BOOKS_RENDERED)
        assert result["locator"] == "ol > li"

    @pytest.mark.asyncio
    async def test_a_beacon_matching_no_page_text_still_returns_none(self):
        page = _NeedlePage(["totally unrelated content"], BOOK_ROW_INFO, 20)
        assert await _find_collection_by_text_traversal(page, BOOKS_FULL) is None

    @pytest.mark.asyncio
    async def test_no_row_container_still_returns_none_after_a_prefix_hit(self):
        # Resolving a beacon is not the same as finding a collection. A
        # rescued beacon that lands outside any repeating container must fail
        # closed rather than invent a locator.
        page = _NeedlePage([BOOKS_RENDERED], None, 0)
        assert await _find_collection_by_text_traversal(page, BOOKS_FULL) is None
