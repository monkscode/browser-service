"""
Task 9 (analysis doc A4/E3) — scoped, layout-safe collection locators.

``_find_collection_by_text_traversal`` used to (1) accept any ancestor whose
class matched ``/row|tr-group|item|record|entry/i`` as a substring — so
"g**row**" in ``flex-grow-1`` made a flexbox utility div a "row container"
(6 confirmed wrong-collection fires in production logs); (2) call any >1
same-tag siblings a collection; and (3) emit a page-global ``.{class}``
selector, preferring the class even when the row was a structural
``<li>``/``<tr>`` — so Bootstrap's grid classes (``.row``, ``.col-xs-6``)
shipped as "collections" counting page chrome as data.

Pinned here:

  1. Semantic class matching is per-segment (``order-row`` yes,
     ``flex-grow-1`` no), with framework layout classes blocklisted as the
     emitted class.
  2. Collections require siblings sharing the row's meaningful class set,
     not just its tag.
  3. Emitted locators are scoped to the row's own parent
     (``.products > div.product-line``, ``ol > li``, ``tbody > tr``) —
     never a bare page-global class.
  4. The beacon prefers an exact text match and understands browser-use's
     truncated expected_text ("A Light in the ..." — literal ellipsis).

Same conventions as test_form_control_semantics.py: real headless Chromium
against a static fixture in ``locator_fixtures/``, marked integration.
"""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.handlers.collection import (
    _find_collection_by_text_traversal,
)

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent / "locator_fixtures"


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page_obj = await ctx.new_page()
        await page_obj.goto((FIXTURES_DIR / "collection_scoping.html").resolve().as_uri())
        try:
            yield page_obj
        finally:
            await browser.close()


@pytest.fixture
async def truncated_page():
    """books.toscrape's real shape — full title in the attribute, truncated
    text node. The collection_scoping fixture renders titles in full, which is
    why its tests passed while the live page failed."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page_obj = await ctx.new_page()
        await page_obj.goto(
            (FIXTURES_DIR / "collection_truncated_render.html").resolve().as_uri()
        )
        try:
            yield page_obj
        finally:
            await browser.close()


async def _matched_texts(page, locator: str) -> list:
    loc = page.locator(locator)
    return [((await loc.nth(i).text_content()) or "").strip() for i in range(await loc.count())]


async def test_bootstrap_rows_scoped_to_container(page):
    """The doc's example: a page-global .row counts header, filter bar and
    footer as products (6). The collection is the 3 product rows only."""
    result = await _find_collection_by_text_traversal(page, "Product A")
    assert result is not None
    assert result["locator"] != ".row"
    assert result["count"] == 3
    texts = await _matched_texts(page, result["locator"])
    assert all("Product" in t for t in texts)


async def test_grid_column_li_scoped_structurally(page):
    """books.toscrape shape (the current log's 22 fires): all classes are
    grid utilities, so the structural ol > li scope must win — and the
    decoy .col-xs-6 outside the list must not be counted."""
    result = await _find_collection_by_text_traversal(page, "A Light in the Attic")
    assert result is not None
    assert result["locator"] != ".col-xs-6"
    assert result["count"] == 4
    texts = await _matched_texts(page, result["locator"])
    assert not any("sidebar promo" in t for t in texts)


async def test_truncated_beacon_prefix_match(page):
    """browser-use truncates expected_text with a literal ellipsis — the
    real traffic shape ('A Light in the ...'). The beacon must still find
    the collection."""
    result = await _find_collection_by_text_traversal(page, "A Light in the ...")
    assert result is not None
    assert result["count"] == 4


async def test_full_title_resolves_when_the_page_renders_it_truncated(truncated_page):
    """The q10 production failure. The page shows 'A Light in the ...' and
    carries the full title only in the title attribute, so neither the exact
    nor the substring lookup matches what the agent reported. Before the
    prefix retry this returned None and the chain answered with
    [title="A Light in the Attic"] — one book, asserted against twenty."""
    result = await _find_collection_by_text_traversal(
        truncated_page, "A Light in the Attic"
    )
    assert result is not None
    assert result["locator"] == "ol > li"
    assert result["count"] == 4


async def test_truncated_form_still_resolves_on_the_same_page(truncated_page):
    """The form that already passed in production must not regress."""
    result = await _find_collection_by_text_traversal(
        truncated_page, "A Light in the ..."
    )
    assert result is not None
    assert result["count"] == 4


async def test_prefix_retry_does_not_invent_a_collection(truncated_page):
    """A beacon that matches nothing on the page must still fail closed —
    the retry may only rescue text that is really there."""
    assert await _find_collection_by_text_traversal(
        truncated_page, "Some Book That Is Not Here"
    ) is None


async def test_flex_grow_is_not_a_row_container(page):
    """The production wrong-collection: 'grow' contains 'row' as a
    substring. flex-grow-1 divs are not a collection — the traversal must
    come up empty so the pipeline falls through."""
    result = await _find_collection_by_text_traversal(page, "Test summary")
    assert result is None


async def test_lone_classed_banner_is_not_a_collection(page):
    """Class-set similarity: alert-row has 'row' as a real segment, but no
    sibling shares the class — one styled banner is not a collection."""
    result = await _find_collection_by_text_traversal(page, "Warning banner XYZ")
    assert result is None


async def test_table_rows_still_found_scoped(page):
    """Structural regression guard: plain table rows keep working, scoped
    to their tbody."""
    result = await _find_collection_by_text_traversal(page, "Alice Smith")
    assert result is not None
    assert result["count"] == 2
    texts = await _matched_texts(page, result["locator"])
    assert any("Alice" in t for t in texts) and any("Bob" in t for t in texts)
