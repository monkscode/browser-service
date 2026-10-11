"""bs #33 — a collection request with no expected text finds its collection from the INDEXED element.

books.toscrape.com, bench q10 ("get the titles of all books ... and verify there are 20 books"): the planner's second
element "all book product items" arrived with is_collection=True and NO expected text, the agent indexed a link
inside the first card, its candidate `article.product_pod` was rejected (an <article>, not the indexed <a>), the class
route found no class, and the generic xpath shortener shipped `//li[1]/article/div[1]/a` — the test counted 1 of 20
(run 1139a69a-c21b-4233-8ad7-6a64832b5c8f, 2026-10-11). With a text hint the handler's walk returned `ol > li` in 13
of 13 runs. The walk now also starts from the indexed element.

Real headless Chromium against static fixtures in ``locator_fixtures/``, marked integration (the fixture lane).
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.classifier import ElementTypeInfo
from browser_service.locators.handlers.collection import find_locator
from browser_service.locators.smart_locator import _generate_locators_from_element_data

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent / "locator_fixtures"
CARD = "html/body/div/div/div/div/section/div[2]/ol/li[1]"
IMAGE_LINK = f"{CARD}/article/div[1]/a"
TITLE_LINK = f"{CARD}/article/h3/a"
DESCRIPTION = "all book product items in the main product grid"


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page_obj = await ctx.new_page()
        await page_obj.goto((FIXTURES_DIR / "collection_card_links.html").resolve().as_uri())
        try:
            yield page_obj
        finally:
            await browser.close()


@pytest.mark.parametrize(
    "indexed_xpath", [IMAGE_LINK, TITLE_LINK], ids=["image-link", "title-link"]
)
async def test_no_text_link_inside_card_ships_the_item_list(page, indexed_xpath):
    """The q10 failure: no expected text, an indexed link inside card 1 -> the item list, all 4 cards."""
    result = await _generate_locators_from_element_data(
        page,
        {"tagName": "a", "className": "", "xpath": indexed_xpath},
        "elem_2",
        DESCRIPTION,
        None,
        page=page,
    )
    assert result is not None
    assert result["best_locator"] == "ol > li"
    assert result["count"] == 4
    assert result["element_type"] == "collection"
    assert result["unique"] is False
    assert await page.locator(result["best_locator"]).count() == 4


async def test_no_text_indexed_item_itself_ships_the_item_list(page):
    """The agent indexed the card's own <li> (no class, no text): the walk includes the start element."""
    result = await _generate_locators_from_element_data(
        page,
        {"tagName": "li", "className": "", "xpath": CARD},
        "elem_2",
        DESCRIPTION,
        None,
        page=page,
    )
    assert result is not None
    assert result["best_locator"] == "ol > li"
    assert result["count"] == 4


async def test_one_item_list_keeps_todays_answer():
    """A list that holds ONE item (a filtered result): no repeated sibling, so the walk finds nothing and the
    request keeps today's one-match answer — never a 'collection' of one built by the new walk."""
    html = (FIXTURES_DIR / "collection_card_links.html").read_text(encoding="utf-8")
    start, end = html.index("<li class=", html.index('<ol class="row">')), html.index("</ol>")
    first_item = html[start : html.index("</li>", start) + len("</li>")]
    one_item_html = html[:start] + first_item + "\n" + html[end:]
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            one = await browser.new_page()
            await one.set_content(one_item_html)
            assert await one.locator("ol > li").count() == 1
            result = await _generate_locators_from_element_data(
                one,
                {"tagName": "a", "className": "", "xpath": IMAGE_LINK},
                "elem_2",
                DESCRIPTION,
                None,
                page=one,
            )
            assert result is not None
            assert result.get("element_type") != "collection"
            assert result["best_locator"] != "ol > li"
            assert await one.locator(result["best_locator"]).count() == 1
        finally:
            await browser.close()


async def test_text_hint_never_reaches_the_indexed_walk(page):
    """With expected text the handler runs exactly as before: its text walk answers, the new walk is not called."""
    type_info = ElementTypeInfo(
        primary_type="collection", framework="list-item", confidence="low", signals=[]
    )
    with patch(
        "browser_service.locators.handlers.collection._find_collection_from_indexed_element",
        new_callable=AsyncMock,
    ) as walk:
        result = await find_locator(
            page=page,
            element_data={"tagName": "a", "className": "", "xpath": IMAGE_LINK},
            type_info=type_info,
            element_id="elem_2",
            element_description=DESCRIPTION,
            expected_text="A Light in the Attic",
            search_context=page,
            iframe_context=None,
            confirmed_coords=None,
        )
    walk.assert_not_awaited()
    assert result is not None
    assert result["best_locator"] == "ol > li"
    assert result["element_info"]["source"] == "text_traversal"


async def test_walk_runs_inside_a_frame():
    """Inside an iframe the walk runs in the frame's context and returns the locator WITHOUT a frame prefix."""
    type_info = ElementTypeInfo(
        primary_type="collection", framework="list-item", confidence="low", signals=[]
    )
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            outer = await browser.new_page()
            await outer.goto((FIXTURES_DIR / "collection_card_links_frame.html").resolve().as_uri())
            frame = outer.frame_locator("#books")
            await frame.locator("ol > li").first.wait_for()
            result = await find_locator(
                page=frame,
                element_data={"tagName": "a", "className": "", "xpath": IMAGE_LINK},
                type_info=type_info,
                element_id="elem_2",
                element_description=DESCRIPTION,
                expected_text=None,
                search_context=frame,
                iframe_context="#books",
                confirmed_coords=None,
            )
            assert result is not None
            assert result["best_locator"] == "ol > li"
            assert result["count"] == 4
            assert ">>>" not in result["best_locator"]
        finally:
            await browser.close()
