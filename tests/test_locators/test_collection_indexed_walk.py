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

from browser_service.agent.actions import _refuse_one_element_for_group
from browser_service.locators.classifier import ElementTypeInfo, classify_element_type
from browser_service.locators.handlers.collection import find_locator
from browser_service.locators.smart_locator import (
    _generate_locators_from_element_data,
    find_unique_locator_at_coordinates,
)

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent / "locator_fixtures"
CARD = "html/body/div/div/div/div/section/div[2]/ol/li[1]"
IMAGE_LINK = f"{CARD}/article/div[1]/a"
TITLE_LINK = f"{CARD}/article/h3/a"
DESCRIPTION = "all book product items in the main product grid"
# The wording of the real failing call (run b9a6f92f-4938-4689-a7ae-8148697d3442): no collection word in it.
B9A6_DESCRIPTION = "book item containers in the main product grid"


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
        is_collection=True,
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
        is_collection=True,
    )
    assert result is not None
    assert result["best_locator"] == "ol > li"
    assert result["count"] == 4


def _one_item_html() -> str:
    html = (FIXTURES_DIR / "collection_card_links.html").read_text(encoding="utf-8")
    start, end = html.index("<li class=", html.index('<ol class="row">')), html.index("</ol>")
    first_item = html[start : html.index("</li>", start) + len("</li>")]
    return html[:start] + first_item + "\n" + html[end:]


async def test_one_item_list_is_not_found_at_step0():
    """A list that holds ONE item (a filtered result): no repeated sibling, so the walk finds nothing.

    Owner decision 2026-11-19 (bs #86, "Not found"): an explicit is_collection=True ships a group or nothing, so
    STEP 0 now returns None here instead of today's one-element answer (the full cascade is refused by the
    actions.py gate - see test_one_item_cascade_answer_is_refused).
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            one = await browser.new_page()
            await one.set_content(_one_item_html())
            assert await one.locator("ol > li").count() == 1
            result = await _generate_locators_from_element_data(
                one,
                {"tagName": "a", "className": "", "xpath": IMAGE_LINK},
                "elem_2",
                DESCRIPTION,
                None,
                page=one,
                is_collection=True,
            )
            assert result is None
        finally:
            await browser.close()


async def test_one_item_cascade_answer_is_refused():
    """Full cascade on the one-item page: True + no text + the <a> ends in a plain one-element answer
    (found, unique, not a collection); the actions.py gate turns that answer into CollectionNotFound."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            one = await browser.new_page()
            await one.set_content(_one_item_html())
            box = await one.locator("ol > li article div a").first.bounding_box()
            result = await find_unique_locator_at_coordinates(
                page=one,
                x=box["x"] + box["width"] / 2,
                y=box["y"] + box["height"] / 2,
                element_id="elem_2",
                element_description=B9A6_DESCRIPTION,
                expected_text=None,
                element_data={"tagName": "a", "className": "", "xpath": IMAGE_LINK},
                is_collection=True,
            )
            assert result["found"] is True
            assert result["unique"] is True
            assert result.get("element_type") != "collection"
            refused = _refuse_one_element_for_group(result, True, None, B9A6_DESCRIPTION)
            assert refused["found"] is False
            assert refused["error_type"] == "CollectionNotFound"
            assert refused["refused_locator"] == result["best_locator"]
        finally:
            await browser.close()


async def test_description_without_collection_words_classifies_unknown():
    """b9a6f92f's description carries no collection word: the classifier alone says 'unknown'."""
    type_info = classify_element_type(
        {"tagName": "a", "className": "", "xpath": IMAGE_LINK}, B9A6_DESCRIPTION
    )
    assert type_info.primary_type == "unknown"


async def test_flag_true_unknown_verdict_routes_to_the_item_list(page, caplog):
    """The b9a6f92f failure: True + an `unknown` verdict + no text -> `ol > li` (4 cards), not the one <a>."""
    with caplog.at_level("INFO"):
        result = await _generate_locators_from_element_data(
            page,
            {"tagName": "a", "className": "", "xpath": IMAGE_LINK},
            "elem_2",
            B9A6_DESCRIPTION,
            None,
            page=page,
            is_collection=True,
        )
    assert result is not None
    assert result["best_locator"] == "ol > li"
    assert result["count"] == 4
    assert result["element_type"] == "collection"
    assert result["unique"] is False
    assert "flag-true-routes-collection" in result["classifier_signals"]
    assert "(signal: flag-true-routes-collection)" in caplog.text


async def test_flag_true_beats_a_specialised_vision_hint(page, caplog):
    """The flag decides, heuristics only when it is absent: a vision hint of 'dropdown' on an `unknown` verdict
    does not keep the discovery route - the call still reaches the collection handler (and ships the item list).

    The classifier is patched to `unknown`: a lone hint is a +3 vote that makes the verdict 'dropdown' itself, so
    `unknown` beside a mapped hint only arises on a vote tie.
    """
    element_data = {"tagName": "a", "className": "", "xpath": IMAGE_LINK}
    tied = ElementTypeInfo(primary_type="unknown", framework="", confidence="low", signals=[])
    with (
        patch("browser_service.locators.smart_locator.classify_element_type", return_value=tied),
        caplog.at_level("INFO"),
    ):
        result = await _generate_locators_from_element_data(
            page,
            element_data,
            "elem_2",
            B9A6_DESCRIPTION,
            None,
            page=page,
            vision_type_hint="dropdown",
            is_collection=True,
        )
    assert "(signal: flag-true-routes-collection)" in caplog.text
    assert result is not None
    assert result["best_locator"] == "ol > li"
    assert result["element_type"] == "collection"


async def test_flag_absent_unknown_verdict_keeps_the_one_element_answer(page):
    """Control: the same call without the flag keeps today's one-element answer."""
    result = await _generate_locators_from_element_data(
        page,
        {"tagName": "a", "className": "", "xpath": IMAGE_LINK},
        "elem_2",
        B9A6_DESCRIPTION,
        None,
        page=page,
        is_collection=None,
    )
    assert result is not None
    assert result["best_locator"] != "ol > li"
    assert result.get("element_type") != "collection"
    assert await page.locator(result["best_locator"]).count() == 1


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
            is_collection=True,
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
                is_collection=True,
            )
            assert result is not None
            assert result["best_locator"] == "ol > li"
            assert result["count"] == 4
            assert ">>>" not in result["best_locator"]
        finally:
            await browser.close()


async def test_unflagged_single_link_keeps_todays_answer(page):
    """A one-element request ("the title link of the first book in the list of books") reaches the collection
    handler through the description words and the DOM probe, but the agent never flagged it as a collection:
    the walk must not run, or a Click on the answer would hit all 4 cards (strict mode)."""
    result = await _generate_locators_from_element_data(
        page,
        {"tagName": "a", "className": "", "xpath": TITLE_LINK},
        "elem_1",
        "the title link of the first book in the list of books",
        None,
        page=page,
    )
    assert result is not None
    assert result["best_locator"] != "ol > li"
    assert result.get("element_type") != "collection"
    assert await page.locator(result["best_locator"]).count() == 1
