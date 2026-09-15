"""
E2a in the cascade, on real Chromium (2026-09-15 unseen-site probe).

Each shape is pinned twice: action=None reproduces the probe defect (the
fixture is faithful), the step's action gives the fixed answer.
"""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.locators import find_unique_locator_at_coordinates

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent / "locator_fixtures"
SEARCH_TEXT = "Search for Products, Brands and More"

ELEMENT_DATA_JS = """(el) => ({
    tagName: el.tagName.toLowerCase(),
    id: el.id || "",
    className: (typeof el.className === 'string') ? el.className : "",
    role: el.getAttribute('role') || "",
    type: el.getAttribute('type') || "",
    name: el.getAttribute('name') || "",
    ariaLabel: el.getAttribute('aria-label') || "",
    placeholder: el.getAttribute('placeholder') || "",
    title: el.getAttribute('title') || "",
    parentClassName: (el.parentElement && typeof el.parentElement.className === 'string')
        ? el.parentElement.className : "",
    textContent: (el.textContent || "").trim().slice(0, 80),
})"""


def _file_url(name: str) -> str:
    return (FIXTURES_DIR / name).resolve().as_uri()


async def _center(page, selector: str):
    box = await page.locator(selector).first.bounding_box()
    assert box, selector
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


async def _resolved_tag(page, locator: str) -> str:
    return await page.locator(locator).evaluate("e => e.tagName.toLowerCase()")


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page_obj = await ctx.new_page()
        try:
            yield page_obj
        finally:
            await browser.close()


# ---- flipkart u02: text-first, no element_data (agent gave element_index=None)


async def _find_search_box(page, action):
    await page.goto(_file_url("input_aria_label_twin.html"), wait_until="domcontentloaded")
    x, y = await _center(page, ".search-wrap input")
    return await find_unique_locator_at_coordinates(
        page=page,
        x=x,
        y=y,
        element_id="elem_1",
        element_description="the search input box in the header",
        expected_text=SEARCH_TEXT,
        element_data=None,
        action=action,
    )


async def test_flipkart_shape_today_resolves_to_the_wrapper(page):
    result = await _find_search_box(page, action=None)
    assert result["found"] is True
    assert await _resolved_tag(page, result["best_locator"]) == "div"


async def test_flipkart_shape_input_action_resolves_to_the_input(page):
    result = await _find_search_box(page, action="input")
    assert result["found"] is True
    assert await _resolved_tag(page, result["best_locator"]) == "input"


# ---- todomvc u10: STEP 0 with the INPUT's element_data, read step


async def _find_todo_title(page, action):
    await page.goto(_file_url("todo_list_next_to_input.html"), wait_until="domcontentloaded")
    data = await page.locator("input.new-todo").evaluate(ELEMENT_DATA_JS)
    x, y = await _center(page, "label[data-testid='todo-title']")
    return await find_unique_locator_at_coordinates(
        page=page,
        x=x,
        y=y,
        element_id="elem_2",
        element_description="the todo title text",
        expected_text="buy milk",
        element_data=data,
        action=action,
    )


async def test_todo_shape_today_reads_the_empty_input(page):
    result = await _find_todo_title(page, action=None)
    assert result["best_locator"] == '[placeholder="What needs to be done?"]'


async def test_todo_shape_get_text_reads_the_list_item(page):
    result = await _find_todo_title(page, action="get_text")
    assert result["found"] is True
    assert (await page.locator(result["best_locator"]).inner_text()).strip() == "buy milk"
