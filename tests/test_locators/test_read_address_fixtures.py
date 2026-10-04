"""
#31 read-address rule on real Chromium (local HTML): the structural address of the element the agent pointed at — no displayed value, exactly one match, the same node — and the shapes R1–R4 of the spec.
"""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.read_target import (
    carries_value,
    resolve_read_address,
    structural_address,
)

pytestmark = pytest.mark.integration

PAGE = (Path(__file__).parent / "locator_fixtures" / "read_address_cases.html").resolve().as_uri()

XPATH_JS = """el => { const p = []; for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
  let i = 1; for (let s = n.previousElementSibling; s; s = s.previousElementSibling) if (s.tagName === n.tagName) i++;
  p.unshift(n.tagName.toLowerCase() + '[' + i + ']'); } return p.join('/'); }"""


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page_obj = await (
            await browser.new_context(viewport={"width": 1280, "height": 900})
        ).new_page()
        await page_obj.goto(PAGE, wait_until="domcontentloaded")
        try:
            yield page_obj
        finally:
            await browser.close()


async def _element_data(page, selector):
    return {"xpath": await page.locator(selector).evaluate(XPATH_JS)}


async def _center(page, selector):
    box = await page.locator(selector).bounding_box()
    return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


async def _assert_rule(page, verdict, target, observed):
    loc = verdict["locator"]
    assert not carries_value(loc, observed), loc
    assert await page.locator(loc).count() == 1, loc
    handle = await page.locator(target).element_handle()
    assert await page.locator(loc).evaluate("(el, t) => el === t", handle), loc


async def test_heading_prefers_the_test_id(page):
    verdict = await resolve_read_address(
        page, "Products", await _element_data(page, ".title"), None
    )
    assert verdict["locator"] == '[data-test="title"]'
    assert verdict["source"] == "pointed element"


async def test_wrong_pointer_is_rescued_by_the_vision_point(page):
    verdict = await resolve_read_address(
        page,
        "Products",
        await _element_data(page, ".select_container"),
        await _center(page, ".title"),
    )
    assert verdict["source"] == "screen point"
    await _assert_rule(page, verdict, ".title", "Products")


async def test_wrong_pointer_and_wrong_point_is_unconfirmed(page):
    verdict = await resolve_read_address(
        page,
        "Products",
        await _element_data(page, ".select_container"),
        await _center(page, ".price >> nth=0"),
    )
    assert "unconfirmed" in verdict
    assert "name (a to z)" in verdict["unconfirmed"].lower()


async def test_one_character_value(page):
    verdict = await resolve_read_address(page, "1", await _element_data(page, ".badge"), None)
    await _assert_rule(page, verdict, ".badge", "1")


async def test_two_tables_scope_the_row_to_its_table(page):
    target = "#t2 tr:nth-child(1) td:nth-child(2)"
    verdict = await resolve_read_address(page, "John", await _element_data(page, target), None)
    await _assert_rule(page, verdict, target, "John")
    assert verdict["locator"].startswith("#t2 >> ")
    await page.evaluate(
        "document.querySelector('#t1 tbody').insertAdjacentHTML('afterbegin',"
        " '<tr><td>New</td><td>Row</td></tr>')"
    )
    assert await page.locator(verdict["locator"]).inner_text() == "John"


async def test_cards_without_a_class_use_the_attribute_name(page):
    target = "[data-id='P1'] a.t"
    verdict = await resolve_read_address(
        page, "Trendy Sports Running Shoes", await _element_data(page, target), None
    )
    await _assert_rule(page, verdict, target, "Trendy Sports Running Shoes")
    assert verdict["locator"].startswith("div[data-id] >> nth=0")
    await page.evaluate("const g = document.getElementById('shoes'); g.prepend(g.lastElementChild)")
    assert await page.locator(verdict["locator"]).inner_text() == "Premium White Sneakers"


async def test_lone_row_is_still_pinned(page):
    target = ".striped td:nth-child(1)"
    verdict = await resolve_read_address(page, "Cierra", await _element_data(page, target), None)
    await _assert_rule(page, verdict, target, "Cierra")
    await page.evaluate(
        "document.querySelector('.striped tbody').insertAdjacentHTML('beforeend',"
        " '<tr><td>Alden</td><td>Cantrell</td></tr><tr><td>Kierra</td><td>Gentry</td></tr>')"
    )
    assert await page.locator(verdict["locator"]).count() == 1
    assert await page.locator(verdict["locator"]).inner_text() == "Cierra"


async def test_first_card_title_is_positional_not_the_name(page):
    target = ".product_pod h3 a >> nth=0"
    verdict = await resolve_read_address(
        page, "A Light in the ...", await _element_data(page, target), None
    )
    await _assert_rule(page, verdict, target, "A Light in the ...")
    await page.evaluate("const o = document.querySelector('ol.row'); o.prepend(o.lastElementChild)")
    assert await page.locator(verdict["locator"]).inner_text() == "Tipping the Velvet"


async def test_volatile_id_is_skipped(page):
    verdict = await resolve_read_address(
        page, "Quarterly report", await _element_data(page, "h2.report-title"), None
    )
    assert "ember1234" not in verdict["locator"]
    await _assert_rule(page, verdict, "h2.report-title", "Quarterly report")


async def test_overlay_point_falls_back_to_nothing_but_pointer_wins(page):
    verdict = await resolve_read_address(
        page,
        "Welcome back",
        await _element_data(page, "h2.banner"),
        await _center(page, "#overlay"),
    )
    assert verdict["source"] == "pointed element"
    await _assert_rule(page, verdict, "h2.banner", "Welcome back")


async def test_hidden_copy_uses_the_visibility_filter(page):
    """A display:none twin makes `b.deal` match twice; Playwright's visible=true keeps the real one."""
    verdict = await resolve_read_address(
        page, "Deal of the day", await _element_data(page, ".promo b.deal"), None
    )
    assert verdict["locator"].endswith(" >> visible=true")
    await _assert_rule(page, verdict, ".promo b.deal", "Deal of the day")


async def test_input_placeholder_is_its_own_text(page):
    verdict = await resolve_read_address(
        page, "Email", await _element_data(page, "input.email"), None
    )
    await _assert_rule(page, verdict, "input.email", "Email")


async def test_no_pointer_and_no_point_is_unconfirmed(page):
    verdict = await resolve_read_address(page, "Products", None, None)
    assert verdict == {"unconfirmed": "the agent pointed at no element"}


async def test_structural_address_never_returns_a_value_bearing_locator(page):
    handle = await page.locator(".title").element_handle()
    got = await structural_address(page, handle, "Products")
    assert got is not None
    assert not carries_value(got["locator"], "Products")


async def test_title_attribute_confirms_when_the_visible_text_differs(page):
    """flipkart 2026-10-04: the agent reported the title attribute; the visible text is shorter."""
    await page.evaluate(
        "document.querySelector(\"[data-id='P1'] a.t\").setAttribute('title',"
        " 'Trendy Sports Running Shoes For Men (Black, 8)')"
    )
    target = "[data-id='P1'] a.t"
    observed = "Trendy Sports Running Shoes For Men (Black, 8)"
    verdict = await resolve_read_address(page, observed, await _element_data(page, target), None)
    await _assert_rule(page, verdict, target, observed)
