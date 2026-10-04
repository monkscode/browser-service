"""
#31 read-address rule on real Chromium (local HTML): the structural address of the element the agent pointed at — no displayed value, exactly one match, the same node — and the shapes R1–R4 of the spec.
"""

import re
from pathlib import Path
from unittest.mock import patch

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import find_unique_locator_action
from browser_service.config import config
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


# ---- Task 3 flow tests -------------------------------------------------------------------------


@pytest.fixture
def action_fit_on():
    with patch.object(config.locator, "enable_action_fit", True):
        yield


async def _find(page, *, action, candidate, pointed, point_at, expected, element_id="elem_4"):
    x, y = await _center(page, point_at)
    return await find_unique_locator_action(
        x=x,
        y=y,
        element_id=element_id,
        element_description="the heading",
        expected_text=expected,
        candidate_locator=candidate,
        element_data=await _element_data(page, pointed),
        page=page,
        is_collection=False,
        action=action,
        vision_point=(x, y),
    )


async def test_flow_text_candidate_with_wrong_index_ships_the_heading(page, action_fit_on):
    result = await _find(
        page,
        action="get_text",
        candidate="text=Products",
        pointed=".select_container",
        point_at=".title",
        expected="Products",
    )
    assert result["found"] is True
    assert not carries_value(result["best_locator"], "Products")
    assert result["read_target_rewritten_from"] == "text=Products"
    assert result["element_info"]["source"] == "read_address_resolved_element"
    assert "title" in result["element_info"]["className"]


async def test_flow_both_wrong_is_not_found_never_text(page, action_fit_on):
    result = await _find(
        page,
        action="get_text",
        candidate="text=Products",
        pointed=".select_container",
        point_at=".price >> nth=0",
        expected="Products",
    )
    assert result["found"] is False
    assert "read_address_unconfirmed" in result


async def test_flow_click_keeps_its_text_locator(page, action_fit_on):
    result = await _find(
        page,
        action="click",
        candidate="text=Products",
        pointed=".title",
        point_at=".title",
        expected="Products",
    )
    assert result["found"] is True
    assert result["best_locator"] == "text=Products"


async def test_flow_value_free_candidate_is_unchanged(page, action_fit_on):
    result = await _find(
        page,
        action="get_text",
        candidate=".title",
        pointed=".title",
        point_at=".title",
        expected="Products",
    )
    assert result["best_locator"] == ".title"
    assert "read_target_rewritten_from" not in result


async def test_flow_flag_off_is_todays_behaviour(page):
    result = await _find(
        page,
        action=None,
        candidate="text=Products",
        pointed=".select_container",
        point_at=".price >> nth=0",
        expected="Products",
    )
    assert result["found"] is True
    assert result["best_locator"] == "text=Products"


async def test_flow_shadow_root_keeps_todays_locator(page, action_fit_on):
    result = await _find(
        page,
        action="get_text",
        candidate='text="Shadow text"',
        pointed="#host",
        point_at="#host",
        expected="Shadow text",
    )
    assert result["found"] is True
    assert result["best_locator"] == 'text="Shadow text"'


# ---- Edge cases: direct text, a confirming ancestor, CSS-hostile and volatile tokens -------------

EDGE = (
    (Path(__file__).parent / "locator_fixtures" / "read_address_edge_cases.html").resolve().as_uri()
)


@pytest.fixture
async def edge():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page_obj = await (
            await browser.new_context(viewport={"width": 1280, "height": 900})
        ).new_page()
        await page_obj.goto(EDGE, wait_until="domcontentloaded")
        try:
            yield page_obj
        finally:
            await browser.close()


async def test_flash_with_a_close_link_is_confirmed_by_its_direct_text(edge):
    """the-internet /secure: the flash shows the message plus a close-link ×; its direct text is the message."""
    observed = "You logged into a secure area!"
    verdict = await resolve_read_address(edge, observed, await _element_data(edge, "#flash"), None)
    assert verdict["source"] == "pointed element"
    await _assert_rule(edge, verdict, "#flash", observed)


async def test_point_on_inline_markup_confirms_the_heading(edge):
    """The vision point lands on the <b> inside the heading; the heading, its parent, confirms."""
    verdict = await resolve_read_address(edge, "Welcome John", None, await _center(edge, "#who"))
    assert verdict["source"] == "screen point"
    await _assert_rule(edge, verdict, "h1.greet", "Welcome John")


async def test_point_on_the_required_star_confirms_the_label(edge):
    """The vision point lands on the required-field star inside the label; the label confirms."""
    verdict = await resolve_read_address(edge, "Email *", None, await _center(edge, ".req"))
    assert verdict["source"] == "screen point"
    await _assert_rule(edge, verdict, "label.lbl", "Email *")


async def test_sort_select_and_an_unrelated_point_stay_unconfirmed(edge):
    """saucedemo's sort select pointed and a point on an unrelated price stay unconfirmed, the reason naming what each source shows: no node or near ancestor shows exactly 'Products' (the header holds it among the sort options). It passes without the walk-up guard too — the guard tests below pin that."""
    verdict = await resolve_read_address(
        edge,
        "Products",
        await _element_data(edge, "select.product_sort_container"),
        await _center(edge, ".inventory_item_price"),
    )
    assert "unconfirmed" in verdict, verdict
    assert "name (a to z)" in verdict["unconfirmed"].lower()
    assert "$29.99" in verdict["unconfirmed"]


@pytest.mark.parametrize(
    "target, observed",
    [("[class~='md:text-lg']", "Total: 42"), ("[class~='w-1/2']", "Half width")],
)
async def test_tailwind_classes_get_no_backslash(edge, target, observed):
    """Robot Framework reads a CSS escape's backslash as its own escape: such a class is written as a quoted attribute."""
    verdict = await resolve_read_address(edge, observed, await _element_data(edge, target), None)
    assert "\\" not in verdict["locator"], verdict["locator"]
    await _assert_rule(edge, verdict, target, observed)


async def test_vue_scoped_hash_is_not_a_card_mark(edge):
    """data-v-<hash> sits on the grid, every card and every node inside a card — not a card mark."""
    target = ".card >> nth=1 >> h3"
    verdict = await resolve_read_address(edge, "Beta", await _element_data(edge, target), None)
    assert verdict["locator"].startswith("div.card >> nth=1 >> "), verdict["locator"]
    await _assert_rule(edge, verdict, target, "Beta")


async def test_volatile_container_id_is_walked_past(edge):
    """Rows inside <div id="ember55"> plus one row elsewhere — a card path, not an xpath."""
    target = "#ember55 .row >> nth=1 >> span"
    verdict = await resolve_read_address(edge, "Two", await _element_data(edge, target), None)
    assert verdict["kind"] == "card-path", verdict
    assert "ember55" not in verdict["locator"]
    await _assert_rule(edge, verdict, target, "Two")


@pytest.mark.parametrize(
    "anchor, observed",
    [("user.name", "Alice"), ("1st", "Bob"), ("form:main", "Carol"), ("-side", "Dan")],
)
async def test_anchor_id_that_needs_css_escaping(edge, anchor, observed):
    """An anchor id CSS.escape would escape, or one NLRF would not prefix with css= (`#-side`): no backslash, no bare `#` + non-letter."""
    target = f'[id="{anchor}"] span.val'
    verdict = await resolve_read_address(edge, observed, await _element_data(edge, target), None)
    assert "\\" not in verdict["locator"], verdict["locator"]
    assert not re.match(r"#[^A-Za-z_]", verdict["locator"]), verdict["locator"]
    await _assert_rule(edge, verdict, target, observed)


async def test_vue_child_component_root_hash_is_not_a_card_mark(edge):
    """A child component's root carries its parent's data-v hash and its own (vuejs.org sponsors)."""
    target = ".spsr-container.platinum a.spsr-item >> nth=1"
    verdict = await resolve_read_address(edge, "Plat Two", await _element_data(edge, target), None)
    assert "data-v-" not in verdict["locator"], verdict["locator"]
    await _assert_rule(edge, verdict, target, "Plat Two")


async def test_shared_attribute_name_on_every_element_is_not_a_card_mark(edge):
    """A data-* NAME on the grid, every card and their insides (Astro's scoped style — not a build hash, so not volatile) marks no card: the parent carries it too."""
    target = ".tile >> nth=1 >> h3"
    verdict = await resolve_read_address(
        edge, "Second tile", await _element_data(edge, target), None
    )
    assert verdict["locator"].startswith("div.tile >> nth=1 >> "), verdict["locator"]
    assert "data-astro-cid" not in verdict["locator"]
    await _assert_rule(edge, verdict, target, "Second tile")


# ---- The walk-up guard: an ancestor confirms only when the pointed node shows nothing or a piece of the text


async def _center_in_view(page, selector):
    await page.locator(selector).scroll_into_view_if_needed()
    return await _center(page, selector)


async def test_point_on_an_edit_button_inside_the_heading_stays_unconfirmed(edge):
    """The point lands on the heading's Edit button: the button shows its own other text, so the heading must not confirm."""
    verdict = await resolve_read_address(
        edge, "Order summary", None, await _center_in_view(edge, "button.edit")
    )
    assert "unconfirmed" in verdict, verdict
    assert "the screen point shows 'edit'" in verdict["unconfirmed"]


async def test_pointed_select_inside_the_crumb_stays_unconfirmed(edge):
    """The agent pointed at the select inside "Products <select>": the select shows its options, so the crumb must not confirm."""
    verdict = await resolve_read_address(
        edge, "Products", await _element_data(edge, "select.sorter"), None
    )
    assert "unconfirmed" in verdict, verdict
    assert "the pointed element shows 'name (a to z) price'" in verdict["unconfirmed"]


async def test_pointed_edit_link_inside_the_cell_stays_unconfirmed(edge):
    """The agent pointed at the cell's edit link: the link shows its own other text, so the cell "John edit" must not confirm."""
    verdict = await resolve_read_address(
        edge, "John", await _element_data(edge, "td.nm >> nth=0 >> a.act"), None
    )
    assert "unconfirmed" in verdict, verdict
    assert "the pointed element shows 'edit'" in verdict["unconfirmed"]


async def test_point_on_a_piece_of_a_split_price_confirms_its_container(edge):
    """Control: the point lands on "29.99", a piece of "$29.99", so the price container confirms."""
    verdict = await resolve_read_address(edge, "$29.99", None, await _center_in_view(edge, ".amt"))
    assert verdict["source"] == "screen point"
    await _assert_rule(edge, verdict, ".price-wrap", "$29.99")
