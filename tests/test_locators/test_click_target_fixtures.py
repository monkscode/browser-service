"""
The click-target detector on real Chromium (bs #63).

Each test sets a small page, points at a container the way browser-use would (its absolute
xpath), and asks ``find_inner_control`` for the one control the step names. The cases that
must return None are the ones where a re-aim would click a wrong element: a control named
after an ACTION on the row, a word inside a longer word, two equal names, a hidden or covered
control, a container that is itself clickable, a password input.
"""

from unittest.mock import MagicMock

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.registration import _extract_dom_node_attributes
from browser_service.locators.click_target import find_inner_control

pytestmark = pytest.mark.integration

POINTED_DATA_JS = """(el) => {
    const parts = [];
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
        let i = 1;
        for (let s = n.previousElementSibling; s; s = s.previousElementSibling) {
            if (s.tagName === n.tagName) i++;
        }
        parts.unshift(n.tagName.toLowerCase() + '[' + i + ']');
    }
    return {tagName: el.tagName.toLowerCase(), xpath: parts.join('/')};
}"""

MENU = (
    '<ul style="width:300px"><li role="menuitem" style="width:300px;height:30px">{inner}</li></ul>'
)
ROW = '<table><tr><td style="width:{w}px">{first}</td>{rest}</tr></table>'


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


async def _find(page, html, selector, expected, description=None, root=None, **over):
    """Set the page, point at ``selector``, ask for the inner control."""
    await page.set_content(html)
    root = root or page
    data = await root.locator(selector).first.evaluate(POINTED_DATA_JS)
    kwargs = dict(
        action="click",
        expected_text=expected,
        is_collection=None,
        element_data=data,
        description=description,
        row_anchor_text=None,
    )
    kwargs.update(over)
    return await find_inner_control(root, **kwargs)


def _tag(result):
    assert result is not None
    return result.element_data["tagName"]


def _text(result):
    assert result is not None
    return result.element_data["textContent"]


# ---------------------------------------------------------------------------
# Pointed at the container: the control comes back
# ---------------------------------------------------------------------------


async def test_inline_link_in_menuitem_li(page):
    r = await _find(page, MENU.format(inner='<a href="#">Products</a>'), "li", "Products")
    assert _tag(r) == "a"
    assert _text(r) == "Products"
    assert r.x > 0 and r.y > 0


async def test_inline_link_in_wide_td(page):
    html = ROW.format(w=600, first='<a href="#">Alice</a>', rest="")
    r = await _find(page, html, "td", "Alice")
    assert _tag(r) == "a"


async def test_other_control_at_the_centre(page):
    html = (
        '<div role="group" style="display:flex;width:200px">'
        '<button style="width:60px;margin:0">Express shipping</button>'
        '<button style="width:140px;margin:0">Standard shipping</button></div>'
    )
    r = await _find(page, html, "div", "Express shipping")
    assert _tag(r) == "button"
    assert _text(r) == "Express shipping"


async def test_icon_link_named_by_aria_label(page):
    html = ROW.format(
        w=600, first='<a href="#" aria-label="Edit"><svg width="16" height="16"></svg></a>', rest=""
    )
    r = await _find(page, html, "td", "Edit")
    assert _tag(r) == "a"
    assert r.element_data["ariaLabel"] == "Edit"


async def test_inside_a_frame(page):
    inner = MENU.format(inner='<a href="#">Products</a>')
    html = (
        '<div style="height:100px">top</div>'
        f'<iframe id="app-frame" width="400" height="200" srcdoc=\'{inner}\'></iframe>'
    )
    r = await _find(page, html, "li", "Products", root=page.frame_locator("#app-frame"))
    assert _tag(r) == "a"
    assert r.y > 100, "page coordinates include the frame's offset"
    assert await r.handle.evaluate("el => el.ownerDocument !== window.top.document")


async def test_case_differs(page):
    r = await _find(
        page,
        MENU.format(inner='<a href="#" style="text-transform:uppercase">Products</a>'),
        "li",
        "Products",
    )
    assert _tag(r) == "a"


async def test_link_text_holds_more(page):
    r = await _find(page, MENU.format(inner='<a href="#">Products (3)</a>'), "li", "Products")
    assert _tag(r) == "a"


async def test_agent_reported_the_whole_row(page):
    html = ROW.format(
        w=300,
        first='<a href="#">&lt;caption&gt;</a>',
        rest="<td>Defines a table caption</td>",
    )
    r = await _find(
        page,
        html,
        "tr",
        "<caption> Defines a table caption",
        description="table row for the <caption> tag",
    )
    assert _tag(r) == "a"
    assert _text(r) == "<caption>"


async def test_name_link_beside_an_action_link(page):
    html = ROW.format(
        w=400,
        first='<a href="#">Order 1234</a>',
        rest='<td>Shipped</td><td><a href="#">Cancel</a></td>',
    )
    r = await _find(
        page, html, "tr", "Order 1234 Shipped Cancel", description="table row for Order 1234"
    )
    assert _text(r) == "Order 1234"


async def test_menu_points_at_the_innermost(page):
    r = await _find(page, MENU.format(inner='<a href="#">Products</a>'), "ul", "Products")
    assert _tag(r) == "a", "never the dead li"


async def test_two_links_the_exact_name_wins(page):
    inner = '<a href="#" id="p">Products</a> <a href="#">Products sale</a>'
    r = await _find(
        page, MENU.format(inner=inner), "li", "Products", description="Products menu item"
    )
    assert _text(r) == "Products"


# ---------------------------------------------------------------------------
# Returns None: a re-aim could click a wrong element, or there is nothing to re-aim
# ---------------------------------------------------------------------------


async def test_block_link_fills_the_container(page):
    r = await _find(
        page,
        MENU.format(inner='<a href="#" style="display:block">Products</a>'),
        "li",
        "Products",
    )
    assert r is None


async def test_two_controls_same_name(page):
    inner = '<a href="#">Products</a> <a href="#">Products</a>'
    assert await _find(page, MENU.format(inner=inner), "li", "Products") is None


async def test_action_link_is_not_the_row(page):
    html = ROW.format(
        w=200,
        first="Order 1234",
        rest="<td>Shipped</td><td><a href='#'>Cancel</a></td>",
    )
    r = await _find(
        page, html, "tr", "Order 1234 Shipped Cancel", description="table row for order 1234"
    )
    assert r is None, "it must never pick Cancel"


@pytest.mark.parametrize(
    "control",
    ['<button aria-label="Delete Alice">x</button>', '<a href="#">Delete Alice</a>'],
    ids=["button-aria-label", "visible-link"],
)
async def test_action_named_after_the_row(page, control):
    html = ROW.format(w=300, first="Alice", rest=f"<td>alice@x.io</td><td>{control}</td>")
    r = await _find(page, html, "tr", "Alice", description="table row for Alice")
    assert r is None


async def test_checkbox_named_after_the_row(page):
    control = '<input type="checkbox" aria-label="Select Alice">'
    html = ROW.format(w=300, first="Alice", rest=f"<td>alice@x.io</td><td>{control}</td>")
    assert await _find(page, html, "tr", "Alice", description="table row for Alice") is None


async def test_expander_named_after_the_item(page):
    html = (
        '<ul style="width:300px"><li role="treeitem" style="width:300px">Reports '
        '<button aria-label="Expand Reports">+</button></li></ul>'
    )
    assert await _find(page, html, "li", "Reports") is None


async def test_word_shared_by_row_name_and_action(page):
    html = ROW.format(
        w=300, first="Remove legacy login", rest="<td>Open</td><td><a href='#'>Remove</a></td>"
    )
    r = await _find(
        page,
        html,
        "tr",
        "Remove legacy login Open Remove",
        description="table row for Remove legacy login",
    )
    assert r is None


async def test_expected_inside_a_longer_word(page):
    html = ROW.format(w=600, first='<a href="#">Credit memo</a>', rest="")
    assert await _find(page, html, "td", "Edit") is None


async def test_word_inside_a_longer_word(page):
    html = ROW.format(w=300, first="Cancelled order 77", rest="<td><a href='#'>Cancel</a></td>")
    r = await _find(
        page,
        html,
        "tr",
        "Cancelled order 77 Cancel",
        description="table row for the Cancelled order",
    )
    assert r is None


async def test_devanagari_word_stays_whole(page):
    rad = "रद्द"
    ridi = rad + "ी"
    # the link is the FIRST text of the row, so `first(c)` holds; the step does not hold the other
    # cell's whole text, so `named` does not refuse either: only `\p{M}` keeps the word whole
    html = f"<table><tr><td><a href='#'>{rad}</a></td><td style='width:500px'>{ridi} scrap</td></tr></table>"
    r = await _find(
        page,
        html,
        "tr",
        f"{rad} {ridi} scrap",
        description=f"table row for {ridi}",
    )
    assert r is None


async def test_symbol_only_name(page):
    html = ROW.format(w=300, first="Next page", rest="<td><a href='#'>→</a></td>")
    r = await _find(page, html, "tr", "Next page →", description="table row for next page →")
    assert r is None


async def test_hidden_control(page):
    inner = '<a href="#" style="display:none">Products</a><span>Products</span>'
    assert await _find(page, MENU.format(inner=inner), "li", "Products") is None


@pytest.mark.parametrize(
    "html",
    [
        '<div style="max-height:0;overflow:hidden"><a href="#">Products</a></div>',
        '<a href="#" style="position:absolute;width:1px;height:1px;overflow:hidden">Products</a>',
        '<a href="#" style="opacity:0">Products</a>',
        '<details><summary>x</summary><a href="#">Products</a></details>',
    ],
    ids=["max-height-0", "one-pixel-skip-link", "opacity-0", "closed-details"],
)
async def test_clipped_control(page, html):
    assert await _find(page, MENU.format(inner=html), "li", "Products") is None


async def test_control_under_a_cover(page):
    html = (
        '<div style="position:relative;width:300px;height:100px">'
        '<a href="#" style="position:absolute;top:0;left:0">Order details</a>'
        '<a href="#" aria-label="" style="position:absolute;top:0;left:0;width:100%;height:100%"></a>'
        "</div>"
    )
    assert await _find(page, html, "div", "Order details") is None


async def test_control_below_the_fold(page):
    # the pointed container's centre is off-screen: the cursor and cover checks read a hit test
    # that answers nothing there, so the container is left as it is
    html = '<div style="height:3000px"></div>' + MENU.format(inner='<a href="#">Products</a>')
    assert await _find(page, html, "li", "Products") is None


async def test_control_below_the_fold_in_a_tall_container(page):
    html = (
        '<ul style="width:300px"><li role="menuitem" style="width:300px;height:1600px;position:relative">'
        '<a href="#" style="position:absolute;top:1300px;left:0">Products</a></li></ul>'
    )
    await page.set_content(html)
    assert await page.evaluate("document.querySelector('a').getBoundingClientRect().top") > 900
    r = await _find(page, html, "li", "Products")
    assert _tag(r) == "a", "only the container's centre has to be on-screen"


async def test_cursor_pointer_on_the_cells_below_the_fold(page):
    html = '<div style="height:1200px"></div><style>td{cursor:pointer}</style>' + ROW.format(
        w=600, first='<a href="#">Alice</a>', rest=""
    )
    assert await _find(page, html, "tr", "Alice") is None


@pytest.mark.parametrize(
    "cells, name",
    [
        (
            '<td style="width:300px">Alice, Smith</td><td style="width:400px">alice@x.io</td>'
            '<td><button aria-label="Delete Alice">x</button></td>',
            "Alice",
        ),
        (
            '<td style="width:300px">(Alice)</td><td style="width:400px">alice@x.io</td>'
            '<td><a href="#">Delete Alice</a></td>',
            "Alice",
        ),
        (
            '<td style="width:400px">Order 1234: Shipped</td><td style="width:200px">2026-10-30</td>'
            '<td><a href="#">Cancel Order 1234</a></td>',
            "Order 1234",
        ),
    ],
    ids=["comma", "parenthesis", "colon"],
)
async def test_name_beside_punctuation(page, cells, name):
    html = f"<table><tr>{cells}</tr></table>"
    assert await _find(page, html, "tr", name, description=f"table row for {name}") is None


async def test_name_only_in_an_input_value(page):
    html = (
        '<table><tr><td style="width:300px"><input value="Alice"></td>'
        '<td style="width:400px">Active</td><td><button>Delete Alice</button></td></tr></table>'
    )
    assert await _find(page, html, "tr", "Alice", description="table row for Alice") is None


@pytest.mark.parametrize(
    "first, name, rest, expected, step",
    [
        (
            '<a href="#">Remove</a>',
            "Remove legacy login",
            "Open",
            "Remove Remove legacy login Open",
            "table row for Remove legacy login",
        ),
        (
            '<a href="#">Close</a>',
            "Will Close",
            "Admin",
            "Close Will Close Admin",
            "table row for Will Close",
        ),
        (
            "<button>Action</button>",
            "Order 1234",
            "Shipped",
            "Action Order 1234 Shipped",
            "table row for Order 1234 (action: click)",
        ),
    ],
    ids=["remove", "close", "action"],
)
async def test_action_column_first(page, first, name, rest, expected, step):
    html = (
        f'<table><tr><td>{first}</td><td style="width:500px">{name}</td>'
        f'<td style="width:200px">{rest}</td></tr></table>'
    )
    assert await _find(page, html, "tr", expected, description=step) is None


async def test_hidden_value_is_not_counted(page):
    # a password's value is not read, so it is not "the name shown elsewhere"
    inner = '<input type="password" value="Products" style="width:60px"> <a href="#">Products</a>'
    r = await _find(page, MENU.format(inner=inner), "li", "Products")
    assert _tag(r) == "a"


async def test_cursor_pointer_on_the_container(page):
    html = (
        '<ul style="width:300px"><li role="menuitem" style="width:300px;cursor:pointer">'
        '<a href="#">Products</a></li></ul>'
    )
    assert await _find(page, html, "li", "Products") is None


async def test_cursor_pointer_on_the_cells(page):
    html = "<style>td{cursor:pointer}</style>" + ROW.format(
        w=600, first='<a href="#">Alice</a>', rest=""
    )
    assert await _find(page, html, "tr", "Alice") is None


async def test_tabindex_zero_on_the_row(page):
    html = '<table><tr tabindex="0"><td style="width:600px"><a href="#">Alice</a></td></tr></table>'
    assert await _find(page, html, "tr", "Alice") is None


async def test_onclick_on_the_container(page):
    html = (
        '<ul style="width:300px"><li role="menuitem" style="width:300px" onclick="void 0">'
        '<a href="#">Products</a></li></ul>'
    )
    assert await _find(page, html, "li", "Products") is None


@pytest.mark.parametrize(
    "control",
    [
        '<input type="password" value="Products" style="width:60px">',
        '<input type="password" value="Products" aria-label="Products" style="width:60px">',
        '<textarea nlrf-secret aria-label="Products" style="width:60px;height:16px"></textarea>',
    ],
    ids=["value", "aria-label", "marked-textarea"],
)
async def test_password_input_is_never_matched(page, control):
    assert await _find(page, MENU.format(inner=control), "li", "Products") is None


async def test_stale_xpath_is_left_alone(page):
    html = MENU.format(inner='<a href="#">Products</a>')
    await page.set_content(html)
    r = await find_inner_control(
        page,
        action="click",
        expected_text="Products",
        is_collection=None,
        element_data={"tagName": "li", "xpath": "html[1]/body[1]/ul[9]/li[1]"},
        description=None,
        row_anchor_text=None,
    )
    assert r is None


async def test_pointed_xpath_matches_two(page):
    html = (
        "<ul>"
        + MENU.format(inner='<a href="#">Products</a>')
        + MENU.format(inner='<a href="#">Products</a>')
        + "</ul>"
    )
    await page.set_content(html)
    r = await find_inner_control(
        page,
        action="click",
        expected_text="Products",
        is_collection=None,
        element_data={"tagName": "li", "xpath": "//li"},
        description=None,
        row_anchor_text=None,
    )
    assert r is None


async def test_read_failure_is_none():
    root = MagicMock()
    root.locator.side_effect = RuntimeError("page closed")
    r = await find_inner_control(
        root,
        action="click",
        expected_text="Products",
        is_collection=None,
        element_data={"tagName": "li", "xpath": "html[1]/body[1]/li[1]"},
        description=None,
        row_anchor_text=None,
    )
    assert r is None


# ---------------------------------------------------------------------------
# No read at all: a MagicMock search_root must never be touched
# ---------------------------------------------------------------------------

LI_DATA = {"tagName": "li", "xpath": "html[1]/body[1]/li[1]"}


async def _no_read(**over):
    root = MagicMock()
    kwargs = dict(
        action="click",
        expected_text="Products",
        is_collection=None,
        element_data=dict(LI_DATA),
        description=None,
        row_anchor_text=None,
    )
    kwargs.update(over)
    assert await find_inner_control(root, **kwargs) is None
    root.locator.assert_not_called()


@pytest.mark.parametrize("action", [None, "get_text", "input", "select"])
async def test_no_read_when_not_a_click(action):
    await _no_read(action=action)


@pytest.mark.parametrize(
    "tag", ["a", "button", "input", "select", "textarea", "label", "summary", "option", "A"]
)
async def test_no_read_for_a_control(tag):
    await _no_read(element_data={"tagName": tag, "xpath": "html[1]/body[1]/x[1]"})


async def test_no_read_with_own_listener():
    await _no_read(
        element_data={"tagName": "li", "hasClickListener": True, "xpath": "html/body/li"}
    )


async def test_no_read_for_a_collection():
    await _no_read(is_collection=True)


@pytest.mark.parametrize("expected", [None, "", "  "])
async def test_no_read_for_blank_expected_text(expected):
    await _no_read(expected_text=expected)


async def test_no_read_with_a_row_anchor():
    await _no_read(row_anchor_text="Alice")


@pytest.mark.parametrize("data", [None, {"tagName": "li"}], ids=["none", "no-xpath"])
async def test_no_read_without_element_data(data):
    await _no_read(element_data=data)


# ---------------------------------------------------------------------------
# is_answer and the shape of element_data
# ---------------------------------------------------------------------------

TWO_LINKS = MENU.format(inner='<a href="#" id="p">Products</a> <a href="#">Products sale</a>')


async def test_is_answer_true_for_the_control_false_for_its_container_and_for_two_matches(page):
    r = await _find(page, TWO_LINKS, "li", "Products", description="Products menu item")
    assert r is not None
    assert await r.is_answer(page, "#p") is True
    assert await r.is_answer(page, "li") is False
    assert await r.is_answer(page, "a") is False, "two matches are not an answer"
    assert await r.is_answer(page, "#nothing") is False


async def test_is_answer_false_when_the_page_went_away(page):
    r = await _find(page, TWO_LINKS, "li", "Products", description="Products menu item")
    assert r is not None
    await page.goto("about:blank")
    await page.set_content(TWO_LINKS)
    assert await r.is_answer(page, "#p") is False


async def test_element_data_keys_match_the_indexed_shape(page):
    r = await _find(page, MENU.format(inner='<a href="#">Products</a>'), "li", "Products")
    assert r is not None
    expected = set(_extract_dom_node_attributes(MagicMock())) | {"textContent"}
    assert set(r.element_data) == expected
    assert r.element_data["xpath"].startswith("html[1]/body[1]/ul[1]/li[1]/a[1]")
