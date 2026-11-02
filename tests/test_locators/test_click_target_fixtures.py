"""
The click-target detector on real Chromium (bs #63).

Each test sets a small page, points at a container the way browser-use would (its absolute
xpath), and asks ``find_inner_control`` for the one control the step names. The cases that
must return None are the ones where a re-aim would click a wrong element: a control named
after an ACTION on the row, a word inside a longer word, two equal names, a hidden or covered
control, a container that is itself clickable, a password input.
"""

import time
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
    assert r.x > 0
    assert r.y > 0


async def test_inline_link_in_wide_td(page):
    html = ROW.format(w=600, first='<a href="#">Alice</a>', rest="")
    r = await _find(page, html, "td", "Alice")
    assert _tag(r) == "a"


async def test_other_control_at_the_centre_stays_unanswered(page):
    # two visible controls in the container: no re-aim
    html = (
        '<div role="group" style="display:flex;width:200px">'
        '<button style="width:60px;margin:0">Express shipping</button>'
        '<button style="width:140px;margin:0">Standard shipping</button></div>'
    )
    assert await _find(page, html, "div", "Express shipping") is None


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


async def test_expected_is_the_control_name(page):
    # the row shows "caption" twice and the link once: only the whole-name clause answers
    html = TABLE.format(
        '<tr class="r-1"><td><a href="#">&lt;caption&gt;</a></td>'
        '<td style="width:500px">Defines a table caption</td><td style="width:200px"></td></tr>'
    )
    r = await _find(
        page, html, "tr.r-1", "<caption>", description="table row for the <caption> tag"
    )
    assert _tag(r) == "a"
    assert _text(r) == "<caption>"


async def test_name_link_beside_an_action_link_stays_unanswered(page):
    html = ROW.format(
        w=400,
        first='<a href="#">Order 1234</a>',
        rest='<td>Shipped</td><td><a href="#">Cancel</a></td>',
    )
    r = await _find(
        page, html, "tr", "Order 1234 Shipped Cancel", description="table row for Order 1234"
    )
    assert r is None


async def test_menu_points_at_the_innermost(page):
    r = await _find(page, MENU.format(inner='<a href="#">Products</a>'), "ul", "Products")
    assert _tag(r) == "a", "never the dead li"


async def test_exact_name_beside_a_longer_one_stays_unanswered(page):
    inner = '<a href="#" id="p">Products</a> <a href="#">Products sale</a>'
    r = await _find(
        page, MENU.format(inner=inner), "li", "Products", description="Products menu item"
    )
    assert r is None


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


async def test_hidden_password_input_counts_as_a_second_control(page):
    # a password input that is not shown still counts as a control: the row holds two (the link
    # and the input), so it is left as today
    html = (
        '<table><tr><td style="width:600px">User</td><td><a href="#">Alice</a></td>'
        '<td><input type="password" value="Alice" style="display:none"></td></tr></table>'
    )
    r = await _find(page, html, "tr", "Alice", description="table row for Alice")
    assert r is None


async def test_hidden_password_value_is_never_counted(page):
    # the link's only content is a password input holding "Alice" (one control: the innermost).
    # Counted, the value would make the cell read "Alice" once more than the link shows, and the
    # step does not hold "Alice", so nothing would answer; skipped, the link answers by its label
    first = (
        '<a href="#" aria-label="Alice" style="display:inline-block;width:120px">'
        '<input type="password" value="Alice" style="width:100px"></a>'
    )
    html = ROW.format(w=600, first=first, rest="")
    r = await _find(page, html, "td", "Alice", description="the link in the first cell")
    assert _tag(r) == "a"


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


TABLE = (
    "<style>td,th{{border:1px solid #ccc;padding:6px}}</style>"
    '<table style="width:900px;border-collapse:collapse"><thead><tr><th>Name</th><th>Info</th>'
    "<th>Actions</th></tr></thead><tbody>{}</tbody></table>"
)
REMOVE = '<a href="#">Remove</a>'


@pytest.mark.parametrize(
    "html, pointed, expected",
    [
        (  # the name cell holds more than the step names
            TABLE.format(
                f'<tr class="r-1"><td>{REMOVE}</td><td style="width:500px">Remove legacy login #12</td>'
                '<td style="width:200px">Open</td></tr>'
            ),
            "tr.r-1",
            "Remove Remove legacy login #12 Open",
        ),
        (  # the same cells one wrapper down
            f'<div class="row" style="width:900px"><div class="cells" style="display:flex"><div>{REMOVE}</div>'
            '<div style="width:500px">Remove legacy login</div><div style="width:200px">Open</div></div></div>',
            "div.row",
            "Remove Remove legacy login Open",
        ),
        (  # a one-row tbody
            f'<table style="width:900px"><tbody><tr><td>{REMOVE}</td><td style="width:500px">'
            'Remove legacy login</td><td style="width:200px">Open</td></tr></tbody></table>',
            "tbody",
            "Remove Remove legacy login Open",
        ),
    ],
    ids=["cell-holds-more", "nested-wrapper", "tbody-one-row"],
)
async def test_step_holds_part_of_a_cell(page, html, pointed, expected):
    r = await _find(page, html, pointed, expected, description="table row for Remove legacy login")
    assert r is None


def _row(a, b, c):
    return TABLE.format(
        f'<tr class="r-1"><td>{a}</td><td style="width:500px">{b}</td>'
        f'<td style="width:200px">{c}</td></tr>'
    )


def _wrapped(a, b, c):
    return (
        '<div class="row" style="width:900px"><div class="cells" style="display:flex">'
        f'<div>{a}</div><div style="width:500px">{b}</div><div style="width:200px">{c}</div>'
        "</div></div>"
    )


def _tbody(a, b, c):
    return (
        '<table style="width:900px"><tbody><tr>'
        f'<td>{a}</td><td style="width:500px">{b}</td><td style="width:200px">{c}</td>'
        "</tr></tbody></table>"
    )


DELETE = '<a href="#">Delete</a>'
ACTION = "<button>Action</button>"


@pytest.mark.parametrize(
    "html, pointed, expected, step",
    [
        (
            _row(DELETE, "Delete", "Allowed"),
            "tr.r-1",
            "Delete Delete Allowed",
            "table row for Delete",
        ),
        (
            _row(REMOVE, "Remove", "Open"),
            "tr.r-1",
            "Remove Remove Open",
            "table row for Remove",
        ),
        (
            _row(DELETE, "Allowed", "Delete"),
            "tr.r-1",
            "Delete Allowed Delete",
            "table row for Delete",
        ),
        (
            _row(ACTION, "Alice", "Shipped"),
            "tr.r-1",
            "Action Alice Shipped",
            "table row for Alice (action: click)",
        ),
        (
            _wrapped(DELETE, "Delete", "Allowed"),
            "div.row",
            "Delete Delete Allowed",
            "table row for Delete",
        ),
        (
            _tbody(DELETE, "Delete", "Allowed"),
            "tbody",
            "Delete Delete Allowed",
            "table row for Delete",
        ),
    ],
    ids=[
        "delete-named-delete",
        "remove-named-remove",
        "delete-name-last",
        "action-one-word",
        "wrapper-one-word",
        "tbody-one-word",
    ],
)
async def test_step_names_another_element_of_the_row(page, html, pointed, expected, step):
    # a one-word cell equal to the control's name holds no run longer than the name:
    # only the element check refuses these
    assert await _find(page, html, pointed, expected, description=step) is None


EDIT_ICON = '<button aria-label="Edit">&#9998;</button>'
DELETE_BUTTON = "<button>Delete</button>"


@pytest.mark.parametrize(
    "html",
    [
        _row(f"{EDIT_ICON} {DELETE_BUTTON}", "Edit rights", "Allowed"),
        TABLE.format(
            f'<tr class="r-1"><td>{EDIT_ICON}</td><td>{DELETE_BUTTON}</td>'
            '<td style="width:500px">Edit rights</td><td style="width:200px">Allowed</td></tr>'
        ),
    ],
    ids=["two_candidates", "two_candidates_cells"],
)
async def test_two_candidates_stay_unanswered(page, html):
    # both buttons carry a name the step holds; refusing Edit (the step names the Delete
    # button beside it) must not leave Delete as the one answer
    r = await _find(
        page,
        html,
        "tr.r-1",
        "Delete Edit rights Allowed",
        description="table row for Edit and Delete",
    )
    assert r is None


BIN_DELETE = '<button aria-label="Delete">&#128465;</button>'


def _t1(ctrl, name='<a href="#">Alice</a>', role="Admin", cls="r-a", act=False):
    td = '<td class="act">' if act else "<td>"
    return (
        f'<tr class="{cls}"><td style="width:300px">{name}</td>'
        f'<td style="width:400px">{role}</td>{td}{ctrl}</td></tr>'
    )


T1_STEP = "Delete button in the row for Alice"
# The Delete control shown only while the row is hovered, or an icon whose click handler a script
# attaches (bench/private/bs63/build_review: rr8_hoverlib.py, m3_hoverlib.py, m3_pages.py)
TRASH_ICON = (
    '<span class="del" data-who="alice" style="cursor:pointer;font-size:16px">&#128465;</span>'
)
SR_ONLY = (
    "<style>.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;"
    "clip:rect(0,0,0,0);white-space:nowrap;border:0}</style>"
)


@pytest.mark.parametrize(
    "html, step",
    [
        (TABLE.format(_t1(BIN_DELETE)), T1_STEP),
        (TABLE.format(_t1('<button title="Delete">&#128465;</button>')), T1_STEP),
        (TABLE.format(_t1('<button><img alt="Delete" width="16" height="16"></button>')), T1_STEP),
        (TABLE.format(_t1("<button>Delete</button>")), T1_STEP),
        (
            TABLE.format(_t1('<a href="#" aria-label="Edit">&#9998;</a>')),
            "Edit icon link in the row for Alice",
        ),
        (
            TABLE.format(
                _t1(BIN_DELETE)
                + _t1(BIN_DELETE, name='<a href="#">Bob</a>', role="User", cls="r-b")
            ),
            T1_STEP,
        ),
        (TABLE.format(_t1(BIN_DELETE)), "the trash icon in the row for Alice"),
        (TABLE.format(_t1(BIN_DELETE)), "remove button in Alice's row"),
        (
            "<style>.act{opacity:0} tr:hover .act{opacity:1}</style>"
            + TABLE.format(_t1(BIN_DELETE, act=True)),
            T1_STEP,
        ),
        (
            "<style>.act{visibility:hidden} tr:hover .act{visibility:visible}</style>"
            + TABLE.format(_t1(BIN_DELETE, act=True)),
            T1_STEP,
        ),
        (
            "<style>.act button{display:none} tr:hover .act button{display:inline}</style>"
            + TABLE.format(_t1(BIN_DELETE, act=True)),
            T1_STEP,
        ),
        (TABLE.format(_t1(TRASH_ICON)), T1_STEP),
        (
            TABLE.format(
                _t1(
                    '<svg class="del" data-who="alice" width="16" height="16" '
                    'style="cursor:pointer"><rect width="16" height="16"/></svg>'
                )
            ),
            T1_STEP,
        ),
        (
            "<style>.act .del{display:none} tr:hover .act .del{display:inline}</style>"
            + TABLE.format(_t1(TRASH_ICON, act=True)),
            T1_STEP,
        ),
        (
            "<style>.act{opacity:0} tr:hover .act{opacity:1}</style>"
            + TABLE.format(
                _t1(
                    '<i class="del" data-who="alice" '
                    'style="cursor:pointer;font-style:normal;font-size:16px">&#128465;</i>',
                    act=True,
                )
            ),
            T1_STEP,
        ),
        (
            SR_ONLY
            + TABLE.format(
                _t1(
                    "a@x",
                    name='<input type="checkbox" class="sr-only" aria-label="Select row"> '
                    '<a href="#">Alice</a>',
                )
            ),
            "Select row checkbox for Alice",
        ),
    ],
    ids=[
        "t1_aria",
        "t1_title",
        "t1_img_alt",
        "t1_visible_ctrl",
        "t1_edit_icon",
        "t1_two_rows",
        "t1_trash",
        "t1_remove",
        "rr8_hover_opacity",
        "rr8_hover_visibility",
        "rr8_hover_display",
        "rr8_icon_span",
        "rr8_icon_svg",
        "a_display",
        "a_i_opacity",
        "m3_sronly_select",
    ],
)
async def test_a_second_control_leaves_the_row_alone(page, html, step):
    # the expected text names the row, the step means its other control: the Alice link
    # must not answer
    assert await _find(page, html, "tr.r-a", "Alice", description=step) is None


async def test_menu_item_with_a_hidden_submenu_stays_unanswered(page):
    # the hidden submenu's two links count as controls, so the item holds three and is left as
    # today, though the step means the shown link (the price of counting hidden controls)
    html = (
        '<ul class="menu" style="list-style:none;padding:0;width:320px">'
        '<li class="mi-p" style="width:300px;height:30px;border:1px solid #ccc;padding:4px">'
        '<a href="#">Products</a><ul class="sub" style="display:none">'
        '<li><a href="#">Shoes</a></li><li><a href="#">Bags</a></li></ul></li></ul>'
    )
    r = await _find(page, html, "li.mi-p", "Products", description="Products menu item")
    assert r is None


def _numeric(rows):
    body = "".join(
        f'<tr><td style="width:120px">{1000 + i}</td><td style="width:120px">{i}.50</td>'
        '<td><button aria-label="Delete">&#128465;</button></td></tr>'
        for i in range(rows)
    )
    return (
        '<!DOCTYPE html><html><body style="margin:8px">'
        f'<table style="width:600px"><tbody>{body}</tbody></table></body></html>'
    )


async def test_numeric_rows_stay_fast(page):
    # 500 rows of Delete buttons: the one-control count refuses the tbody, and this times that count
    await page.set_content(_numeric(500))
    await page.evaluate("document.querySelector('tbody').scrollIntoView({block: 'center'})")
    data = await page.locator("tbody").evaluate(POINTED_DATA_JS)
    start = time.perf_counter()
    r = await find_inner_control(
        page,
        action="click",
        expected_text="Delete row",
        is_collection=None,
        element_data=data,
        description="table row with Delete",
        row_anchor_text=None,
    )
    elapsed = time.perf_counter() - start
    assert r is None
    assert elapsed < 0.5, f"{elapsed:.2f} s"


async def test_later_control_named_by_one_word(page):
    # only `first(c)` refuses this page: the link is not the row's first text
    html = TABLE.format(
        '<tr class="r-1"><td style="width:400px">Legacy login</td><td style="width:300px">Open</td>'
        f"<td>{REMOVE}</td></tr>"
    )
    r = await _find(
        page, html, "tr.r-1", "Legacy login Open Remove", description="table row for Remove"
    )
    assert r is None


async def test_symbol_name_shown_twice(page):
    # a row showing "→" twice: only the plain-string branch of `count` sees it (no word to count)
    r = await _find(page, MENU.format(inner='→ <a href="#">→</a>'), "li", "→")
    assert r is None


async def test_password_input_beside_the_link_stays_unanswered(page):
    # a control that hides its value still counts as a control, by its presence only
    inner = '<input type="password" value="Products" style="width:60px"> <a href="#">Products</a>'
    assert await _find(page, MENU.format(inner=inner), "li", "Products") is None


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

# one control in the container, and a same-named link outside it so that "a" matches two
LINK_AND_DECOY = MENU.format(inner='<a href="#" id="p">Products</a>') + '<a href="#">Products</a>'


async def test_is_answer_true_for_the_control_false_for_its_container_and_for_two_matches(page):
    r = await _find(page, LINK_AND_DECOY, "li", "Products", description="Products menu item")
    assert r is not None
    assert await r.is_answer(page, "#p") is True
    assert await r.is_answer(page, "li") is False
    assert await r.is_answer(page, "a") is False, "two matches are not an answer"
    assert await r.is_answer(page, "#nothing") is False


async def test_is_answer_false_when_the_page_went_away(page):
    r = await _find(page, LINK_AND_DECOY, "li", "Products", description="Products menu item")
    assert r is not None
    await page.goto("about:blank")
    await page.set_content(LINK_AND_DECOY)
    assert await r.is_answer(page, "#p") is False


async def test_element_data_keys_match_the_indexed_shape(page):
    r = await _find(page, MENU.format(inner='<a href="#">Products</a>'), "li", "Products")
    assert r is not None
    expected = set(_extract_dom_node_attributes(MagicMock())) | {"textContent"}
    assert set(r.element_data) == expected
    assert r.element_data["xpath"].startswith("html[1]/body[1]/ul[1]/li[1]/a[1]")
