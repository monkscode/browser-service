"""
The click-target detector on real Chromium (bs #63).

Each test sets a small page, points at a container the way browser-use would (its absolute
xpath), and asks ``find_inner_control`` for the one control the container wraps. The cases that
must return None are the ones where a re-aim could click a wrong element: a container that is
not a bare wrapper, a word inside a longer word, a hidden or covered control, a container that
is itself clickable, a password input.
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


async def _find(page, html, selector, expected, root=None, **over):
    """Set the page, point at ``selector``, ask for the inner control."""
    await page.set_content(html)
    root = root or page
    data = await root.locator(selector).first.evaluate(POINTED_DATA_JS)
    kwargs = dict(
        action="click",
        expected_text=expected,
        is_collection=None,
        element_data=data,
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


async def test_icon_link_named_by_aria_label(page):
    html = ROW.format(
        w=600, first='<a href="#" aria-label="Edit"><svg width="16" height="16"></svg></a>', rest=""
    )
    r = await _find(page, html, "td", "Edit")
    assert _tag(r) == "a"
    assert r.element_data["ariaLabel"] == "Edit"


@pytest.mark.parametrize(
    "control, expected",
    [
        ('<a href="#" title="Edit"><svg width="16" height="16"></svg></a>', "Edit"),
        ('<a href="#"><img alt="Edit" width="16" height="16"></a>', "Edit"),
        ('<input type="submit" value="Save">', "Save"),
    ],
    ids=["title", "image-alt", "button-value"],
)
async def test_control_named_without_text(page, control, expected):
    html = ROW.format(w=600, first=control, rest="")
    assert await _find(page, html, "td", expected) is not None


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


async def test_symbol_only_name(page):
    # a name with no word to compare: only the exact comparison answers
    html = ROW.format(w=600, first='<a href="#">→</a>', rest="")
    assert _tag(await _find(page, html, "td", "→")) == "a"


async def test_card_with_a_heading_link(page):
    html = (
        '<div style="display:flex;gap:20px"><div class="card c-nike" '
        'style="width:300px;height:220px;border:1px solid #ccc">'
        '<h3><a href="#">Nike Air Max 270</a></h3></div></div>'
    )
    r = await _find(page, html, "div.c-nike", "Nike Air Max 270")
    assert _tag(r) == "a"


async def test_one_cell_row(page):
    html = (
        '<table style="width:900px;border-collapse:collapse"><tbody><tr class="r-a">'
        '<td style="width:900px;border:1px solid #ccc;padding:6px"><a href="#">Alice</a></td>'
        "</tr></tbody></table>"
    )
    assert _tag(await _find(page, html, "tr.r-a", "Alice")) == "a"


# ---------------------------------------------------------------------------
# Returns None: a re-aim could click a wrong element, or there is nothing to re-aim
# ---------------------------------------------------------------------------

TABLE = (
    "<style>td,th{{border:1px solid #ccc;padding:6px}}</style>"
    '<table style="width:900px;border-collapse:collapse"><thead><tr><th>Name</th><th>Info</th>'
    "<th>Actions</th></tr></thead><tbody>{}</tbody></table>"
)
REMOVE = '<a href="#">Remove</a>'
EDIT_ICON = '<button aria-label="Edit">&#9998;</button>'
DELETE_BUTTON = "<button>Delete</button>"
BIN_DELETE = '<button aria-label="Delete">&#128465;</button>'
# The Delete control shown only while the row is hovered, or an icon whose click handler a script
# attaches (bench/private/bs63/build_review: rr8_hoverlib.py, m3_hoverlib.py, m3_pages.py)
TRASH_ICON = (
    '<span class="del" data-who="alice" style="cursor:pointer;font-size:16px">&#128465;</span>'
)
QUIET_ICON = '<span class="del" data-who="alice" style="font-size:16px">&#128465;</span>'
SR_ONLY = (
    "<style>.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;"
    "clip:rect(0,0,0,0);white-space:nowrap;border:0}</style>"
)
# The wrapper of the allowlist traps (bench/private/bs63/build_review/m4_traps.py)
BOX = '<div class="w w-docs" style="width:400px;height:40px;border:1px solid #ccc">{}</div>'
BIN_SVG = "<svg width='16' height='16'><rect width='16' height='16'/></svg>"


def _row(a, b, c):
    return TABLE.format(
        f'<tr class="r-1"><td>{a}</td><td style="width:500px">{b}</td>'
        f'<td style="width:200px">{c}</td></tr>'
    )


def _t1(ctrl, name='<a href="#">Alice</a>', role="Admin", cls="r-a", act=False):
    td = '<td class="act">' if act else "<td>"
    return (
        f'<tr class="{cls}"><td style="width:300px">{name}</td>'
        f'<td style="width:400px">{role}</td>{td}{ctrl}</td></tr>'
    )


def _t1_shadow(mode):
    # rr9_lib.py: a custom element with no role and no cursor, a real button in its shadow root
    return (
        TABLE.format(_t1('<x-del data-who="alice"></x-del>'))
        + "<script>customElements.define('x-del', class extends HTMLElement { constructor() { "
        + f"super(); const r = this.attachShadow({{mode: '{mode}'}}); r.innerHTML = "
        + '\'<button aria-label="Delete" style="cursor:pointer">&#128465;</button>\'; '
        + "} });</script>"
    )


@pytest.mark.parametrize(
    "html, pointed, expected",
    [
        (
            '<div role="group" style="display:flex;width:200px">'
            '<button style="width:60px;margin:0">Express shipping</button>'
            '<button style="width:140px;margin:0">Standard shipping</button></div>',
            "div",
            "Express shipping",
        ),
        (
            MENU.format(inner='<a href="#" id="p">Products</a> <a href="#">Products sale</a>'),
            "li",
            "Products",
        ),
        # the walk stops at the li (role menuitem), and the link is a control below it
        (MENU.format(inner='<a href="#">Products</a>'), "ul", "Products"),
        (
            '<ul class="menu" style="list-style:none;padding:0;width:320px">'
            '<li class="mi-p" style="width:300px;height:30px;border:1px solid #ccc;padding:4px">'
            '<a href="#">Products</a><ul class="sub" style="display:none">'
            '<li><a href="#">Shoes</a></li><li><a href="#">Bags</a></li></ul></li></ul>',
            "li.mi-p",
            "Products",
        ),
        (
            MENU.format(
                inner='<input type="password" value="Products" style="width:60px"> '
                '<a href="#">Products</a>'
            ),
            "li",
            "Products",
        ),
        (
            ROW.format(
                w=400,
                first='<a href="#">Order 1234</a>',
                rest='<td>Shipped</td><td><a href="#">Cancel</a></td>',
            ),
            "tr",
            "Order 1234 Shipped Cancel",
        ),
        (
            ROW.format(
                w=200,
                first="Order 1234",
                rest="<td>Shipped</td><td><a href='#'>Cancel</a></td>",
            ),
            "tr",
            "Order 1234 Shipped Cancel",
        ),
        (
            ROW.format(
                w=300,
                first="Alice",
                rest='<td>alice@x.io</td><td><button aria-label="Delete Alice">x</button></td>',
            ),
            "tr",
            "Alice",
        ),
        (
            ROW.format(
                w=300,
                first="Alice",
                rest='<td>alice@x.io</td><td><input type="checkbox" aria-label="Select Alice"></td>',
            ),
            "tr",
            "Alice",
        ),
        (
            '<table><tr><td style="width:300px"><input value="Alice"></td>'
            '<td style="width:400px">Active</td><td><button>Delete Alice</button></td></tr></table>',
            "tr",
            "Alice",
        ),
        (
            '<table><tr><td style="width:600px">User</td><td><a href="#">Alice</a></td>'
            '<td><input type="password" value="Alice" style="display:none"></td></tr></table>',
            "tr",
            "Alice",
        ),
        # the first control is the one the step means, and a multi-cell row still stays as today
        (
            TABLE.format(
                '<tr class="r-1"><td><a href="#">&lt;caption&gt;</a></td>'
                '<td style="width:500px">Defines a table caption</td>'
                '<td style="width:200px"></td></tr>'
            ),
            "tr.r-1",
            "<caption>",
        ),
        (_row("<button>Action</button>", "Alice", "Shipped"), "tr.r-1", "Action Alice Shipped"),
        (
            f'<div class="row" style="width:900px"><div class="cells" style="display:flex">'
            f'<div>{REMOVE}</div><div style="width:500px">Remove legacy login</div>'
            '<div style="width:200px">Open</div></div></div>',
            "div.row",
            "Remove Remove legacy login Open",
        ),
        (
            f'<table style="width:900px"><tbody><tr><td>{REMOVE}</td><td style="width:500px">'
            'Remove legacy login</td><td style="width:200px">Open</td></tr></tbody></table>',
            "tbody",
            "Remove Remove legacy login Open",
        ),
        (
            _row(f"{EDIT_ICON} {DELETE_BUTTON}", "Edit rights", "Allowed"),
            "tr.r-1",
            "Delete Edit rights Allowed",
        ),
        (
            TABLE.format(
                f'<tr class="r-1"><td>{EDIT_ICON}</td><td>{DELETE_BUTTON}</td>'
                '<td style="width:500px">Edit rights</td><td style="width:200px">Allowed</td></tr>'
            ),
            "tr.r-1",
            "Delete Edit rights Allowed",
        ),
        # the expected text names the row, the step means its other control: never the Alice link
        (TABLE.format(_t1(BIN_DELETE)), "tr.r-a", "Alice"),
        (
            TABLE.format(_t1('<button><img alt="Delete" width="16" height="16"></button>')),
            "tr.r-a",
            "Alice",
        ),
        (
            TABLE.format(
                _t1(BIN_DELETE)
                + _t1(BIN_DELETE, name='<a href="#">Bob</a>', role="User", cls="r-b")
            ),
            "tr.r-a",
            "Alice",
        ),
        (
            "<style>.act{opacity:0} tr:hover .act{opacity:1}</style>"
            + TABLE.format(_t1(BIN_DELETE, act=True)),
            "tr.r-a",
            "Alice",
        ),
        (
            "<style>.act{visibility:hidden} tr:hover .act{visibility:visible}</style>"
            + TABLE.format(_t1(BIN_DELETE, act=True)),
            "tr.r-a",
            "Alice",
        ),
        (
            "<style>.act button{display:none} tr:hover .act button{display:inline}</style>"
            + TABLE.format(_t1(BIN_DELETE, act=True)),
            "tr.r-a",
            "Alice",
        ),
        (TABLE.format(_t1(TRASH_ICON)), "tr.r-a", "Alice"),
        (
            TABLE.format(
                _t1(
                    '<svg class="del" data-who="alice" width="16" height="16" '
                    'style="cursor:pointer"><rect width="16" height="16"/></svg>'
                )
            ),
            "tr.r-a",
            "Alice",
        ),
        (
            "<style>.act .del{display:none} tr:hover .act .del{display:inline}</style>"
            + TABLE.format(_t1(TRASH_ICON, act=True)),
            "tr.r-a",
            "Alice",
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
            "tr.r-a",
            "Alice",
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
            "tr.r-a",
            "Alice",
        ),
        # the same row with a Delete control no count can see (rr9_lib.py, m3_pages.py,
        # m3_hoverlib.py): the reason the rule became an allowlist
        (_t1_shadow("open"), "tr.r-a", "Alice"),
        (_t1_shadow("closed"), "tr.r-a", "Alice"),
        (
            TABLE.format(
                _t1(
                    '<span class="del" tabindex="0" aria-label="Delete" data-who="alice" '
                    'style="font-size:16px">&#128465;</span>'
                )
            ),
            "tr.r-a",
            "Alice",
        ),
        (
            TABLE.format(_t1('<label for="del-alice">Delete</label>'))
            + '<input type="checkbox" id="del-alice" style="display:none" data-who="alice">',
            "tr.r-a",
            "Alice",
        ),
        (
            "<style>td.act::before{content:'\\1F5D1';cursor:pointer;font-size:16px}</style>"
            + TABLE.format(_t1("", act=True)),
            "tr.r-a",
            "Alice",
        ),
        (TABLE.format(_t1(QUIET_ICON)), "tr.r-a", "Alice"),
        (
            "<style>.del:hover{cursor:pointer}</style>" + TABLE.format(_t1(QUIET_ICON, act=True)),
            "tr.r-a",
            "Alice",
        ),
        (
            "<style>.act{opacity:0} tr:hover .act{opacity:1} .del:hover{cursor:pointer}</style>"
            + TABLE.format(_t1(QUIET_ICON, act=True)),
            "tr.r-a",
            "Alice",
        ),
        # a closed root reads shadowRoot null: only the hyphen in the tag name refuses the host
        (
            BOX.format('<x-it data-who="alice"><a href="#">Alice</a></x-it>')
            + "<script>customElements.define('x-it', class extends HTMLElement { "
            "connectedCallback() { const r = this.attachShadow({mode: 'closed'}); r.innerHTML = "
            f'`<slot></slot> <button aria-label="Delete" style="cursor:pointer">{BIN_SVG}'
            "</button>`; } });</script>",
            "div.w-docs",
            "Alice",
        ),
        (
            BOX.format(f'<a href="#">Alice <button aria-label="Delete">{BIN_SVG}</button></a>'),
            "div.w-docs",
            "Alice",
        ),
        # the expected text holds the wrapped name, and the real button sits elsewhere
        (
            BOX.format("<button>Save</button>") + '<button class="draft">Save as draft</button>',
            "div.w-docs",
            "Save as draft",
        ),
        (
            ROW.format(
                w=600,
                first='<a href="#" aria-label="Alice" style="display:inline-block;width:120px">'
                '<input type="password" value="Alice" style="width:100px"></a>',
                rest="",
            ),
            "td",
            "Alice",
        ),
        # a plain div hosts the shadow root: its tag name has no hyphen, only shadowRoot refuses it
        (
            ROW.format(w=600, first='<div id="host"><a href="#">Alice</a></div>', rest="")
            + "<script>document.getElementById('host').attachShadow({mode: 'open'}).innerHTML = "
            "'<slot></slot> <button aria-label=\"Delete\">&#128465;</button>'</script>",
            "td",
            "Alice",
        ),
        (
            '<p style="width:600px">Read the <a href="#">docs</a> now</p>',
            "p",
            "docs",
        ),
    ],
    ids=[
        "two-buttons",
        "two-links",
        "role-on-the-item",
        "hidden-submenu",
        "password-beside-the-link",
        "name-link-and-action-link",
        "text-then-action-link",
        "text-then-action-button",
        "text-then-checkbox",
        "name-in-an-input-value",
        "hidden-password-input",
        "name-link-first",
        "button-first",
        "nested-cells",
        "tbody-one-row",
        "two-buttons-in-a-cell",
        "two-buttons-in-two-cells",
        "t1_aria",
        "t1_img_alt",
        "t1_two_rows",
        "rr8_hover_opacity",
        "rr8_hover_visibility",
        "rr8_hover_display",
        "rr8_icon_span",
        "rr8_icon_svg",
        "a_display",
        "a_i_opacity",
        "m3_sronly_select",
        "rr9_shadow_open",
        "rr9_shadow_closed",
        "rr9_tabindex_icon",
        "rr9_label_for",
        "rr9_pseudo_icon",
        "m3_icon_nocursor",
        "m3h_b_hover_cursor",
        "m3h_c_hidden_hover_cursor",
        "m4_closed_slot",
        "m4_control_in_link",
        "m4_part_of_expected",
        "password-input-below-the-link",
        "shadow-root-on-a-div",
        "text-beside-the-control",
    ],
)
async def test_a_non_wrapper_stays_unanswered(page, html, pointed, expected):
    assert await _find(page, html, pointed, expected) is None


async def test_row_reported_whole_keeps_todays_answer(page):
    # two cells: not a bare wrapper, though the link is what the agent meant
    html = ROW.format(
        w=300,
        first='<a href="#">&lt;caption&gt;</a>',
        rest="<td>Defines a table caption</td>",
    )
    assert await _find(page, html, "tr", "<caption> Defines a table caption") is None


async def test_block_link_fills_the_container(page):
    r = await _find(
        page,
        MENU.format(inner='<a href="#" style="display:block">Products</a>'),
        "li",
        "Products",
    )
    assert r is None


async def test_expander_named_after_the_item(page):
    html = (
        '<ul style="width:300px"><li role="treeitem" style="width:300px">Reports '
        '<button aria-label="Expand Reports">+</button></li></ul>'
    )
    assert await _find(page, html, "li", "Reports") is None


async def test_expected_inside_a_longer_word(page):
    html = ROW.format(w=600, first='<a href="#">Credit memo</a>', rest="")
    assert await _find(page, html, "td", "Edit") is None


async def test_word_inside_a_longer_word(page):
    html = ROW.format(w=600, first='<a href="#">Cancelled order 77</a>', rest="")
    assert await _find(page, html, "td", "Cancel") is None


async def test_devanagari_word_stays_whole(page):
    rad = "रद्द"
    ridi = rad + "ी"
    # only `\p{M}` keeps the longer word whole: without it both end in the same letters
    html = ROW.format(w=600, first=f'<a href="#">{ridi} scrap</a>', rest="")
    assert await _find(page, html, "td", rad) is None


async def test_symbols_never_match_as_blank_words(page):
    html = ROW.format(w=600, first='<a href="#">←</a>', rest="")
    assert await _find(page, html, "td", "→") is None


async def test_hidden_control(page):
    # opacity hides the link from the eye only: the hit test and the text still see it
    inner = '<span style="opacity:0"><a href="#">Products</a></span>'
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
    # the cover sits outside the wrapper: only the hit test at the control's centre sees it
    html = (
        '<div style="position:relative;width:300px;height:100px">'
        '<div class="w" style="height:100px"><a href="#">Order details</a></div>'
        '<div style="position:absolute;inset:0"></div></div>'
    )
    assert await _find(page, html, "div.w", "Order details") is None


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


async def test_symbol_name_shown_twice(page):
    # the row shows "→" beside the link: the link's exact name does not excuse the text
    r = await _find(page, MENU.format(inner='→ <a href="#">→</a>'), "li", "→")
    assert r is None


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
    r = await _find(page, LINK_AND_DECOY, "li", "Products")
    assert r is not None
    assert await r.is_answer(page, "#p") is True
    assert await r.is_answer(page, "li") is False
    assert await r.is_answer(page, "a") is False, "two matches are not an answer"
    assert await r.is_answer(page, "#nothing") is False


async def test_is_answer_false_when_the_page_went_away(page):
    r = await _find(page, LINK_AND_DECOY, "li", "Products")
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
