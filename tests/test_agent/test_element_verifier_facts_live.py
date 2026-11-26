"""Element facts read from a real chromium DOM (page.set_content). Synthetic HTML, no customer data.

pytest tests/test_agent/test_element_verifier_facts_live.py -m integration
"""

import pytest

from browser_service.agent.element_verifier import read_element_facts

TABLE = """
<main aria-label="Accounts">
<table>
  <thead><tr><th>Name</th><th>Balance</th><th>Action</th></tr></thead>
  <tbody><tr><td>Cierra Vega</td><td>41.50</td><td><a href="#">View</a></td></tr></tbody>
</table>
</main>
"""

SECRETS = """
<table><tbody>
<tr id="r1"><td>Alice</td>
 <td><input id="pw" type="password" value="hunter2"><span nlrf-secret>topsecret</span><span>visible</span></td></tr>
</tbody></table>
<div id="box">Note: <div contenteditable nlrf-secret>typed-secret</div> end</div>
<div id="own" nlrf-secret>own-secret</div>
"""

LABELS = """
<nav aria-label="Primary"><label for="q">Search products</label><input id="q" name="query" type="text"
 placeholder="Type here" title="finder"></nav>
"""

SHADOW = """
<main><div id="host"></div></main>
<script>
  document.getElementById('host').attachShadow({mode: 'open'}).innerHTML = '<button id="inner">Go</button>';
</script>
"""


# The three icon tiles of the stand-in "image-only UI" page: a div with only a style attribute, an aria-hidden svg
# (bolt / star / heart, the stand-in's own markup), inside <main>.
_SVG = (
    '<svg width="48" height="48" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
    '<path d="{d}" fill="#444"/></svg>'
)
TILES = (
    "<main>"
    + "".join(
        f'<div style="display:flex;width:88px;height:88px">{_SVG.format(d=d)}</div>'
        for d in (
            "M13 2L4 14h6l-1 8 9-12h-6l1-8z",
            "M12 2l2.9 6.3 6.9.8-5.1 4.7 1.4 6.8L12 17.2l-6.1 3.4 1.4-6.8L2.2 9.1l6.9-.8L12 2z",
            "M12 21s-7.5-4.8-10-9.3C.4 8.6 2.3 5 5.7 5c2 0 3.4 1.1 4.3 2.5h4C14.9 6.1 16.3 5 "
            "18.3 5c3.4 0 5.3 3.6 3.7 6.7C19.5 16.2 12 21 12 21z",
        )
    )
    + "</main>"
)

IMG_TILES = """
<div id="a"><img alt="Cart" src="https://cdn.example.test/img/cart.png?v=3#top"></div>
<div id="b"><img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciLz4="></div>
<div id="c"><img alt="" src="/x/y.gif"></div>
"""

SECRET_GRAPHICS = """
<div id="wrap"><div nlrf-secret><svg viewBox="0 0 1 1"><title>topsecret</title><path d="M9 topsecret"/></svg>
  <img alt="topsecret" src="topsecret.png"></div></div>
<div id="own" nlrf-secret><svg viewBox="0 0 1 1"><path d="M9 topsecret"/></svg></div>
<div id="open"><span nlrf-secret><svg viewBox="0 0 1 1"><path d="M9 topsecret"/></svg></span>
  <svg viewBox="0 0 2 2"><path d="M5 visible"/></svg></div>
"""


@pytest.fixture
async def page():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context()
        pg = await ctx.new_page()
        try:
            yield pg
        finally:
            await ctx.close()
            await browser.close()


@pytest.mark.integration
@pytest.mark.fixture
class TestElementFactsOnARealDom:
    async def test_table_cell_has_column_position_and_header(self, page):
        await page.set_content(TABLE)
        facts = await read_element_facts(page.locator("td").nth(1))
        assert facts["tag"] == "td"
        assert facts["column_position"] == "2nd from the left"
        assert facts["column_header"] == "Balance"
        assert facts["row_text"] == "Cierra Vega 41.50 View"
        assert facts["landmark"] == "main: Accounts"
        link = await read_element_facts(page.locator("a"))
        assert link["column_position"] == "3rd from the left"
        assert link["column_header"] == "Action"

    async def test_row_text_drops_secret_descendants_and_passwords(self, page):
        await page.set_content(SECRETS)
        facts = await read_element_facts(page.locator("#pw"))
        assert facts["row_text"] == "Alice visible"
        cell = await read_element_facts(page.locator("td").nth(1))
        assert cell["text"] == "visible"
        everything = str(facts) + str(cell)
        assert "secret" not in everything
        assert "hunter2" not in everything

    async def test_label_and_landmark_are_read(self, page):
        await page.set_content(LABELS)
        facts = await read_element_facts(page.locator("#q"))
        assert facts["label"] == "Search products"
        assert facts["landmark"] == "nav: Primary"
        assert (facts["name"], facts["type"]) == ("query", "text")
        assert (facts["placeholder"], facts["title"]) == ("Type here", "finder")
        assert "text" not in facts and "role" not in facts

    async def test_id_is_read_and_omitted_when_empty(self, page):
        await page.set_content('<button id=" x ">Go</button><button>Stop</button>')
        assert (await read_element_facts(page.locator("button").nth(0)))["id"] == "x"
        assert "id" not in await read_element_facts(page.locator("button").nth(1))

    async def test_element_text_drops_secret_descendant(self, page):
        await page.set_content(SECRETS)
        facts = await read_element_facts(page.locator("#box"))
        assert facts["text"] == "Note: end"
        own = await read_element_facts(page.locator("#own"))
        assert "text" not in own and "secret" not in str(own)

    async def test_shadow_root_reads_empty_landmark(self, page):
        await page.set_content(SHADOW)
        facts = await read_element_facts(page.locator("#host button"))
        assert facts["tag"] == "button" and facts["text"] == "Go"
        # shortcut: closest() stops at the shadow boundary, so the <main> around the host is not seen.
        assert "landmark" not in facts

    async def test_failure_is_none(self, page):
        await page.set_content(TABLE)
        assert await read_element_facts(page.locator("#nope"), timeout_ms=200) is None

    async def test_wordless_svg_tile_reads_its_graphic(self, page):
        await page.set_content(TILES)
        tiles = page.locator("main > div")
        graphics = [(await read_element_facts(tiles.nth(i)))["graphic"] for i in range(3)]
        # exactly the strings Task 1 measured live on the stand-in page
        assert graphics == [
            "svg viewBox=0 0 24 24 path=M13 2L4 14h6l-1 8 9-12h-6l1-8z",
            "svg viewBox=0 0 24 24 path=M12 2l2.9 6.3 6.9.8-5.1 4.7 1.4 6.8L12 17.2l-6.1 3.4 1.4-6.8",
            "svg viewBox=0 0 24 24 path=M12 21s-7.5-4.8-10-9.3C.4 8.6 2.3 5 5.7 5c2 0 3.4 1.1 4.3 2.",
        ]

    async def test_element_with_text_gets_exactly_todays_facts(self, page):
        await page.set_content(
            '<main><button id="b"><svg viewBox="0 0 1 1"><path d="M1 1"/></svg>Go</button></main>'
        )
        facts = await read_element_facts(page.locator("#b"))
        assert list(facts.items()) == [
            ("tag", "button"),
            ("id", "b"),
            ("text", "Go"),
            ("landmark", "main"),
        ]

    async def test_labelled_element_gets_no_graphic(self, page):
        await page.set_content(
            '<button aria-label="Close"><svg viewBox="0 0 1 1"><path d="M1 1"/></svg></button>'
        )
        facts = await read_element_facts(page.locator("button"))
        assert facts == {"tag": "button", "aria_label": "Close"}

    async def test_svg_title_speaks_through_text_not_graphic(self, page):
        await page.set_content(
            '<div id="t"><svg viewBox="0 0 1 1"><title>Home</title><path d="M1 1"/></svg></div>'
        )
        facts = await read_element_facts(page.locator("#t"))
        assert facts["text"] == "Home" and "graphic" not in facts

    async def test_img_tile_reads_alt_and_file_name(self, page):
        await page.set_content(IMG_TILES)
        a = await read_element_facts(page.locator("#a"))
        assert a["graphic"] == "img alt=Cart src=cart.png"
        b = await read_element_facts(page.locator("#b"))
        assert b["graphic"] == "img src=data:image/svg+xml"
        c = await read_element_facts(page.locator("#c"))
        assert c["graphic"] == "img src=y.gif"

    async def test_tile_with_no_picture_has_no_graphic(self, page):
        await page.set_content('<div id="e"><span></span></div>')
        assert "graphic" not in await read_element_facts(page.locator("#e"))

    async def test_secret_graphics_are_never_read(self, page):
        await page.set_content(SECRET_GRAPHICS)
        wrap = await read_element_facts(page.locator("#wrap"))
        own = await read_element_facts(page.locator("#own"))
        inner = await read_element_facts(page.locator("#own svg"))
        assert "graphic" not in wrap and "graphic" not in own and "graphic" not in inner
        assert "secret" not in str(wrap) + str(own) + str(inner)
        # a marked sibling does not hide the visible picture next to it
        open_ = await read_element_facts(page.locator("#open"))
        assert open_["graphic"] == "svg viewBox=0 0 2 2 path=M5 visible"

    async def test_secret_inside_the_svg_is_never_read(self, page):
        await page.set_content(
            '<div id="x"><svg viewBox="0 0 1 1"><g nlrf-secret><title>topsecret</title>'
            '<path d="M9 9"/></g></svg></div>'
        )
        facts = await read_element_facts(page.locator("#x"))
        assert "topsecret" not in str(facts) and "M9 9" not in str(facts)
        assert facts["graphic"] == "svg viewBox=0 0 1 1"

    async def test_a_value_ending_in_equals_is_kept(self, page):
        await page.set_content('<div id="d"><img alt="a=" src="/x/y.gif"></div>')
        assert (await read_element_facts(page.locator("#d")))["graphic"] == "img alt=a= src=y.gif"

    async def test_title_attribute_or_label_text_means_no_graphic(self, page):
        await page.set_content(
            '<div id="t" title="Tip"><svg viewBox="0 0 1 1"><path d="M1 1"/></svg></div>'
            '<label>Name<svg viewBox="0 0 1 1"><path d="M1 1"/></svg><input id="i"></label>'
        )
        tip = await read_element_facts(page.locator("#t"))
        assert list(tip.items()) == [("tag", "div"), ("id", "t"), ("title", "Tip")]
        field = await read_element_facts(page.locator("#i"))
        assert field == {"tag": "input", "id": "i", "label": "Name"}

    async def test_long_values_are_cut(self, page):
        d = "M" + "1" * 80
        await page.set_content(
            f'<div id="s"><svg viewBox="0 0 1 1"><path d="{d}"/></svg></div>'
            f'<div id="i"><img alt="{"a" * 80}" src="/{"f" * 80}.png"></div>'
        )
        assert (await read_element_facts(page.locator("#s")))["graphic"] == (
            "svg viewBox=0 0 1 1 path=" + d[:60]
        )
        assert (await read_element_facts(page.locator("#i")))["graphic"] == (
            "img alt=" + "a" * 60 + " src=" + "f" * 60
        )
