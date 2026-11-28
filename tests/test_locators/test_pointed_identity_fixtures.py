"""The pointed target and the box honesty check on real Chromium (point-then-prove D1).

The target is the indexed node (its browser-use xpath) or the element under the agent's point,
never the smallest indexed box around the point. ``box_is_hit`` tells such a box from the hit:
the box must be the hit itself or the nearest control at or above it. Inline HTML, 1280x900.
"""

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.pointed_identity import (
    box_is_hit,
    count_inner_controls,
    pointed_target,
)

pytestmark = pytest.mark.integration

# browser-use's own xpath shape: no leading slash, an index only where a tag repeats.
XPATH_JS = """(el) => {
    const parts = [];
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
        const same = n.parentElement
            ? Array.from(n.parentElement.children).filter(s => s.tagName === n.tagName) : [n];
        const tag = n.tagName.toLowerCase();
        parts.unshift(same.length > 1 ? tag + '[' + (same.indexOf(n) + 1) + ']' : tag);
    }
    return parts.join('/');
}"""

TODOMVC = """
<section id="root" class="todoapp" style="width:550px;margin:40px auto">
  <header><h1>todos</h1><input class="new-todo" placeholder="What needs to be done?"></header>
  <ul class="todo-list" style="list-style:none">
    <li><div class="view"><input class="toggle" type="checkbox"
        style="width:40px;height:40px"><label>Buy milk</label><button class="destroy">x</button></div></li>
    <li><div class="view"><input class="toggle" type="checkbox"
        style="width:40px;height:40px"><label>Walk the dog</label><button class="destroy">x</button></div></li>
  </ul>
</section>"""

ICON_BUTTON = (
    '<button id="del" style="position:absolute;left:100px;top:100px;padding:20px">'
    '<svg id="icon" width="40" height="40"><title>Delete</title></svg></button>'
)


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


async def _centre(locator):
    box = await locator.bounding_box()
    return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


async def _id(handle):
    assert handle is not None
    return await handle.evaluate("el => el.id")


async def _data(locator):
    return {"xpath": await locator.evaluate(XPATH_JS)}


async def test_target_by_index_xpath(page):
    await page.set_content('<button id="a">A</button><button id="b">B</button>')
    data = await _data(page.locator("#b"))
    point = await _centre(page.locator("#a"))  # the point says A; the index says B

    assert await _id(await pointed_target(page, None, data, True, point)) == "b"


async def test_target_by_point_returns_innermost(page):
    await page.set_content(ICON_BUTTON)
    hit = await pointed_target(page, None, None, False, await _centre(page.locator("#icon")))

    assert await _id(hit) == "icon"


async def test_target_point_on_body_is_none(page):
    await page.set_content(ICON_BUTTON)

    assert await pointed_target(page, None, None, False, (900, 700)) is None


async def test_target_in_iframe_uses_relative_point(page):
    await page.set_content(
        '<div style="height:300px"></div>'
        '<iframe id="f" style="margin-left:400px;width:400px;height:200px" '
        "srcdoc=\"<button id='inner' style='margin:60px'>Go</button>\"></iframe>"
    )
    frame = page.frame_locator('iframe[id="f"]')
    inner = frame.locator("#inner")
    await inner.wait_for()
    point = await _centre(inner)  # page coordinates; the same point in the main frame is the iframe

    hit = await pointed_target(page, 'iframe[id="f"]', None, False, point)
    assert await _id(hit) == "inner"
    assert await pointed_target(page, None, None, False, point) is None  # the iframe element itself
    # the box check's path inside a frame: the xpath handle and the hit compare as one element
    box = await pointed_target(page, 'iframe[id="f"]', {"xpath": "html/body/button"}, True, None)
    assert await box_is_hit(box, hit) is True


async def test_target_in_open_shadow_root(page):
    await page.set_content(
        '<div id="host" style="margin:100px"></div><script>'
        "document.getElementById('host').attachShadow({mode: 'open'}).innerHTML ="
        ' \'<button id="inner" style="padding:20px">Go</button>\';</script>'
    )
    point = await _centre(page.locator("#inner"))  # Playwright's css pierces open shadow roots
    hit = await pointed_target(page, None, None, False, point)

    assert await _id(hit) == "inner"
    assert await hit.evaluate("el => el.getRootNode() instanceof ShadowRoot")


async def test_xpath_unreachable_falls_back_to_point(page):
    await page.set_content(ICON_BUTTON)
    box = await page.locator("#del").bounding_box()
    point = (box["x"] + 5, box["y"] + 5)  # the button's padding, off the svg
    hit = await pointed_target(page, None, {"xpath": "html/body/div[9]/span"}, True, point)

    assert await _id(hit) == "del"


async def test_target_by_index_on_scrolled_page(page):
    place = "position:absolute;left:100px;width:200px;height:60px"
    await page.set_content(
        f'<button id="far" style="{place};top:100px">Far</button>'
        f'<button id="top" style="{place};top:700px">Top</button>'
        '<div style="height:3000px"></div>'
    )
    data = await _data(page.locator("#far"))
    before = await _centre(page.locator("#far"))
    await page.evaluate("window.scrollTo(0, 600)")
    # the indexed node's old screen point now lies on another element
    assert await _id(await pointed_target(page, None, None, False, before)) == "top"

    assert await _id(await pointed_target(page, None, data, True, before)) == "far"


async def test_box_is_hit_same_and_control_above(page):
    await page.set_content(ICON_BUTTON)
    button = await page.locator("#del").element_handle()
    hit = await pointed_target(page, None, None, False, await _centre(page.locator("#icon")))

    assert await box_is_hit(button, hit) is True  # the svg's nearest control is the button
    assert await box_is_hit(hit, hit) is True
    assert await box_is_hit(hit, button) is False  # the button's nearest control is itself


async def test_box_is_hit_false_for_container(page):
    await page.set_content(TODOMVC)
    toggle = page.locator(".toggle").first
    box = await pointed_target(page, None, await _data(page.locator("#root")), True, None)
    hit = await pointed_target(page, None, None, False, await _centre(toggle))

    assert await hit.evaluate("el => el.className") == "toggle"
    assert await box_is_hit(box, hit) is False


# D6: the controls a pick holds (the element verifier's silence rule ships only at 0).


async def test_count_inner_controls_root_container(page):
    await page.set_content(TODOMVC)
    # the new-todo input, then a checkbox, a label and a button in each of the 2 rows
    assert await count_inner_controls(page.locator("#root")) == 7


async def test_count_inner_controls_label_wrapping_its_checkbox_is_zero(page):
    await page.set_content('<label id="agree"><input type="checkbox"> I agree</label>')
    assert await count_inner_controls(page.locator("#agree")) == 0


async def test_count_inner_controls_label_holding_another_control_counts_it(page):
    await page.set_content(
        '<label id="agree"><input type="checkbox"> I agree to <a href="#t">the terms</a></label>'
    )
    assert await count_inner_controls(page.locator("#agree")) == 1


async def test_count_inner_controls_icon_button_by_element_handle(page):
    await page.set_content(ICON_BUTTON)
    assert await count_inner_controls(await page.locator("#del").element_handle()) == 0


async def test_count_inner_controls_reads_roles_and_contenteditable(page):
    await page.set_content(
        '<div id="w"><div role="button">x</div><p contenteditable>y</p>'
        '<p contenteditable="false">z</p><span>w</span></div>'
    )
    assert await count_inner_controls(page.locator("#w")) == 2


async def test_count_inner_controls_crosses_open_shadow_roots(page):
    await page.set_content('<div id="host"></div>')
    await page.evaluate(
        """() => document.getElementById('host').attachShadow({mode: 'open'}).innerHTML =
            '<span><button>Save</button></span><div role="tab">A</div>'"""
    )
    assert await count_inner_controls(page.locator("#host")) == 2
