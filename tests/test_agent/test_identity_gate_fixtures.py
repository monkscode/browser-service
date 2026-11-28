"""Proven on real Chromium (point-then-prove D2): a performed step's locator lands on the target.

``proven`` strips the frame hop, counts the locator in the frame, and accepts one match that is the
target itself (``same``), the nearest control at or above it (``control-above``), or the only
control of its kind beside it (``only-control``: a label and its control, or the one control of
its tag / type / role in a common ancestor at most 3 levels above both). A collection is proven
when the target is a match or holds one. The target is built the way the gate builds it
(``pointed_target``). Inline HTML, 1280x900.
"""

import logging

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import find_unique_locator_action
from browser_service.locators.frame_locator import strip_frame
from browser_service.locators.handlers.checkbox import resolve_hidden_input_proxy
from browser_service.locators.pointed_identity import pointed_target, proven
from tests.test_locators.test_pointed_identity_fixtures import TODOMVC, _centre, _data

pytestmark = pytest.mark.integration

FRAME = 'iframe[id="f"]'
FIRST_TOGGLE = "li:first-child .toggle"


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


async def _judge(page, locator, point=None, *, frame=None, data=None, collection=False):
    """``proven`` for ``locator`` against the target the gate would build."""
    target = await pointed_target(page, frame, data, data is not None, point)
    assert target is not None, "the fixture's point or index must name an element"
    return await proven(page, frame, locator, target, collection)


async def _at(page, selector, frame=None):
    root = page.frame_locator(frame) if frame else page
    element = root.locator(selector)
    await element.wait_for()
    return await _centre(element)


async def _named(page, selector, name):
    """How a refusal names the element ``selector`` finds: ``name`` and its box centre."""
    x, y = await _at(page, selector)
    return f"{name} at ({round(x)}, {round(y)})"


async def test_sibling_label_checkbox_point_on_box(page):
    await page.set_content(TODOMVC)

    assert await _judge(page, FIRST_TOGGLE, await _at(page, FIRST_TOGGLE)) == (True, "same")


async def test_sibling_label_checkbox_point_on_label(page):
    """TodoMVC's <label> has no `for`: the checkbox is the only one in their div.view."""
    await page.set_content(TODOMVC)
    point = await _at(page, "li:first-child label")

    assert await _judge(page, FIRST_TOGGLE, point) == (True, "only-control")


async def test_wrapping_label(page):
    await page.set_content(
        '<div class="row"><label id="wrap" style="display:inline-block;padding:10px">'
        '<input id="remember" type="checkbox"> Remember me on this computer</label></div>'
    )
    box = await page.locator("#wrap").bounding_box()
    point = (box["x"] + box["width"] - 15, box["y"] + box["height"] / 2)  # the label's text

    assert await _judge(page, "#remember", point) == (True, "only-control")
    assert await _judge(page, "#wrap", await _at(page, "#remember")) == (True, "only-control")


async def test_label_for_control_far_away_proven(page):
    """A form with labels in their own column: no common ancestor within 3 levels."""
    await page.set_content(
        '<table><tr><td><label for="email" id="lab">Email</label></td></tr>'
        '<tr><td><div><div><div><input id="email"></div></div></div></td></tr></table>'
    )

    assert await _judge(page, "#email", await _at(page, "#lab")) == (True, "only-control")


async def test_app_root_candidate_refused(page):
    await page.set_content(TODOMVC)
    ok, reason = await _judge(page, "id=root", await _at(page, FIRST_TOGGLE))

    assert ok is False
    found = await _named(page, "#root", "<section id=root class=todoapp>")
    pointed = await _named(page, FIRST_TOGGLE, "<input type=checkbox class=toggle>")
    assert reason == f"finds {found}; the element you pointed at is {pointed}"


async def test_indexed_icon_tile_kept(page):
    await page.set_content(
        '<div class="grid">'
        + "".join(
            f'<div class="tile" id="t{i}" style="display:inline-block;width:80px;height:80px">'
            f"<span>{i}</span></div>"
            for i in range(1, 4)
        )
        + "</div>"
    )
    data = await _data(page.locator("#t2"))

    assert await _judge(page, ".tile >> nth=1", data=data) == (True, "same")


async def test_indexless_svg_hit_in_button_kept(page):
    await page.set_content(
        '<button id="del" style="padding:20px">'
        '<svg id="icon" width="40" height="40"><path d="M0 0H40V40H0z"></path></svg></button>'
    )

    assert await _judge(page, "#del", await _at(page, "#icon")) == (True, "control-above")


async def test_indexless_div_tile_refused(page):
    """Stated limit: a listener-only <div> is no control, so a hit on its inner span is not it."""
    await page.set_content(
        '<div class="tile" id="t1" onclick="void 0" style="width:120px;padding:20px">'
        "<span>Icons</span></div>"
    )
    ok, reason = await _judge(page, "#t1", await _at(page, "#t1 span"))

    assert ok is False
    found = await _named(page, "#t1", "<div id=t1 class=tile>")
    assert reason == f"finds {found}; the element you pointed at is " + await _named(
        page, "#t1 span", "<span>"
    )


async def test_flatpickr_button_to_input(page):
    await page.set_content(
        '<div class="flatpickr"><input class="flatpickr-input" type="text" placeholder="Pick">'
        '<a class="input-button" title="toggle" style="display:inline-block;padding:8px">'
        '<i class="icon-calendar" style="display:inline-block;width:20px;height:20px"></i></a>'
        "</div>"
    )
    point = await _at(page, ".icon-calendar")

    assert await _judge(page, ".flatpickr-input", point) == (True, "only-control")


async def test_same_control_other_item_refused(page):
    await page.set_content(TODOMVC)
    ok, reason = await _judge(page, FIRST_TOGGLE, await _at(page, "li:nth-child(2) label"))

    assert ok is False
    assert reason.endswith(
        "the element you pointed at is " + await _named(page, "li:nth-child(2) label", "<label>")
    )


async def test_two_links_are_told_apart_by_position(page):
    """Two bare <a>: the refusal names each by its tag and its box centre, never its text."""
    await page.set_content(
        '<table><tr><td><a href="#">edit</a></td></tr><tr><td><a href="#">edit</a></td></tr>'
        "</table>"
    )
    first, second = "tr:nth-child(1) a", "tr:nth-child(2) a"
    ok, reason = await _judge(page, first, await _at(page, second))

    assert ok is False
    found, pointed = await _named(page, first, "<a>"), await _named(page, second, "<a>")
    assert found != pointed
    assert reason == f"finds {found}; the element you pointed at is {pointed}"


async def test_two_checkboxes_in_one_item_refused(page):
    await page.set_content(
        '<ul><li><div class="view"><input type="checkbox" class="toggle">'
        '<input type="checkbox" class="star" style="width:30px;height:30px">'
        "<label>Buy milk</label></div></li></ul>"
    )

    assert (await _judge(page, ".toggle", await _at(page, "label")))[0] is False
    assert (await _judge(page, ".toggle", await _at(page, ".star")))[0] is False
    assert await _judge(page, ".star", await _at(page, ".star")) == (True, "same")


async def test_overlay_covering_target_refused(page):
    await page.set_content(
        '<div id="app"><div class="card"><button id="buy">Buy</button></div></div>'
        '<div id="cookie" style="position:fixed;inset:0;background:rgba(0,0,0,.3)">'
        "We use cookies</div>"
    )
    ok, reason = await _judge(page, "#buy", await _centre(page.locator("#buy")))

    assert ok is False
    found = await _named(page, "#buy", "<button id=buy>")
    pointed = await _named(page, "#cookie", "<div id=cookie>")
    assert reason == f"finds {found}; the element you pointed at is {pointed}"


async def test_detached_target_unreadable(page):
    """A React re-render between the hit-test and the check: the locator finds the new node."""
    await page.set_content(TODOMVC)
    target = await pointed_target(page, None, None, False, await _at(page, FIRST_TOGGLE))
    await page.evaluate(
        "() => { const li = document.querySelector('li'); li.replaceWith(li.cloneNode(true)); }"
    )

    assert await proven(page, None, FIRST_TOGGLE, target, False) == (False, "unreadable")


async def test_frame_element_proven_in_frame(page):
    await page.set_content(
        '<div style="height:300px"></div>'
        '<iframe id="f" style="margin-left:400px;width:400px;height:200px" '
        "srcdoc=\"<button id='inner' style='margin:60px'>Go</button>\"></iframe>"
    )
    point = await _at(page, "#inner", FRAME)

    assert await _judge(page, f"{FRAME} >>> id=inner", point, frame=FRAME) == (True, "same")


async def test_open_shadow_element_proven(page):
    await page.set_content(
        '<div id="host" style="margin:100px"></div><script>'
        "document.getElementById('host').attachShadow({mode: 'open'}).innerHTML ="
        ' \'<button id="inner" style="padding:20px">Go</button>\';</script>'
    )

    assert await _judge(page, "#inner", await _at(page, "#inner")) == (True, "same")


async def test_collection_member_proven(page):
    await page.set_content(
        '<ul id="list" style="padding:30px"><li><a href="#">One</a></li>'
        '<li><a href="#">Two</a></li><li><a href="#">Three</a></li></ul><p id="out">Other</p>'
    )
    box = await page.locator("#list").bounding_box()
    on_list = (box["x"] + 5, box["y"] + 5)  # the list's own padding: the list holds the matches

    assert await _judge(page, "#list a", await _at(page, "text=Two"), collection=True) == (
        True,
        "collection",
    )
    assert await _judge(page, "#list a", on_list, collection=True) == (True, "collection")
    ok, _ = await _judge(page, "#list a", await _at(page, "#out"), collection=True)
    assert ok is False


async def test_select_option_refused(page):
    await page.set_content(
        '<select id="dropdown"><option value="1">Option 1</option>'
        '<option value="2">Option 2</option></select>'
    )
    data = await _data(page.locator("#dropdown"))
    ok, _ = await _judge(page, '#dropdown option[value="2"]', data=data)

    assert ok is False


async def test_hidden_input_proxy_kept(page):
    await page.set_content(
        '<div class="sw"><input type="checkbox" id="real" '
        'style="opacity:0;position:absolute;width:0;height:0">'
        '<span class="slider" style="display:inline-block;width:60px;height:30px"></span></div>'
    )
    proxy = (await resolve_hidden_input_proxy(page, "id=real"))["locator"]

    assert await _judge(page, proxy, await _at(page, ".slider")) == (True, "same")


async def test_visible_filter_locator_proven(page):
    await page.set_content(
        '<button class="save" style="display:none">Save</button><button class="save">Save</button>'
    )
    point = await _at(page, "button.save >> visible=true")

    assert await _judge(page, "button.save >> visible=true", point) == (True, "same")


async def test_contenteditable_editor_in_frame_proven(page):
    """bench q09: the editor body ships; the hit is its inner <p>."""
    frame = 'iframe[id="mce_0_ifr"]'
    await page.set_content(
        '<iframe id="mce_0_ifr" style="width:500px;height:200px" srcdoc="'
        "<body id='tinymce' contenteditable='true'><p style='margin:40px'>Your content</p></body>"
        '"></iframe>'
    )
    point = await _at(page, "p", frame)

    assert await _judge(page, f"{frame} >>> id=tinymce", point, frame=frame) == (
        True,
        "control-above",
    )


EDITOR_FRAME = 'iframe[id="mce_0_ifr"]'


async def _editor_call(page, caplog, x, y, data, *, by_index, expected_text=None):
    """The q09 input step with the agent's candidate ``id=tinymce``, as registration calls it."""
    with caplog.at_level(logging.INFO, logger="browser_service.agent.actions"):
        return await find_unique_locator_action(
            x=x,
            y=y,
            element_id="elem_1",
            element_description="the body of the text editor inside the iframe",
            expected_text=expected_text,
            candidate_locator="id=tinymce",
            element_data=data,
            page=page,
            iframe_context=EDITOR_FRAME,
            is_collection=False,
            action="input",
            pointed_by_index=by_index,
        )


def _assert_editor_body_proven(out, caplog):
    assert out["found"] is True
    assert strip_frame(out["best_locator"], EDITOR_FRAME) == "id=tinymce"
    assert any(m.endswith("(signal: candidate-proven)") for m in caplog.messages)
    assert any(m.endswith("(signal: pointed-identity-proven)") for m in caplog.messages)


@pytest.mark.parametrize("height", [200, 300])
async def test_editor_frame_centre_after_the_frame_swap_proven(page, caplog, height):
    """July TinyMCE q09: the agent indexed the <iframe>, the frame lookup swapped in another box
    (not by index), so the call is judged at the frame's centre — below a one-line editor body, on
    the frame's <html>. A contenteditable body is a control; that point names it."""
    await page.set_content(
        f'<iframe id="mce_0_ifr" style="width:600px;height:{height}px" srcdoc="'
        "<body id='tinymce' contenteditable='true' style='margin:16px'>"
        "<p>Your content goes here.</p></body>"
        '"></iframe>'
    )
    await page.frame_locator(EDITOR_FRAME).locator("p").wait_for()
    box = await page.locator(EDITOR_FRAME).bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    swapped = {"tagName": "div", "textContent": "notice", "xpath": "html/body/div"}

    out = await _editor_call(
        page, caplog, x, y, swapped, by_index=False, expected_text="Your content goes here."
    )

    _assert_editor_body_proven(out, caplog)


async def test_indexed_editor_body_proven(page, caplog):
    """The agent indexed the editor's own contenteditable <body>."""
    await page.set_content(
        '<iframe id="mce_0_ifr" style="width:500px;height:200px" srcdoc="'
        "<body id='tinymce' class='mce-content-body' contenteditable='true'>"
        "<p style='margin:40px'>Your content goes here.</p></body>"
        '"></iframe>'
    )
    x, y = await _at(page, "p", EDITOR_FRAME)
    data = {
        "tagName": "body",
        "id": "tinymce",
        "className": "mce-content-body",
        "textContent": "Your content goes here.",
        "xpath": "html/body",
    }

    out = await _editor_call(page, caplog, x, y, data, by_index=True)

    _assert_editor_body_proven(out, caplog)


async def test_label_result_for_input_hit_proven(page):
    await page.set_content(
        '<div><label for="x">Name</label></div><div><div><input id="x"></div></div>'
    )

    assert await _judge(page, 'label[for="x"]', await _at(page, "#x")) == (True, "only-control")
