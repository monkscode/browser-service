"""
A click aimed at the one control inside a dead container, in the test and in discovery (bs #63).

The Chromium tests drive the real ``find_unique_locator_action`` on small pages whose links push
a name onto ``window.__acts``; each judges the returned locator by CLICKING it on a fresh copy of
the page. The fast tests drive the wrapper with fakes (the paths a real page cannot reach on
demand) and the registration wiring with ``_do_interaction`` replaced at its seam.

Only the Chromium tests carry the ``integration`` marker; the module has no ``pytestmark``.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import _through_inner_control, find_unique_locator_action
from browser_service.agent.registration import register_custom_actions
from browser_service.locators.click_target import InnerControl
from browser_service.locators.frame_locator import strip_frame
from browser_service.locators.validation import convert_to_playwright_locator
from tests.test_agent.test_vision_escalation import FakeAgent, FakeBrowserSession, _fake_playwright

chromium = pytest.mark.integration

FRAME = 'iframe[id="app-frame"]'
SIGNAL = "signal: click-aimed-at-inner-control"
ELSEWHERE = "signal: inner-control-answer-elsewhere"

ELEMENT_DATA_JS = """(el) => {
    const parts = [];
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
        let i = 1;
        for (let s = n.previousElementSibling; s; s = s.previousElementSibling) {
            if (s.tagName === n.tagName) i++;
        }
        parts.unshift(n.tagName.toLowerCase() + '[' + i + ']');
    }
    return {
        tagName: el.tagName.toLowerCase(),
        id: el.id || '',
        className: typeof el.className === 'string' ? el.className : '',
        role: el.getAttribute('role') || '',
        textContent: (el.textContent || '').trim().slice(0, 80),
        xpath: parts.join('/'),
        hasClickListener: false,
    };
}"""

ACTS = "<script>window.__acts = [];</script>"
STYLE = "<style>body{margin:8px} ul{list-style:none;margin:0;padding:0} li{width:300px;height:30px}</style>"


def _link(act, text, extra=""):
    return f'<a href="#" {extra} onclick="window.__acts.push(\'{act}\');return false">{text}</a>'


MENU_ITEMS = (
    f'<li role="menuitem" class="mi-products">{_link("products", "Products")}</li>'
    f'<li role="menuitem" class="mi-orders">{_link("orders", "Orders")}</li>'
)
MENU = f'{ACTS}{STYLE}<ul class="menu">{MENU_ITEMS}</ul>'
MENU_WITH_FOOTER = (
    f'{ACTS}{STYLE}<ul class="menu">{MENU_ITEMS}</ul>'
    f'<div style="margin-top:80px"><a href="#" class="foot" '
    f"onclick=\"window.__acts.push('footer');return false\">Products</a></div>"
)
BLOCK = 'style="display:block"'
MENU_BLOCK = (
    f'{ACTS}{STYLE}<ul class="menu">'
    f'<li role="menuitem" class="mi-products">'
    f"{_link('products', 'Products', BLOCK)}</li>"
    f'<li role="menuitem" class="mi-orders">{_link("orders", "Orders")}</li></ul>'
)
CAPTION = (
    f"{ACTS}<table>"
    f'<tr class="r-cap"><td style="width:300px">{_link("caption", "&lt;caption&gt;")}</td>'
    f"<td>Defines a table caption</td></tr>"
    f'<tr class="r-col"><td style="width:300px">{_link("col", "&lt;col&gt;")}</td>'
    f"<td>Defines a column</td></tr></table>"
)
CANCEL = (
    f"{ACTS}<table>"
    f'<tr class="r-1"><td style="width:200px">Order 1234</td><td>Shipped</td>'
    f"<td>{_link('cancel-1', 'Cancel')}</td></tr>"
    f'<tr class="r-2"><td style="width:200px">Order 5678</td><td>Pending</td>'
    f"<td>{_link('cancel-2', 'Cancel')}</td></tr></table>"
)
SRCDOC = MENU.replace("'", "&#39;")
OUTER = (
    '<div style="height:60px">top</div>'
    f'<iframe id="app-frame" width="500" height="200" srcdoc=\'{SRCDOC}\'></iframe>'
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


async def _centre(root, selector):
    box = await root.locator(selector).first.bounding_box()
    assert box, selector
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


async def _call(page, html, pointed, *, frame=None, **over):
    """Set the page, point at ``pointed`` (centre + its element data), call the real action."""
    await page.set_content(html)
    root = page.frame_locator(frame) if frame else page
    if frame:
        await root.locator(pointed).first.wait_for()
    x, y = await _centre(root, pointed)
    data = await root.locator(pointed).first.evaluate(ELEMENT_DATA_JS)
    kwargs = dict(
        x=x,
        y=y,
        element_id="elem_1",
        element_description="the Products menu item",
        expected_text="Products",
        element_data=data,
        page=page,
        iframe_context=frame,
        action="click",
    )
    kwargs.update(over)
    return await find_unique_locator_action(**kwargs), kwargs


async def _acts_after_click(page, html, best, frame=None):
    """A fresh copy of the page, a click on ``best``: what the generated test would do."""
    await page.set_content(html)
    root = page.frame_locator(frame) if frame else page
    selector = convert_to_playwright_locator(strip_frame(best, frame))[0]
    if frame:
        await root.locator(selector).first.wait_for()
    await root.locator(selector).click(timeout=5000)
    return await page.evaluate("window.__acts") if not frame else await _frame_acts(page)


async def _frame_acts(page):
    return await page.frames[1].evaluate("window.__acts")


def _signal_lines(caplog, signal):
    return [r for r in caplog.records if signal in r.getMessage()]


# ---------------------------------------------------------------------------
# Real Chromium, the real action
# ---------------------------------------------------------------------------


@chromium
async def test_candidate_on_li_ships_the_link(page, caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.actions"):
        out, _ = await _call(page, MENU, "li.mi-products", candidate_locator="li.mi-products")

    assert out["found"] is True
    assert out["inner_control"] is True
    assert len(_signal_lines(caplog, SIGNAL)) == 1
    assert await _acts_after_click(page, MENU, out["best_locator"]) == ["products"]


@chromium
async def test_no_candidate_ships_the_link(page, caplog):
    # today: the pointed li is one of two under ".menu > li"
    await page.set_content(MENU)
    assert await page.locator(".menu > li").count() == 2

    with caplog.at_level(logging.INFO, logger="browser_service.agent.actions"):
        out, _ = await _call(page, MENU, "li.mi-products")

    assert out["found"] is True
    assert out["inner_control"] is True
    assert len(_signal_lines(caplog, SIGNAL)) == 1
    assert await _acts_after_click(page, MENU, out["best_locator"]) == ["products"]


@chromium
async def test_whole_row_text_keeps_todays_answer(page):
    # two cells: not a bare wrapper, though the link is what the agent meant
    out, kwargs = await _call(
        page,
        CAPTION,
        "tr.r-cap",
        candidate_locator="tr.r-cap",
        element_description="table row for the <caption> tag",
        expected_text="<caption> Defines a table caption",
    )
    today = await find_unique_locator_action.__wrapped__(**kwargs)

    assert "inner_control" not in out
    assert out["best_locator"] == today["best_locator"]


@chromium
async def test_action_link_is_never_clicked(page):
    kwargs_over = dict(
        candidate_locator="tr.r-1",
        element_description="table row for order 1234",
        expected_text="Order 1234 Shipped Cancel",
    )
    out, kwargs = await _call(page, CANCEL, "tr.r-1", **kwargs_over)
    today = await find_unique_locator_action.__wrapped__(**kwargs)

    assert "inner_control" not in out
    assert out["best_locator"] == today["best_locator"]
    assert await _acts_after_click(page, CANCEL, out["best_locator"]) == []


ALICE_ROW = (
    f"{ACTS}<table>"
    '<tr class="r-a"><td style="width:300px">Alice, Smith</td><td style="width:400px">alice@x.io</td>'
    '<td><button aria-label="Delete Alice" onclick="window.__acts.push(\'delete-alice\')">x'
    "</button></td></tr></table>"
)
DELETE_ROW = (
    f"{ACTS}<table>"
    f'<tr class="r-1"><td>{_link("delete", "Delete")}</td>'
    '<td style="width:500px">Delete</td><td style="width:200px">Allowed</td></tr></table>'
)
REMOVE_ROW = (
    f"{ACTS}<table>"
    f'<tr class="r-1"><td>{_link("remove", "Remove")}</td>'
    '<td style="width:500px">Remove legacy login</td><td style="width:200px">Open</td></tr></table>'
)
EDIT_DELETE_ROW = (
    f"{ACTS}<table>"
    '<tr class="r-1"><td><button aria-label="Edit" onclick="window.__acts.push(\'edit\')">&#9998;'
    "</button> <button onclick=\"window.__acts.push('delete')\">Delete</button></td>"
    '<td style="width:500px">Edit rights</td><td style="width:200px">Allowed</td></tr></table>'
)
T1_TWO_ROWS = (
    f"{ACTS}<style>td,th{{border:1px solid #ccc;padding:6px}}</style>"
    '<table style="width:900px;border-collapse:collapse"><thead><tr><th>Name</th><th>Info</th>'
    "<th>Actions</th></tr></thead><tbody>"
    f'<tr class="r-a"><td style="width:300px">{_link("profile-alice", "Alice")}</td>'
    '<td style="width:400px">Admin</td><td><button aria-label="Delete" '
    "onclick=\"window.__acts.push('delete-alice')\">&#128465;</button></td></tr>"
    f'<tr class="r-b"><td style="width:300px">{_link("profile-bob", "Bob")}</td>'
    '<td style="width:400px">User</td><td><button aria-label="Delete" '
    "onclick=\"window.__acts.push('delete-bob')\">&#128465;</button></td></tr></tbody></table>"
)
# T1's two rows with the Delete control shown only while its row is hovered: a button
# (bench/private/bs63/build_review/rr8_hoverlib.py rr8_hover_opacity_two) and a span icon whose
# handler a script attaches (m3_hoverlib.py a_display_two)
T1_HOVER_TWO_ROWS = (
    f"{ACTS}<style>.act{{opacity:0}} tr:hover .act{{opacity:1}}</style>"
    "<style>td,th{border:1px solid #ccc;padding:6px}</style>"
    '<table style="width:900px;border-collapse:collapse"><thead><tr><th>Name</th><th>Info</th>'
    "<th>Actions</th></tr></thead><tbody>"
    f'<tr class="r-a"><td style="width:300px">{_link("profile-alice", "Alice")}</td>'
    '<td style="width:400px">Admin</td><td class="act"><button aria-label="Delete" '
    "onclick=\"window.__acts.push('delete-alice')\">&#128465;</button></td></tr>"
    f'<tr class="r-b"><td style="width:300px">{_link("profile-bob", "Bob")}</td>'
    '<td style="width:400px">User</td><td class="act"><button aria-label="Delete" '
    "onclick=\"window.__acts.push('delete-bob')\">&#128465;</button></td></tr></tbody></table>"
)
T1_HOVER_ICON_TWO_ROWS = (
    f"{ACTS}<style>.act .del{{display:none}} tr:hover .act .del{{display:inline}}</style>"
    "<style>td,th{border:1px solid #ccc;padding:6px}</style>"
    '<table style="width:900px;border-collapse:collapse"><thead><tr><th>Name</th><th>Info</th>'
    "<th>Actions</th></tr></thead><tbody>"
    f'<tr class="r-a"><td style="width:300px">{_link("profile-alice", "Alice")}</td>'
    '<td style="width:400px">Admin</td><td class="act"><span class="del" data-who="alice" '
    'style="cursor:pointer;font-size:16px">&#128465;</span></td></tr>'
    f'<tr class="r-b"><td style="width:300px">{_link("profile-bob", "Bob")}</td>'
    '<td style="width:400px">User</td><td class="act"><span class="del" data-who="bob" '
    'style="cursor:pointer;font-size:16px">&#128465;</span></td></tr></tbody></table>'
    "<script>document.querySelectorAll('.del').forEach(d => d.addEventListener('click', "
    "() => window.__acts.push('delete-' + d.dataset.who)))</script>"
)
# T1's two rows with the Delete button in a custom element's open shadow root, no role and no
# cursor on the host (bench/private/bs63/build_review/rr9_lib.py rr9_shadow_open_two)
T1_SHADOW_TWO_ROWS = (
    f"{ACTS}<style>td,th{{border:1px solid #ccc;padding:6px}}</style>"
    '<table style="width:900px;border-collapse:collapse"><thead><tr><th>Name</th><th>Info</th>'
    "<th>Actions</th></tr></thead><tbody>"
    f'<tr class="r-a"><td style="width:300px">{_link("profile-alice", "Alice")}</td>'
    '<td style="width:400px">Admin</td><td><x-del data-who="alice"></x-del></td></tr>'
    f'<tr class="r-b"><td style="width:300px">{_link("profile-bob", "Bob")}</td>'
    '<td style="width:400px">User</td><td><x-del data-who="bob"></x-del></td></tr></tbody></table>'
    "<script>customElements.define('x-del', class extends HTMLElement { constructor() { super(); "
    "const r = this.attachShadow({mode: 'open'}); "
    'r.innerHTML = \'<button aria-label="Delete" style="cursor:pointer">&#128465;</button>\'; '
    "r.querySelector('button').addEventListener('click', "
    "() => window.__acts.push('delete-' + this.dataset.who)); } });</script>"
)
# The wrapper inside a card with its own onclick, two cards (rr10_lib.py rr10_inside_onclick_two):
# the card is the target, so its click records nothing; the Track link is the wrong click
CARD = (
    '<div class="card" onclick="void 0" '
    'style="width:500px;padding:10px;margin-bottom:6px;border:1px solid #999">'
    '<div class="w w-{0}" style="width:400px;height:40px;border:1px solid #ccc;margin-bottom:6px">'
    "{1}</div></div>"
)
INSIDE_ONCLICK_TWO = (
    ACTS
    + CARD.format("docs", _link("track-alice", "Track"))
    + CARD.format("blog", _link("track-bob", "Track"))
)


@chromium
@pytest.mark.parametrize(
    "html, pointed, over",
    [
        (
            ALICE_ROW,
            "tr.r-a",
            dict(expected_text="Alice", element_description="table row for Alice"),
        ),
        (
            REMOVE_ROW,
            "tr.r-1",
            dict(
                expected_text="Remove Remove legacy login Open",
                element_description="table row for Remove legacy login",
            ),
        ),
        (
            DELETE_ROW,
            "tr.r-1",
            dict(
                expected_text="Delete Delete Allowed",
                element_description="table row for Delete",
            ),
        ),
        (
            EDIT_DELETE_ROW,
            "tr.r-1",
            dict(
                expected_text="Delete Edit rights Allowed",
                element_description="table row for Edit and Delete",
            ),
        ),
        (
            T1_TWO_ROWS,
            "tr.r-a",
            dict(expected_text="Alice", element_description="Delete button in the row for Alice"),
        ),
        (
            T1_HOVER_TWO_ROWS,
            "tr.r-a",
            dict(expected_text="Alice", element_description="Delete button in the row for Alice"),
        ),
        (
            T1_HOVER_ICON_TWO_ROWS,
            "tr.r-a",
            dict(expected_text="Alice", element_description="Delete button in the row for Alice"),
        ),
        (
            T1_SHADOW_TWO_ROWS,
            "tr.r-a",
            dict(expected_text="Alice", element_description="Delete button in the row for Alice"),
        ),
        (
            INSIDE_ONCLICK_TWO,
            "div.w-docs",
            dict(expected_text="Track", element_description="the Track order card"),
        ),
    ],
    ids=[
        "delete-beside-punctuation",
        "action-column-first",
        "one-word-name-cell",
        "two_candidates",
        "t1_two_rows",
        "rr8_hover_opacity_two",
        "a_display_two",
        "rr9_shadow_open_two",
        "rr10_inside_onclick_two",
    ],
)
async def test_destructive_control_is_never_clicked(page, html, pointed, over):
    out, kwargs = await _call(page, html, pointed, candidate_locator=pointed, **over)
    today = await find_unique_locator_action.__wrapped__(**kwargs)

    assert "inner_control" not in out
    assert out["best_locator"] == today["best_locator"]
    try:  # today's own answer may not click at all; what matters is that nothing destructive does
        acts = await _acts_after_click(page, html, out["best_locator"])
    except Exception:
        acts = []
    assert acts == []


@chromium
async def test_off_screen_row_keeps_todays_answer(page):
    html = (
        '<div style="height:1200px"></div><style>td{cursor:pointer}</style>'
        f'{ACTS}<table><tr class="r-a"><td style="width:600px">{_link("profile", "Alice")}</td></tr></table>'
    )
    out, kwargs = await _call(
        page,
        html,
        "tr.r-a",
        candidate_locator="tr.r-a",
        expected_text="Alice",
        element_description="table row for Alice",
    )
    today = await find_unique_locator_action.__wrapped__(**kwargs)

    assert "inner_control" not in out
    assert out["best_locator"] == today["best_locator"]


@chromium
async def test_right_candidate_on_the_link_is_not_refused(page):
    # today: the identity check refuses it (the candidate resolves to the <a>, the pointed one is the <li>)
    out, kwargs = await _call(
        page, MENU, "li.mi-products", candidate_locator="role=menuitem >> text=Products"
    )

    assert out["found"] is True
    assert out["inner_control"] is True
    assert await _acts_after_click(page, MENU, out["best_locator"]) == ["products"]
    assert kwargs["element_data"]["tagName"] == "li"


@chromium
async def test_in_frame_keeps_the_frame(page):
    out, _ = await _call(page, OUTER, "li.mi-products", frame=FRAME)

    assert out["found"] is True
    assert out["inner_control"] is True
    assert out["best_locator"].startswith(f"{FRAME} >>> ")
    assert await _acts_after_click(page, OUTER, out["best_locator"], FRAME) == ["products"]


@chromium
async def test_decoy_same_named_link_elsewhere(page):
    out, _ = await _call(page, MENU_WITH_FOOTER, "li.mi-products")

    assert out["found"] is True
    assert out["inner_control"] is True
    assert await _acts_after_click(page, MENU_WITH_FOOTER, out["best_locator"]) == ["products"]


@chromium
async def test_block_link_unchanged(page):
    out, kwargs = await _call(
        page, MENU_BLOCK, "li.mi-products", candidate_locator="li.mi-products"
    )
    today = await find_unique_locator_action.__wrapped__(**kwargs)

    assert "inner_control" not in out
    assert out["best_locator"] == today["best_locator"]


@chromium
async def test_read_step_unchanged(page):
    out, kwargs = await _call(
        page, MENU, "li.mi-products", candidate_locator="li.mi-products", action="get_text"
    )
    today = await find_unique_locator_action.__wrapped__(**kwargs)

    assert "inner_control" not in out
    assert out["best_locator"] == today["best_locator"]


@chromium
async def test_indexed_tag_still_names_what_the_agent_pointed_at(page):
    out, _ = await _call(page, MENU, "li.mi-products", candidate_locator="li.mi-products")

    assert out["inner_control"] is True
    assert out["approach_metrics"]["indexed_tag"] == "li"


# ---------------------------------------------------------------------------
# The wrapper with fakes
# ---------------------------------------------------------------------------

CALL = dict(
    x=1.0,
    y=2.0,
    element_id="elem_1",
    element_description="the Products menu item",
    expected_text="Products",
    candidate_locator="li.mi-products",
    element_data={"tagName": "li", "xpath": "html[1]/body[1]/li[1]"},
    page=MagicMock(),
    iframe_context=None,
    action="click",
)


def _fake_action(results):
    calls = []

    async def fake(
        x,
        y,
        element_id,
        element_description,
        expected_text=None,
        candidate_locator=None,
        element_data=None,
        page=None,
        iframe_context=None,
        is_collection=None,
        row_anchor_text=None,
        action=None,
    ):
        calls.append(
            dict(
                x=x,
                y=y,
                element_id=element_id,
                element_description=element_description,
                expected_text=expected_text,
                candidate_locator=candidate_locator,
                element_data=element_data,
                page=page,
                iframe_context=iframe_context,
                is_collection=is_collection,
                row_anchor_text=row_anchor_text,
                action=action,
            )
        )
        return results[len(calls) - 1]

    return fake, calls


def _control():
    return InnerControl(
        element_data={"tagName": "a", "textContent": "Products", "xpath": "html[1]/body[1]/a[1]"},
        x=10.0,
        y=20.0,
        handle=None,
    )


async def test_not_found_keeps_todays_answer():
    fake, calls = _fake_action([{"found": False}, {"found": True, "best_locator": "x"}])
    control = _control()
    with patch(
        "browser_service.agent.actions.find_inner_control", new=AsyncMock(return_value=control)
    ):
        out = await _through_inner_control(fake)(**CALL)

    assert out == {"found": True, "best_locator": "x"}
    assert len(calls) == 2
    assert calls[0]["element_data"] == control.element_data
    assert (calls[0]["x"], calls[0]["y"], calls[0]["candidate_locator"]) == (10.0, 20.0, None)
    assert calls[1] == {**CALL, "is_collection": None, "row_anchor_text": None}


async def test_answer_elsewhere_keeps_todays_answer(caplog):
    fake, calls = _fake_action(
        [{"found": True, "best_locator": "css=#other"}, {"found": True, "best_locator": "x"}]
    )
    with (
        patch(
            "browser_service.agent.actions.find_inner_control",
            new=AsyncMock(return_value=_control()),
        ),
        patch.object(InnerControl, "is_answer", new=AsyncMock(return_value=False)),
        caplog.at_level(logging.INFO, logger="browser_service.agent.actions"),
    ):
        out = await _through_inner_control(fake)(**CALL)

    assert out == {"found": True, "best_locator": "x"}
    assert len(calls) == 2
    assert "inner_control" not in out
    assert len(_signal_lines(caplog, ELSEWHERE)) == 1


async def test_no_control_calls_once():
    fake, calls = _fake_action([{"found": True, "best_locator": "x"}])
    with patch(
        "browser_service.agent.actions.find_inner_control", new=AsyncMock(return_value=None)
    ):
        out = await _through_inner_control(fake)(**CALL)

    assert out == {"found": True, "best_locator": "x"}
    assert len(calls) == 1
    assert calls[0] == {**CALL, "is_collection": None, "row_anchor_text": None}


async def test_page_none_goes_straight_to_the_action():
    fake, calls = _fake_action([{"found": True, "best_locator": "x"}])
    finder = AsyncMock()
    with patch("browser_service.agent.actions.find_inner_control", new=finder):
        await _through_inner_control(fake)(**{**CALL, "page": None})

    finder.assert_not_awaited()
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Registration: discovery's own click
# ---------------------------------------------------------------------------

CLICK = {"id": "elem_1", "action": "click", "value": ""}
PARAMS = {"x": 500, "y": 300, "element_description": "the thing", "element_index": 5}


async def _discover(engine_result):
    interaction = AsyncMock(return_value=("", "auto_ok", "click", ""))
    agent = FakeAgent()
    with (
        patch(
            "browser_service.agent.actions.find_unique_locator_action",
            new=AsyncMock(return_value=engine_result),
        ),
        patch("browser_service.agent.registration._do_interaction", new=interaction),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
    ):
        assert register_custom_actions(agent, elements=[CLICK]) is True
        result = await agent.tools.registry.execute_action(
            "find_unique_locator",
            {**PARAMS, "element_id": "elem_1"},
            browser_session=FakeBrowserSession(),
        )
    return result, interaction


def _found(**extra):
    return {
        "found": True,
        "best_locator": "css=#a",
        "validated": True,
        "element_id": "elem_1",
        **extra,
    }


async def test_discovery_click_uses_the_locator_when_re_aimed():
    result, interaction = await _discover(_found(inner_control=True))

    interaction.assert_awaited_once()
    assert interaction.await_args.args[4] is None
    assert "inner_control" not in result.metadata


async def test_discovery_click_keeps_the_index_otherwise():
    result, interaction = await _discover(_found())

    interaction.assert_awaited_once()
    assert interaction.await_args.args[4] == 5
    assert "inner_control" not in result.metadata
