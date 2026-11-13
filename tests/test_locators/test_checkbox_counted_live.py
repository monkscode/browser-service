"""bs #64 on real Chromium: the checkbox label finder ships only a string it counted.

The finder counted one selector (``label >> input[type="checkbox"]``) and then returned
another (``id=rm``, ``input[type="checkbox"][name="prefs"]``) it never counted. On a page
with a duplicate id or a shared name the shipped string matches 2 elements and the test
fails Playwright strict mode. Each row runs the real find_unique_locator_action on an
inline page and pins the exact string shipped, its Playwright count (1, inside the frame
for a framed string) and that a click on it reaches the remember-me control.

Pages and call inputs are the bs #64 evaluation's probe pages (the pointed element is
``.rm``, the call is made at its box centre, ``element_data`` is the production shape or
None). Frame rows serve the same page in ``iframe#app-frame`` via ``srcdoc``. On d4e6331
the duplicate-id, shared-name, label-for and adjacent rows fail (they shipped ``id=rm``,
``input[type="checkbox"][name="prefs"]``, ...); proxy_dup and nested_uniq pass on both.

    pytest tests/test_locators/test_checkbox_counted_live.py -m fixture
"""

import html

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import find_unique_locator_action
from browser_service.locators import convert_to_playwright_locator

pytestmark = [pytest.mark.integration, pytest.mark.fixture]

FRAME = 'iframe[id="app-frame"]'
F = f"{FRAME} >>> "
NESTED = 'label:text-is("Remember me") >> input[type="checkbox"]'
RADIO = 'label:text-is("Remember me") >> input[type="radio"]'
ADJ = ':text("Remember me") >> xpath=preceding-sibling::input[@type="checkbox"]'

_ON = "act(this.checked ? 'remember-on' : 'remember-off')"
_OTHER = "act('other-' + this.checked)"
_HIDE = 'style="display:none"'
_SLIDER = '<span style="display:inline-block;width:30px;height:14px;background:#ccc"></span>'
_FAR = '<div style="margin-top:40px">'

PAGES = {
    "nested_dup": (
        f'<label class="rm"><input type="checkbox" id="rm" onchange="{_ON}"> Remember me</label>'
        f'{_FAR}<input type="checkbox" id="rm" class="other" onchange="{_OTHER}"> '
        "<span>Receive mail</span></div>"
    ),
    "space_dup": (
        f'<label class="rm"><input type="checkbox" id="rm 1" onchange="{_ON}"> Remember me</label>'
        f'{_FAR}<input type="checkbox" id="rm 1" class="other" onchange="{_OTHER}"> '
        "<span>Receive mail</span></div>"
    ),
    "name_group": (
        f'<label class="rm"><input type="checkbox" name="prefs" value="remember" onchange="{_ON}">'
        f' Remember me</label>{_FAR}<label><input type="checkbox" name="prefs" value="mail" '
        f'class="other" onchange="{_OTHER}"> Receive mail</label></div>'
    ),
    "radio_dup": (
        f'<label class="rm"><input type="radio" name="r" value="a" id="rm" onchange="{_ON}">'
        f' Remember me</label>{_FAR}<input type="radio" name="r2" value="b" id="rm" '
        f'class="other" onchange="{_OTHER}"> <span>Receive mail</span></div>'
    ),
    "radio_name": (
        f'<label class="rm"><input type="radio" name="r" onchange="{_ON}"> Remember me</label>'
        f'{_FAR}<label><input type="radio" name="r" class="other" onchange="{_OTHER}">'
        " Receive mail</label></div>"
    ),
    "for_dup": (
        f'<input type="checkbox" id="rm" onchange="{_ON}"><label for="rm" class="rm">'
        'Remember me</label><div id="rm" style="margin-top:40px">Receive mail</div>'
    ),
    "adj_dup": (
        f'<div><input type="checkbox" id="rm" onchange="{_ON}"> <span class="rm">Remember me'
        f'</span></div>{_FAR}<input type="checkbox" id="rm" class="other" '
        f'onchange="{_OTHER}"> <span>Receive mail</span></div>'
    ),
    "adj_name": (
        f'<div><input type="checkbox" name="prefs" onchange="{_ON}"> <span class="rm">'
        f'Remember me</span></div>{_FAR}<input type="checkbox" name="prefs" class="other" '
        f'onchange="{_OTHER}"> <span>Receive mail</span></div>'
    ),
    "proxy_dup": (
        '<label class="rm" style="display:inline-block;padding:4px">'
        f'<input type="checkbox" id="rm" {_HIDE} onchange="{_ON}">'
        '<span class="slider" style="display:inline-block;width:30px;height:14px;'
        f'background:#ccc"></span> Remember me</label>{_FAR}<input type="checkbox" id="rm" '
        f'class="other" onchange="{_OTHER}"> <span>Receive mail</span></div>'
    ),
    "nested_uniq": (
        f'<label class="rm"><input type="checkbox" id="rm" onchange="{_ON}"> Remember me</label>'
        f'{_FAR}<input type="checkbox" id="rm2" class="other" onchange="{_OTHER}"> '
        "<span>Receive mail</span></div>"
    ),
    "proxy_name": (
        '<label class="rm" style="display:inline-block;padding:4px">'
        f'<input type="checkbox" name="prefs" value="remember" {_HIDE} onchange="{_ON}">'
        f"{_SLIDER} Remember me</label>{_FAR}<label>"
        f'<input type="checkbox" name="prefs" value="mail" class="other" {_HIDE} '
        f'onchange="{_OTHER}">{_SLIDER} Receive mail</label></div>'
    ),
}

# element_data as the production page script builds it (bs53_eval/lib.py PROD_ED_JS)
ED_JS = """(el) => {
    const a = (n) => el.getAttribute(n) || "";
    const parts = [];
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
        const tag = n.tagName.toLowerCase();
        const sibs = n.parentElement ? Array.from(n.parentElement.children).filter(s => s.tagName === n.tagName) : [n];
        parts.unshift(sibs.length > 1 ? tag + '[' + (sibs.indexOf(n) + 1) + ']' : tag);
    }
    const p = el.parentElement;
    return {
        tagName: el.tagName.toLowerCase(), id: a('id'), name: a('name'), className: a('class'),
        ariaLabel: a('aria-label'), ariaInvalid: a('aria-invalid'),
        parentClassName: p ? (p.getAttribute('class') || "") : "", placeholder: a('placeholder'), title: a('title'),
        href: a('href'), role: a('role'), dataTestId: a('data-testid') || a('data-test'),
        dataTestAttr: (!a('data-testid') && a('data-test')) ? 'data-test' : 'data-testid',
        type: a('type'), value: a('value'), xpath: parts.join('/'), backendNodeId: null,
        textContent: ((el.innerText || el.textContent || '').trim()).slice(0, 300)
    };
}"""

_SCRIPT = "<script>window.__acts=[];function act(n){window.__acts.push(n)}</script>"


def _doc(name):
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>bs64</title></head>'
        f'<body style="margin:8px">{_SCRIPT}{PAGES[name]}</body></html>'
    )


def _host(name):
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>host</title></head>'
        '<body style="margin:8px"><h1>Host</h1>'
        f'<iframe id="app-frame" srcdoc="{html.escape(_doc(name), quote=True)}" '
        'style="width:900px;height:420px"></iframe></body></html>'
    )


@pytest.fixture
async def page():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True)
        try:
            yield await b.new_page(viewport={"width": 1280, "height": 900})
        finally:
            await b.close()


async def _load(page, name, level):
    await page.set_content(_host(name) if level == "frame" else _doc(name), wait_until="load")
    root = page.frame_locator(FRAME) if level == "frame" else page
    await root.locator(".rm").wait_for(state="attached", timeout=5000)
    return root


def _applied(page, shipped):
    """The shipped string as Playwright applies it: inside the frame when it has the prefix."""
    prefix, inner = shipped.split(" >>> ", 1) if " >>> " in shipped else (None, shipped)
    selector, _ = convert_to_playwright_locator(inner)
    return (page.frame_locator(prefix) if prefix else page).locator(selector)


async def _call(page, name, level, view, desc):
    root = await _load(page, name, level)
    target = root.locator(".rm")
    box = await target.bounding_box()
    data = await target.evaluate(ED_JS) if view == "ed" else None
    return await find_unique_locator_action(
        x=box["x"] + box["width"] / 2,
        y=box["y"] + box["height"] / 2,
        element_id="elem_1",
        element_description=desc,
        expected_text="Remember me",
        candidate_locator=None,
        element_data=data,
        page=page,
        iframe_context=FRAME if level == "frame" else None,
        action="click",
        is_collection=None,
        vision_type_hint=None,
        vision_framework_hint=None,
    )


async def _assert_row(page, name, level, view, expected, desc="Remember me checkbox"):
    out = await _call(page, name, level, view, desc)
    assert out.get("found") is True, out
    shipped = F + expected if level == "frame" else expected
    assert out["best_locator"] == shipped
    assert await _applied(page, shipped).count() == 1
    # a click on a fresh page reaches the remember-me control
    root = await _load(page, name, level)
    await _applied(page, shipped).click(timeout=3000)
    assert await root.locator("body").evaluate("() => window.__acts") == ["remember-on"]


def _rows(names, views, expected):
    return [
        pytest.param(n, lv, vw, expected, id=f"{n}-{lv}-{vw}") for n in names for lv, vw in views
    ]


THREE = [("top", "ed"), ("top", "no-ed"), ("frame", "no-ed")]


@pytest.mark.parametrize(
    "name,level,view,expected",
    _rows(["nested_dup", "space_dup", "name_group"], THREE, NESTED)
    + _rows(["adj_dup", "adj_name"], THREE, ADJ)
    + _rows(["proxy_dup"], THREE, 'label:has(input[id="rm"])')
    + _rows(["nested_uniq"], THREE, "id=rm")
    + [pytest.param("for_dup", "frame", "no-ed", 'input[id="rm"]', id="for_dup-frame-no-ed")],
)
async def test_checkbox_rows_ship_a_counted_string(page, name, level, view, expected):
    await _assert_row(page, name, level, view, expected)


@pytest.mark.parametrize("name", ["radio_dup", "radio_name"])
async def test_radio_rows_ship_a_counted_string(page, name):
    await _assert_row(page, name, "top", "ed", RADIO, desc="Remember me radio button")


async def test_proxy_name_hidden_input_in_a_shared_name_group_gets_no_proxy(page):
    """stated limit: a hidden input with no id in a shared-name group gets no proxy
    (`_proxy_candidates` builds none from a `>>` string). The counted string ships (count 1)
    but the input is display:none, so a click on it would time out; no click assertion."""
    out = await _call(page, "proxy_name", "top", "ed", "Remember me checkbox")
    assert out.get("found") is True, out
    assert out["best_locator"] == NESTED
    shipped = _applied(page, NESTED)
    assert await shipped.count() == 1
    assert await shipped.is_visible() is False
