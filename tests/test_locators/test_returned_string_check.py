"""bs #53 part 1 — the string STEP 2.5 hands on is counted where the test applies it.

STEP 2.5 of ``find_unique_locator_action`` (the accessibility fallback) counted one query and
returned another string: a partial-name ``role=button[name="CSV"]`` that matches nothing (Playwright's
role engine matches the WHOLE name unless told otherwise), or an ``[aria-label="Say "hi""]`` that
is not even a valid selector — and bs reported it validated. The returned string is now counted in
the search context (the frame when there is one); a single element needs exactly 1 match, a
collection (``unique`` False) at least 1, as a live list may grow between the two counts. A string
that fails is dropped (signal ``accessibility-string-unconfirmed``) and the cascade carries on.

Two tests pin silent wrong clicks the owner accepted and handed to bs #61 (comment 6334760388):
the dead string used to hide them. When #61 fixes either one, its test changes.

Real headless Chromium against inline HTML (``page.set_content``), marked integration (the fixture
lane). ``element_data`` is None so STEP 2.5 is reached.
"""

from unittest.mock import AsyncMock, patch

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import find_unique_locator_action
from browser_service.locators import smart_locator
from browser_service.locators.frame_locator import strip_frame

pytestmark = pytest.mark.integration

FRAME = 'iframe[id="app-frame"]'
FAR = '<div id="far" style="margin-top:300px;width:200px;height:30px">far</div>'
EXPORT = "<main><h2>Reports</h2><button>Export CSV</button> <button>Print</button></main>"
QUOTE = '<input type="number" aria-label=\'Say "hi"\'> <input type="number" aria-label="Count">'
CAND = (
    '<main><button>Export CSV</button> <span aria-disabled="true" style="margin-left:40px">'
    "CSV help</span> <button>Print</button></main>"
)
NEIGH = (
    '<button class="exp" style="width:120px">Export CSV</button> '
    '<a href="#" class="imp">Import CSV file</a>'
)
LIST3 = '<ul id="l"><li>Apple</li><li>Banana</li><li>Cherry</li></ul>'
LIST1 = '<ul id="l"><li>Banana</li></ul>'
GROW = (
    "for(let k=0;k<5;k++){const li=document.createElement('li');li.textContent='Item '+k;"
    "document.getElementById('l').appendChild(li)}"
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


def _root(page, in_frame):
    return page.frame_locator(FRAME) if in_frame else page


async def _run(
    page,
    body,
    *,
    in_frame,
    target,
    pointed=None,
    description,
    expected,
    action,
    candidate=None,
    is_collection=False,
):
    """Load ``body`` (in the frame or not) and call the action at the centre of ``pointed``."""
    if in_frame:
        await page.set_content(
            '<p>Outer page</p><iframe id="app-frame" width="900" height="600"></iframe>'
        )
        await page.frames[1].set_content(body)
    else:
        await page.set_content(body)
    box = await _root(page, in_frame).locator(pointed or target).first.bounding_box()
    assert box, pointed or target
    out = await find_unique_locator_action(
        x=box["x"] + box["width"] / 2,
        y=box["y"] + box["height"] / 2,
        element_id="elem_1",
        element_description=description,
        expected_text=expected,
        candidate_locator=candidate,
        element_data=None,
        page=page,
        iframe_context=FRAME if in_frame else None,
        action=action,
        is_collection=is_collection,
    )
    if in_frame and out.get("best_locator"):
        assert out["best_locator"].startswith(f"{FRAME} >>> "), out["best_locator"]
    return out


async def _is_target(page, out, target, in_frame):
    """The best locator matches one element, and it is ``target``."""
    assert out["found"] is True, out.get("error")
    root = _root(page, in_frame)
    inner = strip_frame(out["best_locator"], FRAME)
    assert await root.locator(inner).count() == 1, inner
    a = await root.locator(inner).element_handle()
    b = await root.locator(target).first.element_handle()
    return await a.evaluate("(a, b) => a === b", b)


def _best(out):
    return strip_frame(out.get("best_locator") or "", FRAME)


@pytest.mark.parametrize("in_frame", [False, True])
async def test_dead_partial_role_string_is_dropped(page, in_frame):
    t = 'button:text-is("Export CSV")'
    out = await _run(
        page,
        EXPORT,
        in_frame=in_frame,
        target=t,
        description="the export control",
        expected="CSV",
        action="click",
    )
    assert _best(out) == 'text="Export CSV"'
    assert await _is_target(page, out, t, in_frame)


@pytest.mark.parametrize("in_frame", [False, True])
async def test_invalid_quote_string_is_dropped(page, in_frame):
    t = "input >> nth=0"
    out = await _run(
        page,
        QUOTE,
        in_frame=in_frame,
        target=t,
        description="the greeting box",
        expected='Say "hi"',
        action="input",
    )
    # The precondition: the string STEP 2.5 builds for this name is not a valid selector.
    with pytest.raises(Exception):
        await _root(page, in_frame).locator('[aria-label="Say "hi""]').count()
    assert await _is_target(page, out, t, in_frame)


LIVE = [
    (
        '<button class="exp">Export\n      CSV</button>' + FAR,
        "button.exp",
        "the export control",
        "Export CSV",
        'role=button[name="Export CSV"]',
    ),
    (
        '<button class="exp">Export\n      CSV</button>' + FAR,
        "button.exp",
        "the export control",
        "Export CSV ",
        'role=button[name="Export CSV"]',
    ),
    (
        '<button class="ds">Don\'t save</button>' + FAR,
        "button.ds",
        "the discard control",
        "Don't save",
        'role=button[name="Don\'t save"]',
    ),
]


@pytest.mark.parametrize("in_frame", [False, True])
@pytest.mark.parametrize("body,target,desc,expected,string", LIVE, ids=["ws", "trail", "apos"])
async def test_live_string_is_kept(page, in_frame, body, target, desc, expected, string):
    out = await _run(
        page,
        body,
        in_frame=in_frame,
        target=target,
        pointed="#far",
        description=desc,
        expected=expected,
        action="click",
    )
    assert _best(out) == string
    assert await _is_target(page, out, target, in_frame)


# Top only: STEP 2.5b (the collection route) cannot run in a frame today.
@pytest.mark.parametrize(
    "body,target,n", [(LIST3, 'li:text-is("Banana")', 3), (LIST1, "li", 1)], ids=["3", "1"]
)
async def test_collection_string_is_kept(page, body, target, n):
    out = await _run(
        page,
        body,
        in_frame=False,
        target=target,
        description="the fruit items",
        expected=None,
        action="get_text",
        is_collection=True,
    )
    assert out["best_locator"] == "role=listitem"
    assert await page.locator("role=listitem").count() == n


@pytest.mark.parametrize(
    "body,target,n", [(LIST3, 'li:text-is("Banana")', 8), (LIST1, "li", 6)], ids=["3", "1"]
)
async def test_growing_collection_is_kept(page, body, target, n):
    """A live list that grows after the site's count still passes (collection = at least 1)."""
    real = smart_locator._find_element_via_accessibility
    grew = []

    async def grow_after(*a, **kw):
        r = await real(*a, **kw)
        if r:
            await kw["page"].evaluate(GROW)
            grew.append(r.get("locator"))
        return r

    with patch.object(
        smart_locator, "_find_element_via_accessibility", AsyncMock(side_effect=grow_after)
    ):
        out = await _run(
            page,
            body,
            in_frame=False,
            target=target,
            description="the fruit items",
            expected=None,
            action="get_text",
            is_collection=True,
        )
    assert grew
    assert out["best_locator"] == "role=listitem"
    assert await page.locator("role=listitem").count() == n


@pytest.mark.parametrize("in_frame", [False, True])
async def test_known_61_parked_disabled_candidate_is_handed_back(page, in_frame):
    # A KNOWN silent wrong click, pinned on purpose: the dead string used to hide it, and now
    # _settle_action_fit hands back the parked disabled candidate. Handed to bs #61 (comment
    # 6334760388) — a #61 fix is expected to change this test.
    s = 'span[aria-disabled="true"]'
    out = await _run(
        page,
        CAND,
        in_frame=in_frame,
        target='button:text-is("Export CSV")',
        pointed=s,
        candidate=s,
        description="the export control",
        expected="CSV",
        action="click",
    )
    assert _best(out) == s


@pytest.mark.parametrize("in_frame", [False, True])
async def test_known_61_neighbour_is_accepted(page, in_frame):
    # A KNOWN silent wrong click, pinned on purpose: STEP 3's substring check accepts a
    # neighbour holding the partial text ("Import CSV file" for "CSV"). Handed to bs #61
    # (comment 6334760388) — a #61 fix is expected to change this test.
    out = await _run(
        page,
        NEIGH,
        in_frame=in_frame,
        target="button.exp",
        pointed="a.imp",
        description="the export control",
        expected="CSV",
        action="click",
    )
    assert _best(out) == 'text="Import CSV file"'


@pytest.mark.parametrize("in_frame", [False, True])
async def test_neighbour_twin_finds_the_target(page, in_frame):
    out = await _run(
        page,
        NEIGH,
        in_frame=in_frame,
        target="button.exp",
        pointed="button.exp",
        description="the export control",
        expected="CSV",
        action="click",
    )
    assert _best(out) == 'text="Export CSV"'
    assert await _is_target(page, out, "button.exp", in_frame)
