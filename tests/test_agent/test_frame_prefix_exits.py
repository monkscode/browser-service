"""
Every exit of find_unique_locator_action, with the element inside an iframe,
on real Chromium (browser-service #35).

A locator found inside a frame is only usable by the generated test when it
carries the frame: ``iframe[id="x"] >>> <locator>``. The rule lived in four
nested functions and each exit decided for itself whether to call one, so
this file drives every exit through the same frame and asserts the same
thing of each — the frame on the best locator AND on every alternative
(tasks/workflow.py's re-ranker and id check only choose among
``all_locators`` entries, they never build a string).

The exits that already worked were pinned here before anything was refactored;
the candidate, collection-candidate, parked-candidate and coordinate-fallback
exits returned the locator bare. Each test also asserts
``approach_metrics.locator_approach``, so a test cannot quietly start
exercising a different exit.

The fixture: ``frame_prefix_outer.html`` hosts ``frame_prefix_inner.html`` in
``iframe#editor-frame`` and holds a twin of the frame's editable paragraph,
so a locator that lost its frame resolves at page level to the wrong element.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import find_unique_locator_action
from browser_service.locators.frame_locator import strip_frame

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "test_locators" / "locator_fixtures"

FRAME = 'iframe[id="editor-frame"]'
PREFIX = f"{FRAME} >>> "

# browser-use-style element_data for the INDEXED element, including the
# absolute xpath browser-use supplies (html[1]/body[1]/...).
INDEXED_DATA_JS = """(el) => {
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
        id: el.id || "",
        className: (typeof el.className === 'string') ? el.className : "",
        textContent: (el.textContent || "").trim().slice(0, 80),
        xpath: parts.join('/'),
    };
}"""


def _file_url(name: str) -> str:
    return (FIXTURES_DIR / name).resolve().as_uri()


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


async def _centre(root, selector: str):
    box = await root.locator(selector).first.bounding_box()
    assert box, selector
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


async def _in_frame(page, target: str, *, frame: str = FRAME, indexed: bool = False, **kwargs):
    """Call the action for ``target``, an element inside the editor frame."""
    await page.goto(_file_url("frame_prefix_outer.html"), wait_until="load")
    root = page.frame_locator(frame)
    x, y = await _centre(root, target)
    if indexed:
        kwargs["element_data"] = await root.locator(target).first.evaluate(INDEXED_DATA_JS)
    return await find_unique_locator_action(
        x=x, y=y, element_id="elem_1", page=page, iframe_context=frame, **kwargs
    )


async def _top_level(page, target: str, *, indexed: bool = False, **kwargs):
    """Call the action for ``target`` on the same document loaded with no frame."""
    await page.goto(_file_url("frame_prefix_inner.html"), wait_until="load")
    x, y = await _centre(page, target)
    if indexed:
        kwargs["element_data"] = await page.locator(target).first.evaluate(INDEXED_DATA_JS)
    return await find_unique_locator_action(x=x, y=y, element_id="elem_1", page=page, **kwargs)


def _approach(out) -> str:
    return (out.get("approach_metrics") or {}).get("locator_approach", "")


def _assert_carries_the_frame(out, inner: str, approach: str, prefix: str = PREFIX):
    assert out["found"] is True, out.get("error")
    assert _approach(out) == approach
    assert out["best_locator"] == f"{prefix}{inner}"
    entries = [entry["locator"] for entry in out["all_locators"]]
    assert entries, "a found result lists at least its own locator"
    assert [e for e in entries if not e.startswith(prefix)] == []


# ---------------------------------------------------------------------------
# Pins: exits that carried the frame before #35 was fixed
# ---------------------------------------------------------------------------


async def test_element_data_step_carries_the_frame(page):
    out = await _in_frame(
        page,
        "#save-btn",
        indexed=True,
        element_description="the Save draft button",
        expected_text="Save draft",
    )
    _assert_carries_the_frame(out, "#save-btn", "element_data")


async def test_text_first_step_carries_the_frame(page):
    out = await _in_frame(
        page,
        "#save-btn",
        element_description="the Save draft button",
        expected_text="Save draft",
    )
    _assert_carries_the_frame(out, 'text="Save draft"', "text_first")


async def test_collection_by_text_carries_the_frame(page):
    out = await _in_frame(
        page,
        "li.chapter-item",
        element_description="all chapters",
        expected_text="Chapter one",
        is_collection=True,
    )
    _assert_carries_the_frame(out, "#chapters > li.chapter-item", "collection")
    assert out["count"] == 3


async def test_accessibility_step_carries_the_frame(page):
    """The label's text is unique but sits >100px from the input, so text-first
    declines it; the description names nothing; the role API finds the input."""
    out = await _in_frame(
        page,
        "#email-in",
        element_description="the work email box",
        expected_text="Work email",
    )
    _assert_carries_the_frame(out, 'role=textbox[name="Work email"]', "accessibility")


async def test_description_step_carries_the_frame(page):
    """No expected_text: text-first is skipped and the description answers."""
    out = await _in_frame(page, "#save-btn", element_description="Save draft button")
    _assert_carries_the_frame(out, 'role=button[name="Save draft"]', "semantic")


async def test_candidate_that_names_the_frame_itself_falls_back_and_is_prefixed_once(page):
    """The agent sometimes writes the frame into its candidate. Counted inside
    the frame that matches nothing, so the cascade answers — one frame hop."""
    out = await _in_frame(
        page,
        "p[contenteditable]",
        indexed=True,
        element_description="the editable paragraph",
        candidate_locator=f"{PREFIX}p[contenteditable='true']",
    )
    assert out["found"] is True, out.get("error")
    assert _approach(out) == "element_data"
    inner = strip_frame(out["best_locator"], FRAME)
    assert inner != out["best_locator"]
    assert ">>>" not in inner
    assert await page.frame_locator(FRAME).locator(inner).inner_text() == "frame paragraph"


# ---------------------------------------------------------------------------
# #35: exits that returned the locator WITHOUT the frame
# ---------------------------------------------------------------------------


@pytest.fixture
def cascade_finds_nothing():
    """The cascade misses, so a parked candidate is the answer of last resort."""
    miss = AsyncMock(return_value={"found": False, "error": "nothing fits"})
    with patch("browser_service.locators.find_unique_locator_at_coordinates", new=miss):
        yield miss


async def _paragraph_candidate(page, action=None):
    """Bench q09's shape: the agent proposes the editable paragraph's selector."""
    return await _in_frame(
        page,
        "p[contenteditable]",
        indexed=True,
        element_description="the editable paragraph",
        candidate_locator="p[contenteditable='true']",
        action=action,
    )


@pytest.mark.parametrize("action", [None, "input"], ids=["action-fit-off", "action-fit-on"])
async def test_candidate_accepted_in_a_frame_carries_the_frame(page, action):
    out = await _paragraph_candidate(page, action)
    _assert_carries_the_frame(out, "p[contenteditable='true']", "actions_candidate")
    assert out["iframe_context"] == FRAME


async def test_candidate_in_a_frame_resolves_to_the_frame_element_not_the_page_twin(page):
    """Without the frame the locator still matches exactly one element at page
    level — the main page's twin — so the test would pass on the wrong one."""
    out = await _paragraph_candidate(page)
    best = out["best_locator"]
    inner = strip_frame(best, FRAME)
    # What the generated test does with it: through the frame when the locator
    # names one, at page level otherwise.
    target = page.frame_locator(FRAME).locator(inner) if inner != best else page.locator(best)
    assert await target.count() == 1
    assert await target.inner_text() == "frame paragraph"


async def test_collection_candidate_in_a_frame_carries_the_frame(page):
    out = await _in_frame(
        page,
        "li.chapter-item",
        indexed=True,
        element_description="all chapters",
        candidate_locator="li.chapter-item",
        is_collection=True,
    )
    _assert_carries_the_frame(out, "li.chapter-item", "actions_candidate_collection")
    assert out["count"] == 3


async def test_misfit_candidate_kept_as_last_resort_carries_the_frame(page, cascade_finds_nothing):
    """An empty input cannot perform a read of "buy milk": the candidate is
    parked, the cascade finds nothing, the parked candidate is returned."""
    out = await _in_frame(
        page,
        "#todo-in",
        indexed=True,
        element_description="the todo text",
        expected_text="buy milk",
        candidate_locator="#todo-in",
        action="get_text",
    )
    assert out["action_fit"] == "misfit"
    _assert_carries_the_frame(out, "#todo-in", "actions_candidate")


async def test_volatile_candidate_kept_as_last_resort_carries_the_frame(
    page, cascade_finds_nothing
):
    out = await _in_frame(
        page,
        "#ext-gen1042",
        element_description="the panel",
        candidate_locator="id=ext-gen1042",
        action="click",
    )
    assert out["stability"] == "volatile"
    _assert_carries_the_frame(out, "id=ext-gen1042", "actions_candidate")


async def test_coordinate_fallback_in_a_frame_carries_the_frame(page):
    """No element_data, no expected_text, a description that names nothing and
    no role at the point: only the coordinate strategies are left."""
    out = await _in_frame(page, "span.sync-chip", element_description="status chip")
    assert out["found"] is True, out.get("error")
    assert _approach(out) == "coordinate_fallback"
    entries = [entry["locator"] for entry in out["all_locators"]]
    assert len(entries) > 1, "the fallback validated several alternatives"
    assert [e for e in [out["best_locator"], *entries] if not e.startswith(PREFIX)] == []
    inner = strip_frame(out["best_locator"], FRAME)
    assert await page.frame_locator(FRAME).locator(inner).inner_text() == "Synced"


async def test_candidate_in_an_ordinal_frame_is_positional(page):
    """`iframe >> nth=1 >>> #save-btn` re-resolves the second iframe: the
    result is positional however stable the inner locator is."""
    ordinal = "iframe >> nth=1"
    out = await _in_frame(
        page,
        "#save-btn",
        frame=ordinal,
        indexed=True,
        element_description="the Save draft button",
        candidate_locator="#save-btn",
    )
    _assert_carries_the_frame(out, "#save-btn", "actions_candidate", prefix=f"{ordinal} >>> ")
    assert out["stability"] == "positional"
    assert [entry["stability"] for entry in out["all_locators"]] == ["positional"]


async def test_nested_frame_with_the_hosts_selector_still_gets_the_frame_hop(page):
    """The frame hosts a nested <iframe> with its own id. That element's
    in-frame locator IS the host's selector, so "starts with the frame text"
    would read it as already framed and ship it bare — and at page level the
    bare string is the HOST frame, the wrong element."""
    out = await _in_frame(
        page,
        FRAME,
        indexed=True,
        element_description="the nested preview frame",
        candidate_locator=FRAME,
    )
    assert out["found"] is True, out.get("error")
    assert _approach(out) == "actions_candidate"
    assert out["best_locator"] == f"{PREFIX}{FRAME}"
    inner = strip_frame(out["best_locator"], FRAME)
    nested = page.frame_locator(FRAME).locator(inner)
    assert await nested.get_attribute("title") == "nested preview"


# ---------------------------------------------------------------------------
# Control: with no frame, nothing is added
# ---------------------------------------------------------------------------


def _assert_no_frame(out, locator: str, approach: str):
    assert out["found"] is True, out.get("error")
    assert _approach(out) == approach
    assert out["best_locator"] == locator
    assert [entry["locator"] for entry in out["all_locators"]] == [locator]
    assert "iframe_context" not in out


async def test_no_frame_candidate_is_returned_as_written(page):
    out = await _top_level(
        page,
        "p[contenteditable]",
        indexed=True,
        element_description="the editable paragraph",
        candidate_locator="p[contenteditable='true']",
    )
    _assert_no_frame(out, "p[contenteditable='true']", "actions_candidate")


async def test_no_frame_collection_candidate_is_returned_as_written(page):
    out = await _top_level(
        page,
        "li.chapter-item",
        indexed=True,
        element_description="all chapters",
        candidate_locator="li.chapter-item",
        is_collection=True,
    )
    _assert_no_frame(out, "li.chapter-item", "actions_candidate_collection")


async def test_no_frame_cascade_result_is_returned_as_written(page):
    out = await _top_level(
        page,
        "#save-btn",
        indexed=True,
        element_description="the Save draft button",
        expected_text="Save draft",
    )
    _assert_no_frame(out, "#save-btn", "element_data")
