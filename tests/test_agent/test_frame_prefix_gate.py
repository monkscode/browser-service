"""
find_unique_locator_action returns every result through the frame gate
(browser-service #35) — mocked page, no browser.

The real-Chromium proof of each exit is test_frame_prefix_exits.py. These
tests pin the WIRING, with the cascade replaced by a sentinel: whatever an
exit returns — including an exit that does not exist yet — leaves with the
frame, and a result that needs nothing leaves as the very same object.
"""

import copy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_service.agent.actions import find_unique_locator_action

FRAME = 'iframe[id="editor-frame"]'


def _cascade_returning(result):
    spy = AsyncMock(return_value=result)
    return patch("browser_service.locators.find_unique_locator_at_coordinates", new=spy)


@pytest.fixture
def framed_page():
    """A page whose frame_locator() hands back a root that matches one element."""
    locator = MagicMock()
    locator.count = AsyncMock(return_value=1)
    locator.evaluate = AsyncMock(return_value={"tagName": "p", "id": ""})
    locator.nth = MagicMock(return_value=locator)
    root = MagicMock()
    root.locator = MagicMock(return_value=locator)
    page = MagicMock()
    page.url = "https://example.com/editor"
    page.locator = MagicMock(return_value=locator)
    page.frame_locator = MagicMock(return_value=root)
    return page


def _bare_cascade_result():
    return {
        "element_id": "elem_1",
        "found": True,
        "best_locator": "css=.from-a-new-exit",
        "all_locators": [
            {"locator": "css=.from-a-new-exit", "stability": "stable"},
            {"locator": "id=alternative", "stability": "stable"},
        ],
        "stability": "stable",
    }


async def _call(page, **kwargs):
    return await find_unique_locator_action(
        x=100,
        y=200,
        element_id="elem_1",
        element_description="the editable paragraph",
        page=page,
        **kwargs,
    )


async def test_bare_result_from_any_exit_leaves_with_the_frame(framed_page):
    with _cascade_returning(_bare_cascade_result()):
        out = await _call(framed_page, iframe_context=FRAME)
    assert out["best_locator"] == f"{FRAME} >>> css=.from-a-new-exit"
    assert [entry["locator"] for entry in out["all_locators"]] == [
        f"{FRAME} >>> css=.from-a-new-exit",
        f"{FRAME} >>> id=alternative",
    ]
    assert out["iframe_context"] == FRAME


async def test_frame_passed_positionally_is_gated_too(framed_page):
    with _cascade_returning(_bare_cascade_result()):
        out = await find_unique_locator_action(
            100, 200, "elem_1", "the editable paragraph", None, None, None, framed_page, FRAME
        )
    assert out["best_locator"] == f"{FRAME} >>> css=.from-a-new-exit"


async def test_already_prefixed_result_is_returned_as_the_same_object(framed_page):
    prefixed = {
        "element_id": "elem_1",
        "found": True,
        "best_locator": f'{FRAME} >>> text="Save draft"',
        "all_locators": [{"locator": f'{FRAME} >>> text="Save draft"', "stability": "stable"}],
        "stability": "stable",
    }
    before = copy.deepcopy(prefixed)
    with _cascade_returning(prefixed):
        out = await _call(framed_page, iframe_context=FRAME)
    assert out is prefixed
    assert out == before


async def test_no_frame_returns_the_cascade_result_as_the_same_object(framed_page):
    bare = _bare_cascade_result()
    before = copy.deepcopy(bare)
    with _cascade_returning(bare):
        out = await _call(framed_page)
    assert out is bare
    assert out == before


async def test_candidate_accepted_in_a_frame_carries_the_frame(framed_page):
    with _cascade_returning(_bare_cascade_result()) as cascade:
        out = await _call(
            framed_page, iframe_context=FRAME, candidate_locator="p[contenteditable='true']"
        )
    cascade.assert_not_called()
    assert out["best_locator"] == f"{FRAME} >>> p[contenteditable='true']"
    assert [entry["locator"] for entry in out["all_locators"]] == [out["best_locator"]]
    # The resolved-tag stamp read the element through the frame root, bare.
    framed_page.frame_locator.assert_called_with(FRAME)
    assert out["approach_metrics"]["element_tag_source"] == "resolved"


async def test_not_found_result_in_a_frame_is_returned_untouched(framed_page):
    miss = {
        "element_id": "elem_1",
        "found": False,
        "error": "nothing",
        "candidate_locators": ["#x"],
    }
    before = copy.deepcopy(miss)
    with _cascade_returning(miss):
        out = await _call(framed_page, iframe_context=FRAME)
    assert out is miss
    assert out == before


async def test_input_error_in_a_frame_is_returned_untouched():
    out = await _call(None, iframe_context=FRAME)
    assert out["found"] is False
    assert out["error_type"] == "PageObjectError"
    assert "iframe_context" not in out
