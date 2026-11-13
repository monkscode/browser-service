"""
The returned-string count at the action's exit (browser-service #64).

``_through_frame_gate`` counts the locator string a found result hands on, where
the generated test applies it (the frame hop stripped, the count run inside the
frame), and withholds it unless it matches exactly one element (at least one for
a collection). Unit cases use a stub action and a fake page; the live case runs
real Chromium on a page with a duplicated id.
"""

import logging

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import (
    RETURNED_COUNT_ERROR,
    ROBOT_UNSAFE_ERROR,
    _through_frame_gate,
)

FRAME = 'iframe[id="editor-frame"]'
DUP_PAGE = """<!DOCTYPE html><html><body>
<label class="rm"><input type="checkbox" id="rm"> Remember me</label>
<div><input type="checkbox" id="rm" class="other"> <span>Receive mail</span></div>
</body></html>"""


class _Locator:
    def __init__(self, page, selector, via_frame):
        self._page, self._selector, self._via_frame = page, selector, via_frame

    async def count(self):
        self._page.counted.append((self._selector, self._via_frame))
        if isinstance(self._page.count_value, Exception):
            raise self._page.count_value
        return self._page.count_value


class _Root:
    def __init__(self, page, via_frame):
        self._page, self._via_frame = page, via_frame

    def locator(self, selector):
        return _Locator(self._page, selector, self._via_frame)


class _FakePage(_Root):
    def __init__(self, count_value):
        super().__init__(self, None)
        self.count_value = count_value
        self.counted = []  # (selector, frame it was counted in or None)

    def frame_locator(self, ctx):
        return _Root(self, ctx)


def _gated(result):
    @_through_frame_gate
    async def action(x, y, page=None, iframe_context=None):
        return result

    return action


def _found(best="id=cb", **extra):
    return {"found": True, "best_locator": best, "all_locators": [], **extra}


async def test_count_two_is_withheld(caplog):
    page = _FakePage(2)
    with caplog.at_level(logging.WARNING):
        out = await _gated(_found())(1, 2, page=page)
    assert out["found"] is False
    assert out["best_locator"] is None
    assert out["error"] == RETURNED_COUNT_ERROR
    assert "signal: returned-string-count-withheld" in caplog.text
    assert page.counted == [("id=cb", None)]


async def test_count_one_returns_the_same_object():
    result = _found()
    out = await _gated(result)(1, 2, page=_FakePage(1))
    assert out is result


async def test_collection_count_three_is_kept():
    result = _found(unique=False)
    out = await _gated(result)(1, 2, page=_FakePage(3))
    assert out is result


async def test_collection_count_zero_is_withheld():
    out = await _gated(_found(unique=False))(1, 2, page=_FakePage(0))
    assert out["found"] is False
    assert out["best_locator"] is None


async def test_count_that_raises_is_withheld():
    out = await _gated(_found())(1, 2, page=_FakePage(RuntimeError("bad selector")))
    assert out["found"] is False
    assert out["error"] == RETURNED_COUNT_ERROR


async def test_framed_result_is_counted_in_the_frame_with_the_hop_stripped():
    page = _FakePage(1)
    out = await _gated(_found("css=.x"))(1, 2, page=page, iframe_context=FRAME)
    assert out["best_locator"] == f"{FRAME} >>> css=.x"
    assert page.counted == [("css=.x", FRAME)]


async def test_no_page_leaves_the_result_alone():
    result = _found()
    out = await _gated(result)(1, 2, page=None)
    assert out is result


async def test_not_found_result_is_not_counted():
    page = _FakePage(2)
    result = {"found": False, "best_locator": "id=cb"}
    out = await _gated(result)(1, 2, page=page)
    assert out is result
    assert page.counted == []


async def test_robot_unsafe_best_is_withheld_without_a_count():
    page = _FakePage(1)
    out = await _gated(_found("id=cb" + chr(92) + "x"))(1, 2, page=page)
    assert out["error"] == ROBOT_UNSAFE_ERROR
    assert page.counted == []


@pytest.mark.integration
@pytest.mark.fixture
async def test_duplicate_id_is_withheld_and_the_label_locator_is_kept():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.set_content(DUP_PAGE)
            dup = _found("id=rm")
            out = await _gated(dup)(1, 2, page=page)
            assert out["found"] is False
            assert out["best_locator"] is None

            ok = _found('label:text-is("Remember me") >> input[type="checkbox"]')
            assert await _gated(ok)(1, 2, page=page) is ok
        finally:
            await browser.close()
