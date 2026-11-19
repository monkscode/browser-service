"""
bs #73 option (b) on real Chromium: a READ inside an iframe no longer ships a locator built from the value it reads.

In a frame the read locator (frame hop stripped) must match exactly once inside the frame, show the
observed text by its own text, and have a structural address; the result then gets the frame back.
"""

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import _refresh_read_address_info
from browser_service.locators.read_target import apply_read_target_policy

pytestmark = pytest.mark.integration

FRAME = 'iframe[title="coupon-frame"]'
PREFIX = f"{FRAME} >>> "
LITERAL = f'{PREFIX}text="SAVE-1234"'

COUPON = '<p class="note">Your code</p><p class="coupon-code">SAVE-1234</p>'
TWICE = '<p class="coupon-code">SAVE-1234</p><p class="coupon-code">SAVE-1234</p>'
SHADOW = (
    '<div id="host"></div><script>document.getElementById("host").attachShadow({mode: "open"})'
    '.innerHTML = "<p>SAVE-1234</p>";</script>'
)


def _page_html(inner):
    return f"<h1>Shop</h1><iframe title=\"coupon-frame\" srcdoc='{inner}'></iframe>"


async def _gate(inner, locator, locate_only=False, expected="SAVE-1234", then=None):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.set_content(_page_html(inner))
            frame = page.frame_locator(FRAME)
            await frame.locator("body").wait_for()
            out = await apply_read_target_policy(
                frame,
                {
                    "found": True,
                    "best_locator": locator,
                    "all_locators": [{"locator": locator}],
                    "element_id": "elem_1",
                },
                "get_text",
                expected,
                FRAME,
                element_data={"xpath": "html[1]/body[1]/p[2]"},
                locate_only=locate_only,
            )
            if then:
                await then(page, frame, out)
            return out
        finally:
            await browser.close()


async def _after_the_coupon_changes(page, frame, out):
    inner = out["best_locator"].removeprefix(PREFIX)
    assert await frame.locator(inner).count() == 1
    await frame.locator(".coupon-code").evaluate("e => { e.textContent = 'SAVE-9999'; }")
    assert await frame.locator(inner).text_content() == "SAVE-9999"
    # control: the literal the old code shipped no longer finds anything
    assert await frame.locator('text="SAVE-1234"').count() == 0


async def test_a_framed_read_ships_the_frame_and_an_address_without_the_value():
    out = await _gate(COUPON, LITERAL, then=_after_the_coupon_changes)
    assert out["found"] is True
    assert out["best_locator"].startswith(PREFIX)
    assert "SAVE-1234" not in out["best_locator"]
    assert out["iframe_context"] == FRAME
    assert [e["locator"] for e in out["all_locators"]] == [out["best_locator"]]
    assert out["read_target_rewritten_from"] == LITERAL
    assert out["read_address_source"] == "single match in the frame"


async def test_two_matches_in_the_frame_are_not_found_with_the_count():
    out = await _gate(TWICE, LITERAL)
    assert out["found"] is False
    assert "2 elements" in out["read_address_unconfirmed"]
    assert out["read_address_rejected"] == LITERAL


async def test_a_framed_locate_only_step_is_untouched():
    out = await _gate(COUPON, LITERAL, locate_only=True)
    assert out["best_locator"] == LITERAL
    assert "read_target_rewritten_from" not in out


async def test_a_framed_value_free_locator_is_untouched():
    out = await _gate(COUPON, f"{PREFIX}.coupon-code")
    assert out["best_locator"] == f"{PREFIX}.coupon-code"
    assert "read_target_rewritten_from" not in out


async def test_a_match_in_a_shadow_root_keeps_the_locator():
    out = await _gate(SHADOW, LITERAL)
    assert out["best_locator"] == LITERAL
    assert "read_target_rewritten_from" not in out


async def test_a_single_match_that_does_not_show_the_text_is_not_found():
    # the locator finds one wrapper that holds the value in a child; the wrapper's own text is "Other"
    inner = '<div class="w">Other<p>SAVE-1234</p></div>'
    out = await _gate(inner, f'{PREFIX}div:has-text("SAVE-1234")')
    assert out["found"] is False
    assert "the match in the frame shows 'other save-1234', not" in out["read_address_unconfirmed"]


async def test_the_refresh_reads_the_rewritten_framed_element():
    seen = {}

    async def refresh(page, frame, out):
        await _refresh_read_address_info(out, frame)
        seen.update(out["element_info"])

    out = await _gate(COUPON, LITERAL, then=refresh)
    assert out["found"] is True
    assert seen["source"] == "read_address_resolved_element"
    assert seen["text"] == "SAVE-1234"
