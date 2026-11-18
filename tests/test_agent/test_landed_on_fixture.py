"""bs #91 on real Chromium: the cascade's "landed on BODY / HTML" failure names the tag in ``landed_on``.

The agent sends page-pixel points, bs reads them as 0-1000 grid points and the point lands on
empty page. Registration reads ``landed_on`` to name the unit in the reply. The value is exactly
what the cascade's tag check holds: ``"body"`` or ``"html"``.

    pytest tests/test_agent/test_landed_on_fixture.py -m fixture
"""

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import find_unique_locator_action

pytestmark = [pytest.mark.integration, pytest.mark.fixture]

ABSENT_TEXT = "Checkout total"  # not on any page below; a call with no element data and no text is refused first

PAGE_TALL_BODY = '<body style="margin:0;min-height:100vh"><p>Hello there</p></body>'
# The doctype matters: without it Chromium's quirks mode stretches <body> to the viewport and the point lands on BODY.
PAGE_SHORT_BODY = "<!doctype html><body><p>Hello there</p></body>"


@pytest.fixture
async def browser():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True)
        try:
            yield b
        finally:
            await b.close()


async def _call_at(browser, html, point):
    """One no-evidence-but-text call at ``point`` on ``html``, as registration.py makes it."""
    ctx = await browser.new_context(viewport={"width": 1920, "height": 1080})
    try:
        page = await ctx.new_page()
        await page.set_content(html, wait_until="domcontentloaded")
        return await find_unique_locator_action(
            x=point[0],
            y=point[1],
            element_id="elem_1",
            element_description="the checkout total",
            expected_text=ABSENT_TEXT,
            candidate_locator=None,
            element_data=None,
            page=page,
            iframe_context=None,
            is_collection=None,
            browser_session=None,
            action="get_text",
            vision_point=point,
        )
    finally:
        await ctx.close()


async def test_point_on_the_body_names_body(browser):
    """A full-height body: the point lands on <body>."""
    out = await _call_at(browser, PAGE_TALL_BODY, (960, 600))
    assert out["found"] is False
    assert out["landed_on"] == "body"


async def test_point_below_a_short_body_names_html(browser):
    """A short body: the point below it lands on <html>."""
    out = await _call_at(browser, PAGE_SHORT_BODY, (960, 900))
    assert out["found"] is False
    assert out["landed_on"] == "html"


async def test_control_a_point_on_an_element_carries_no_landed_on(browser):
    """Control, unchanged by the fix: the point is on the <p>, so the failure is not a
    BODY / HTML landing and the key is absent. (The found outcome is asserted as observed below.)"""
    out = await _call_at(browser, PAGE_SHORT_BODY, (30, 20))
    assert out["found"] is False
    assert "landed_on" not in out
