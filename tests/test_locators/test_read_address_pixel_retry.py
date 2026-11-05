"""bs #75 E1: an index-less read the gate cannot confirm at the agent's point read as a 0-1000 grid point is tried once more at the raw point read as pixels — last, and only when exactly one element shows the observed text (G1).

Viewport 1280x900. The agent's raw point is (960, 122) — the message "Access granted". Scaled as a grid
point it is (int(960/1000*1280), int(122/1000*900)) = (1228, 109): the "Cart total" box, a different element.
"""

from unittest.mock import AsyncMock, patch

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import find_unique_locator_action
from browser_service.agent.registration import _pixel_point
from browser_service.config import config
from browser_service.locators import read_target
from browser_service.locators.read_target import carries_value, resolve_read_address

pytestmark = pytest.mark.integration

RAW = (960, 122)  # the agent's raw point; the message's centre
GRID = (1228, 109)  # RAW scaled as a 0-1000 grid point on a 1280x900 viewport: the box
VIEWPORT = (1280, 900)

STYLE = (
    "<style>body{margin:0} .abs{position:absolute;height:44px;margin:0}"
    " #msg{left:900px;top:100px;width:120px} #box{left:1180px;top:80px;width:100px;height:60px}"
    " #a{left:100px;top:300px;width:80px} #b{left:400px;top:300px;width:80px}</style>"
)
MESSAGE_ONCE = (
    STYLE + '<p id="msg" class="abs">Access granted</p><div id="box" class="abs">Cart total</div>'
)
MESSAGE_TWICE = (
    MESSAGE_ONCE + '<p id="twin" class="abs" style="left:100px;top:600px">Access granted</p>'
)
PRICE_TWICE = STYLE + (
    '<p id="a" class="abs">$10</p><p id="b" class="abs">$10</p>'
    '<div id="box" class="abs">Sort by</div>'
)
PRICE_ONCE = STYLE + '<p id="a" class="abs">$10</p><div id="box" class="abs">Sort by</div>'


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]})
        page_obj = await ctx.new_page()
        try:
            yield page_obj
        finally:
            await browser.close()


@pytest.fixture
def action_fit_on():
    with patch.object(config.locator, "enable_action_fit", True):
        yield


async def _centre(page, selector):
    box = await page.locator(selector).bounding_box()
    return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


async def test_geometry_of_the_fixture(page):
    """The positive case needs the grid point and the pixel point on different elements."""
    await page.set_content(MESSAGE_ONCE)
    on_grid = await page.evaluate("([x, y]) => document.elementFromPoint(x, y).id", list(GRID))
    on_raw = await page.evaluate("([x, y]) => document.elementFromPoint(x, y).id", list(RAW))
    assert (on_grid, on_raw) == ("box", "msg")


async def test_pixel_only_page_is_refused_today_and_confirmed_with_the_pixel_point(page):
    await page.set_content(MESSAGE_ONCE)
    today = await resolve_read_address(page, "Access granted", None, GRID)
    assert "unconfirmed" in today and "cart total" in today["unconfirmed"]
    verdict = await resolve_read_address(page, "Access granted", None, GRID, pixel_point=RAW)
    assert verdict["source"] == "screen point read as pixels"
    assert not carries_value(verdict["locator"], "Access granted")
    assert await page.locator(verdict["locator"]).inner_text() == "Access granted"


async def test_a_grid_read_confirmed_today_never_reads_the_pixel_source(page):
    await page.set_content(MESSAGE_ONCE)
    with patch.object(read_target, "_pixel_source", new=AsyncMock()) as pixel:
        verdict = await resolve_read_address(
            page, "Access granted", None, await _centre(page, "#msg"), pixel_point=GRID
        )
    assert verdict["source"] == "screen point"
    pixel.assert_not_awaited()


async def test_a_pixel_point_showing_other_text_is_still_refused(page):
    await page.set_content(MESSAGE_ONCE)
    other = await _centre(page, "#box")
    verdict = await resolve_read_address(page, "Access granted", None, GRID, pixel_point=other)
    assert "unconfirmed" in verdict
    assert "cart total" in verdict["unconfirmed"]


async def test_g1_same_text_twice_keeps_the_refusal(page):
    """Two cards show '$10'; the grid point is the sort box, the pixel point is card b: the wrong-element risk."""
    await page.set_content(PRICE_TWICE)
    grid = await _centre(page, "#box")
    pixel = await _centre(page, "#b")
    verdict = await resolve_read_address(page, "$10", None, grid, pixel_point=pixel)
    assert "unconfirmed" in verdict, verdict


async def test_g1_same_text_once_is_confirmed(page):
    await page.set_content(PRICE_ONCE)
    grid = await _centre(page, "#box")
    pixel = await _centre(page, "#a")
    verdict = await resolve_read_address(page, "$10", None, grid, pixel_point=pixel)
    assert verdict["source"] == "screen point read as pixels"
    assert await page.locator(verdict["locator"]).inner_text() == "$10"


async def test_g1_a_lookup_failure_skips_the_source(page):
    await page.set_content(MESSAGE_ONCE)
    with patch.object(page, "get_by_text", side_effect=RuntimeError("boom")):
        verdict = await resolve_read_address(page, "Access granted", None, GRID, pixel_point=RAW)
    assert "unconfirmed" in verdict


async def _flow(page, pixel_point, candidate="text=Access granted"):
    return await find_unique_locator_action(
        x=GRID[0],
        y=GRID[1],
        element_id="elem_1",
        element_description="the message",
        expected_text="Access granted",
        candidate_locator=candidate,
        element_data=None,
        page=page,
        is_collection=False,
        action="get_text",
        vision_point=GRID,
        pixel_point=pixel_point,
    )


async def test_flow_candidate_path_refused_today_confirmed_with_the_pixel_point(
    page, action_fit_on
):
    """Run 58919ce7's shape end to end: the agent's text candidate, the grid point on the wrong element."""
    await page.set_content(MESSAGE_ONCE)
    today = await _flow(page, None)
    assert today["found"] is False and "read_address_unconfirmed" in today
    result = await _flow(page, RAW)
    assert result["found"] is True
    assert result["read_address_source"] == "screen point read as pixels"
    assert not carries_value(result["best_locator"], "Access granted")
    assert await page.locator(result["best_locator"]).inner_text() == "Access granted"


async def test_flow_candidate_path_same_text_twice_stays_refused(page, action_fit_on):
    await page.set_content(MESSAGE_TWICE)
    result = await _flow(page, RAW, candidate="text=Access granted >> nth=0")
    assert result["found"] is False
    assert "read_address_unconfirmed" in result


async def test_flow_cascade_path_forwards_the_pixel_point(page, action_fit_on):
    """The cascade path hands pixel_point to the gate (a real cascade here fails its own text check first)."""
    from browser_service.agent import actions

    await page.set_content(MESSAGE_ONCE)

    async def passthrough(_ctx, result, *_args, **_kwargs):
        return result

    gate = AsyncMock(side_effect=passthrough)
    with patch.object(actions, "apply_read_target_policy", new=gate):
        await _flow(page, RAW, candidate=None)
    assert gate.await_args.kwargs["pixel_point"] == RAW


async def test_pixel_signal_is_logged_when_the_source_confirms(page, caplog):
    await page.set_content(MESSAGE_ONCE)
    with caplog.at_level("INFO", logger=read_target.logger.name):
        await resolve_read_address(page, "Access granted", None, GRID, pixel_point=RAW)
    assert "signal: read-address-pixel-point" in caplog.text


class _P:
    def __init__(self, x, y, element_index=None):
        self.x, self.y, self.element_index = x, y, element_index


@pytest.mark.parametrize(
    "params, scaled, expected",
    [
        (_P(960, 122), (1228, 109), (960, 122)),  # index-less, inside, differs
        (_P(960, 122, element_index=7), (1228, 109), None),  # indexed: no pixel source
        (_P(1500, 122), (1920, 109), None),  # x outside the viewport
        (_P(960, 950), (1228, 855), None),  # y outside the viewport
        (_P(0, 0), (0, 0), None),  # equal to the scaled point
        (_P(1280, 10), (1638, 9), None),  # x == viewport width is outside
    ],
)
def test_pixel_point_rules(params, scaled, expected):
    assert _pixel_point(params, scaled[0], scaled[1], *VIEWPORT) == expected
