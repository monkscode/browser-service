"""
E2b on real Chromium, through find_unique_locator_action, on the amazon u01
shape: each accept path (element_data, agent candidate, text-first) must
end on a container-ordinal locator; a long static heading must not.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import find_unique_locator_action
from browser_service.config import config

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent / "locator_fixtures"
CARD = '[data-component-type="s-search-result"]'
NAMES = [
    "Portronics Toad 23 Wireless Optical Mouse with 2.4GHz USB Nano Dongle",
    "Logitech B170 Wireless Mouse, 2.4 GHz with USB Nano Receiver",
    "Zebronics Zeb-Transformer-M Wireless Gaming Mouse with RGB Lights",
]
HEADING = "Results for wireless mouse in Computers and Accessories"

H2_DATA_JS = """(el) => ({
    tagName: el.tagName.toLowerCase(),
    id: el.id || "",
    className: (typeof el.className === 'string') ? el.className : "",
    ariaLabel: el.getAttribute('aria-label') || "",
    textContent: (el.textContent || "").trim().slice(0, 120),
})"""


async def _center(page, selector: str):
    box = await page.locator(selector).first.bounding_box()
    assert box, selector
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page_obj = await ctx.new_page()
        await page_obj.goto(
            (FIXTURES_DIR / "product_card_aria_label.html").resolve().as_uri(),
            wait_until="domcontentloaded",
        )
        try:
            yield page_obj
        finally:
            await browser.close()


@pytest.fixture
def action_fit_on():
    with patch.object(config.locator, "enable_action_fit", True):
        yield


async def _read(page, target, expected, action, candidate=None, element_data=None):
    x, y = await _center(page, target)
    return await find_unique_locator_action(
        x=x,
        y=y,
        element_id="elem_2",
        element_description="the name of the product in the results",
        expected_text=expected,
        candidate_locator=candidate,
        element_data=element_data,
        page=page,
        is_collection=False,
        action=action,
    )


async def test_element_data_path_today_embeds_the_name(page):
    data = await page.locator(f"{CARD} >> nth=0 >> h2").evaluate(H2_DATA_JS)
    out = await _read(page, f"{CARD} >> nth=0 >> h2", NAMES[0], None, element_data=data)
    assert out["best_locator"] == f'[aria-label="{NAMES[0]}"]'


async def test_element_data_path_rewrites_to_the_card(page, action_fit_on):
    data = await page.locator(f"{CARD} >> nth=0 >> h2").evaluate(H2_DATA_JS)
    out = await _read(page, f"{CARD} >> nth=0 >> h2", NAMES[0], "get_text", element_data=data)
    assert out["best_locator"] == f"{CARD} >> nth=0 >> h2"
    assert len(out["all_locators"]) == 1  # nothing left for PHASE-2 to re-promote
    assert (await page.locator(out["best_locator"]).inner_text()).strip() == NAMES[0]


async def test_candidate_path_rewrites_to_the_card(page, action_fit_on):
    candidate = "a[title*='Logitech B170 Wireless Mouse, 2.4 GHz with USB']"  # u02 r2 shape
    out = await _read(page, f"{CARD} >> nth=1 >> a", NAMES[1], "get_text", candidate=candidate)
    assert out["best_locator"].startswith(f"{CARD} >> nth=1 >> ")
    assert NAMES[1] in await page.locator(out["best_locator"]).inner_text()


async def test_text_first_path_rewrites_to_the_card(page, action_fit_on):
    out = await _read(page, f"{CARD} >> nth=2 >> h2", NAMES[2], "get_text")
    assert out["best_locator"].startswith(f"{CARD} >> nth=2 >> ")
    assert (await page.locator(out["best_locator"]).inner_text()).strip() == NAMES[2]


async def test_long_static_heading_is_left_alone(page, action_fit_on):
    out = await _read(page, "h1.s-title", HEADING, "get_text")
    assert out["best_locator"] == f'text="{HEADING}"'
    assert "read_target_rewritten_from" not in out
