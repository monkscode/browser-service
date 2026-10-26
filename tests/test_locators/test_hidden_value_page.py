"""
The legacy (page) path of validate_semantic_match never reads back a hidden value
(NLRF #131 Part 2, owner decision D8). Real Chromium; made-up value only.
"""

import logging

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.smart_locator import validate_semantic_match

pytestmark = pytest.mark.integration

V = "Zq131-made-up-7"


async def _observe(html, caplog):
    caplog.set_level(logging.DEBUG)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.set_content(html)
            await page.locator("#f").fill(V)
            return await validate_semantic_match(None, "Sign in button", page=page, locator="#f")
        finally:
            await browser.close()


async def test_password_input_value_not_read(caplog):
    ok, observed = await _observe('<input id="f" type="password">', caplog)
    assert ok is False
    assert V not in observed
    assert V not in caplog.text


async def test_marked_text_input_value_not_read(caplog):
    ok, observed = await _observe('<input id="f" type="text" nlrf-secret="">', caplog)
    assert ok is False
    assert V not in observed
    assert V not in caplog.text


async def test_unmarked_text_input_value_still_read(caplog):
    """The known positive: the probe can see a value at all (accepted residual D4)."""
    _, observed = await _observe('<input id="f" type="text">', caplog)
    assert observed == V
