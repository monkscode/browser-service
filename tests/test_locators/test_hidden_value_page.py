"""
The legacy (page) path of validate_semantic_match never reads back a hidden value
(NLRF #131 Part 2, owner decision D8). Real Chromium; made-up value only.
"""

import json
import logging

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import _read_resolved_element
from browser_service.agent.registration import _mark_secret_field
from browser_service.locators.hidden_value import SECRET_MARK
from browser_service.locators.smart_locator import (
    find_unique_locator_at_coordinates,
    validate_semantic_match,
)

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


# ---- D12 / D13: a marked textarea or contenteditable keeps the value as CHILD text (what React
# leaves in the DOM), a label's mark goes to the field it controls. Set directly in the HTML.

TEXTAREA_HTML = '<textarea id="f" role="none" nlrf-secret="">' + V + "</textarea>"
EDITABLE_HTML = '<div id="f" contenteditable nlrf-secret="">' + V + "</div>"
UNMARKED_EDITABLE_HTML = '<div id="f" contenteditable>' + V + "</div>"


async def _on_page(html, fn):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.set_content(html)
            return await fn(page)
        finally:
            await browser.close()


@pytest.mark.parametrize("html", [TEXTAREA_HTML, EDITABLE_HTML], ids=["textarea", "editable"])
async def test_legacy_path_never_reads_marked_child_text(html, caplog):
    caplog.set_level(logging.DEBUG)

    async def run(page):
        return await validate_semantic_match(None, "Sign in button", page=page, locator="#f")

    ok, observed = await _on_page(html, run)
    assert ok is False
    assert V not in observed
    assert V not in caplog.text


async def test_legacy_path_unmarked_child_text_still_read(caplog):
    """The known positive: the probe can see child text at all."""
    caplog.set_level(logging.DEBUG)

    async def run(page):
        return await validate_semantic_match(None, "Sign in button", page=page, locator="#f")

    _, observed = await _on_page(UNMARKED_EDITABLE_HTML, run)
    assert observed == V


async def _coordinate_probe(html, caplog):
    caplog.set_level(logging.DEBUG)

    async def run(page):
        box = await page.locator("#f").bounding_box()
        return await find_unique_locator_at_coordinates(
            page,
            box["x"] + box["width"] / 2,
            box["y"] + box["height"] / 2,
            "elem_1",
            "Sign in button",
        )

    result = await _on_page(html, run)
    return json.dumps(result, default=str), caplog.text


@pytest.mark.parametrize("html", [TEXTAREA_HTML, EDITABLE_HTML], ids=["textarea", "editable"])
async def test_coordinate_path_never_reads_marked_child_text(html, caplog):
    dumped, log = await _coordinate_probe(html, caplog)
    assert V not in dumped
    assert V not in log


async def test_coordinate_path_unmarked_child_text_still_read(caplog):
    dumped, log = await _coordinate_probe(UNMARKED_EDITABLE_HTML, caplog)
    assert V in dumped or V in log


async def test_resolved_element_read_blanks_a_marked_textarea():
    async def run(page):
        return await _read_resolved_element(page, "#f")

    assert (await _on_page(TEXTAREA_HTML, run))["textContent"] == ""
    assert (await _on_page('<textarea id="f">' + V + "</textarea>", run))["textContent"] == V


async def test_mark_goes_to_the_field_a_label_points_to():
    html = '<label for="f" id="l">Token</label><input id="f">'

    async def run(page):
        await _mark_secret_field(page, "#l", None)
        return [
            await page.locator("#l").get_attribute(SECRET_MARK),
            await page.locator("#f").get_attribute(SECRET_MARK),
        ]

    assert await _on_page(html, run) == [None, ""]


async def test_mark_on_a_plain_field_is_unchanged():
    async def run(page):
        await _mark_secret_field(page, "#f", None)
        return await page.locator("#f").get_attribute(SECRET_MARK)

    assert await _on_page('<input id="f">', run) == ""
