"""
The read-address reason never prints text typed into a hidden-value field
(NLRF #131 Part 2, owner decision D11). Real Chromium; made-up value only.
"""

import json
import logging

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.read_target import (
    apply_read_target_policy,
    normalize_text,
    resolve_read_address,
)

pytestmark = pytest.mark.integration

V = "Zq131-made-up-7"

XPATH_JS = """el => { const p = []; for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
  let i = 1; for (let s = n.previousElementSibling; s; s = s.previousElementSibling) if (s.tagName === n.tagName) i++;
  p.unshift(n.tagName.toLowerCase() + '[' + i + ']'); } return p.join('/'); }"""


async def _probe(html, caplog, fill=True, via_policy=False):
    caplog.set_level(logging.DEBUG)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.set_content(html)
            if fill:
                await page.locator("#f").fill(V)
            data = {"xpath": await page.locator("#f").evaluate(XPATH_JS)}
            if via_policy:
                result = {"found": True, "best_locator": 'text="Account secret"'}
                return await apply_read_target_policy(
                    page, result, "get_text", "Account secret", element_data=data
                )
            return await resolve_read_address(page, "Account secret", data, None)
        finally:
            await browser.close()


def _no_value(verdict, caplog):
    assert "unconfirmed" in verdict or verdict.get("found") is False
    for text in (json.dumps(verdict), caplog.text):
        assert V not in text
        assert normalize_text(V) not in text


@pytest.mark.parametrize(
    "html,fill",
    [
        ('<input id="f" type="password">', True),
        ('<input id="f" type="text" nlrf-secret="">', True),
        ('<textarea id="f" nlrf-secret="">' + V + "</textarea>", False),
        ('<div id="f" contenteditable nlrf-secret="">' + V + "</div>", False),
    ],
    ids=["password", "marked-text", "marked-textarea-child", "marked-contenteditable"],
)
async def test_hidden_field_text_not_in_reason(html, fill, caplog):
    _no_value(await _probe(html, caplog, fill=fill), caplog)


async def test_hidden_field_text_not_in_policy_result_or_log(caplog):
    verdict = await _probe('<input id="f" type="password">', caplog, via_policy=True)
    assert verdict["found"] is False
    _no_value(verdict, caplog)


async def test_unmarked_text_input_value_still_shown(caplog):
    """The known positive: the probe sees a value at all (accepted residual D4)."""
    verdict = await _probe('<input id="f" type="text">', caplog)
    assert normalize_text(V) in verdict["unconfirmed"]


async def test_hidden_field_placeholder_still_reported(caplog):
    verdict = await _probe('<input id="f" type="password" placeholder="Password">', caplog)
    assert "'password'" in verdict["unconfirmed"]
    assert normalize_text(V) not in verdict["unconfirmed"]
