"""
FIT_FACTS_JS on real Chromium: the facts action_misfit() reasons over must
be what the browser actually reports (disabled fieldsets, aria-disabled
ancestors, readonly, contenteditable, the iframe that hosts an editor).
"""

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.action_fit import check_action_fit

pytestmark = pytest.mark.integration

HTML = """<!doctype html><html><body>
<input id="t" type="text">
<input id="cb" type="checkbox">
<input id="ro" type="text" readonly>
<input id="dis" type="text" disabled>
<textarea id="ta"></textarea>
<div id="ce" contenteditable="true">edit me</div>
<div id="twin" aria-label="Search for Products, Brands and More">wrapper</div>
<iframe id="mce_0_ifr" srcdoc="<body contenteditable='true'><p>Your content goes here.</p></body>"></iframe>
<div aria-disabled="true"><div id="opt" role="option">Blue</div></div>
<button id="btn" disabled>Save</button>
<fieldset disabled><input id="fs" type="text"></fieldset>
<input id="empty" placeholder="What needs to be done?">
<input id="filled" value="NewYork">
<label id="lbl" for="t">Name</label>
</body></html>"""


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page_obj = await browser.new_page()
        await page_obj.set_content(HTML)
        try:
            yield page_obj
        finally:
            await browser.close()


@pytest.mark.parametrize(
    "selector, action, expected_text, want",
    [
        ("#t", "input", None, ""),
        ("#cb", "input", None, "cannot be filled"),
        ("#ro", "input", None, "read-only"),
        ("#dis", "input", None, "disabled"),
        ("#ta", "input", None, ""),
        ("#ce", "input", None, ""),
        ("#twin", "input", None, "cannot be filled"),
        ("#mce_0_ifr", "input", None, "cannot be filled"),
        ("#opt", "click", None, "aria-disabled"),
        ("#btn", "click", None, "disabled"),
        ("#fs", "input", None, "disabled"),
        ("#empty", "get_text", "buy milk", "empty"),
        ("#filled", "get_text", "NewYork", ""),
        ("#lbl", "input", None, ""),
    ],
)
async def test_fitness_on_real_dom(page, selector, action, expected_text, want):
    reason = await check_action_fit(page, selector, action, expected_text)
    if want:
        assert want in reason, reason
    else:
        assert reason == "", reason


async def test_editor_body_inside_the_iframe_fits(page):
    frame = page.frame_locator("#mce_0_ifr")
    assert await check_action_fit(frame, "body", "input") == ""
