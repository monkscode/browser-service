"""
NLRF #162 on real Chromium (inline HTML): a locate-only step (wait / hover / count) keeps its text
locator when the text lands on the element the agent pointed at, is re-aimed when it lands on
another element, and is never refused. A real read of the same pages is rewritten as before.
"""

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.read_target import apply_read_target_policy, carries_value

pytestmark = pytest.mark.integration

XPATH_JS = """el => { const p = []; for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
  let i = 1; for (let s = n.previousElementSibling; s; s = s.previousElementSibling) if (s.tagName === n.tagName) i++;
  p.unshift(n.tagName.toLowerCase() + '[' + i + ']'); } return p.join('/'); }"""

# The heading's word sits one level inside the element the agent points at (the OrangeHRM shape).
HEADING = """<header><div class="topbar-title"><span><h6 class="crumb">Dashboard</h6></span></div></header>
<nav><a href="#">Dashboard</a><a href="#">PIM</a></nav>"""

# The product name and its image are different elements; the image's alt holds the name.
CARD = """<div class="item"><a id="img-link"><img data-test="backpack-img" alt="Sauce Labs Backpack"
  width="40" height="40"></a><a id="title-link"><div class="name">Sauce Labs Backpack</div></a></div>"""

# The issue's own example: a labelled button the gate re-addressed by position.
CART = (
    """<nav class="bar"><button aria-label="Go to cart">Cart</button><button>Help</button></nav>"""
)


async def _policy(html, locator, expected, pointed, locate_only):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.set_content(html)
            data = {"xpath": await page.locator(pointed).evaluate(XPATH_JS)}
            out = await apply_read_target_policy(
                page,
                {"found": True, "best_locator": locator, "element_id": "elem_1"},
                "get_text",
                expected,
                element_data=data,
                locate_only=locate_only,
            )
            if out.get("found"):
                assert await page.locator(out["best_locator"]).count() == 1
                out["_tag"] = await page.locator(out["best_locator"]).evaluate("e => e.tagName")
            return out
        finally:
            await browser.close()


@pytest.mark.parametrize(
    "html, locator, expected, pointed",
    [
        (HEADING, "h6:has-text('Dashboard')", "Dashboard", ".topbar-title"),
        (CART, '[aria-label="Go to cart"]', "Go to cart", "nav.bar button >> nth=0"),
    ],
)
async def test_a_wait_keeps_the_text_that_names_the_pointed_element(
    html, locator, expected, pointed
):
    out = await _policy(html, locator, expected, pointed, locate_only=True)
    assert out["best_locator"] == locator
    assert "read_target_rewritten_from" not in out


async def test_a_wait_is_re_aimed_when_the_text_lands_on_another_element():
    """'verify the Sauce Labs Backpack image is displayed': the text finds the NAME, the agent pointed at the IMAGE."""
    out = await _policy(
        CARD, 'text="Sauce Labs Backpack"', "Sauce Labs Backpack", "img", locate_only=True
    )
    assert out["_tag"] == "IMG"
    assert out["read_target_rewritten_from"] == 'text="Sauce Labs Backpack"'


async def test_a_wait_is_never_refused():
    """The pointed element shows other words: a read is refused, a wait keeps its validated locator."""
    args = (HEADING, "h6:has-text('Dashboard')", "Dashboard", "nav a >> nth=1")
    assert (await _policy(*args, locate_only=False))["found"] is False
    out = await _policy(*args, locate_only=True)
    assert out["found"] is True and out["best_locator"] == "h6:has-text('Dashboard')"


@pytest.mark.parametrize(
    "html, locator, expected, pointed",
    [
        (HEADING, "h6:has-text('Dashboard')", "Dashboard", ".topbar-title"),
        (CART, '[aria-label="Go to cart"]', "Go to cart", "nav.bar button >> nth=0"),
    ],
)
async def test_a_real_read_of_the_same_page_is_still_rewritten(html, locator, expected, pointed):
    out = await _policy(html, locator, expected, pointed, locate_only=False)
    assert out["read_target_rewritten_from"] == locator
    assert not carries_value(out["best_locator"], expected)
