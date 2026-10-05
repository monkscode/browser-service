"""
F1 read_change on real Chromium (inline HTML via page.set_content): whether a read's own
locator, evaluated on the HTML kept before an action, showed other text than it shows now.
"""

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.read_change import read_changed_since, snapshot_html

pytestmark = pytest.mark.integration


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page_obj = await (
            await browser.new_context(viewport={"width": 1280, "height": 900})
        ).new_page()
        try:
            yield page_obj
        finally:
            await browser.close()


GRID_BEFORE = "<ul id='grid'><li class='item'><span class='name'>Phone A</span></li><li class='item'><span class='name'>Phone B</span></li></ul>"
GRID = "css=#grid >> li.item >> nth=0 >> span.name"


async def _run(page, before, mutate_js, locator, after_doc=None):
    await page.set_content(before)
    html = await snapshot_html(page)
    if after_doc is not None:
        await page.set_content(after_doc)
    else:
        await page.evaluate(mutate_js)
    return await read_changed_since(page, locator, html)


async def test_replaced_with_other_text_is_changed(page):  # demoblaze Laptops
    js = "document.getElementById('grid').innerHTML = \"<li class='item'><span class='name'>Laptop A</span></li>\""
    assert await _run(page, GRID_BEFORE, js, GRID) == "changed"


async def test_same_text_after_rerender_is_same(page):  # demoblaze Phones
    js = "document.getElementById('grid').innerHTML = \"<li class='item'><span class='name'>Phone A</span></li>\""
    assert await _run(page, GRID_BEFORE, js, GRID) == "same"


async def test_in_place_text_update_is_changed(page):  # cart badge 1 -> 2
    js = "document.querySelector('#grid span.name').textContent = 'Laptop A'"
    assert await _run(page, GRID_BEFORE, js, GRID) == "changed"


async def test_absent_before_is_absent(page):  # saucedemo .title, verifier #newrow
    js = "document.body.insertAdjacentHTML('beforeend', \"<p id='late'>Late</p>\")"
    assert await _run(page, GRID_BEFORE, js, "id=late") == "absent"


async def test_row_inserted_above_keeps_same(page):  # verifier #x
    before = "<div id='list'><div class='row'>A</div><div class='row' id='x'>B</div><div class='row'>C</div></div>"
    js = "var d=document.createElement('div');d.className='row';d.textContent='Z';document.getElementById('list').prepend(d)"
    assert await _run(page, before, js, "id=x") == "same"


async def test_new_document_with_other_text_is_changed(page):  # navigation (books Travel)
    after = GRID_BEFORE.replace("Phone A", "Laptop A")
    assert await _run(page, GRID_BEFORE, None, GRID, after_doc=after) == "changed"


async def test_two_matches_before_is_ambiguous(page):  # flipkart: 19 matches before
    before = "<p class='t'>a</p><p class='t'>b</p>"
    js = "document.querySelectorAll('p.t')[1].remove()"
    assert await _run(page, before, js, "css=p.t") == "ambiguous"


async def test_xpath_address_is_evaluated(page):
    js = "document.querySelector('#grid span.name').textContent = 'Laptop A'"
    assert await _run(page, GRID_BEFORE, js, "xpath=//ul[@id='grid']/li[1]/span") == "changed"


async def test_unsupported_engine_is_unknown(
    page,
):  # text= / :has-text cannot be evaluated on parsed HTML
    js = "document.querySelector('#grid span.name').textContent = 'Laptop A'"
    assert await _run(page, GRID_BEFORE, js, "span:has-text('Laptop A')") == "unknown"


CARD_BEFORE = "<div id='card'><span>Phone A</span><noscript><img src='a.png'><b>fallback</b></noscript></div><p id='other'>x</p>"


async def test_unchanged_element_with_noscript_is_same(
    page,
):  # DOMParser parses <noscript> children as elements
    js = "document.getElementById('other').textContent = 'y'"
    assert await _run(page, CARD_BEFORE, js, "id=card") == "same"


async def test_changed_text_beside_noscript_is_changed(page):
    js = "document.querySelector('#card span').textContent = 'Laptop A'"
    assert await _run(page, CARD_BEFORE, js, "id=card") == "changed"


async def test_a_noscript_element_itself_is_unknown(page):
    js = "document.getElementById('other').textContent = 'y'"
    assert await _run(page, CARD_BEFORE, js, "css=#card >> noscript") == "unknown"
