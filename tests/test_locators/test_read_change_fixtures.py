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


# --- final-review fixes -------------------------------------------------------------------------


async def _run_built(page, build_js, action_js, locator):
    """A page built by JavaScript (the DOM may hold nesting the HTML parser never produces)."""
    await page.set_content("<div id='root'></div><p id='other'>x</p>")
    await page.evaluate(build_js)
    html = await snapshot_html(page)
    await page.evaluate(action_js)  # an action that does not touch the read
    return await read_changed_since(page, locator, html)


UNRELATED = "document.getElementById('other').textContent = 'y'"


async def test_p_holding_a_div_built_by_js_is_unknown(page):  # React validateDOMNesting shape
    build = """
      const p = document.createElement('p'); p.className = 'price';
      const d = document.createElement('div'); d.textContent = '$10';
      p.appendChild(d); document.getElementById('root').appendChild(p);
    """
    assert await _run_built(page, build, UNRELATED, "css=p.price") == "unknown"


async def test_anchor_holding_an_anchor_built_by_js_is_unknown(page):
    build = """
      const a = document.createElement('a'); a.className = 'card'; a.href = '#c';
      a.append('Phone A ');
      const i = document.createElement('a'); i.href = '#s'; i.textContent = 'Seller';
      a.appendChild(i); a.append(' $10');
      document.getElementById('root').appendChild(a);
    """
    assert await _run_built(page, build, UNRELATED, "css=a.card") == "unknown"


async def test_positional_xpath_below_a_misnested_block_is_unknown(page):
    build = """
      const root = document.getElementById('root');
      const p = document.createElement('p');
      const d = document.createElement('div'); d.textContent = 'banner';
      p.appendChild(d); root.appendChild(p);
      const t = document.createElement('div'); t.textContent = 'target'; root.appendChild(t);
    """
    assert await _run_built(page, build, UNRELATED, "xpath=//div[@id='root']/div[1]") == "unknown"


async def test_two_live_matches_is_unknown(page):  # live-identity guard
    before = "<p class='t'>a</p>"
    js = "document.body.insertAdjacentHTML('beforeend', \"<p class='t'>b</p>\"); document.querySelector('p.t').textContent='z'"
    assert await _run(page, before, js, "css=p.t") == "unknown"


async def test_an_element_in_an_open_shadow_root_is_unknown(page):  # live-identity guard
    await page.set_content("<div id='h'></div>")
    await page.evaluate(
        "document.getElementById('h').attachShadow({mode:'open'}).innerHTML = \"<span id='s'>old</span>\""
    )
    html = await snapshot_html(page)
    await page.evaluate(
        "document.getElementById('h').shadowRoot.getElementById('s').textContent='new'"
    )
    assert await read_changed_since(page, "css=#s", html) == "unknown"


async def test_text_engine_is_unknown(page):  # refused by the engine whitelist
    js = "document.getElementById('x').textContent='new'"
    assert await _run(page, "<p id='x'>old</p>", js, "text=new") == "unknown"


async def test_visible_true_suffix_is_unknown(page):  # engine whitelist; bs's containment collapse
    js = "document.querySelector('p.t').textContent='new'"
    assert (
        await _run(page, "<p class='t'>old</p>", js, "css=p.t >> visible=true >> nth=0")
        == "unknown"
    )


async def test_nth_minus_one_is_evaluated(page):
    js = "document.querySelectorAll('li')[1].textContent='c'"
    assert await _run(page, "<ul><li>a</li><li>b</li></ul>", js, "css=li >> nth=-1") == "changed"


async def test_xpath_after_css_is_scoped_to_the_css_match(page):  # Playwright prefixes '.'
    before = "<span>Old</span><div id='grid'></div>"
    after = "<div id='grid'><span>New</span></div>"
    assert await _run(page, before, None, "css=#grid >> xpath=//span", after_doc=after) == "absent"


async def test_hidden_text_change_is_unknown(page):  # Get Text reads innerText, not textContent
    before = "<div id='card'><span>Phone A</span><span id='h' style='display:none'>0</span></div>"
    js = "document.getElementById('h').textContent = '1'"
    assert await _run(page, before, js, "id=card") == "unknown"


async def test_page_without_a_doctype_is_re_parsed_in_its_own_mode(page):  # quirks mode
    before = "<p id='x'>Total <table><tr><td>10</td></tr></table></p>"
    js = "document.querySelector('td').textContent = '12'"
    assert await _run(page, before, js, "id=x") == "changed"


async def test_only_the_live_identity_guard_decides_a_shadow_last_match(page):
    # Playwright's last css=span.s match is the span in the open shadow root; a plain
    # querySelectorAll on the light DOM finds B. Only the guard sees the two differ.
    await page.set_content("<div id='h'></div><span class='s'>B</span>")
    await page.evaluate(
        "document.getElementById('h').attachShadow({mode:'open'}).innerHTML = \"<span class='s'>A</span>\""
    )
    html = await snapshot_html(page)
    await page.evaluate("document.querySelector('body > span.s').textContent = 'A'")
    assert await read_changed_since(page, "css=span.s >> nth=-1", html) == "unknown"
