"""
F1 read_change on real Chromium (inline HTML via page.set_content): whether a read's own
locator, evaluated on the HTML kept before an action, showed other text than it shows now.
"""

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.read_change import (
    CAME_BACK,
    NOT_CAME_BACK,
    UNCHECKED,
    read_came_back,
    read_changed_since,
    snapshot_html,
)

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


# --- came_back (R16): the read's live text equals its text on an earlier kept page, and a later
# kept page differs. read_came_back answers CAME_BACK (a real one), NOT_CAME_BACK, or UNCHECKED
# (a value that cannot be compared: the fail-safe) ---------------------------------------------


async def _came_back(page, kept_docs, live_doc, locator):
    """Keep one page per document in ``kept_docs`` (in order), then show ``live_doc`` live."""
    htmls = []
    for doc in kept_docs:
        await page.set_content(doc)
        htmls.append(await snapshot_html(page))
    await page.set_content(live_doc)
    return await read_came_back(page, locator, htmls)


def _cart(n):
    return f"<span id='cart'>{n}</span><p id='first'>Samsung galaxy s6</p>"


async def test_add_then_remove_came_back(page):  # cart 0, 1; live 0
    assert await _came_back(page, [_cart(0), _cart(1)], _cart(0), "id=cart") == CAME_BACK


async def test_add_alone_did_not_come_back(page):  # cart 0; live 1
    assert await _came_back(page, [_cart(0)], _cart(1), "id=cart") == NOT_CAME_BACK


async def test_type_then_search_button_did_not_come_back(page):
    popular = "<ul id='res'><li>Popular item</li></ul>"
    live = "<ul id='res'><li>Result for laptop</li></ul>"
    assert await _came_back(page, [popular, popular], live, "css=#res >> li") == NOT_CAME_BACK


async def test_z1_zero_one_zero_came_back(page):
    kept = [_cart(0), _cart(1), _cart(0)]
    assert await _came_back(page, kept, _cart(0), "id=cart") == CAME_BACK


async def test_z2_zero_one_live_one_did_not_come_back(page):
    assert await _came_back(page, [_cart(0), _cart(1)], _cart(1), "id=cart") == NOT_CAME_BACK


async def test_absent_before_did_not_come_back(page):
    live = "<p id='msg'>Welcome</p>"
    assert await _came_back(page, ["<p>nothing</p>"], live, "id=msg") == NOT_CAME_BACK


async def test_absent_after_a_match_came_back(page):  # [0, absent]; live 0
    gone = "<p>nothing</p>"
    assert await _came_back(page, [_cart(0), gone], _cart(0), "id=cart") == CAME_BACK


async def test_a_text_locator_is_not_comparable_so_unchecked(page):
    assert await _came_back(page, [_cart(0)], _cart(1), "text=1") == UNCHECKED


async def test_an_xpath_locator_is_evaluated(page):
    xp = "xpath=//span[@id='cart']"
    assert await _came_back(page, [_cart(0), _cart(1)], _cart(0), xp) == CAME_BACK
    assert await _came_back(page, [_cart(0)], _cart(1), xp) == NOT_CAME_BACK


async def test_a_gap_in_the_kept_pages_is_unchecked(page):
    await page.set_content(_cart(1))
    assert await read_came_back(page, "id=cart", [None]) == UNCHECKED


async def test_a_noscript_kept_match_is_unchecked(page):
    # The kept page's match is a <noscript> element (not comparable); the live match is a <p>.
    kept = "<noscript id='n'>hi</noscript>"
    live = "<p id='n'>hi</p>"
    assert await _came_back(page, [_cart(0), kept], live, "id=n") == UNCHECKED
    # a <noscript> element on the live side is not comparable either
    assert await _came_back(page, [kept], kept, "id=n") == UNCHECKED
    # a <noscript> INSIDE the target is ignored on both sides: comparable, text unchanged
    inside = "<div id='y'>A<noscript><b>x</b></noscript></div>"
    assert await _came_back(page, [inside], inside, "id=y") == NOT_CAME_BACK


async def test_hidden_live_text_is_unchecked(page):  # Get Text reads innerText, not textContent
    before = "<div id='card'><span>Phone A</span><span id='h' style='display:none'>0</span></div>"
    assert await _came_back(page, [before], before, "id=card") == UNCHECKED


async def test_read_came_back_answers_like_read_changed_since_on_one_page(page):
    # one engine: a one-page list is "came back" only when the read's text matches then and the
    # LATER page differs, so a single page can never be CAME_BACK while the engine says comparable.
    await page.set_content(_cart(0))
    html = await snapshot_html(page)
    await page.set_content(_cart(1))
    assert await read_changed_since(page, "id=cart", html) == "changed"
    assert await read_came_back(page, "id=cart", [html]) == NOT_CAME_BACK


# --- form controls: Get Text returns .value for INPUT / TEXTAREA, which the kept HTML does not
# hold and the engine's text never sees, so such a read is not comparable (final review) ------


def _qty(n):
    return f"<input id='qty' value='{n}'><span id='badge'>{n}</span>"


def _note(text):
    return f"<textarea id='note'>{text}</textarea>"


async def test_input_value_one_two_one_is_unchecked_not_did_not_come_back(page):
    # the quantity FIELD goes 1 -> 2 -> 1 (set through .value, as a +/- widget does)
    await page.set_content(_qty(1))
    first = await snapshot_html(page)
    await page.evaluate("document.getElementById('qty').value = '2'")
    second = await snapshot_html(page)
    await page.evaluate("document.getElementById('qty').value = '1'")
    assert await read_came_back(page, "id=qty", [first, second]) == UNCHECKED


async def test_textarea_a_b_a_is_unchecked(page):
    await page.set_content(_note("A"))
    first = await snapshot_html(page)
    await page.evaluate("document.getElementById('note').value = 'B'")
    second = await snapshot_html(page)
    await page.evaluate("document.getElementById('note').value = 'A'")
    assert await read_came_back(page, "id=note", [first, second]) == UNCHECKED


async def test_an_empty_textarea_typed_into_and_cleared_is_unchecked(page):
    # an EMPTY textarea has no text to compare (innerText == textContent == ''), so only the tag
    # rule catches it
    await page.set_content(_note(""))
    first = await snapshot_html(page)
    await page.evaluate("document.getElementById('note').value = 'B'")
    second = await snapshot_html(page)
    await page.evaluate("document.getElementById('note').value = ''")
    assert await read_came_back(page, "id=note", [first, second]) == UNCHECKED


async def test_an_input_read_after_one_action_is_unknown_and_unchecked(page):
    await page.set_content(_qty(1))
    html = await snapshot_html(page)
    await page.evaluate("document.getElementById('qty').value = '2'")
    assert await read_changed_since(page, "id=qty", html) == "unknown"
    assert await read_came_back(page, "id=qty", [html]) == UNCHECKED


async def test_a_textarea_read_after_one_action_is_unknown(page):
    await page.set_content(_note("A"))
    html = await snapshot_html(page)
    await page.set_content(_note("B"))
    assert await read_changed_since(page, "id=note", html) == "unknown"


async def test_a_text_badge_zero_one_zero_still_came_back(page):  # control
    kept = [_qty(0), _qty(1)]
    assert await _came_back(page, kept, _qty(0), "id=badge") == CAME_BACK


async def test_a_text_badge_zero_one_still_did_not_come_back(page):  # control
    assert await _came_back(page, [_qty(0)], _qty(1), "id=badge") == NOT_CAME_BACK


async def test_a_text_badge_zero_to_one_is_still_changed(page):  # control: the mark is unchanged
    await page.set_content(_qty(0))
    html = await snapshot_html(page)
    await page.set_content(_qty(1))
    assert await read_changed_since(page, "id=badge", html) == "changed"


# --- the backslash escape of an id= locator survives the assembled JS string (it was lost once) --


async def test_an_id_locator_with_a_backslash_is_evaluated_as_before(page):
    before = r"<p id='a\b'>old</p>"
    after = r"<p id='a\b'>new</p>"
    assert await _run(page, before, None, r"id=a\b", after_doc=after) == "changed"
    assert await _came_back(page, [before, after], before, r"id=a\b") == CAME_BACK
