"""Item facts (point-then-prove D5): the repeated item a pick sits in and its place among its peers.

Real chromium DOM (page.set_content), synthetic HTML, no customer data.
pytest tests/test_agent/test_element_verifier_item_facts.py -m integration
"""

import pytest

from browser_service.agent.element_verifier import read_element_facts

pytestmark = pytest.mark.integration

# The reader as it was before item facts (bench/private/ptp/facts_today.js in the NLRF repo), verbatim.
TODAY_FACTS_JS = r"""
(el) => {
  const hides = (el => String(el.type || '').toLowerCase() === 'password' || !!el.closest('[nlrf-secret]'));
  const squash = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const safeText = (node, max) => {
    if (!node || hides(node)) return '';
    const clone = node.cloneNode(true);
    clone.querySelectorAll('[nlrf-secret], input[type=password], script, style')
      .forEach((n) => n.remove());
    const parts = [];
    const walker = document.createTreeWalker(clone, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) parts.push(walker.currentNode.nodeValue);
    return squash(parts.join(' ')).slice(0, max);
  };
  const attr = (n) => el.getAttribute(n) || '';
  const labelNode = (el.labels && el.labels[0]) || el.closest('label')
    || (attr('aria-labelledby') ? document.getElementById(attr('aria-labelledby').split(/\s+/)[0]) : null);
  const NAMES = {banner: 'header', navigation: 'nav', complementary: 'aside', contentinfo: 'footer'};
  const lm = el.parentElement && el.parentElement.closest(
    'header,nav,main,aside,footer,form,dialog,[role=banner],[role=navigation],[role=main],'
    + '[role=complementary],[role=contentinfo],[role=form],[role=dialog]');
  let landmark = '';
  if (lm) {
    const r = lm.getAttribute('role');
    landmark = r ? (NAMES[r] || r) : lm.tagName.toLowerCase();
    const l = lm.getAttribute('aria-label');
    if (l) landmark += ': ' + l.trim();
  }
  const row = el.closest('tr,[role=row]');
  const cell = el.closest('td,th');
  let position = '', header = '';
  if (cell && cell.parentElement) {
    const n = Array.prototype.indexOf.call(cell.parentElement.children, cell) + 1;
    const suffix = (n % 100 >= 11 && n % 100 <= 13) ? 'th' : ({1: 'st', 2: 'nd', 3: 'rd'}[n % 10] || 'th');
    position = n + suffix + ' from the left';
    const table = cell.closest('table');
    const headRow = table && (table.querySelector('thead tr') || table.querySelector('tr'));
    const h = headRow && headRow.children[n - 1];
    if (h && h.tagName === 'TH') header = safeText(h, 80);
  }
  const facts = {
    tag: el.tagName.toLowerCase(),
    id: attr('id').trim(),
    text: safeText(el, 80),
    role: attr('role'),
    aria_label: attr('aria-label').trim(),
    title: attr('title').trim(),
    placeholder: attr('placeholder').trim(),
    label: safeText(labelNode, 80),
    type: attr('type'),
    name: attr('name'),
    landmark: landmark,
    row_text: safeText(row, 120),
    column_position: position,
    column_header: header,
  };
  // graphic: a wordless element's picture as text (first svg: viewBox, path start; else first img: alt, file name).
  // shortcut: a sprite icon (<use href>), a bounding-box first path (M0 0h24v24H0z) or an icon drawn without a path gives
  // an uninformative graphic, which the call site refuses on "unsure"; upgrade if a sweep shows a right icon refused so.
  // shortcut: only display:none (own or an ancestor's) is skipped, so a visibility:hidden or zero-opacity svg / img still counts;
  // upgrade if one is mis-described.
  if (!facts.text && !facts.aria_label && !facts.title && !facts.label && !hides(el)) {
    const first = (sel) => [el, ...el.querySelectorAll(sel)]
      .find((n) => n.matches(sel) && !hides(n) && n.getClientRects().length > 0);
    const fileName = (s) => /^data:/i.test(s)
      ? 'data:' + s.slice(5).split(/[;,]/)[0]
      : s.split(/[?#]/)[0].split('/').pop().slice(0, 60);
    const join = (kind, pairs) => [kind, ...pairs.filter((p) => p[1]).map((p) => p[0] + '=' + p[1])].join(' ');
    const visible = (root, sel) => [...root.querySelectorAll(sel)].find((n) => !hides(n));
    const svg = first('svg');
    const img = svg ? null : first('img');
    if (svg) {
      const path = visible(svg, 'path');
      facts.graphic = join('svg', [
        ['viewBox', squash(svg.getAttribute('viewBox'))],
        ['path', squash(path && path.getAttribute('d')).slice(0, 60)],
      ]);
    } else if (img) {
      facts.graphic = join('img', [
        ['alt', squash(img.getAttribute('alt')).slice(0, 60)],
        ['src', fileName(squash(img.getAttribute('src')))],
      ]);
    }
  }
  for (const k of Object.keys(facts)) if (!facts[k]) delete facts[k];
  return facts;
}"""

TODO = """
<ul class="todo-list">
  <li><div class="view"><input class="toggle" type="checkbox"><label>Buy milk</label><button class="destroy"></button></div></li>
  <li><div class="view"><input class="toggle" type="checkbox"><label>Walk the dog</label><button class="destroy"></button></div></li>
</ul>
"""

FIGURES = "".join(
    f'<div class="figure"><img src="/a/user{n}.png" alt="">'
    f'<div class="figcaption" style="display:none"><h5>name: user{n}</h5></div></div>'
    for n in (1, 2, 3)
)

TABLE = "<table><tr><th>Name</th><th>Qty</th></tr><tr><td>Apple</td><td>3</td></tr></table>"

TILES = (
    "<main>"
    + "".join(
        '<div style="display:flex;width:88px;height:88px">'
        '<svg width="48" height="48" viewBox="0 0 24 24" aria-hidden="true"><path d="M1 2h3"/></svg></div>'
        for _ in range(3)
    )
    + "</main>"
)

OUTSIDE = (
    '<form><button id="go">Go</button></form>'
    '<label for="q">Search</label><input id="q" placeholder="find">'
    "<h1>Title</h1>"
)

SECRET_LI = (
    "<ul><li id='s'>Note <span nlrf-secret>s3cr3t</span> and <input type=password value=pw> end</li>"
    "<li>Other</li></ul>"
)


@pytest.fixture
async def page():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context()
        pg = await ctx.new_page()
        try:
            yield pg
        finally:
            await ctx.close()
            await browser.close()


async def test_todo_items_get_item_text_and_position(page):
    await page.set_content(TODO + TODO)
    first = await read_element_facts(page.locator(".toggle").nth(0))
    second = await read_element_facts(page.locator(".toggle").nth(1))
    assert (first["item_text"], first["item_position"]) == ("Buy milk", "1 of 2")
    assert (second["item_text"], second["item_position"]) == ("Walk the dog", "2 of 2")


async def test_figures_get_position(page):
    await page.set_content(FIGURES)
    facts = await read_element_facts(page.locator("img").nth(1))
    assert facts["item_position"] == "2 of 3"
    assert "name: user2" in facts["item_text"]


async def test_table_cell_keeps_row_facts_only(page):
    await page.set_content(TABLE)
    facts = await read_element_facts(page.locator("td").nth(0))
    assert facts["row_text"] == "Apple 3"
    assert "item_text" not in facts and "item_position" not in facts


async def test_classless_sibling_tiles_are_not_items(page):
    await page.set_content(TILES)
    facts = await read_element_facts(page.locator("div").nth(1))
    assert "item_text" not in facts and "item_position" not in facts
    assert "graphic" in facts


async def test_outside_items_facts_equal_today(page):
    await page.set_content(OUTSIDE)
    for sel in ("#go", "#q", "h1"):
        today = await page.locator(sel).evaluate(TODAY_FACTS_JS)
        assert await read_element_facts(page.locator(sel)) == today, sel


async def test_item_text_drops_secret_and_password(page):
    await page.set_content(SECRET_LI)
    facts = await read_element_facts(page.locator("#s"))
    assert facts["item_text"] == "Note and end"
    assert "s3cr3t" not in str(facts) and "pw" not in facts["item_text"].split()
