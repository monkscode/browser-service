"""Element facts read from a real chromium DOM (page.set_content). Synthetic HTML, no customer data.

pytest tests/test_agent/test_element_verifier_facts_live.py -m integration
"""

import pytest

from browser_service.agent.element_verifier import read_element_facts

TABLE = """
<main aria-label="Accounts">
<table>
  <thead><tr><th>Name</th><th>Balance</th><th>Action</th></tr></thead>
  <tbody><tr><td>Cierra Vega</td><td>41.50</td><td><a href="#">View</a></td></tr></tbody>
</table>
</main>
"""

SECRETS = """
<table><tbody>
<tr id="r1"><td>Alice</td>
 <td><input id="pw" type="password" value="hunter2"><span nlrf-secret>topsecret</span><span>visible</span></td></tr>
</tbody></table>
<div id="box">Note: <div contenteditable nlrf-secret>typed-secret</div> end</div>
<div id="own" nlrf-secret>own-secret</div>
"""

LABELS = """
<nav aria-label="Primary"><label for="q">Search products</label><input id="q" name="query" type="text"
 placeholder="Type here" title="finder"></nav>
"""

SHADOW = """
<main><div id="host"></div></main>
<script>
  document.getElementById('host').attachShadow({mode: 'open'}).innerHTML = '<button id="inner">Go</button>';
</script>
"""


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


@pytest.mark.integration
@pytest.mark.fixture
class TestElementFactsOnARealDom:
    async def test_table_cell_has_column_position_and_header(self, page):
        await page.set_content(TABLE)
        facts = await read_element_facts(page.locator("td").nth(1))
        assert facts["tag"] == "td"
        assert facts["column_position"] == "2nd from the left"
        assert facts["column_header"] == "Balance"
        assert facts["row_text"] == "Cierra Vega 41.50 View"
        assert facts["landmark"] == "main: Accounts"
        link = await read_element_facts(page.locator("a"))
        assert link["column_position"] == "3rd from the left"
        assert link["column_header"] == "Action"

    async def test_row_text_drops_secret_descendants_and_passwords(self, page):
        await page.set_content(SECRETS)
        facts = await read_element_facts(page.locator("#pw"))
        assert facts["row_text"] == "Alice visible"
        cell = await read_element_facts(page.locator("td").nth(1))
        assert cell["text"] == "visible"
        everything = str(facts) + str(cell)
        assert "secret" not in everything
        assert "hunter2" not in everything

    async def test_label_and_landmark_are_read(self, page):
        await page.set_content(LABELS)
        facts = await read_element_facts(page.locator("#q"))
        assert facts["label"] == "Search products"
        assert facts["landmark"] == "nav: Primary"
        assert (facts["name"], facts["type"]) == ("query", "text")
        assert (facts["placeholder"], facts["title"]) == ("Type here", "finder")
        assert "text" not in facts and "role" not in facts

    async def test_element_text_drops_secret_descendant(self, page):
        await page.set_content(SECRETS)
        facts = await read_element_facts(page.locator("#box"))
        assert facts["text"] == "Note: end"
        own = await read_element_facts(page.locator("#own"))
        assert "text" not in own and "secret" not in str(own)

    async def test_shadow_root_reads_empty_landmark(self, page):
        await page.set_content(SHADOW)
        facts = await read_element_facts(page.locator("#host button"))
        assert facts["tag"] == "button" and facts["text"] == "Go"
        # shortcut: closest() stops at the shadow boundary, so the <main> around the host is not seen.
        assert "landmark" not in facts

    async def test_failure_is_none(self, page):
        await page.set_content(TABLE)
        assert await read_element_facts(page.locator("#nope"), timeout_ms=200) is None
