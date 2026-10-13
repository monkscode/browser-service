"""
An ambiguous row-anchor pick is addressed inside its anchored rows
(browser-service #45; bench query q04), real Chromium.

The story: "the 'edit' link in the row containing 'Smith' in the first table"
on a page whose two tables both hold a Smith row. With no candidate, the text
route finds ``text="edit"`` on 8 links, tries the row rescue
``tr:has-text("Smith") >> text="edit"`` (2 links, one per table) and declares
it ambiguous, so the cascade ships the page-wide index ``text="edit" >> nth=0``
-- the first edit link on the page, whatever row that is.

The fix never changes WHICH link is picked (the coordinate pick is untouched).
It only addresses that same link inside the chain's matches
(``... >> nth=0``) when it is one of them, so the address follows the Smith row
when the rows move. Guards pin that every other case stays byte for byte
today: a pointer on another row, a single table, a read step.
"""

import logging
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import find_unique_locator_action
from browser_service.locators.smart_locator import _address_pick_in_row_chain

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).resolve().parent / "locator_fixtures"

FRAME = 'iframe[id="tables"]'
SIGNAL = "row-anchor-indexed-in-rows"

SMITH_EDIT = "#table1 tbody tr:has(td:text-is('Smith')) a[href='#edit']"
SMITH_EDIT_2 = "#table2 tbody tr:has(td:text-is('Smith')) a[href='#edit']"
BACH_EDIT = "#table1 tbody tr:has(td:text-is('Bach')) a[href='#edit']"
CONWAY_EDIT = "#table1 tbody tr:has(td:text-is('Conway')) a[href='#edit']"
SMITH_DUE = "#table1 tbody tr:has(td:text-is('Smith')) td:nth-child(4)"
DESC = "the 'edit' link in the row containing 'Smith' in the first table (action: click)"
DUE_DESC = "the Due cell in the row containing 'Smith' in the first table (action: get_text)"
LANDS_JS = "e => e.closest('tr').cells[0].textContent.trim() + '|' + e.closest('table').id"
ROWS = 'tr:has-text("Smith")'
CHAIN = f'{ROWS} >> text="edit"'

# browser-use-style element_data for the INDEXED element.
INDEXED_DATA_JS = """(el) => {
    const parts = [];
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
        let i = 1;
        for (let s = n.previousElementSibling; s; s = s.previousElementSibling) {
            if (s.tagName === n.tagName) i++;
        }
        parts.unshift(n.tagName.toLowerCase() + '[' + i + ']');
    }
    return {
        tagName: el.tagName.toLowerCase(),
        id: el.id || "",
        className: (typeof el.className === 'string') ? el.className : "",
        placeholder: el.getAttribute('placeholder') || "",
        parentClassName: (el.parentElement && typeof el.parentElement.className === 'string')
            ? el.parentElement.className : "",
        textContent: (el.textContent || "").trim().slice(0, 80),
        xpath: parts.join('/'),
    };
}"""


def _file_url(name: str) -> str:
    return (FIXTURES_DIR / name).resolve().as_uri()


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page_obj = await ctx.new_page()
        try:
            yield page_obj
        finally:
            await browser.close()


async def _open(page):
    await page.goto(_file_url("row_anchor_two_tables.html"), wait_until="load")


async def _call(
    page,
    target: str,
    *,
    expected_text,
    action,
    description: str = DESC,
    right_of: bool = False,
):
    """Call the action with NO candidate for the element ``target`` names.

    The pointer is the centre of ``target``, or 6 px past its right edge when
    ``right_of`` is set (element data is always that of ``target``).
    """
    box = await page.locator(target).first.bounding_box()
    assert box, target
    x = box["x"] + (box["width"] + 6 if right_of else box["width"] / 2)
    y = box["y"] + box["height"] / 2
    element_data = await page.locator(target).first.evaluate(INDEXED_DATA_JS)
    return await find_unique_locator_action(
        x=x,
        y=y,
        element_id="elem_1",
        element_description=description,
        expected_text=expected_text,
        candidate_locator=None,
        element_data=element_data,
        page=page,
        is_collection=False,
        row_anchor_text="Smith",
        action=action,
    )


async def _click_call(page, target: str, **kwargs):
    return await _call(page, target, expected_text="edit", action="click", **kwargs)


class TestPickIsAddressedInsideItsRows:
    async def test_pointer_on_the_first_table_smith_link(self, page, caplog):
        await _open(page)
        with caplog.at_level(logging.INFO):
            out = await _click_call(page, SMITH_EDIT)
        assert out["found"] is True
        assert out["best_locator"] == f"{CHAIN} >> nth=0"
        assert out["row_anchor_ambiguous"] is True
        assert not out.get("row_anchored")
        assert out["stability"] == "positional"
        assert SIGNAL in caplog.text
        assert await page.locator(out["best_locator"]).evaluate(LANDS_JS) == "Smith|table1"

    async def test_pointer_on_the_second_table_smith_link(self, page):
        await _open(page)
        out = await _click_call(page, SMITH_EDIT_2)
        assert out["best_locator"] == f"{CHAIN} >> nth=1"
        assert await page.locator(out["best_locator"]).evaluate(LANDS_JS) == "Smith|table2"

    async def test_pointer_just_right_of_the_link(self, page):
        await _open(page)
        out = await _click_call(page, SMITH_EDIT, right_of=True)
        assert out["best_locator"] == f"{CHAIN} >> nth=0"

    async def test_address_follows_the_smith_row_when_the_rows_move(self, page):
        await _open(page)
        out = await _click_call(page, SMITH_EDIT)
        await page.evaluate(
            "() => { const b = document.querySelector('#table1 tbody'); b.appendChild(b.rows[0]); }"
        )
        assert await page.locator(out["best_locator"]).evaluate(LANDS_JS) == "Smith|table1"
        # The contrast: today's page-wide index now lands on Bach.
        assert await page.locator('text="edit" >> nth=0').evaluate(LANDS_JS) == "Bach|table1"


class TestEverythingElseIsByteForByteToday:
    """Guards: green before and after the change."""

    async def test_pointer_on_another_row_keeps_the_page_wide_index(self, page, caplog):
        await _open(page)
        with caplog.at_level(logging.INFO):
            out = await _click_call(page, BACH_EDIT)
        assert out["best_locator"] == 'text="edit" >> nth=1'
        assert out["row_anchor_ambiguous"] is True
        assert SIGNAL not in caplog.text

    async def test_pointer_on_the_last_row_keeps_the_page_wide_index(self, page, caplog):
        await _open(page)
        with caplog.at_level(logging.INFO):
            out = await _click_call(page, CONWAY_EDIT)
        assert out["best_locator"] == 'text="edit" >> nth=3'
        assert out["row_anchor_ambiguous"] is True
        assert SIGNAL not in caplog.text

    async def test_one_table_is_still_row_anchored(self, page, caplog):
        await _open(page)
        await page.evaluate("() => document.querySelector('#table2').remove()")
        with caplog.at_level(logging.INFO):
            out = await _click_call(page, SMITH_EDIT)
        assert out["best_locator"] == CHAIN
        assert out["row_anchored"] is True
        assert "row_anchor_ambiguous" not in out
        assert SIGNAL not in caplog.text

    async def test_read_step_keeps_its_read_address(self, page, caplog):
        await _open(page)
        with caplog.at_level(logging.INFO):
            out = await _call(
                page,
                SMITH_DUE,
                expected_text="$50.00",
                action="get_text",
                description=DUE_DESC,
            )
        assert out["best_locator"] == "#table1 >> tbody > tr >> nth=0 >> td:nth-of-type(4)"
        assert "read-address-rewritten" in caplog.text
        assert not out.get("row_anchored")


class TestReadStepIsAddressedInRowsFirst:
    async def test_the_signal_fires_before_the_read_address_replaces_it(self, page, caplog):
        await _open(page)
        with caplog.at_level(logging.INFO):
            await _call(
                page,
                SMITH_DUE,
                expected_text="$50.00",
                action="get_text",
                description=DUE_DESC,
            )
        assert SIGNAL in caplog.text


class TestAddressPickInRowChainLive:
    """The helper against real Chromium: ``text="edit"`` matches 8 links,
    the Smith chain matches 2 (table 1's, then table 2's)."""

    async def test_table_two_smith_is_the_second_chain_match(self, page):
        await _open(page)
        got = await _address_pick_in_row_chain(page, ROWS, CHAIN, 'text="edit" >> nth=4')
        assert got == f"{CHAIN} >> nth=1"

    async def test_a_pick_outside_the_chain_is_none(self, page):
        await _open(page)
        assert await _address_pick_in_row_chain(page, ROWS, CHAIN, 'text="edit" >> nth=1') is None

    async def test_a_pick_that_resolves_to_many_elements_is_none(self, page):
        await _open(page)
        assert await _address_pick_in_row_chain(page, ROWS, CHAIN, 'text="edit"') is None

    async def test_a_pick_that_resolves_to_nothing_is_none(self, page):
        await _open(page)
        assert await _address_pick_in_row_chain(page, ROWS, CHAIN, 'text="nope"') is None

    async def test_a_hidden_match_still_counts_so_the_address_is_the_same_element(self, page):
        await _open(page)
        pick = 'text="edit" >> nth=4'
        await page.evaluate(
            "() => { document.querySelector('#table1 tbody').rows[0].style.display = 'none'; }"
        )
        handle = await page.locator(pick).element_handle()
        got = await _address_pick_in_row_chain(page, ROWS, CHAIN, pick)
        assert got is not None
        assert await page.locator(got).evaluate("(e, o) => e === o", handle) is True

    async def test_inside_a_frame(self, page):
        await page.goto(_file_url("row_anchor_two_tables_frame.html"), wait_until="load")
        root = page.frame_locator(FRAME)
        await root.locator("#table1").wait_for()
        got = await _address_pick_in_row_chain(root, ROWS, CHAIN, 'text="edit" >> nth=4')
        assert got == f"{CHAIN} >> nth=1"


WRAP_IN_LAYOUT_TABLE_JS = """() => {
    const table = document.createElement('table');
    table.id = 'layout';
    const cell = table.createTBody().insertRow().insertCell();
    while (document.body.firstChild) cell.appendChild(document.body.firstChild);
    document.body.appendChild(table);
}"""


async def _open_in_layout_table(page):
    """The fixture wrapped in an old-style layout table, so the layout row
    also matches ``tr:has-text("Smith")`` and the anchored rows nest."""
    await _open(page)
    await page.evaluate(WRAP_IN_LAYOUT_TABLE_JS)


class TestAnchoredRowsThatNestKeepTodaysLocator:
    """A layout table around the data makes the chain match all 8 edit links
    and scope nothing, so today's page-wide index stays, byte for byte."""

    async def test_the_chain_matches_every_edit_link(self, page):
        await _open_in_layout_table(page)
        assert await page.locator(CHAIN).count() == 8

    async def test_pointer_on_another_row_keeps_the_page_wide_index(self, page, caplog):
        await _open_in_layout_table(page)
        with caplog.at_level(logging.INFO):
            out = await _click_call(page, BACH_EDIT)
        assert out["best_locator"] == 'text="edit" >> nth=1'
        assert out["row_anchor_ambiguous"] is True
        assert SIGNAL not in caplog.text

    async def test_pointer_on_the_smith_link_keeps_the_page_wide_index(self, page, caplog):
        await _open_in_layout_table(page)
        with caplog.at_level(logging.INFO):
            out = await _click_call(page, SMITH_EDIT)
        assert out["best_locator"] == 'text="edit" >> nth=0'
        assert SIGNAL not in caplog.text

    async def test_the_helper_answers_none(self, page):
        await _open_in_layout_table(page)
        got = await _address_pick_in_row_chain(page, ROWS, CHAIN, 'text="edit" >> nth=4')
        assert got is None
