"""
A positional candidate that skips the row the step names is demoted, and shipped
only if no answer that names the row can replace it (browser-service #61 cut +
#49 (a)), real Chromium, no LLM.

The story: for "the 'edit' link in the row containing 'Smith' in the first
table" the agent sent ``#table1 tbody tr:nth-child(1) a[href='#edit']``. It lands
on Smith today because Smith is row 1 — and on another surname after a re-sort
or a changed data order, with no signal. The candidate is parked and the cascade
runs; a found answer that names ``Smith`` (a ``tr:has-text('Smith')`` hop) that
resolves to the same element replaces it. A step that names no row datum (q05,
"the first row and second column"), a read whose cure would hold the value read,
a collection and the flag-off path all ship the candidate unchanged.
"""

import logging
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import find_unique_locator_action

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "test_locators" / "locator_fixtures"

EDIT_CANDIDATE = "#table1 tbody tr:nth-child(1) a[href='#edit']"
EDIT_DESC = "the 'edit' link in the row containing 'Smith' in the first table (Example 1)"
SMITH_EDIT = "#table1 tbody tr:has(td:text-is('Smith')) a[href='#edit']"
LANDS_JS = "e => e.closest('tr').cells[0].textContent.trim() + '|' + e.closest('table').id"
MOVE_SMITH_THIRD_JS = (
    "() => { const b = document.querySelector('#table1 tbody'); "
    "b.insertBefore(b.rows[0], b.rows[2]); }"
)

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


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page_obj = await ctx.new_page()
        try:
            await page_obj.goto(
                (FIXTURES_DIR / "row_anchor_two_tables.html").resolve().as_uri(),
                wait_until="load",
            )
            yield page_obj
        finally:
            await browser.close()


async def _call(
    page,
    target: str,
    candidate: str,
    *,
    description: str,
    expected_text,
    action,
    is_collection: bool = False,
    row_anchor_text=None,
):
    """Call the action for ``target`` (element data and centre read from it)."""
    box = await page.locator(target).first.bounding_box()
    assert box, target
    return await find_unique_locator_action(
        x=box["x"] + box["width"] / 2,
        y=box["y"] + box["height"] / 2,
        element_id="elem_1",
        element_description=description,
        expected_text=expected_text,
        candidate_locator=candidate,
        element_data=await page.locator(target).first.evaluate(INDEXED_DATA_JS),
        page=page,
        is_collection=is_collection,
        row_anchor_text=row_anchor_text,
        action=action,
    )


class TestPositionalCandidateThatSkipsTheRow:
    async def test_reorder_the_shipped_locator_still_names_smith(self, page, caplog):
        with caplog.at_level(logging.INFO):
            out = await _call(
                page,
                SMITH_EDIT,
                EDIT_CANDIDATE,
                description=EDIT_DESC,
                expected_text="edit",
                action="click",
            )
        shipped = out["best_locator"]
        assert await page.locator(shipped).evaluate(LANDS_JS) == "Smith|table1"
        await page.evaluate(MOVE_SMITH_THIRD_JS)
        assert await page.locator(shipped).count() == 1
        assert await page.locator(shipped).evaluate(LANDS_JS) == "Smith|table1"
        assert "candidate-demoted" in caplog.text
        assert "row-datum-replaced" in caplog.text

    async def test_q05_names_no_row_datum_and_is_never_demoted(self, page, caplog):
        candidate = "#table1 tbody tr:nth-child(1) td:nth-child(2)"
        with caplog.at_level(logging.INFO):
            out = await _call(
                page,
                candidate,
                candidate,
                description="the cell in the first row and second column of Table 1 (Example 1)",
                expected_text="John",
                action="get_text",
            )
        assert "candidate-demoted" not in caplog.text
        assert out["best_locator"] == candidate

    async def test_read_cure_that_holds_the_value_is_refused(self, page, caplog):
        candidate = "#table1 tbody tr:nth-child(1) td:nth-child(3)"
        with caplog.at_level(logging.INFO):
            out = await _call(
                page,
                candidate,
                candidate,
                description="the email in the row containing 'Smith' in the first table",
                expected_text="jsmith@gmail.com",
                action="get_text",
            )
        assert "candidate-demoted" in caplog.text
        assert "row-datum-replaced" not in caplog.text
        assert out["best_locator"] == candidate

    async def test_collection_is_not_demoted(self, page, caplog):
        with caplog.at_level(logging.INFO):
            out = await _call(
                page,
                SMITH_EDIT,
                EDIT_CANDIDATE,
                description=EDIT_DESC,
                expected_text="edit",
                action="click",
                is_collection=True,
            )
        assert "candidate-demoted" not in caplog.text
        assert out["best_locator"] == EDIT_CANDIDATE

    async def test_flag_off_ships_the_candidate(self, page, caplog):
        with caplog.at_level(logging.INFO):
            out = await _call(
                page,
                SMITH_EDIT,
                EDIT_CANDIDATE,
                description=EDIT_DESC,
                expected_text="edit",
                action=None,
            )
        assert "candidate-demoted" not in caplog.text
        assert out["best_locator"] == EDIT_CANDIDATE
