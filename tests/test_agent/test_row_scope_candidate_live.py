"""
A row-scoped candidate is kept when the element it lands on does not carry the
row anchor (browser-service #45, bench query q04), real Chromium.

The q04 story (run ``fd766e52-e05f-4acf-80c5-2b57c4e8007a``, 2026-10-01): the
agent asked for "the 'edit' link in the row containing 'Smith' in the first
table" and passed ``row_anchor_text='Smith'`` with a correct row-scoped
candidate such as ``#table1 tbody tr:has-text('Smith') a:has-text('edit')``.
The candidate matched exactly 1 element and was rejected anyway, because a
string test reads "the anchor appears in the last segment" as "this is the
anchor cell itself". The cascade then shipped ``text="edit" >> nth=0`` — the
first edit link on the page, which belongs to whatever row is first.

The fix reads the element the candidate lands on. The reject stands only when
that element itself carries the anchor; a link that merely sits inside the
Smith row does not, so the candidate is kept as written.

Part one drives ``find_unique_locator_action`` over the-internet's /tables
shape (two tables, "Smith" in both). Guards pin that the fix never
over-withdraws: the anchor cell, the whole row, a row whose anchor lies past
500 characters, and an unreadable landed element all still reject. Part two
drives the helper itself.
"""

import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import find_unique_locator_action
from browser_service.locators.frame_locator import strip_frame

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "test_locators" / "locator_fixtures"

FRAME = 'iframe[id="tables"]'

SMITH_EDIT = "#table1 tbody tr:has(td:text-is('Smith')) a[href='#edit']"
BACH_EDIT = "#table1 tbody tr:has(td:text-is('Bach')) a[href='#edit']"
SMITH_DUE = "#table1 tbody tr:has(td:text-is('Smith')) td:nth-child(4)"
DESC = "the 'edit' link in the row containing 'Smith' in the first table (action: click)"
DUE_DESC = "the Due cell in the row containing 'Smith' in the first table (action: get_text)"
LANDS_JS = "e => e.closest('tr').cells[0].textContent.trim() + '|' + e.closest('table').id"

# The four most-sent stored q04 candidate shapes.
STORED_SHAPES = [
    "#table1 tbody tr:has-text('Smith') a:has-text('edit')",
    "table#table1 tr:has-text('Smith') a[href='#edit']",
    "xpath=//table[@id='table1']//tr[td[text()='Smith']]//a[text()='edit']",
    "xpath=//table[@id='table1']//tr[contains(.,'Smith')]//a[text()='edit']",
]
FIRST_SHAPE = STORED_SHAPES[0]

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


async def _center(root, selector: str):
    box = await root.locator(selector).first.bounding_box()
    assert box, selector
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


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
    candidate: str,
    *,
    expected_text,
    action,
    description: str = DESC,
    frame: str | None = None,
):
    """Call the action for ``target`` (element data and centre read from it)."""
    root = page.frame_locator(frame) if frame else page
    x, y = await _center(root, target)
    element_data = await root.locator(target).first.evaluate(INDEXED_DATA_JS)
    kwargs = {"iframe_context": frame} if frame else {}
    return await find_unique_locator_action(
        x=x,
        y=y,
        element_id="elem_1",
        element_description=description,
        expected_text=expected_text,
        candidate_locator=candidate,
        element_data=element_data,
        page=page,
        is_collection=False,
        row_anchor_text="Smith",
        action=action,
        **kwargs,
    )


class TestRowScopedCandidateIsKept:
    @pytest.mark.parametrize("candidate", STORED_SHAPES)
    async def test_row_scoped_candidate_is_accepted_as_written(self, page, caplog, candidate):
        await _open(page)
        with caplog.at_level(logging.INFO):
            out = await _call(page, SMITH_EDIT, candidate, expected_text="edit", action="click")
        assert out["found"] is True
        assert out["best_locator"] == candidate
        assert out["all_locators"][0]["type"] == "candidate"
        assert "row-anchor-scope-not-target" in caplog.text
        assert "row-anchor-rejects-candidate" not in caplog.text
        assert await page.locator(out["best_locator"]).evaluate(LANDS_JS) == "Smith|table1"

    async def test_accepted_candidate_follows_smith_when_the_rows_move(self, page):
        await _open(page)
        out = await _call(page, SMITH_EDIT, FIRST_SHAPE, expected_text="edit", action="click")
        await page.evaluate(
            "() => { const b = document.querySelector('#table1 tbody'); b.appendChild(b.rows[0]); }"
        )
        assert await page.locator(out["best_locator"]).evaluate(LANDS_JS) == "Smith|table1"

    async def test_action_none_accepts_the_row_scoped_candidate(self, page):
        await _open(page)
        out = await _call(page, SMITH_EDIT, FIRST_SHAPE, expected_text="edit", action=None)
        assert out["best_locator"] == FIRST_SHAPE

    async def test_read_step_keeps_the_row_scoped_cell_candidate(self, page):
        await _open(page)
        candidate = "#table1 tr:has-text('Smith') td:nth-child(4)"
        out = await _call(
            page,
            SMITH_DUE,
            candidate,
            expected_text="$50.00",
            action="get_text",
            description=DUE_DESC,
        )
        assert out["best_locator"] == candidate

    async def test_candidate_wins_over_a_pointer_on_another_row(self, page):
        await _open(page)
        out = await _call(page, BACH_EDIT, FIRST_SHAPE, expected_text="edit", action="click")
        assert out["best_locator"] == FIRST_SHAPE
        assert await page.locator(out["best_locator"]).evaluate(LANDS_JS) == "Smith|table1"

    async def test_row_scoped_candidate_is_accepted_inside_a_frame(self, page, caplog):
        await page.goto(_file_url("row_anchor_two_tables_frame.html"), wait_until="load")
        await page.frame_locator(FRAME).locator("#table1").wait_for()
        with caplog.at_level(logging.INFO):
            out = await _call(
                page,
                SMITH_EDIT,
                FIRST_SHAPE,
                expected_text="edit",
                action="click",
                frame=FRAME,
            )
        assert out["found"] is True
        assert strip_frame(out["best_locator"], FRAME) == FIRST_SHAPE
        assert "row-anchor-scope-not-target" in caplog.text
        landed = await page.frame_locator(FRAME).locator(FIRST_SHAPE).evaluate(LANDS_JS)
        assert landed == "Smith|table1"


class TestKeptCandidateStillFacesTheLaterChecks:
    """Withdrawing the anchor reject must not skip the identity and text checks."""

    async def test_kept_candidate_still_meets_the_later_checks(self, page, caplog):
        await _open(page)
        candidate = "#table1 tbody tr:has-text('Smith') a[href='#delete']"
        with caplog.at_level(logging.INFO):
            out = await _call(page, SMITH_EDIT, candidate, expected_text="edit", action="click")
        assert "row-anchor-scope-not-target" in caplog.text
        assert out["best_locator"] != candidate
        assert await page.locator(out["best_locator"]).evaluate(LANDS_JS) == "Smith|table1"
        assert await page.locator(out["best_locator"]).get_attribute("href") == "#edit"


class TestRejectStillStandsWhereTheElementCarriesTheAnchor:
    """Guards: green before and after the fix. The fix may only withdraw a
    reject when the landed element provably does not carry the anchor."""

    async def _assert_rejected(self, out, caplog, candidate):
        assert "row-anchor-rejects-candidate" in caplog.text
        assert "row-anchor-scope-not-target" not in caplog.text
        assert out["best_locator"] != candidate

    async def test_anchor_cell_candidate_is_still_rejected(self, page, caplog):
        await _open(page)
        candidate = "xpath=//table[@id='table1']//td[text()='Smith']"
        with caplog.at_level(logging.INFO):
            out = await _call(page, SMITH_EDIT, candidate, expected_text=None, action="click")
        await self._assert_rejected(out, caplog, candidate)

    async def test_whole_row_candidate_is_still_rejected(self, page, caplog):
        await _open(page)
        candidate = "#table1 tbody tr:has-text('Smith')"
        with caplog.at_level(logging.INFO):
            out = await _call(page, SMITH_EDIT, candidate, expected_text="edit", action="click")
        await self._assert_rejected(out, caplog, candidate)

    async def test_long_row_with_the_anchor_past_500_characters_is_still_rejected(
        self, page, caplog
    ):
        await _open(page)
        await page.evaluate(
            "() => { const r = document.querySelector('#table1 tbody').rows[0]; "
            "const c = r.insertCell(0); c.textContent = 'x '.repeat(300); }"
        )
        candidate = "#table1 tbody tr:has-text('Smith')"
        with caplog.at_level(logging.INFO):
            out = await _call(
                page,
                "#table1 tbody tr >> nth=0 >> a[href='#edit']",
                candidate,
                expected_text="edit",
                action="click",
            )
        await self._assert_rejected(out, caplog, candidate)

    async def test_unreadable_landed_element_keeps_the_reject(self, page, caplog):
        await _open(page)
        with (
            caplog.at_level(logging.INFO),
            patch(
                "browser_service.agent.actions._resolved_element_carries_anchor",
                AsyncMock(return_value=None),
                create=True,
            ),
        ):
            out = await _call(page, SMITH_EDIT, FIRST_SHAPE, expected_text="edit", action="click")
        assert "row-anchor-rejects-candidate" in caplog.text
        assert out["best_locator"] != FIRST_SHAPE


class TestCarriesAnchorRead:
    """The helper against real Chromium, anchor 'Smith'."""

    @staticmethod
    def _helper():
        from browser_service.agent.actions import _resolved_element_carries_anchor

        return _resolved_element_carries_anchor

    async def test_edit_link_does_not_carry_the_anchor(self, page):
        await _open(page)
        assert await self._helper()(page, SMITH_EDIT, "Smith") is False

    async def test_anchor_cell_carries_it(self, page):
        await _open(page)
        assert await self._helper()(page, "#table1 td:text-is('Smith')", "Smith") is True

    async def test_email_cell_carries_it_as_a_case_insensitive_substring(self, page):
        await _open(page)
        locator = "#table1 td:text-is('jsmith@gmail.com')"
        assert await self._helper()(page, locator, "Smith") is True

    @pytest.mark.parametrize("attribute", ["title", "aria-label", "placeholder", "alt"])
    async def test_attribute_carries_it(self, page, attribute):
        await _open(page)
        await page.locator(BACH_EDIT).evaluate(
            "(e, [n, v]) => e.setAttribute(n, v)", [attribute, "John   SMITH jr"]
        )
        assert await self._helper()(page, BACH_EDIT, "Smith") is True

    async def test_input_value_carries_it(self, page):
        await _open(page)
        await page.evaluate(
            "() => { const i = document.createElement('input'); i.id = 'probe'; "
            "i.value = 'Smith'; document.body.appendChild(i); }"
        )
        assert await self._helper()(page, "#probe", "Smith") is True

    async def test_several_matches_is_unknown(self, page):
        await _open(page)
        assert await self._helper()(page, "#table1 a[href='#edit']", "Smith") is None

    async def test_no_match_is_unknown(self, page):
        await _open(page)
        assert await self._helper()(page, "#nope", "Smith", timeout_ms=300) is None

    @pytest.mark.parametrize("anchor", ["", None])
    async def test_empty_anchor_is_unknown(self, page, anchor):
        await _open(page)
        assert await self._helper()(page, SMITH_EDIT, anchor) is None

    async def test_inside_a_frame(self, page):
        await page.goto(_file_url("row_anchor_two_tables_frame.html"), wait_until="load")
        root = page.frame_locator(FRAME)
        await root.locator("#table1").wait_for()
        assert await self._helper()(root, SMITH_EDIT, "Smith") is False
        assert await self._helper()(root, "#table1 td:text-is('Smith')", "Smith") is True
