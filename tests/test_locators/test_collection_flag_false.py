"""bs #93 — an explicit is_collection=False is followed at STEP 0 and at the ARIA role step.

NLRF sweep d16 (2026-11-18): request "first team name cell in the search results table", expected text
"Toronto Maple Leafs", is_collection=False, the indexed element a <td>. The classifier voted 'collection' on the
description words ("table", "results"), the DOM probe "confirmed" it (any cell has tr/tbody ancestors) and the handler
shipped `tbody > tr` (22 matches) for a request the agent had said is ONE element. At the ARIA role step a `<li>`
(role listitem, an "ARIA collection") was accepted with count > 1, typed `listitem` — so NLRF read it as one element
and strict mode failed (2 of 3 sealed runs).

The STEP 2.5 table detection (description words) is deliberately NOT changed: bench q06 sent a wrong False five
times and that rescue kept all five passing — the control below pins it.

Real headless Chromium against ``locator_fixtures/collection_flag_results_table.html`` (the fixture lane).
"""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.smart_locator import (
    _find_element_via_accessibility,
    _generate_locators_from_element_data,
    find_unique_locator_at_coordinates,
)

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent / "locator_fixtures"
FIRST_NAME_CELL = "html/body/div/section/div/table/tbody/tr[2]/td[1]"
CELL_DESCRIPTION = "first team name cell in the search results table"
CELL_DATA = {"tagName": "td", "className": "name", "xpath": FIRST_NAME_CELL}


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page_obj = await ctx.new_page()
        await page_obj.goto(
            (FIXTURES_DIR / "collection_flag_results_table.html").resolve().as_uri()
        )
        try:
            yield page_obj
        finally:
            await browser.close()


async def _center(page, selector: str) -> tuple[float, float]:
    box = await page.locator(selector).first.bounding_box()
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


async def _step0(page, is_collection, expected_text="Toronto Maple Leafs"):
    return await _generate_locators_from_element_data(
        page,
        CELL_DATA,
        "elem_3",
        CELL_DESCRIPTION,
        expected_text,
        page=page,
        is_collection=is_collection,
    )


async def test_step0_false_with_text_no_longer_ships_the_rows(page):
    """The d16 failure: with the expected text, False must not end in `tbody > tr` (7 matches).

    STEP 0 now declines (the element has no semantic attribute and the text is known, so TEXT-FIRST owns it);
    anything it did answer must be the one cell.
    """
    result = await _step0(page, False)
    if result is not None:
        assert result["element_type"] != "collection"
        locator = page.locator(result["best_locator"])
        assert await locator.count() == 1
        assert (await locator.inner_text()).strip() == "Toronto Maple Leafs"


async def test_step0_false_without_text_ships_the_one_cell(page, caplog):
    """No expected text: STEP 0 answers the first team cell alone, and the log says the flag decided."""
    with caplog.at_level("INFO"):
        result = await _step0(page, False, expected_text=None)
    assert result is not None
    assert result["element_type"] != "collection"
    locator = page.locator(result["best_locator"])
    assert await locator.count() == 1
    assert (await locator.inner_text()).strip() == "Toronto Maple Leafs"
    assert "agent flagged is_collection=False" in caplog.text


async def test_step0_absent_flag_keeps_the_collection_answer(page):
    """Control: flag absent -> today's collection verdict, the description words still vote."""
    result = await _step0(page, None)
    assert result is not None
    assert result["element_type"] == "collection"
    assert "flag-false-suppresses-collection" not in result["classifier_signals"]


async def test_aria_false_refuses_a_many_match_role(page):
    """A <li> (role listitem) matching 4 is not accepted for a request the agent called one element."""
    x, y = await _center(page, "ul.fruit li")
    result = await _find_element_via_accessibility(
        page, x, y, "an item in the fruit list", None, is_collection=False
    )
    assert result is None or result["count"] == 1


async def test_aria_absent_flag_keeps_the_many_match_answer(page):
    """Control: flag absent -> the ARIA collection role still ships its count > 1 answer."""
    x, y = await _center(page, "ul.fruit li")
    result = await _find_element_via_accessibility(page, x, y, "an item in the fruit list", None)
    assert result is not None
    assert result["count"] == 4
    assert result["locator"] == "role=listitem"


async def test_aria_table_detection_ignores_false(page):
    """Control (the kept q06 rescue): description words still get the rows answer under False."""
    x, y = await _center(page, "tr.team td.name")
    desc = "all the table rows in the search results"
    with_false = await _find_element_via_accessibility(page, x, y, desc, None, is_collection=False)
    with_none = await _find_element_via_accessibility(page, x, y, desc, None)
    assert with_none is not None
    assert with_none["element_type"] == "collection"
    assert with_false is not None
    assert with_false["element_type"] == "collection"
    assert with_false["locator"] == with_none["locator"]


# ---- 1d: the pointed element's own text is the evidence (full cascade) ----
FIRST_ROW = "html/body/div/section/div/table/tbody/tr[2]"
FIRST_ROW_TEXT = "Toronto Maple Leafs 1917 38"


async def _cascade(page, element_data, selector, description, is_collection, expected_text):
    x, y = await _center(page, selector)
    return await find_unique_locator_at_coordinates(
        page=page,
        x=x,
        y=y,
        element_id="elem_3",
        element_description=description,
        expected_text=expected_text,
        element_data=element_data,
        is_collection=is_collection,
    )


def _cell_data():
    return {**CELL_DATA, "textContent": "Toronto Maple Leafs"}


async def test_cascade_false_with_matching_cell_text_ships_the_one_cell(page):
    """d16 end to end: the indexed cell's own text IS the expected text, so False is believed."""
    result = await _cascade(
        page, _cell_data(), "tr.team td.name", CELL_DESCRIPTION, False, "Toronto Maple Leafs"
    )
    assert result["found"] is True
    assert result.get("element_type") != "collection"
    locator = page.locator(result["best_locator"])
    assert await locator.count() == 1
    assert (await locator.inner_text()).strip() == "Toronto Maple Leafs"


async def test_cascade_absent_flag_keeps_the_rows_answer(page):
    """Control: flag absent -> today's rows answer."""
    result = await _cascade(
        page, _cell_data(), "tr.team td.name", CELL_DESCRIPTION, None, "Toronto Maple Leafs"
    )
    assert result["found"] is True
    assert result["element_type"] == "collection"


async def test_cascade_false_on_a_whole_row_keeps_todays_answer(page):
    """Control (q06 shape): False, pointed at a whole row whose text is not the expected text -> unchanged."""
    row = {"tagName": "tr", "className": "team", "xpath": FIRST_ROW, "textContent": FIRST_ROW_TEXT}
    args = (page, row, "tr.team", "table rows with data in the results table")
    flagged = await _cascade(*args, False, "Toronto Maple Leafs")
    absent = await _cascade(*args, None, "Toronto Maple Leafs")
    assert flagged["found"] is True
    assert flagged["best_locator"] == absent["best_locator"]
    assert flagged.get("element_type") == absent.get("element_type")
