"""
Task 10 (E1) — stability-aware candidate ordering and payload marking,
exercised against real Chromium.

STEP 0 (element-data candidates):
- A session-generated id (``ext-gen1042``) must no longer beat a stable
  ``name=`` on the same element; the value-blind ordering emitted the id
  and never even tested the name (verified 2026-07-06 on this exact
  fixture shape).
- A volatile-only element is still returned — marked
  ``stability:"volatile"``, never rejected (locked decision #3), so
  found=false cannot rise.
- Stable ids and bare all-digit ids (bench q01's ``id=880667900``) keep
  winning byte-identically — the narrowed digit rule regression guard.

Same conventions as test_text_first_identity.py: real headless Chromium,
static fixtures in ``locator_fixtures/``, marked integration.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import find_unique_locator_action
from browser_service.config import config
from browser_service.locators.smart_locator import (
    _build_coordinate_strategies,
    _generate_locators_from_element_data,
    find_unique_locator_at_coordinates,
)

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent / "locator_fixtures"


def _file_url(name: str) -> str:
    return (FIXTURES_DIR / name).resolve().as_uri()


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page_obj = await ctx.new_page()
        await page_obj.goto(_file_url("stability_ids.html"))
        try:
            yield page_obj
        finally:
            await browser.close()


def _element_data(**overrides) -> dict:
    base = {
        "tagName": "input",
        "id": "",
        "name": "",
        "className": "",
        "ariaLabel": "",
        "placeholder": "",
        "title": "",
        "role": "",
        "dataTestId": "",
        "type": "",
        "xpath": "",
        "textContent": "",
    }
    base.update(overrides)
    return base


async def test_volatile_id_demoted_stable_name_wins(page):
    """id=ext-gen1042 + name=username on one input: the stable name must
    be emitted. Before Task 10 the id won on priority alone and the name
    was never even tested."""
    result = await _generate_locators_from_element_data(
        search_context=page,
        element_data=_element_data(
            id="ext-gen1042",
            name="username",
            placeholder="Enter username",
            type="text",
        ),
        element_id="elem_1",
        element_description="username field",
        page=page,
    )
    assert result is not None and result["found"] is True
    assert result["best_locator"] == '[name="username"]'
    assert result["stability"] == "stable"
    assert result["all_locators"][0]["stability"] == "stable"


async def test_volatile_only_element_returned_marked(page):
    """A button whose ONLY usable attribute is a framework id is still
    returned (found=true) — marked volatile, never hard-rejected
    (locked decision #3)."""
    result = await _generate_locators_from_element_data(
        search_context=page,
        element_data=_element_data(
            tagName="button",
            id="ext-gen2001",
            textContent="Refresh grid",
        ),
        element_id="elem_2",
        element_description="refresh button",
        page=page,
    )
    assert result is not None and result["found"] is True
    assert result["best_locator"] == "#ext-gen2001"
    assert result["stability"] == "volatile"
    assert result["all_locators"][0]["stability"] == "volatile"


async def test_stable_id_still_wins(page):
    """Hand-authored id beats name exactly as before — no behavior change
    on healthy traffic."""
    result = await _generate_locators_from_element_data(
        search_context=page,
        element_data=_element_data(
            id="save_button",
            name="save",
            type="text",
        ),
        element_id="elem_3",
        element_description="save field",
        page=page,
    )
    assert result is not None and result["found"] is True
    assert result["best_locator"] == "#save_button"
    assert result["stability"] == "stable"


def _coord_element_data(**overrides) -> dict:
    """Full-key element_data for _build_coordinate_strategies (direct indexing)."""
    base = {
        "tagName": "input",
        "id": "",
        "name": "",
        "className": "",
        "ariaLabel": "",
        "placeholder": "",
        "title": "",
        "role": "",
        "dataTestId": "",
        "dataTest": "",
        "dataQa": "",
        "type": "text",
        "xpath": "",
        "textContent": "",
        "innerText": "",
        "href": "",
        "siblingIndex": None,
        "parentId": "",
        "parentClass": "",
    }
    base.update(overrides)
    return base


def test_step3_strategies_carry_stability_tiers():
    """Every STEP-3 strategy is annotated: volatile framework id, stable
    name, positional nth-child / xpath-position / first-of-class."""
    strategies = _build_coordinate_strategies(
        _coord_element_data(
            id="ext-gen1042",
            name="username",
            className="form-control",
            siblingIndex=2,
            parentClass="form-group",
        )
    )
    by_type = {s["type"]: s for s in strategies}

    assert by_type["id"]["stability"] == "volatile"
    assert by_type["name"]["stability"] == "stable"
    assert by_type["css-nth-child"]["stability"] == "positional"
    assert by_type["xpath-class-position"]["stability"] == "positional"
    assert by_type["xpath-first-of-class"]["stability"] == "positional"


def test_step3_dynamic_text_strategy_is_volatile():
    """text="Cart (3 items)" dies when the count changes — marked volatile."""
    strategies = _build_coordinate_strategies(
        _coord_element_data(
            tagName="button",
            innerText="Cart (3 items)",
        )
    )
    by_type = {s["type"]: s for s in strategies}
    assert by_type["text"]["stability"] == "volatile"
    assert by_type["xpath-text"]["stability"] == "volatile"


def test_step3_bare_digit_id_stays_stable():
    """Narrowed digit rule flows through STEP-3 too (q01 shape)."""
    strategies = _build_coordinate_strategies(
        _coord_element_data(
            tagName="a",
            id="880667900",
        )
    )
    by_type = {s["type"]: s for s in strategies}
    assert by_type["id"]["stability"] == "stable"


async def test_step3_end_to_end_volatile_id_loses_to_name(page):
    """Full coordinate path: the target has NO accessible name (no label,
    placeholder or aria), so STEP 1/2/2.5 fail and STEP 3 reads the DOM at
    (x, y) — id=ext-gen3003 AND name=payload_ref. The emitted locator must
    be the stable name, stability reported in the payload."""
    box = await page.locator("#ext-gen3003").bounding_box()
    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2

    result = await find_unique_locator_at_coordinates(
        page=page,
        x=x,
        y=y,
        element_id="elem_step3",
        element_description="unnamed entry widget",
        expected_text=None,
        element_data=None,
    )
    assert result["found"] is True
    assert result["best_locator"] == '[name="payload_ref"]'
    assert result["stability"] == "stable"


async def test_bare_digit_id_stays_stable(page):
    """The q01 GitHub shape: id=880667900 is a database id, empirically
    session-stable (bench 3/3). The narrowed digit rule must not demote it."""
    result = await _generate_locators_from_element_data(
        search_context=page,
        element_data=_element_data(
            tagName="a",
            id="880667900",
            textContent="my-repo",
        ),
        element_id="elem_4",
        element_description="first pinned repository link",
        page=page,
    )
    assert result is not None and result["found"] is True
    assert result["best_locator"] == '[id="880667900"]'
    assert result["stability"] == "stable"


# --- bs #58: framework counters in every id segment (real find_unique_locator_action) ---


@pytest.fixture
def action_fit_on():
    with patch.object(config.locator, "enable_action_fit", True):
        yield


@pytest.fixture
async def blank_page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page_obj = await browser.new_page(viewport={"width": 1280, "height": 900})
        try:
            yield page_obj
        finally:
            await browser.close()


async def _locate(page, selector, element_data, **kwargs):
    box = await page.locator(selector).bounding_box()
    return await find_unique_locator_action(
        x=box["x"] + box["width"] / 2,
        y=box["y"] + box["height"] / 2,
        element_id="elem_1",
        page=page,
        element_data=element_data,
        **kwargs,
    )


async def test_jsf_counter_id_is_shipped_marked_volatile(blank_page, action_fit_on):
    """The t1 repro: a JSF id (j_idt45) and its twin name are both counters; the id still ships (found=true is
    never lost) but is labelled volatile."""
    await blank_page.set_content(
        '<form><button id="j_idt45" name="j_idt45" type="button">Save</button>'
        '<button id="j_idt46" name="j_idt46" type="button">Delete</button></form>'
    )
    r = await _locate(
        blank_page,
        "#j_idt45",
        {"tagName": "button", "id": "j_idt45", "name": "j_idt45", "type": "button"},
        element_description="Save button",
        expected_text="Save",
        action="click",
    )
    assert r["best_locator"] == "#j_idt45"
    assert r["stability"] == "volatile"


async def test_hand_written_id_stays_stable(blank_page, action_fit_on):
    await blank_page.set_content(
        '<form><button id="saveBtn" name="save" type="button">Save</button>'
        '<button id="delBtn" name="delete" type="button">Delete</button></form>'
    )
    r = await _locate(
        blank_page,
        "#saveBtn",
        {"tagName": "button", "id": "saveBtn", "name": "save", "type": "button"},
        element_description="Save button",
        expected_text="Save",
        action="click",
    )
    assert r["best_locator"] == "#saveBtn"
    assert r["stability"] == "stable"


async def test_read_step_keeps_todays_address_and_labels_it_volatile(blank_page, action_fit_on):
    """D2: the read gate scores its tokens with segments=False, so an all-generated page still ships today's
    [id="j_idt5:j_idt30"] read address (no positional trade) — only the label changes to volatile."""
    await blank_page.set_content(
        '<form id="j_idt5"><div id="j_idt5:j_idt28"><span id="j_idt5:j_idt30">Order 1234 confirmed</span>'
        '<span id="j_idt5:j_idt31">Status</span></div></form>'
    )
    r = await _locate(
        blank_page,
        '[id="j_idt5:j_idt30"]',
        {
            "xpath": "/html[1]/body[1]/form[1]/div[1]/span[1]",
            "tagName": "span",
            "id": "j_idt5:j_idt30",
            "textContent": "Order 1234 confirmed",
        },
        element_description="order confirmation",
        expected_text="Order 1234 confirmed",
        candidate_locator='text="Order 1234 confirmed"',
        action="get_text",
    )
    assert r["best_locator"] == '[id="j_idt5:j_idt30"]'
    assert r["stability"] == "volatile"
