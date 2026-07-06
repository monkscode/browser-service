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

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.smart_locator import (
    _generate_locators_from_element_data,
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
            id="ext-gen1042", name="username",
            placeholder="Enter username", type="text",
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
            tagName="button", id="ext-gen2001", textContent="Refresh grid",
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
            id="save_button", name="save", type="text",
        ),
        element_id="elem_3",
        element_description="save field",
        page=page,
    )
    assert result is not None and result["found"] is True
    assert result["best_locator"] == "#save_button"
    assert result["stability"] == "stable"


async def test_bare_digit_id_stays_stable(page):
    """The q01 GitHub shape: id=880667900 is a database id, empirically
    session-stable (bench 3/3). The narrowed digit rule must not demote it."""
    result = await _generate_locators_from_element_data(
        search_context=page,
        element_data=_element_data(
            tagName="a", id="880667900", textContent="my-repo",
        ),
        element_id="elem_4",
        element_description="first pinned repository link",
        page=page,
    )
    assert result is not None and result["found"] is True
    assert result["best_locator"] == '[id="880667900"]'
    assert result["stability"] == "stable"
