"""
E2 on the agent-candidate path, real Chromium (2026-09-15 probe shapes).

- react-select shape: a candidate resolving to an aria-disabled option
  (the live u11 option is disabled by the site, so u11 stays unpassable;
  this fixture adds an enabled twin to test the rule), and a counter-id
  candidate with a stable alternative on the page.
- todomvc u10 r2: after the todo was added (so the input was empty), the
  agent indexed the INPUT (element_index=5) while its candidate named the
  list (`ul.todo-list`). The identity guard must yield when the indexed
  element provably cannot perform the read. The `label:has-text` variant
  is r1's candidate STRING on r2's page state — in r1 itself the todo did
  not exist yet, which is outside E2.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import find_unique_locator_action
from browser_service.config import config

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "test_locators" / "locator_fixtures"

# browser-use-style element_data for the INDEXED element, including the
# absolute xpath browser-use supplies (html[1]/body[1]/...).
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


async def _center(page, selector: str):
    box = await page.locator(selector).first.bounding_box()
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


@pytest.fixture
def action_fit_on():
    with patch.object(config.locator, "enable_action_fit", True):
        yield


async def _pick_option(page, candidate, target, expected, action):
    await page.goto(_file_url("react_select_disabled_option.html"), wait_until="domcontentloaded")
    x, y = await _center(page, target)
    return await find_unique_locator_action(
        x=x,
        y=y,
        element_id="elem_2",
        element_description=f"the {expected} option in the open colour menu",
        expected_text=expected,
        candidate_locator=candidate,
        element_data=None,
        page=page,
        is_collection=False,
        action=action,
    )


async def test_disabled_candidate_today_is_accepted(page):
    out = await _pick_option(
        page, "id=react-select-3-option-1", "#react-select-3-option-4", "Blue", None
    )
    assert out["best_locator"] == "id=react-select-3-option-1"


async def test_disabled_candidate_click_falls_to_the_enabled_option(page, action_fit_on):
    out = await _pick_option(
        page, "id=react-select-3-option-1", "#react-select-3-option-4", "Blue", "click"
    )
    assert out["found"] is True
    assert await page.locator(out["best_locator"]).get_attribute("id") == "react-select-3-option-4"


async def test_volatile_candidate_kept_with_flag_off(page):
    out = await _pick_option(
        page, "id=react-select-3-option-2", "#react-select-3-option-2", "Purple", None
    )
    assert out["best_locator"] == "id=react-select-3-option-2"


async def test_volatile_candidate_yields_to_a_stable_locator(page, action_fit_on):
    out = await _pick_option(
        page, "id=react-select-3-option-2", "#react-select-3-option-2", "Purple", "click"
    )
    assert out["best_locator"] == 'text="Purple"'
    assert out["stability"] == "stable"


TODO_CANDIDATES = [
    'label:has-text("buy milk")',
    "ul.todo-list",
]  # u10 r1/r2 strings; r2's page state


async def _read_todo(page, candidate, action):
    await page.goto(_file_url("todo_list_next_to_input.html"), wait_until="domcontentloaded")
    data = await page.locator("input.new-todo").evaluate(INDEXED_DATA_JS)
    x, y = await _center(page, "label[data-testid='todo-title']")
    return await find_unique_locator_action(
        x=x,
        y=y,
        element_id="elem_2",
        element_description="the list item text element in the main todo list area",
        expected_text="buy milk",
        candidate_locator=candidate,
        element_data=data,
        page=page,
        is_collection=False,
        action=action,
    )


@pytest.mark.parametrize("candidate", TODO_CANDIDATES)
async def test_todo_candidate_rejected_today(page, candidate):
    out = await _read_todo(page, candidate, None)
    assert out["best_locator"] == '[placeholder="What needs to be done?"]'


@pytest.mark.parametrize("candidate", TODO_CANDIDATES)
async def test_todo_candidate_wins_over_an_unfit_index(page, action_fit_on, candidate):
    out = await _read_todo(page, candidate, "get_text")
    assert out["best_locator"] == candidate
    assert "buy milk" in await page.locator(candidate).inner_text()
