"""
The has_semantic_locators gate: which candidate types may enable the xpath path.

Regression cover for the locator-quality change of 2026-07-22 (``9e3fe36``,
"read parentClassName so STEP-0 parent-context CSS fires"). That commit
repaired a dead fallback and, as a side effect, handed
``_generate_locators_from_element_data`` a non-empty candidate list for every
no-id/no-name element whose parent carries a class. ``has_semantic_locators``
was computed as ``len(locator_candidates) > 0``, so the mere EXISTENCE of a
parent-context candidate flipped the gate and enabled the shortened-xpath
branch — including on elements where the parent-context candidate then failed
validation and contributed nothing.

Measured effect on ``logs/browser_use.log*``, holding browser-use at 0.12.6
across the boundary: shortened-xpath went 5.2% -> 38.9% of resolutions and
text-first 43.0% -> 22.1%. On the ASTPP sidebar the emitted locator flipped
from ``text="Accounts"`` (15/17 runs passed) to a positional xpath suffix
(4/8 passed), because the suffix ``_shorten_xpath`` accepts depends on which
page happens to be loaded at validation time.

The fix counts only genuinely semantic candidate types toward the gate.
Nothing is removed from the candidate list — parent-context CSS still wins
when it validates uniquely, which ``test_parent_class_css_still_wins_when_unique``
holds in place.

The fixture-driven tests are marked ``integration``/``fixture`` in line with
the rest of ``tests/test_locators`` — they drive a real headless Chromium
against inline HTML and need ``playwright install chromium``:

    pytest tests/test_locators/test_semantic_gate.py -m integration -v

The unit tests below need neither and run in the default fast lane.
"""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.smart_locator import (
    _build_element_data_candidates,
    _has_semantic_locators,
    find_unique_locator_at_coordinates,
)

FIXTURES_DIR = Path(__file__).parent / "locator_fixtures"


def _file_url(name: str) -> str:
    return (FIXTURES_DIR / name).resolve().as_uri()


# ----------------------------------------------------------------------
# Unit layer — the gate itself, over candidate-type sets
# ----------------------------------------------------------------------

# Types _build_element_data_candidates can emit. Kept explicit rather than
# derived so that adding a candidate type without deciding which side of the
# gate it belongs on fails test_every_emitted_type_is_classified below.
SEMANTIC_TYPES = [
    "id",
    "id-attr",
    "data-testid",
    "data-test",
    "name",
    "aria-role",
    "aria-label",
    "placeholder",
]
PARENT_CONTEXT_TYPES = ["parent-id-css", "parent-class-css"]


def _candidates(*types: str) -> list[dict]:
    return [{"locator": f"<{t}>", "type": t, "priority": 1} for t in types]


@pytest.mark.parametrize("candidate_type", SEMANTIC_TYPES)
def test_direct_attribute_types_open_the_gate(candidate_type):
    """An attribute that identifies the element itself is a real semantic
    locator — the xpath branch stays available behind it."""
    assert _has_semantic_locators(_candidates(candidate_type)) is True


@pytest.mark.parametrize("candidate_type", PARENT_CONTEXT_TYPES)
def test_parent_context_types_do_not_open_the_gate(candidate_type):
    """Parent-context CSS describes the element's NEIGHBOURHOOD, not the
    element. It may or may not be unique, and its existence says nothing
    about whether this element can be identified semantically — so it must
    not decide that TEXT-FIRST should be skipped."""
    assert _has_semantic_locators(_candidates(candidate_type)) is False


def test_parent_context_does_not_open_the_gate_even_alongside_itself():
    """Both parent-context branches together are still not a semantic
    identity. (Only one can fire per element today — the STEP-0 payload
    carries parentClassName but no parentId — but the gate must not depend
    on that.)"""
    assert _has_semantic_locators(_candidates(*PARENT_CONTEXT_TYPES)) is False


def test_semantic_type_alongside_parent_context_still_opens_the_gate():
    """The parent-context candidate is not subtractive — an element with a
    real id keeps the xpath branch it always had."""
    assert _has_semantic_locators(_candidates("parent-class-css", "id")) is True


def test_empty_candidate_list_keeps_the_gate_shut():
    assert _has_semantic_locators([]) is False


def test_every_emitted_type_is_classified():
    """Anti-rot guard.

    ``_build_element_data_candidates`` is the only producer feeding the gate,
    so the allowlist only has to cover what that function emits. If a new
    candidate type is added there and nobody decides which side of the gate
    it belongs on, this fails rather than silently defaulting it to
    "not semantic".
    """
    emitted = set()
    for element_data in (
        {"id": "login"},
        {"id": "12345"},
        {"dataTestId": "submit", "dataTestAttr": "data-testid"},
        {"dataTestId": "submit", "dataTestAttr": "data-test"},
        {"name": "email"},
        {"ariaLabel": "Close", "role": "button"},
        {"ariaLabel": "Close"},
        {"placeholder": "Search"},
        {"tagName": "input", "parentId": "formbox"},
        {"tagName": "a", "parentClassName": "menuclass"},
    ):
        emitted.update(c["type"] for c in _build_element_data_candidates(element_data))

    known = set(SEMANTIC_TYPES) | set(PARENT_CONTEXT_TYPES)
    assert emitted <= known, (
        f"_build_element_data_candidates emits unclassified candidate "
        f"type(s) {sorted(emitted - known)} — decide whether each is a "
        f"semantic identity for the element and add it to SEMANTIC_TYPES "
        f"or PARENT_CONTEXT_TYPES, then to _SEMANTIC_CANDIDATE_TYPES."
    )


# ----------------------------------------------------------------------
# Fixture layer — the whole STEP 0 -> TEXT-FIRST chain on a real DOM
# ----------------------------------------------------------------------

_ELEMENT_DATA_JS = """
(el) => ({
    tagName: el.tagName.toLowerCase(),
    id: el.id || "",
    name: el.getAttribute('name') || "",
    className: (typeof el.className === 'string') ? el.className : "",
    ariaLabel: el.getAttribute('aria-label') || "",
    placeholder: el.getAttribute('placeholder') || "",
    title: el.getAttribute('title') || "",
    href: el.getAttribute('href') || "",
    role: el.getAttribute('role') || "",
    dataTestId: el.getAttribute('data-testid') || el.getAttribute('data-test') || "",
    dataTestAttr: "data-testid",
    type: el.getAttribute('type') || "",
    value: el.value || "",
    textContent: (el.textContent || "").trim().slice(0, 80),
    parentClassName: el.parentElement
        ? ((typeof el.parentElement.className === 'string') ? el.parentElement.className : "")
        : "",
    xpath: (() => {
        // browser-use emits an unprefixed, 1-indexed path with the index
        // omitted when the tag is the only one of its kind among siblings.
        const parts = [];
        for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
            const tag = n.tagName.toLowerCase();
            const twins = n.parentElement
                ? Array.from(n.parentElement.children).filter(c => c.tagName === n.tagName)
                : [n];
            parts.unshift(twins.length > 1 ? `${tag}[${twins.indexOf(n) + 1}]` : tag);
        }
        return parts.join('/');
    })(),
})
"""


async def _coords_and_data(page, selector: str):
    handle = page.locator(selector).first
    await handle.scroll_into_view_if_needed()
    bbox = await handle.bounding_box()
    if not bbox:
        raise RuntimeError(f"No bounding box for {selector!r}")
    data = await handle.evaluate(_ELEMENT_DATA_JS)
    return bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2, data


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1920, "height": 1080})
        pg = await ctx.new_page()
        try:
            yield pg
        finally:
            await ctx.close()
            await browser.close()


@pytest.mark.integration
@pytest.mark.fixture
async def test_sidebar_nav_link_routes_to_text_first(page):
    """The regression case.

    16 sibling nav links, no id/name/aria-label/test-attr on any of them, each
    wrapped in a ``.menuclass`` div. Every STEP-0 candidate is ambiguous:
    ``.menuclass a``, ``a[role="button"]`` and ``.dropdown-toggle`` all count
    16. Before the fix the parent-context candidate opened the xpath gate and
    a positional suffix shipped; TEXT-FIRST never ran because STEP 0 had
    already returned a result.
    """
    await page.goto(_file_url("sidebar_parent_class_gate.html"), wait_until="domcontentloaded")
    x, y, data = await _coords_and_data(page, ".menuclass a[href='#navAccounts']")

    # Preconditions — if these drift the fixture has stopped reproducing the bug.
    assert data["parentClassName"] == "menuclass"
    assert not data["id"]
    assert not data["name"]
    assert not data["ariaLabel"]
    assert not data["dataTestId"]
    assert not data["placeholder"]
    assert await page.locator(".menuclass a").count() == 16
    assert await page.locator('a[role="button"]').count() == 16

    result = await find_unique_locator_at_coordinates(
        page,
        x,
        y,
        element_id="elem_4",
        element_description="Accounts item in the left sidebar navigation menu",
        expected_text="Accounts",
        element_data=data,
    )

    assert result["found"] is True
    assert result["all_locators"][0]["type"] == "text-first"
    assert "xpath" not in result["best_locator"].lower()
    assert result["stability"] != "positional"


@pytest.mark.integration
@pytest.mark.fixture
async def test_text_first_locator_survives_the_span_nesting(page):
    """The label sits in a ``<span>`` inside the ``<a>``, as the real control
    does. Playwright's text engine resolves to the innermost element, so the
    emitted locator points at the span — that is fine (a click bubbles), but
    it must still be exactly one element and it must be the one at the
    reported coordinates, or TEXT-FIRST would skip it and fall through to a
    coordinate strategy.
    """
    await page.goto(_file_url("sidebar_parent_class_gate.html"), wait_until="domcontentloaded")
    x, y, data = await _coords_and_data(page, ".menuclass a[href='#navAccounts']")

    assert await page.locator('text="Accounts"').count() == 1

    result = await find_unique_locator_at_coordinates(
        page,
        x,
        y,
        element_id="elem_4",
        element_description="Accounts item in the left sidebar navigation menu",
        expected_text="Accounts",
        element_data=data,
    )

    assert await page.locator(result["best_locator"]).count() == 1


@pytest.mark.integration
@pytest.mark.fixture
async def test_parent_class_css_still_wins_when_unique(page):
    """The capability ``9e3fe36`` added must survive the gate change.

    Same candidate family, but here ``.loginbox input`` matches exactly one
    element. Parent-context CSS stays in the candidate list and still wins —
    the fix removes it from the GATE, not from the cascade.
    """
    await page.goto(_file_url("parent_class_unique_wins.html"), wait_until="domcontentloaded")
    x, y, data = await _coords_and_data(page, ".loginbox input")

    assert data["parentClassName"] == "loginbox"
    assert not data["id"]
    assert not data["name"]
    assert await page.locator(".loginbox input").count() == 1

    result = await find_unique_locator_at_coordinates(
        page,
        x,
        y,
        element_id="elem_1",
        element_description="Account field in the login form",
        expected_text="admin",
        element_data=data,
    )

    assert result["found"] is True
    assert result["all_locators"][0]["type"] == "parent-class-css"
    # The branch narrows inputs by type on purpose, so the emitted selector is
    # '.loginbox input[type="text"]' rather than the bare '.loginbox input'.
    assert result["best_locator"].startswith(".loginbox input")
    assert await page.locator(result["best_locator"]).count() == 1
