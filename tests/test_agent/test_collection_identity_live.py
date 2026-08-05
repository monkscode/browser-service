"""The collection identity gate, driven against a real chromium DOM.

``test_collection_candidate_accept.py`` pins the same accept with a mocked
page: ``FakeLocator`` returns canned dicts and the real ``_RESOLVED_ELEMENT_JS``
never executes. That pins the DECISION, not the path — and the id-comparison
defect lived exactly in the gap between the two. The real read returns an
``id`` field because the JS emits one; the mock returned one because it was
told to, so a mocked suite can pass while the live path rejects.

THE DEFECT THIS FILE CONCLUDES
------------------------------
``_identity_mismatch`` was written for the unique accept, where "is this the
same element" is a coherent question. The collection accept reused it whole,
including the id comparison, which is unsound there: the read is pinned to
``.nth(0)`` while the agent indexes whichever member it clicked, and the
members of a keyed list carry distinct ids by construction. The reject fell
through to the cascade, which returns the ANCESTOR and hands the assembler back
the unvalidated child selector the accept exists to eliminate.

WHY A FIXTURE AND NOT A BENCH
-----------------------------
No bench can see this. Across nlrf's captured corpus, 0 of 330 collection
elements carry an id — books.toscrape (``<a>``, 182) and demoqa/herokuapp
(``<tr>``, 137) dominate it, and neither keys its rows. Server-rendered admin
tables routinely do, so the input simply never occurs in the benched
population. A 30-query run confirms no regression and proves nothing about the
fix; this fixture is what actually exercises it.

The fixture HTML is synthetic — reconstructed from the SHAPE of a billing
table, with invented account numbers. Nothing is customer-derived.

    pytest tests/test_agent/test_collection_identity_live.py -m integration
"""

from pathlib import Path

import pytest

from browser_service.agent.actions import _identity_mismatch, find_unique_locator_action

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "test_locators" / "locator_fixtures"
_FIXTURE = _FIXTURES_DIR / "id_keyed_collection.html"

# The row the agent indexes — deliberately NOT the first, which is the only row
# the pre-fix gate ever let through.
INDEXED_ROW = "#row-4471902"
ROW_COLLECTION = 'tbody tr[id^="row-"]'
ROW_COUNT = 6

# The same defect one level down, where a reject actually costs something. The
# agent identifies an <a>; if the candidate is rejected the cascade generalises
# up to the ROW, so the service ships a <tr> address for an <a> and leaves the
# assembler to invent a child selector — the chain both captured collection
# failures start from.
INDEXED_LINK = "#cust-4471902"
LINK_COLLECTION = "tbody a.customer-link"

# What the cascade would return if the candidate were rejected: an ancestor,
# which is the outcome this whole branch exists to stop.
CASCADE_RESULT = {
    "element_id": "elem_1",
    "found": True,
    "best_locator": "table > tbody",
    "element_type": "collection",
    "all_locators": [{"type": "collection", "locator": "table > tbody"}],
    "element_info": {"tagName": "tbody"},
    "approach_metrics": {"locator_approach": "collection", "fallback_depth": 3},
}

# browser-use's element_data shape, read off the live DOM rather than invented.
_ROW_ELEMENT_DATA_JS = """
(el) => ({
    tagName: el.tagName.toLowerCase(),
    id: el.id || "",
    name: el.getAttribute('name') || "",
    className: (typeof el.className === 'string') ? el.className : "",
    ariaLabel: el.getAttribute('aria-label') || "",
    placeholder: el.getAttribute('placeholder') || "",
    role: el.getAttribute('role') || "",
    type: el.getAttribute('type') || "",
    textContent: (el.textContent || "").trim().slice(0, 80),
    xpath: (() => {
        const parts = [];
        for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
            const tag = n.tagName.toLowerCase();
            const twins = n.parentElement
                ? Array.from(n.parentElement.children).filter(c => c.tagName === n.tagName)
                : [n];
            parts.unshift(twins.length > 1 ? tag + '[' + (twins.indexOf(n) + 1) + ']' : tag);
        }
        return parts.join('/');
    })(),
})
"""


@pytest.fixture
async def live_page():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1920, "height": 1080})
        pg = await ctx.new_page()
        try:
            await pg.goto(_FIXTURE.resolve().as_uri(), wait_until="domcontentloaded")
            yield pg
        finally:
            await ctx.close()
            await browser.close()


async def _coords_and_data(page, selector: str):
    handle = page.locator(selector).first
    await handle.scroll_into_view_if_needed()
    bbox = await handle.bounding_box()
    if not bbox:
        raise RuntimeError(f"No bounding box for {selector!r}")
    data = await handle.evaluate(_ROW_ELEMENT_DATA_JS)
    return bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2, data


async def _find(page, data, x, y, candidate, description="all rows in the customer results table"):
    return await find_unique_locator_action(
        x=x,
        y=y,
        element_id="elem_1",
        element_description=description,
        expected_text="Cierra Vega",
        candidate_locator=candidate,
        element_data=data,
        is_collection=True,
        page=page,
    )


@pytest.mark.integration
@pytest.mark.fixture
class TestIdKeyedCollectionOnARealDom:
    async def test_the_defect_is_present_on_this_fixture(self, live_page):
        """Precondition, run on the REAL payloads.

        If the indexed row and ``.nth(0)`` ever agreed on id, the acceptance
        test below would pass for the wrong reason and this file would be
        proving nothing.
        """
        _, _, indexed = await _coords_and_data(live_page, INDEXED_ROW)
        first = await live_page.locator(ROW_COLLECTION).nth(0).evaluate(_ROW_ELEMENT_DATA_JS)

        assert await live_page.locator(ROW_COLLECTION).count() == ROW_COUNT
        assert indexed["tagName"] == first["tagName"] == "tr"  # the tag agrees
        assert indexed["id"]
        assert first["id"]
        assert indexed["id"] != first["id"]  # ...and the id does not

        # The pre-fix predicate on those same real payloads: it rejects.
        assert _identity_mismatch(indexed, first, compare_id=True)
        # The shipped one: it does not.
        assert _identity_mismatch(indexed, first, compare_id=False) == ""

    async def test_the_candidate_is_accepted(self, live_page):
        """The point of the fix: the agent's own validated address survives."""
        x, y, data = await _coords_and_data(live_page, INDEXED_ROW)

        result = await _find(live_page, data, x, y, ROW_COLLECTION)

        assert result["found"] is True
        assert result["best_locator"] == ROW_COLLECTION
        assert result["element_type"] == "collection"
        assert result["count"] == ROW_COUNT
        assert result["unique"] is False

    async def test_the_accepted_locator_resolves_to_the_rows(self, live_page):
        """An accept is only worth having if the address works.

        Guards the ancestor problem in the other direction: a locator that
        validates inside the service but resolves to something else on the page.
        """
        x, y, data = await _coords_and_data(live_page, INDEXED_ROW)

        result = await _find(live_page, data, x, y, ROW_COLLECTION)

        locator = live_page.locator(result["best_locator"])
        assert await locator.count() == ROW_COUNT
        assert await locator.nth(0).evaluate("el => el.tagName.toLowerCase()") == "tr"
        # The row the agent actually indexed is inside the accepted set.
        assert await live_page.locator(f"{result['best_locator']}{INDEXED_ROW}").count() == 1

    async def test_a_real_tag_mismatch_is_still_rejected(self, live_page):
        """The half of the gate that must survive.

        The agent indexed a ``<tr>``; this candidate resolves to ``<td>``. Tag
        is provable, so reject — verified on the real DOM rather than against a
        canned dict, which is the whole reason this file exists.
        """
        from unittest.mock import patch

        x, y, data = await _coords_and_data(live_page, INDEXED_ROW)

        with patch(
            "browser_service.locators.find_unique_locator_at_coordinates",
            return_value=dict(CASCADE_RESULT),
        ):
            result = await _find(live_page, data, x, y, "tbody td")

        assert result["best_locator"] == "table > tbody"  # cascade, not the candidate


@pytest.mark.integration
@pytest.mark.fixture
class TestTheRejectIsExpensiveOneLevelDown:
    """The rows case proves the gate mis-fires. This one proves it costs.

    When the reject lands on a collection of LINKS inside the rows, the cascade
    has nothing anchor-like to fall back to and generalises upward — so the
    service ships an address for the row while the agent identified the anchor.
    Every read the assembler writes against that address then has to guess its
    way back down, and the guess is never validated here.
    """

    async def test_the_link_collection_is_accepted(self, live_page):
        x, y, data = await _coords_and_data(live_page, INDEXED_LINK)
        assert data["tagName"] == "a"
        assert data["id"] == "cust-4471902"

        result = await _find(
            live_page,
            data,
            x,
            y,
            LINK_COLLECTION,
            description="the customer name links in the results table",
        )

        assert result["best_locator"] == LINK_COLLECTION
        assert result["element_type"] == "collection"
        assert result["count"] == ROW_COUNT

    async def test_the_accepted_address_is_the_anchor_not_its_row(self, live_page):
        """The property that matters downstream: what the assembler is handed
        must BE the element, not an ancestor it has to descend from."""
        x, y, data = await _coords_and_data(live_page, INDEXED_LINK)

        result = await _find(
            live_page,
            data,
            x,
            y,
            LINK_COLLECTION,
            description="the customer name links in the results table",
        )

        resolved = live_page.locator(result["best_locator"])
        assert await resolved.count() == ROW_COUNT
        tags = {
            await resolved.nth(i).evaluate("el => el.tagName.toLowerCase()")
            for i in range(ROW_COUNT)
        }
        assert tags == {"a"}, f"shipped an address resolving to {tags}, not the anchors"

        # And the read the assembler would write needs no invented child step.
        texts = [await resolved.nth(i).inner_text() for i in range(ROW_COUNT)]
        assert "Cierra Vega" in texts
