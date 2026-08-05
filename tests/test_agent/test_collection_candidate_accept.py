"""Tests for accepting an agent candidate that matches MANY on a collection.

The recorded defect — bench q10, books.toscrape.com, three captured runs
(641ff45a, 11a4fbf9, cfff6942 and others):

    element_description = "the links containing book titles ..."
    element_data        = <a>, xpath .../ol/li[1]/article/h3/a
    candidate_locator   = ".product_pod h3 a"
    is_collection       = True
    page.locator(".product_pod h3 a").count() -> 20

The service measured the agent's own address matching all 20 anchors and then
discarded it, because the accept branch required ``count == 1``. The traversal
fallback returned ``ol > li`` — the list ITEM, an ANCESTOR of the element the
agent identified — so the assembler had to invent a child selector to reach
the anchor. That invention is never validated by this service (only the
collection locator is), and when it is wrong the run dies:

    31a66a98  Get Attribute ${li} title -> AttributeError: 'title' not found
    q06 rep2  tbody tr -> div[role="gridcell"]:first-child -> 30s timeout

For a COLLECTION request, count > 1 is the correct answer, not a failure.

IDENTITY GATE — why this is not just ``or is_collection``:
the three guards on the unique path are all single-element.
``_read_resolved_element`` calls ``Locator.evaluate``, which raises Playwright
strict-mode on a 20-match locator and returns None — and None is documented as
"unknown, never mismatch", i.e. ACCEPT. Widening the branch without a new gate
would silently disarm the q08 identity guard. So the collection accept reads
the FIRST match and applies the same provable tag/id comparison.

Deliberately checks the first match only, matching the unique path's strength
(tag + id equality) rather than inventing a stricter set-wide rule: a
collection may legitimately mix element kinds, and over-rejecting here costs a
correct locator.
"""

import logging
from unittest.mock import patch

import pytest

from browser_service.agent.actions import find_unique_locator_action

CANDIDATE = ".product_pod h3 a"

# element_data as browser-use hands over the indexed <a> on the q10 page.
ANCHOR_ELEMENT_DATA = {
    "tagName": "a",
    "id": "",
    "className": "",
    "textContent": "A Light in the ...",
    "xpath": "html/body/div/div/div/div/section/div[2]/ol/li[1]/article/h3/a",
}

RESOLVED_ANCHOR = {
    "tagName": "a",
    "id": "",
    "className": "",
    "textContent": "A Light in the ...",
}

# What the traversal returns today: the <li> ANCESTOR, not the anchor.
RESOLVED_LIST_ITEM = {
    "tagName": "li",
    "id": "",
    "className": "col-xs-6 col-sm-4 col-md-3 col-lg-3",
    "textContent": "A Light in the ... £51.77 In stock",
}


class FakeLocator:
    def __init__(self, count: int, resolved: dict | None = None):
        self._count = count
        self._resolved = resolved or {}

    async def count(self) -> int:
        return self._count

    async def bounding_box(self):
        return None

    def nth(self, i: int) -> "FakeLocator":
        return FakeLocator(1, self._resolved)

    async def evaluate(self, js: str, arg=None, *, timeout: float = None):
        text = self._resolved.get("textContent", "")
        if "labelledbyText" in js:  # semantic validation probe
            return {
                "textContent": text,
                "textContentLength": len(text),
                "innerText": text,
                "placeholder": "",
                "ariaLabel": "",
                "value": "",
                "labelText": "",
                "labelledbyText": "",
                "tagName": self._resolved.get("tagName", ""),
                "isContentEditable": False,
            }
        return dict(self._resolved)  # identity probe


class FakePage:
    url = "https://books.toscrape.com"

    def __init__(self, locators: dict):
        self._locators = locators

    def locator(self, selector: str) -> FakeLocator:
        return self._locators.get(selector, FakeLocator(0))

    async def evaluate(self, *args, **kwargs):
        return None


# What the cascade would return if the candidate is rejected — the container.
CASCADE_RESULT = {
    "element_id": "elem_1",
    "found": True,
    "best_locator": "ol > li",
    "element_type": "collection",
    "all_locators": [{"type": "collection", "locator": "ol > li"}],
    "element_info": {"tagName": "li"},
    "approach_metrics": {"locator_approach": "collection", "fallback_depth": 3},
}


async def _call(page, **overrides):
    kwargs = dict(
        x=455,
        y=410,
        element_id="elem_1",
        element_description="the links containing book titles within the main product gallery grid",
        expected_text="A Light in the ...",
        candidate_locator=CANDIDATE,
        element_data=dict(ANCHOR_ELEMENT_DATA),
        is_collection=True,
        page=page,
    )
    kwargs.update(overrides)
    with patch(
        "browser_service.locators.find_unique_locator_at_coordinates",
        return_value=dict(CASCADE_RESULT),
    ):
        return await find_unique_locator_action(**kwargs)


class TestQ10Replay:
    """The agent's own validated address must survive a collection request."""

    @pytest.mark.asyncio
    async def test_multi_match_candidate_is_accepted_for_a_collection(self):
        page = FakePage({CANDIDATE: FakeLocator(20, RESOLVED_ANCHOR)})

        result = await _call(page)

        assert result["found"] is True
        assert result["best_locator"] == CANDIDATE
        assert result["count"] == 20

    @pytest.mark.asyncio
    async def test_accept_does_not_claim_uniqueness(self):
        """20 matches is the answer here, but it is still not unique — the
        payload must say so or downstream reads it as a single element."""
        page = FakePage({CANDIDATE: FakeLocator(20, RESOLVED_ANCHOR)})

        result = await _call(page)

        assert result["unique"] is False
        assert result["all_locators"][0]["unique"] is False

    @pytest.mark.asyncio
    async def test_accept_is_stamped_as_a_collection(self):
        """nlrf routes the assembler's FOR-loop block on element_type ==
        'collection' (tasks._needs_loop). Without the stamp the assembler
        points a single-element keyword at a 20-match locator."""
        page = FakePage({CANDIDATE: FakeLocator(20, RESOLVED_ANCHOR)})

        result = await _call(page)

        assert result["best_locator"] == CANDIDATE  # the accept, not the cascade
        assert result["element_type"] == "collection"

    @pytest.mark.asyncio
    async def test_accept_records_the_collection_approach(self):
        page = FakePage({CANDIDATE: FakeLocator(20, RESOLVED_ANCHOR)})

        result = await _call(page)

        assert result["approach_metrics"]["is_collection"] is True
        assert result["approach_metrics"]["success"] is True


class TestIdentityGate:
    """The unique path's guards cannot run on a multi-match locator, so the
    collection accept brings its own — same provable comparison, first match."""

    @pytest.mark.asyncio
    async def test_candidate_resolving_to_a_different_tag_is_rejected(self, caplog):
        """Agent indexed <a>; this candidate's matches are <li>. Reject into
        the cascade rather than ship an ancestor as if it were the element."""
        page = FakePage({CANDIDATE: FakeLocator(20, RESOLVED_LIST_ITEM)})

        with caplog.at_level(logging.INFO):
            result = await _call(page)

        assert result["best_locator"] == "ol > li"  # cascade, not the candidate
        assert "collection-candidate-resolves-to-different-element" in caplog.text

    @pytest.mark.asyncio
    async def test_absent_element_data_still_accepts(self):
        """Nothing to compare against is not evidence of a mismatch — the
        unique path treats this the same way."""
        page = FakePage({CANDIDATE: FakeLocator(20, RESOLVED_ANCHOR)})

        result = await _call(page, element_data=None)

        assert result["best_locator"] == CANDIDATE

    @pytest.mark.asyncio
    async def test_members_carrying_distinct_ids_are_not_a_mismatch(self):
        """The id comparison is unsound on THIS path and must not run.

        On the unique path two different non-empty ids prove two different
        elements. On a collection they prove nothing: the members of a keyed
        list carry distinct ids BY CONSTRUCTION, and the read is pinned to
        .nth(0) while the agent indexes whichever member it clicked. Any
        server-rendered table keyed by database id hits this on every row but
        the first.

        Left in, it rejects a correct locator into the cascade, which returns
        the ANCESTOR and hands the assembler back the unvalidated child
        selector this whole branch exists to eliminate. Tag stays compared —
        that is the half with evidence behind it (the bench rejected
        `article.product_pod` as "indexed <a> but resolves to <article>").
        """
        indexed_third_row = {
            "tagName": "tr",
            "id": "row-3",
            "className": "",
            "textContent": "Cierra Vega",
            "xpath": "html/body/table/tbody/tr[3]",
        }
        resolved_first_row = {
            "tagName": "tr",
            "id": "row-1",
            "className": "",
            "textContent": "Alden Cantrell",
        }
        locator = "tr[id^=row]"
        page = FakePage({locator: FakeLocator(20, resolved_first_row)})

        result = await _call(
            page,
            candidate_locator=locator,
            element_data=indexed_third_row,
            element_description="all rows in the results table",
            expected_text="Cierra Vega",
        )

        assert result["best_locator"] == locator
        assert result["element_type"] == "collection"

    @pytest.mark.asyncio
    async def test_a_differing_id_still_rejects_on_the_unique_path(self):
        """Scope guard: dropping the id comparison is scoped to collections.

        count == 1 keeps both halves — there, two different non-empty ids are
        exactly the q08 proof that the candidate found a different element.
        """
        page = FakePage({CANDIDATE: FakeLocator(1, {"tagName": "a", "id": "other"})})

        result = await _call(page, element_data={"tagName": "a", "id": "indexed"})

        assert result["best_locator"] == "ol > li"  # cascade, not the candidate


class TestScope:
    """Everything outside `is_collection and count > 1` keeps today's path."""

    @pytest.mark.asyncio
    async def test_multi_match_without_collection_flag_still_falls_through(self):
        """A non-collection request wanting ONE element must keep rejecting a
        many-match candidate — that guard is why q02/q08 resolve correctly."""
        page = FakePage({CANDIDATE: FakeLocator(20, RESOLVED_ANCHOR)})

        result = await _call(page, is_collection=False)

        assert result["best_locator"] == "ol > li"  # cascade

    @pytest.mark.asyncio
    async def test_zero_match_candidate_still_falls_through(self):
        page = FakePage({CANDIDATE: FakeLocator(0)})

        result = await _call(page)

        assert result["best_locator"] == "ol > li"  # cascade

    @pytest.mark.asyncio
    async def test_single_match_collection_takes_the_unique_path(self):
        """count == 1 must keep the existing accept, which reports unique."""
        page = FakePage({CANDIDATE: FakeLocator(1, RESOLVED_ANCHOR)})

        result = await _call(page)

        assert result["best_locator"] == CANDIDATE
        assert result["unique"] is True


class TestHasIdIsOnePredicate:
    """Both accept branches report the same locator shapes, so both must score
    ``approach_metrics.has_id`` the same way.

    nlrf's tools/analyze_locator_patterns.py counts this field across
    approaches; two definitions silently make that count mean two things. The
    collection branch shipped a narrower inline copy (``#``/``[id=`` only),
    which scored `[id="books"] li` and `id=books >> li` differently depending
    only on which branch accepted them.
    """

    @pytest.mark.parametrize(
        "locator, expected",
        [
            ("#books li", True),
            ('[id="books"] li', True),
            ('[id^="row"]', True),
            ("id=books >> li", True),  # the shape the narrow copy missed
            ("li >> id=title", True),  # engine token mid-chain
            ('xpath=//div[@id="books"]', True),  # 64 occurrences in the corpus
            ('[data-testid="book"] a', False),  # ends "...test|id=|"
            ('[aria-invalid="true"]', False),  # ends "...inval|id=|"
            ('[data-qa="book"] a', False),
            (".product_pod h3 a", False),
            ("text=Accounts", False),
        ],
    )
    @pytest.mark.asyncio
    async def test_both_branches_agree(self, locator, expected):
        collection = await _call(
            FakePage({locator: FakeLocator(20, RESOLVED_ANCHOR)}), candidate_locator=locator
        )
        unique = await _call(
            FakePage({locator: FakeLocator(1, RESOLVED_ANCHOR)}), candidate_locator=locator
        )

        assert collection["best_locator"] == locator  # both accepted
        assert unique["best_locator"] == locator
        assert collection["approach_metrics"]["has_id"] is expected
        assert unique["approach_metrics"]["has_id"] is expected
