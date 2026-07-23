"""
Tests for the row-anchor guards on the agent-input trust boundary
(ASTPP gate q02, root-caused 2026-07-16).

The recorded failure (5 of 6 gate elem_4 runs, gate-r2 baseline): the
vision agent called find_unique_locator with a self-contradiction —
    element_description = "Edit icon button inside the table row
                           containing customer ID 4727985745"
    expected_text       = "4727985745"        <- the ROW ANCHOR DATUM
    row_anchor_text     = "4727985745"
    candidate_locator   = "a[title='Edit']"   <- healthy, count=9, rejected
    element_data        = <a> textContent="Edit"  <- DOM ground truth
The cascade's TEXT-FIRST step then built text="4727985745" from the
corrupt expected_text; the account CELL is unique on the page, so
count==1 and the row-anchor rescue (which only fires on count>1) never
ran. The cell shipped as the Edit locator in every gate-r2 test.robot.

Two guards, one trust boundary:
  1. expected_text correction — when expected_text IS the anchor datum
     and element_data's own text disagrees, trust the DOM: search with
     the element's text, which repeats across rows and triggers the
     existing row-anchor rescue. Signal: row-anchor-corrects-expected-text
  2. candidate anchor-datum reject — a unique candidate that targets the
     anchor datum itself while the element's text disagrees is the CELL,
     not the row's control; reject it into the cascade even when
     expected_text is absent (semantic validation is skipped there).
     Signal: row-anchor-rejects-candidate
"""

import logging

import pytest

from browser_service.agent.actions import find_unique_locator_action

ANCHOR = "4727985745"
DESC = "Edit icon button inside the table row containing customer ID 4727985745"
TEXT_EDIT = 'text="Edit"'
COMPOSITE = f'tr:has-text("{ANCHOR}") >> {TEXT_EDIT}'
# DOM-confirmed coordinates of the Edit link from the gate-r2 r1 trace.
X, Y = 494, 682
EDIT_BOX = {"x": 480.0, "y": 670.0, "width": 30.0, "height": 20.0}

Q02_ELEMENT_DATA = {
    "tagName": "a",
    "id": "",
    "className": "",
    "textContent": "Edit",
    "xpath": "html/body/main/div[3]/div/div/div[2]/table/tbody/tr[9]/td[2]/div/a[1]",
}


class FakeLocator:
    """Playwright Locator stand-in: count() for validation,
    bounding_box() for the coordinate identity check, evaluate() for
    semantic validation."""

    def __init__(self, count: int, text: str = "", box: dict = None):
        self._count = count
        self._text = text
        self._box = box

    async def count(self) -> int:
        return self._count

    async def bounding_box(self):
        return self._box

    def nth(self, i: int) -> "FakeLocator":
        return FakeLocator(1, text=self._text, box=self._box)

    async def evaluate(self, js: str) -> dict:
        return {
            "textContent": self._text,
            "textContentLength": len(self._text),
            "innerText": self._text,
            "placeholder": "",
            "ariaLabel": "",
            "value": "",
            "labelText": "",
            "labelledbyText": "",
        }


class FakePage:
    """Lenient page stand-in: unknown selectors count 0 so the cascade
    walks past them; evaluate() (coordinate fallback JS) yields nothing."""

    url = "https://sujal.astppbilling.org/accounts/customer_list/"

    def __init__(self, locators: dict):
        self._locators = locators
        self.probed = []

    def locator(self, selector: str) -> FakeLocator:
        self.probed.append(selector)
        return self._locators.get(selector, FakeLocator(0))

    async def evaluate(self, *args, **kwargs):
        return None


def _q02_page() -> FakePage:
    return FakePage(
        {
            "a[title='Edit']": FakeLocator(9),
            "text=4727985745": FakeLocator(1, text=ANCHOR),
            'text="4727985745"': FakeLocator(1, text=ANCHOR),
            TEXT_EDIT: FakeLocator(9),
            COMPOSITE: FakeLocator(1, text="Edit", box=EDIT_BOX),
        }
    )


async def _call(page, **overrides):
    kwargs = dict(
        x=X,
        y=Y,
        element_id="elem_4",
        element_description=DESC,
        expected_text=ANCHOR,
        candidate_locator="a[title='Edit']",
        element_data=dict(Q02_ELEMENT_DATA),
        page=page,
        is_collection=False,
        row_anchor_text=ANCHOR,
    )
    kwargs.update(overrides)
    return await find_unique_locator_action(**kwargs)


class TestQ02Replay:
    """The gate-r2 elem_4 call, replayed verbatim — the engine must emit
    the row-scoped composite, never the bare anchor cell."""

    @pytest.mark.asyncio
    async def test_q02_emits_row_scoped_composite(self, caplog):
        with caplog.at_level(logging.INFO):
            result = await _call(_q02_page())
        assert result["found"] is True
        assert result["best_locator"] == COMPOSITE
        assert result["best_locator"] != 'text="4727985745"'  # the shipped bug
        assert result.get("row_anchored") is True
        assert "row-anchor-corrects-expected-text" in caplog.text


class TestCandidateAnchorReject:
    """A unique candidate that targets the anchor datum itself (the
    cell) while the indexed element's text disagrees must be rejected
    into the cascade."""

    @pytest.mark.asyncio
    async def test_anchor_datum_candidate_rejected_with_expected_text(self, caplog):
        with caplog.at_level(logging.INFO):
            result = await _call(_q02_page(), candidate_locator="text=4727985745")
        assert "row-anchor-rejects-candidate" in caplog.text
        assert result["found"] is True
        assert result["best_locator"] == COMPOSITE
        assert result["all_locators"][0]["type"] != "candidate"

    @pytest.mark.asyncio
    async def test_anchor_datum_candidate_rejected_without_expected_text(self, caplog):
        """expected_text absent -> semantic validation is skipped, so
        only the explicit guard stands between the cell candidate and a
        wrong accept."""
        with caplog.at_level(logging.INFO):
            result = await _call(
                _q02_page(),
                candidate_locator="text=4727985745",
                expected_text=None,
            )
        assert "row-anchor-rejects-candidate" in caplog.text
        if result.get("found"):
            assert result["all_locators"][0]["type"] != "candidate"
            assert result["best_locator"] != "text=4727985745"

    @pytest.mark.asyncio
    async def test_legit_anchor_click_accepted_unchanged(self):
        """User genuinely clicks the datum ('click the 4727985745
        link'): element text EQUALS the anchor — accept stands."""
        page = FakePage(
            {
                "text=4727985745": FakeLocator(1, text=ANCHOR, box=EDIT_BOX),
            }
        )
        result = await _call(
            page,
            candidate_locator="text=4727985745",
            element_data={"tagName": "a", "textContent": ANCHOR},
        )
        assert result["found"] is True
        assert result["all_locators"][0]["type"] == "candidate"
        assert result["best_locator"] == "text=4727985745"

    @pytest.mark.asyncio
    async def test_row_scoped_candidate_not_rejected(self):
        """Anchor text in the SCOPING segment is the correct shape —
        the guard must not fire on it."""
        candidate = f'tr:has-text("{ANCHOR}") >> a[title="Edit"]'
        page = FakePage(
            {
                candidate: FakeLocator(1, text="Edit", box=EDIT_BOX),
            }
        )
        result = await _call(page, candidate_locator=candidate, expected_text="Edit")
        assert result["found"] is True
        assert result["all_locators"][0]["type"] == "candidate"
        assert result["best_locator"] == candidate

    @pytest.mark.asyncio
    async def test_healthy_candidate_with_anchor_set_accepted(self):
        """row_anchor_text present must not disturb ordinary accepts
        (login fields carry no anchor overlap)."""
        page = FakePage(
            {
                "id=username": FakeLocator(1, text="Enter Account Number or Email", box=EDIT_BOX),
            }
        )
        result = await _call(
            page,
            candidate_locator="id=username",
            expected_text="Enter Account Number or Email",
            element_data={
                "tagName": "input",
                "textContent": "Enter Account Number or Email",
            },
            row_anchor_text=None,
        )
        assert result["found"] is True
        assert result["all_locators"][0]["type"] == "candidate"
