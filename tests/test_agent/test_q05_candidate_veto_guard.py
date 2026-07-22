"""
q05 guard (i), actions level — the corrupt-expected_text veto replay
(ASTPP gate q05 r3, gate-r2 baseline, root-caused 2026-07-16).

Verbatim traffic: the vision agent attached the LABEL's text to the
INPUT —
    element_description = "Email input field in the customer add form"
    expected_text       = "Email *"            <- the label's text
    candidate_locator   = "input[name='email']" <- correct, count=1
    element surface     = ENTIRELY EMPTY (broken <label for=>, no
                          placeholder/aria/value — live-verified)
The semantic check vetoed the correct candidate (an input's own text is
''), the cascade fell to the ELEMENT-DATA path with no classifier stamp,
and the assembler emitted its designed EXPECTED_STATE_CLASS placeholder.

Guard (i): the candidate check passes accept_empty_interactive=True, so
a unique+valid candidate whose surface is EMPTY is accepted on
uniqueness. Signal: empty-surface-interactive-accept.

Same conventions as test_row_anchor_candidate_guard.py.
"""

import logging

import pytest

from browser_service.agent.actions import find_unique_locator_action

DESC = "Email input field in the customer add form"
LABEL_TEXT = "Email *"
CANDIDATE = "input[name='email']"
X, Y = 640, 480
EMAIL_BOX = {"x": 620.0, "y": 470.0, "width": 200.0, "height": 24.0}

EMAIL_ELEMENT_DATA = {
    "tagName": "input",
    "id": "",
    "className": "text field medium form-control",
    "textContent": "",
}


class FakeLocator:
    """Playwright Locator stand-in. evaluate() returns what the real
    validate_semantic_match JS returns, including the element's tag."""

    def __init__(self, count: int, tag: str = "INPUT", surface: dict = None, box: dict = None):
        self._count = count
        self._tag = tag
        self._surface = surface or {}
        self._box = box

    async def count(self) -> int:
        return self._count

    async def bounding_box(self):
        return self._box

    def nth(self, i: int) -> "FakeLocator":
        return FakeLocator(1, tag=self._tag, surface=self._surface, box=self._box)

    async def evaluate(self, js: str) -> dict:
        text = self._surface.get("textContent", "")
        return {
            "textContent": text,
            "textContentLength": len(text),
            "innerText": self._surface.get("innerText", text),
            "placeholder": self._surface.get("placeholder", ""),
            "ariaLabel": self._surface.get("ariaLabel", ""),
            "value": self._surface.get("value", ""),
            "labelText": self._surface.get("labelText", ""),
            "labelledbyText": self._surface.get("labelledbyText", ""),
            "tagName": self._tag,
            "isContentEditable": False,
        }


class FakePage:
    url = "https://sujal.astppbilling.org/accounts/customer_add/"

    def __init__(self, locators: dict):
        self._locators = locators
        self.probed = []

    def locator(self, selector: str) -> FakeLocator:
        self.probed.append(selector)
        return self._locators.get(selector, FakeLocator(0))

    async def evaluate(self, *args, **kwargs):
        return None


async def _call(page, **overrides):
    kwargs = dict(
        x=X,
        y=Y,
        element_id="elem_9",
        element_description=DESC,
        expected_text=LABEL_TEXT,
        candidate_locator=CANDIDATE,
        element_data=dict(EMAIL_ELEMENT_DATA),
        page=page,
        is_collection=False,
    )
    kwargs.update(overrides)
    return await find_unique_locator_action(**kwargs)


class TestQ05Replay:
    """The corrupt-expected_text q05 call, replayed verbatim — the
    unique surface-less input candidate must be accepted."""

    @pytest.mark.asyncio
    async def test_empty_surface_candidate_accepted(self, caplog):
        page = FakePage(
            {
                CANDIDATE: FakeLocator(1, tag="INPUT", box=EMAIL_BOX),
            }
        )
        with caplog.at_level(logging.INFO):
            result = await _call(page)
        assert result["found"] is True
        assert result["best_locator"] == CANDIDATE
        assert result["all_locators"][0]["type"] == "candidate"
        assert "empty-surface-interactive-accept" in caplog.text

    @pytest.mark.asyncio
    async def test_surface_bearing_mismatch_still_vetoed(self, caplog):
        """The carve-out must not weaken the probe-06 veto: a candidate
        whose surface genuinely disagrees still falls to the cascade."""
        page = FakePage(
            {
                "input[name='username']": FakeLocator(
                    1,
                    tag="INPUT",
                    surface={"placeholder": "Enter Username"},
                    box=EMAIL_BOX,
                ),
            }
        )
        with caplog.at_level(logging.INFO):
            result = await _call(page, candidate_locator="input[name='username']")
        if result.get("found"):
            assert result["all_locators"][0]["type"] != "candidate"
        assert "empty-surface-interactive-accept" not in caplog.text

    @pytest.mark.asyncio
    async def test_empty_surface_non_interactive_not_carved_out(self, caplog):
        """A surface-less non-interactive node (bare div) is not rescued
        by the carve-out even as an agent candidate."""
        page = FakePage(
            {
                "div.decor": FakeLocator(1, tag="DIV", box=EMAIL_BOX),
            }
        )
        with caplog.at_level(logging.INFO):
            result = await _call(
                page,
                candidate_locator="div.decor",
                element_data={"tagName": "div", "textContent": ""},
            )
        if result.get("found"):
            assert result["all_locators"][0]["type"] != "candidate"
        assert "empty-surface-interactive-accept" not in caplog.text
