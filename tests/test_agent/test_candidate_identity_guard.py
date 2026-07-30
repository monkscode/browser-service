"""
Tests for the candidate identity guard (q08 select/option defect, 2026-07-30).

The recorded failure — q08 rep1, the-internet.herokuapp.com/dropdown:
    element_data      = <select id="dropdown">   <- the element the agent INDEXED
    expected_text     = "Option 2"
    candidate_locator = "css=select#dropdown option[selected='selected']"
                                                 <- resolves to an <option>

The semantic check passed *correctly*: Chromium moves the `selected` content
attribute onto the chosen option when browser-use selects, so the candidate
really did resolve to an element reading "Option 2". What went wrong is that
the accept described a DIFFERENT element than the one it returned —
`_candidate_element_info` and `_candidate_tier0_stamp` both read element_data,
so nlrf received `element_type: select` + `select_id: dropdown` beside an
<option> locator. The assembler paired that with `Get Selected Options`, which
runs Array.from(el.selectedOptions) — undefined on an <option>:
    TypeError: undefined is not iterable

Two changes, one trust boundary (owner-approved 2026-07-30):
  1. identity reject — when the resolved element is PROVABLY not the indexed
     element (different tag, or two different non-empty ids), reject into the
     cascade. Signal: candidate-resolves-to-different-element
  2. describe what was accepted — element_info and the Tier-0 stamp derive
     from the resolved element, falling back to element_data only when the
     read fails.

The gate is deliberately conservative: it fires only on provable mismatch.
An unreadable element, absent element_data, or a missing id on either side
leaves the pre-existing accept behaviour untouched — this guard must never
turn a working accept into a found=false.
"""

import logging
from unittest.mock import patch

import pytest

from browser_service.agent.actions import find_unique_locator_action

Q08_CANDIDATE = "css=select#dropdown option[selected='selected']"

# element_data as browser-use hands it over for index [7] on the q08 page.
Q08_ELEMENT_DATA = {
    "tagName": "select",
    "id": "dropdown",
    "className": "",
    "textContent": "Please select an option\nOption 1\nOption 2",
    "xpath": "html/body/div[2]/div/div/select",
}

# What Q08_CANDIDATE actually resolves to once Option 2 is selected.
Q08_RESOLVED_OPTION = {
    "id": "",
    "tagName": "OPTION",
    "textContent": "Option 2",
    "className": "",
    "ariaInvalid": "",
    "parentClassName": "",
    "name": "",
    "dataTestId": "",
    "role": "",
    "type": "",
}


class FakeLocator:
    """Playwright Locator stand-in.

    evaluate() serves both probes off one resolved-element dict: the semantic
    check's payload is recognised by its labelledbyText key, everything else
    is the identity read.
    """

    def __init__(
        self,
        count: int,
        resolved: dict = None,
        evaluate_raises: bool = False,
        evaluate_returns=None,
    ):
        self._count = count
        self._resolved = resolved or {}
        self._evaluate_raises = evaluate_raises
        self._evaluate_returns = evaluate_returns

    async def count(self) -> int:
        return self._count

    async def bounding_box(self):
        return None

    def nth(self, i: int) -> "FakeLocator":
        return FakeLocator(1, self._resolved)

    async def evaluate(self, js: str):
        if self._evaluate_raises:
            raise RuntimeError("Element is not attached to the DOM")
        if self._evaluate_returns is not None and "labelledbyText" not in js:
            return self._evaluate_returns
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
    """Unknown selectors count 0 so the cascade walks past them."""

    url = "https://the-internet.herokuapp.com/dropdown"

    def __init__(self, locators: dict):
        self._locators = locators

    def locator(self, selector: str) -> FakeLocator:
        return self._locators.get(selector, FakeLocator(0))

    async def evaluate(self, *args, **kwargs):
        return None


CASCADE_RESULT = {
    "element_id": "elem_2",
    "found": True,
    "best_locator": "id=dropdown",
    "all_locators": [{"type": "id", "locator": "id=dropdown"}],
    "element_info": {"tagName": "select", "id": "dropdown"},
    "approach_metrics": {"locator_approach": "smart_locator", "fallback_depth": 1},
}


async def _call(page, **overrides):
    kwargs = dict(
        x=960,
        y=99,
        element_id="elem_2",
        element_description="the selected option within the dropdown on the main page",
        expected_text="Option 2",
        candidate_locator=Q08_CANDIDATE,
        element_data=dict(Q08_ELEMENT_DATA),
        page=page,
    )
    kwargs.update(overrides)
    with patch(
        "browser_service.locators.find_unique_locator_at_coordinates",
        return_value=dict(CASCADE_RESULT),
    ):
        return await find_unique_locator_action(**kwargs)


class TestQ08Replay:
    """The captured q08 call — an <option> locator must never ship stamped
    as the <select> the agent indexed."""

    @pytest.mark.asyncio
    async def test_option_under_select_is_rejected_into_cascade(self, caplog):
        page = FakePage({Q08_CANDIDATE: FakeLocator(1, Q08_RESOLVED_OPTION)})
        with caplog.at_level(logging.INFO):
            result = await _call(page)

        assert "candidate-resolves-to-different-element" in caplog.text
        assert result["all_locators"][0]["type"] != "candidate"
        assert result["best_locator"] == "id=dropdown"

    @pytest.mark.asyncio
    async def test_rejected_candidate_never_stamps_select_id(self):
        """select_id beside an <option> locator is what detonated downstream."""
        page = FakePage({Q08_CANDIDATE: FakeLocator(1, Q08_RESOLVED_OPTION)})
        result = await _call(page)

        assert result.get("select_id") is None
        assert result["best_locator"] != Q08_CANDIDATE

    @pytest.mark.asyncio
    async def test_semantic_check_alone_would_have_accepted(self):
        """Guard against a false-green replay: the resolved <option> really
        does read 'Option 2', so semantic validation cannot reject it. Only
        the identity gate stands here."""
        from browser_service.locators import validate_semantic_match

        page = FakePage({Q08_CANDIDATE: FakeLocator(1, Q08_RESOLVED_OPTION)})
        ok, observed = await validate_semantic_match(
            None, "Option 2", page=page, locator=Q08_CANDIDATE
        )
        assert ok is True
        assert observed == "Option 2"


class TestIdentityGateBoundaries:
    """Fires only on provable mismatch — never on uncertainty."""

    @pytest.mark.asyncio
    async def test_matching_tag_and_id_accepts(self):
        resolved = {"id": "email", "tagName": "INPUT", "textContent": "", "className": "form-control"}
        page = FakePage({"id=email": FakeLocator(1, resolved)})
        result = await _call(
            page,
            candidate_locator="id=email",
            expected_text=None,
            element_data={"tagName": "input", "id": "email"},
        )
        assert result["all_locators"][0]["type"] == "candidate"

    @pytest.mark.asyncio
    async def test_same_tag_different_id_rejects(self, caplog):
        resolved = {"id": "password", "tagName": "INPUT", "textContent": ""}
        page = FakePage({"id=password": FakeLocator(1, resolved)})
        with caplog.at_level(logging.INFO):
            result = await _call(
                page,
                candidate_locator="id=password",
                expected_text=None,
                element_data={"tagName": "input", "id": "email"},
            )
        assert "candidate-resolves-to-different-element" in caplog.text
        assert result["all_locators"][0]["type"] != "candidate"

    @pytest.mark.asyncio
    async def test_missing_id_on_one_side_is_not_provable(self):
        """An empty id proves nothing — accept, as before the guard."""
        resolved = {"id": "", "tagName": "INPUT", "textContent": ""}
        page = FakePage({"[name='email']": FakeLocator(1, resolved)})
        result = await _call(
            page,
            candidate_locator="[name='email']",
            expected_text=None,
            element_data={"tagName": "input", "id": "email"},
        )
        assert result["all_locators"][0]["type"] == "candidate"

    @pytest.mark.asyncio
    async def test_absent_element_data_still_accepts(self):
        resolved = {"id": "go", "tagName": "BUTTON", "textContent": "Go"}
        page = FakePage({"id=go": FakeLocator(1, resolved)})
        result = await _call(
            page, candidate_locator="id=go", expected_text=None, element_data=None
        )
        assert result["all_locators"][0]["type"] == "candidate"

    @pytest.mark.asyncio
    async def test_unreadable_element_falls_back_to_element_data(self):
        """A failed identity read must not turn a working accept into a
        rejection — degrade to the pre-guard behaviour."""
        page = FakePage({"id=email": FakeLocator(1, evaluate_raises=True)})
        result = await _call(
            page,
            candidate_locator="id=email",
            expected_text=None,
            element_data={"tagName": "input", "id": "email", "className": "form-control"},
        )
        assert result["all_locators"][0]["type"] == "candidate"
        assert result["element_info"]["className"] == "form-control"

    @pytest.mark.asyncio
    async def test_non_dict_evaluate_payload_falls_back(self):
        """An unexpected evaluate() shape is unknown evidence, not a mismatch —
        it must not raise inside the accept and drop a valid candidate."""
        page = FakePage({"id=email": FakeLocator(1, evaluate_returns="not-a-dict")})
        result = await _call(
            page,
            candidate_locator="id=email",
            expected_text=None,
            element_data={"tagName": "input", "id": "email"},
        )
        assert result["all_locators"][0]["type"] == "candidate"
        assert result["element_info"]["source"] == "candidate_element_data"


class TestDescribesWhatWasAccepted:
    """element_info and the Tier-0 stamp come from the resolved element."""

    @pytest.mark.asyncio
    async def test_stamp_derives_from_resolved_not_stale_element_data(self):
        """element_data carries no className; the live element is a flatpickr
        input. Deriving from element_data loses the date-picker routing."""
        resolved = {
            "id": "customer_cdr_from_date",
            "tagName": "INPUT",
            "textContent": "",
            "className": "form-control input flatpickr-input",
            "type": "text",
        }
        page = FakePage({"id=customer_cdr_from_date": FakeLocator(1, resolved)})
        result = await _call(
            page,
            candidate_locator="id=customer_cdr_from_date",
            expected_text=None,
            element_data={"tagName": "input", "id": "customer_cdr_from_date", "className": ""},
        )
        assert result["element_type"] == "date-picker"
        assert result["datepicker_framework"] == "flatpickr"
        assert result["element_info"]["className"] == "form-control input flatpickr-input"

    @pytest.mark.asyncio
    async def test_element_info_reports_resolved_source(self):
        resolved = {"id": "go", "tagName": "BUTTON", "textContent": "Go", "className": "btn"}
        page = FakePage({"id=go": FakeLocator(1, resolved)})
        result = await _call(
            page,
            candidate_locator="id=go",
            expected_text=None,
            element_data={"tagName": "button", "id": "go"},
        )
        assert result["element_info"]["source"] == "candidate_resolved_element"
        assert result["element_info"]["tagName"] == "BUTTON"
