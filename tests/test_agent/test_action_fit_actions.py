"""
E2 decision logic in agent/actions.py, no browser.

_settle_action_fit is the "never worse than today" rule; the candidate
tests drive find_unique_locator_action with the cascade and the fitness
read mocked (same conventions as test_actions_mocked.py).
"""

from unittest.mock import AsyncMock, patch

import pytest

from browser_service.agent.actions import _settle_action_fit, find_unique_locator_action

FOUND_STABLE = {"found": True, "best_locator": 'text="Blue"', "stability": "stable"}
FOUND_POSITIONAL = {"found": True, "best_locator": "xpath=//div[3]", "stability": "positional"}
NOT_FOUND = {
    "found": False,
    "error": "Action misfit: none of the 2 unique locators can perform 'input'",
}
CAND_VOLATILE = {
    "found": True,
    "best_locator": "id=react-select-3-option-2",
    "stability": "volatile",
}
CAND_MISFIT = {
    "found": True,
    "best_locator": "#twin",
    "stability": "stable",
    "action_fit": "misfit",
    "action_fit_reason": "<div> cannot be filled",
}


def _cascade(*answers):
    calls = []

    async def run(cascade_action):
        calls.append(cascade_action)
        return dict(answers[len(calls) - 1])

    return run, calls


class TestSettle:
    async def test_stable_cascade_beats_volatile_candidate(self):
        run, calls = _cascade()
        out = await _settle_action_fit(dict(FOUND_STABLE), dict(CAND_VOLATILE), run, "click")
        assert out["best_locator"] == 'text="Blue"'
        assert calls == []

    async def test_positional_cascade_loses_to_volatile_candidate(self):
        run, calls = _cascade()
        out = await _settle_action_fit(dict(FOUND_POSITIONAL), dict(CAND_VOLATILE), run, "click")
        assert out["best_locator"] == "id=react-select-3-option-2"

    async def test_any_found_cascade_beats_a_misfit_candidate(self):
        run, calls = _cascade()
        out = await _settle_action_fit(dict(FOUND_POSITIONAL), dict(CAND_MISFIT), run, "input")
        assert out["best_locator"] == "xpath=//div[3]"

    async def test_misfit_candidate_kept_when_the_cascade_finds_nothing(self):
        run, calls = _cascade()
        out = await _settle_action_fit(dict(NOT_FOUND), dict(CAND_MISFIT), run, "input")
        assert out["best_locator"] == "#twin"
        assert out["action_fit"] == "misfit"
        assert calls == []

    async def test_found_fit_result_passes_through(self):
        run, calls = _cascade()
        out = await _settle_action_fit(dict(FOUND_STABLE), None, run, "click")
        assert out == FOUND_STABLE
        assert calls == []

    async def test_fit_mode_not_found_reruns_todays_cascade(self):
        run, calls = _cascade({"found": True, "best_locator": "#twin", "stability": "stable"})
        out = await _settle_action_fit(dict(NOT_FOUND), None, run, "input")
        assert calls == [None]
        assert out["best_locator"] == "#twin"
        assert out["action_fit"] == "misfit"
        assert "none of the 2" in out["action_fit_reason"]

    async def test_todays_not_found_stays_not_found(self):
        run, calls = _cascade({"found": False, "error": "no element"})
        out = await _settle_action_fit(dict(NOT_FOUND), None, run, "input")
        assert out["found"] is False
        assert "action_fit" not in out


SEARCH_TEXT = "Search for Products, Brands and More"


class FakeLocator:
    def __init__(self, count: int):
        self._count = count

    async def count(self) -> int:
        return self._count

    def nth(self, i: int) -> "FakeLocator":
        return self

    async def bounding_box(self):
        return {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}

    async def evaluate(self, js, *args, **kwargs):
        return {
            "textContent": "",
            "textContentLength": 0,
            "innerText": "",
            "placeholder": "",
            "ariaLabel": SEARCH_TEXT,
            "value": "",
            "labelText": "",
            "labelledbyText": "",
            "tagName": "DIV",
            "isContentEditable": False,
        }


class FakePage:
    url = "https://www.flipkart.com/"

    def __init__(self, locators: dict):
        self._locators = locators

    def locator(self, selector: str) -> FakeLocator:
        return self._locators.get(selector, FakeLocator(0))

    def frame_locator(self, selector: str):
        return self

    async def evaluate(self, *args, **kwargs):
        return None


CANDIDATE = f'[aria-label="{SEARCH_TEXT}"]'


async def _call(action, cascade):
    page = FakePage({CANDIDATE: FakeLocator(1)})
    with patch("browser_service.locators.find_unique_locator_at_coordinates", new=cascade):
        return await find_unique_locator_action(
            x=100,
            y=100,
            element_id="elem_1",
            element_description="the search input box",
            expected_text=SEARCH_TEXT,
            candidate_locator=CANDIDATE,
            element_data=None,
            page=page,
            is_collection=False,
            action=action,
        )


class TestCandidatePath:
    async def test_flag_off_accepts_the_candidate_without_a_fitness_read(self):
        cascade = AsyncMock(return_value=dict(FOUND_STABLE))
        with patch("browser_service.agent.actions.check_action_fit", new=AsyncMock()) as fit:
            out = await _call(None, cascade)
        assert out["best_locator"] == CANDIDATE
        cascade.assert_not_awaited()
        fit.assert_not_awaited()

    async def test_misfit_candidate_loses_to_a_found_cascade_result(self):
        cascade = AsyncMock(return_value=dict(FOUND_STABLE))
        with patch(
            "browser_service.agent.actions.check_action_fit",
            new=AsyncMock(return_value="<div> cannot be filled"),
        ):
            out = await _call("input", cascade)
        assert out["best_locator"] == 'text="Blue"'
        assert cascade.await_args.kwargs["action"] == "input"

    async def test_misfit_candidate_kept_when_the_cascade_finds_nothing(self):
        cascade = AsyncMock(return_value=dict(NOT_FOUND))
        with patch(
            "browser_service.agent.actions.check_action_fit",
            new=AsyncMock(return_value="<div> cannot be filled"),
        ):
            out = await _call("input", cascade)
        assert out["best_locator"] == CANDIDATE
        assert out["action_fit"] == "misfit"
        assert cascade.await_count == 1  # the candidate IS today's answer — no legacy rerun
