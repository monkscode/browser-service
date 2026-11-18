"""bs #91: an index-less point that found nothing gets the 0-1000 grid unit named in the retry reply.

The agent sometimes sends page-pixel points; bs reads every index-less point as a 0-1000 grid point,
so the point lands on empty page. The first failure reply says what happened to the point and how to
convert, only when the point is plausibly pixels (inside the viewport, not equal to its scaled self)
and no DOM box confirmed it.
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from browser_service.agent.registration import register_custom_actions
from tests.test_agent.test_indexed_read_screen_point import FakeNode
from tests.test_agent.test_no_evidence_retry_cap import ELEMENTS, ERROR, _final, _refusal
from tests.test_agent.test_vision_escalation import (
    FakeAgent,
    FakeBrowserSession,
    _fake_playwright,
)

# d09's point: grid (815, 294.28) on a 1920x1080 page -> page pixel (1564, 317).
D09 = {"x": 815, "y": 294.28, "element_description": "the status message"}
BODY_LANDING = {
    "element_id": "elem_1",
    "description": "the status message",
    "found": False,
    "error": "Semantic approach failed and coordinates (1564, 317) landed on BODY (empty space)",
    "landed_on": "body",
    "coordinates": {"x": 1564, "y": 317},
}
NO_UNIQUE = {
    "element_id": "elem_1",
    "description": "the status message",
    "found": False,
    "error": "Could not find unique locator",
}
HINT_LINE = "(signal: coordinate-unit-hint)"


class SelectorMapSession(FakeBrowserSession):
    """A session whose selector map holds the given nodes."""

    def __init__(self, selector_map):
        self._map = selector_map

    async def get_selector_map(self):
        return self._map


async def _run(calls, params=None, session=None):
    """Execute one find_unique_locator per (element_id, engine_result) in ``calls``, in order."""
    results = iter(result for _, result in calls)
    engine = AsyncMock(side_effect=lambda **kwargs: next(results))
    agent = FakeAgent()
    with (
        patch("browser_service.agent.actions.find_unique_locator_action", new=engine),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
    ):
        assert register_custom_actions(agent, elements=ELEMENTS) is True
        return [
            await agent.tools.registry.execute_action(
                "find_unique_locator",
                {**(params or D09), "element_id": eid},
                browser_session=session or FakeBrowserSession(),
            )
            for eid, _ in calls
        ]


@pytest.mark.asyncio
async def test_first_no_evidence_refusal_names_the_unit(caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        (first,) = await _run([("elem_1", _refusal("elem_1"))])

    assert first.error.startswith("Fallback strategies failed for 'elem_1'.")
    assert ERROR in first.error
    assert "x and y are read on a 0-1000 grid of the screenshot" in first.error
    # nothing was checked at the point, so the hint must not claim a miss there
    assert (
        "Your (815, 294.28) became page pixel (1564, 317) on this 1920x1080 page. If" in first.error
    )
    assert "nothing was found" not in first.error
    assert (
        "(your point would be x=424.5, y=272.5), with the element's visible text as "
        "expected_text if it has any." in first.error
    )
    assert any(m.endswith(f"(1564, 317) not checked {HINT_LINE}") for m in caplog.messages)


@pytest.mark.asyncio
async def test_body_landing_names_the_unit(caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        (first,) = await _run([("elem_1", BODY_LANDING)])

    assert first.error.startswith("Fallback strategies failed for 'elem_1'.")
    assert "x and y are read on a 0-1000 grid of the screenshot" in first.error
    assert (
        "Your (815, 294.28) became page pixel (1564, 317) on this 1920x1080 page, "
        "where it landed on BODY (empty page). If" in first.error
    )
    assert "(your point would be x=424.5, y=272.5)" in first.error
    assert any(m.endswith(f"(1564, 317) landed on body {HINT_LINE}") for m in caplog.messages)


@pytest.mark.asyncio
async def test_indexed_refusal_has_no_hint():
    (first,) = await _run([("elem_1", _refusal("elem_1"))], params={**D09, "element_index": 7})

    assert first.error.startswith("FAILED: Could not find unique locator for elem_1.")
    assert "0-1000 grid" not in first.error


@pytest.mark.asyncio
async def test_other_index_less_failure_has_no_hint(caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        (first,) = await _run([("elem_1", NO_UNIQUE)])

    assert first.error.startswith("Fallback strategies failed for 'elem_1'.")
    assert "0-1000 grid" not in first.error
    assert not any(HINT_LINE in m for m in caplog.messages)


@pytest.mark.asyncio
async def test_second_refusal_is_the_existing_not_found():
    _, second = await _run([("elem_1", _refusal("elem_1")), ("elem_1", _refusal("elem_1"))])

    assert second.error == _final("elem_1")


@pytest.mark.asyncio
async def test_point_outside_the_viewport_read_as_pixels_has_no_hint():
    (first,) = await _run([("elem_1", _refusal("elem_1"))], params={**D09, "x": 1950, "y": 300})

    assert "0-1000 grid" not in first.error


@pytest.mark.asyncio
async def test_dom_confirmed_point_has_no_hint():
    node = FakeNode(x=1500, y=300, width=140, height=50)  # holds the scaled point (1564, 317)
    (first,) = await _run([("elem_1", BODY_LANDING)], session=SelectorMapSession({3: node}))

    assert first.error.startswith("Fallback strategies failed for 'elem_1'.")
    assert "0-1000 grid" not in first.error
