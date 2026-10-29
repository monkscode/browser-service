"""bs #72: a call refused for lack of evidence gets ONE retry per element, then a not-found.

The find_unique_locator action refuses a call with no element data and no expected_text
(``evidence: "none"``). The agent may repeat the same blind call forever, so the registration
counts refusals per element_id: the first gets today's retry message, every later one ends
as NOT FOUND (no screenshot, a "do not retry" order). Other elements keep their own count,
and a call that carries evidence is processed as before at any time.
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from browser_service.agent.registration import register_custom_actions
from tests.test_agent.test_vision_escalation import (
    FakeAgent,
    FakeBrowserSession,
    _fake_playwright,
)

ELEMENTS = [
    {"id": "elem_1", "action": "get_text", "value": ""},
    {"id": "elem_2", "action": "get_text", "value": ""},
]
ERROR = (
    "No element evidence: no element_index matched an element and no expected_text was given, "
    "so nothing was checked. Send the element's index, or its visible text as expected_text."
)
PARAMS = {"x": 500, "y": 300, "element_description": "the status message"}
FOUND = {"found": True, "best_locator": "css=#status", "validated": True}


def _refusal(element_id):
    return {
        "element_id": element_id,
        "description": "the status message",
        "found": False,
        "error": ERROR,
        "error_type": "NoElementEvidence",
        "evidence": "none",
        "coordinates": {"x": 500, "y": 300},
    }


def _final(element_id):
    return (
        f"'{element_id}' named no element again (no element_index matched and no expected_text "
        f"was given); it is recorded as NOT FOUND. Do not retry it — continue with the "
        f"remaining elements."
    )


async def _run(calls, element_index=None):
    """Execute one find_unique_locator per (element_id, engine_result) in ``calls``, in order."""
    results = iter(result for _, result in calls)
    engine = AsyncMock(side_effect=lambda **kwargs: next(results))
    agent = FakeAgent()
    params = dict(PARAMS)
    if element_index is not None:
        params["element_index"] = element_index
    with (
        patch("browser_service.agent.actions.find_unique_locator_action", new=engine),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
    ):
        assert register_custom_actions(agent, elements=ELEMENTS) is True
        return [
            await agent.tools.registry.execute_action(
                "find_unique_locator",
                {**params, "element_id": eid},
                browser_session=FakeBrowserSession(),
            )
            for eid, _ in calls
        ]


@pytest.mark.asyncio
async def test_first_refusal_without_an_index_asks_for_the_index_with_a_screenshot():
    (first,) = await _run([("elem_1", _refusal("elem_1"))])

    assert first.error.startswith("Fallback strategies failed for 'elem_1'.")
    assert ERROR in first.error  # the refusal's own words reach the agent
    assert first.metadata.get("include_screenshot") is True
    assert first.is_done is not True


@pytest.mark.asyncio
async def test_first_refusal_with_an_index_keeps_todays_failure_message():
    (first,) = await _run([("elem_1", _refusal("elem_1"))], element_index=7)

    assert first.error.startswith("FAILED: Could not find unique locator for elem_1.")
    assert first.metadata.get("include_screenshot") is True
    assert first.is_done is False


@pytest.mark.asyncio
async def test_second_refusal_for_the_same_element_is_not_found(caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        _, second = await _run([("elem_1", _refusal("elem_1")), ("elem_1", _refusal("elem_1"))])

    assert second.error == _final("elem_1")
    assert second.extracted_content == _final("elem_1")
    assert second.is_done is False
    assert not second.metadata.get("include_screenshot")
    assert second.metadata["evidence"] == "none"
    assert (
        "⛔ NO EVIDENCE NOT FOUND for elem_1 after 2 attempts (signal: no-evidence-not-found)"
        in caplog.messages
    )


@pytest.mark.asyncio
async def test_the_count_is_per_element():
    _, _, other = await _run(
        [
            ("elem_1", _refusal("elem_1")),
            ("elem_1", _refusal("elem_1")),
            ("elem_2", _refusal("elem_2")),
        ]
    )

    assert other.error.startswith("Fallback strategies failed for 'elem_2'.")
    assert other.metadata.get("include_screenshot") is True


@pytest.mark.asyncio
async def test_a_call_with_evidence_after_the_cap_is_processed_as_found():
    *_, found = await _run(
        [
            ("elem_1", _refusal("elem_1")),
            ("elem_1", _refusal("elem_1")),
            ("elem_1", {**FOUND, "element_id": "elem_1"}),
        ]
    )

    assert found.error is None
    assert "css=#status" in found.extracted_content
