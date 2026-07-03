"""LOCATOR_TIMER instrumentation around find_unique_locator_at_coordinates.

Task-2 bench harness needs per-element locator latency. actions.py must:
1. Always log `LOCATOR_TIMER element_id=... duration_ms=... found=...` around the
   smart-locator call — including the asyncio.TimeoutError path.
2. Inject duration_ms into result['approach_metrics'] ONLY when that dict already
   exists (smart_locator builds it on success; failure results may lack it).
"""

import asyncio
import logging
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_service.agent.actions import find_unique_locator_action

TIMER_RE = re.compile(
    r"LOCATOR_TIMER element_id=(?P<element_id>\S+) "
    r"duration_ms=(?P<duration_ms>[\d.]+) "
    r"found=(?P<found>True|False)"
)


def _timer_records(caplog):
    return [m for m in (TIMER_RE.search(r.getMessage()) for r in caplog.records) if m]


def _page():
    page = MagicMock()
    page.url = "https://example.com"
    return page


@pytest.mark.asyncio
async def test_timer_logged_and_duration_injected_on_success(caplog):
    found_result = {
        "found": True,
        "element_id": "elem_1",
        "description": "login button",
        "best_locator": "#login",
        "validation_summary": {},
        "approach_metrics": {"locator_approach": "element_data", "success": True},
    }
    with patch(
        "browser_service.locators.find_unique_locator_at_coordinates",
        new=AsyncMock(return_value=found_result),
    ), caplog.at_level(logging.INFO, logger="browser_service.agent.actions"):
        result = await find_unique_locator_action(
            x=10, y=20, element_id="elem_1",
            element_description="login button", page=_page(),
        )

    timers = _timer_records(caplog)
    assert len(timers) == 1, "exactly one LOCATOR_TIMER line per element"
    assert timers[0]["element_id"] == "elem_1"
    assert timers[0]["found"] == "True"
    assert float(timers[0]["duration_ms"]) >= 0.0

    assert "duration_ms" in result["approach_metrics"]
    assert result["approach_metrics"]["duration_ms"] >= 0.0


@pytest.mark.asyncio
async def test_timer_logged_but_not_injected_when_approach_metrics_absent(caplog):
    not_found = {
        "found": False,
        "element_id": "elem_2",
        "description": "missing thing",
        "error": "No unique locator found",
        "validation_summary": {},
    }
    with patch(
        "browser_service.locators.find_unique_locator_at_coordinates",
        new=AsyncMock(return_value=not_found),
    ), caplog.at_level(logging.INFO, logger="browser_service.agent.actions"):
        result = await find_unique_locator_action(
            x=10, y=20, element_id="elem_2",
            element_description="missing thing", page=_page(),
        )

    timers = _timer_records(caplog)
    assert len(timers) == 1
    assert timers[0]["element_id"] == "elem_2"
    assert timers[0]["found"] == "False"

    assert "approach_metrics" not in result, (
        "duration must be injected only into an EXISTING approach_metrics dict"
    )


@pytest.mark.asyncio
async def test_timer_logged_on_timeout_path(caplog):
    with patch(
        "browser_service.locators.find_unique_locator_at_coordinates",
        new=AsyncMock(side_effect=asyncio.TimeoutError),
    ), caplog.at_level(logging.INFO, logger="browser_service.agent.actions"):
        result = await find_unique_locator_action(
            x=10, y=20, element_id="elem_3",
            element_description="slow page", page=_page(),
        )

    timers = _timer_records(caplog)
    assert len(timers) == 1, "TimeoutError path must still emit LOCATOR_TIMER"
    assert timers[0]["element_id"] == "elem_3"
    assert timers[0]["found"] == "False"

    assert result["found"] is False
    assert result["error_type"] == "TimeoutError"
