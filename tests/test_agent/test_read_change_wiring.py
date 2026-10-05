"""F1: the find action keeps the page before each action and marks changed reads.

Defect: "click Laptops, get the first laptop name" reads the OLD card, because the
click refreshes the grid in the background. bs keeps the page HTML right before
each action it performs (only while a read is still to come); when a later READ
element is located, that read's own locator is compared with the kept HTML
(``read_change.read_changed_since``) and, when the answer is "changed", the read's
result dict (the ActionResult ``metadata``) carries
``changed_by_action = "<element id of the action>"``.

No browser: the locator engine, Playwright, ``_do_interaction``, ``snapshot_html``
and ``read_changed_since`` are replaced at their seams. The tests read the order
of calls, and the metadata the closure returns.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_service.agent.registration import register_custom_actions
from tests.test_agent.test_interaction_frame_wiring import APP_FRAME, CHECKBOX
from tests.test_agent.test_interaction_frame_wiring import (
    FakeBrowserSession as FrameBrowserSession,
)
from tests.test_agent.test_vision_escalation import FakeAgent, FakeBrowserSession, _fake_playwright

CLICK = {"id": "elem_1", "action": "click", "value": ""}
READ = {"id": "elem_2", "action": "get_text", "value": ""}
PARAMS = {"x": 500, "y": 300, "element_description": "the thing", "element_index": 7}
LOCATORS = {"elem_1": "css=#a", "elem_2": "css=#r"}
INTERACTING = frozenset({"input", "type", "click", "submit", "select", "check", "uncheck"})


async def _engine(**kwargs):
    eid = kwargs["element_id"]
    return {
        "found": True,
        "best_locator": LOCATORS[eid],
        "validated": True,
        "element_id": eid,
    }


async def _run(
    elements,
    calls,
    *,
    changed="changed",
    snapshot_effect=None,
    changed_effect=None,
    sessions=None,
    params_for=None,
):
    """Register the action, call it once per id in ``calls``.

    Returns (results, snapshot_mock, interaction_mock, changed_mock, manager).
    ``manager`` records snapshot_html / _do_interaction calls in ONE order.
    ``sessions`` maps an element id to the browser session used for that call.
    """
    specs = {e["id"]: e for e in elements}

    async def _interaction(browser_session, active_page, locator_str, element_id, *args, **kwargs):
        performed_actions = args[2]
        action = specs[element_id]["action"]
        if action in INTERACTING:
            performed_actions.add(element_id)  # the real function does the same
        return "", "auto_ok", action, ""

    snapshot = AsyncMock(return_value="<html>before</html>", side_effect=snapshot_effect)
    interaction = AsyncMock(side_effect=_interaction)
    read_changed = AsyncMock(return_value=changed, side_effect=changed_effect)
    manager = MagicMock()
    manager.attach_mock(snapshot, "snapshot")
    manager.attach_mock(interaction, "interaction")

    agent = FakeAgent()
    sessions = sessions or {}
    with (
        patch("browser_service.agent.actions.find_unique_locator_action", new=_engine),
        patch("browser_service.agent.registration._do_interaction", new=interaction),
        patch("browser_service.agent.registration.snapshot_html", new=snapshot),
        patch("browser_service.agent.registration.read_changed_since", new=read_changed),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
    ):
        assert register_custom_actions(agent, elements=elements) is True
        results = [
            await agent.tools.registry.execute_action(
                "find_unique_locator",
                {**PARAMS, **(params_for or {}).get(eid, {}), "element_id": eid},
                browser_session=sessions.get(eid) or FakeBrowserSession(),
            )
            for eid in calls
        ]
    return results, snapshot, interaction, read_changed, manager


@pytest.mark.asyncio
async def test_snapshot_is_taken_before_the_action_and_the_read_is_marked():
    results, snapshot, interaction, read_changed, manager = await _run(
        [CLICK, READ], ["elem_1", "elem_2"]
    )

    snapshot.assert_awaited_once()
    interaction.assert_awaited()
    names = [c[0] for c in manager.mock_calls if c[0] in ("snapshot", "interaction")]
    assert names[:2] == ["snapshot", "interaction"], names  # snapshot BEFORE elem_1's action
    read_changed.assert_awaited_once()
    assert read_changed.await_args.args[1] == "css=#r"  # the READ's own locator
    assert read_changed.await_args.args[2] == "<html>before</html>"
    assert results[1].metadata["changed_by_action"] == "elem_1"
    assert "changed_by_action" not in results[0].metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["same", "absent", "ambiguous", "unknown"])
async def test_no_mark_unless_the_status_is_changed(status):
    results, _, _, read_changed, _ = await _run([CLICK, READ], ["elem_1", "elem_2"], changed=status)

    read_changed.assert_awaited_once()
    assert "changed_by_action" not in results[1].metadata


@pytest.mark.asyncio
async def test_no_snapshot_when_no_read_remains():
    results, snapshot, interaction, read_changed, _ = await _run([CLICK], ["elem_1"])

    snapshot.assert_not_awaited()
    interaction.assert_awaited_once()
    read_changed.assert_not_awaited()
    assert "changed_by_action" not in results[0].metadata


@pytest.mark.asyncio
async def test_a_read_inside_an_iframe_is_never_compared():
    results, snapshot, interaction, read_changed, _ = await _run(
        [CLICK, READ],
        ["elem_1", "elem_2"],
        sessions={"elem_2": FrameBrowserSession({7: APP_FRAME, 12: CHECKBOX})},
        # the point and index of test_interaction_frame_wiring: inside APP_FRAME
        params_for={"elem_2": {"x": 70, "y": 230, "element_index": 12}},
    )

    assert interaction.await_args_list[-1].kwargs["iframe_context"]  # the frame was detected
    snapshot.assert_awaited_once()  # the click still snapshotted
    read_changed.assert_not_awaited()
    assert "changed_by_action" not in results[1].metadata


@pytest.mark.asyncio
async def test_a_read_before_any_action_is_never_compared():
    first_read = {"id": "elem_1", "action": "get_text", "value": ""}
    later_click = {"id": "elem_2", "action": "click", "value": ""}
    results, snapshot, _, read_changed, _ = await _run([first_read, later_click], ["elem_1"])

    read_changed.assert_not_awaited()
    snapshot.assert_not_awaited()
    assert "changed_by_action" not in results[0].metadata


@pytest.mark.asyncio
async def test_a_failing_snapshot_does_not_stop_the_action(caplog):
    with caplog.at_level(logging.WARNING, logger="browser_service.agent.registration"):
        results, snapshot, interaction, read_changed, _ = await _run(
            [CLICK, READ], ["elem_1", "elem_2"], snapshot_effect=RuntimeError("boom")
        )

    snapshot.assert_awaited_once()
    interaction.assert_awaited()  # the click still ran
    read_changed.assert_not_awaited()  # nothing was kept to compare with
    assert all("changed_by_action" not in r.metadata for r in results)
    assert len([r for r in caplog.records if "F1 read-change step skipped" in r.getMessage()]) == 1


@pytest.mark.asyncio
async def test_a_failing_comparison_returns_the_read_normally(caplog):
    with caplog.at_level(logging.WARNING, logger="browser_service.agent.registration"):
        results, _, _, read_changed, _ = await _run(
            [CLICK, READ], ["elem_1", "elem_2"], changed_effect=RuntimeError("boom")
        )

    read_changed.assert_awaited_once()
    assert results[1].metadata is not None
    assert results[1].metadata["best_locator"] == "css=#r"
    assert "changed_by_action" not in results[1].metadata
    skipped = [r for r in caplog.records if "F1 read-change step skipped" in r.getMessage()]
    assert len(skipped) == 1
    assert skipped[0].levelno == logging.WARNING
    assert skipped[0].name == "browser_service.agent.registration"


@pytest.mark.asyncio
async def test_a_requery_of_the_performed_action_does_not_snapshot_again():
    # elem_1 is found twice (a re-query after it was performed). _do_interaction's
    # mock adds elem_1 to the performed set, as the real function does, so the
    # second call sees it as performed.
    results, snapshot, interaction, read_changed, _ = await _run(
        [CLICK, READ], ["elem_1", "elem_1", "elem_2"]
    )

    snapshot.assert_awaited_once()
    assert interaction.await_count >= 1
    # the kept snapshot is still the one taken before the first (real) action
    assert read_changed.await_args.args[2] == "<html>before</html>"
    assert results[2].metadata["changed_by_action"] == "elem_1"
