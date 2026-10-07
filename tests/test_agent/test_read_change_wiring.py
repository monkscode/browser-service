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
LOCATORS = {"elem_1": "css=#a", "elem_2": "css=#r", "elem_3": "css=#n", "elem_4": "css=#p"}
INTERACTING = frozenset({"input", "type", "click", "submit", "select", "check", "uncheck"})


def _make_engine(weak_first=()):
    """A locator engine: found, validated; the FIRST answer for an id in
    ``weak_first`` is weak (``validated: False``) so the D3 downgrade guard lets
    a later strong re-query through to the F1 block."""
    seen: dict = {}

    async def _engine(**kwargs):
        eid = kwargs["element_id"]
        seen[eid] = seen.get(eid, 0) + 1
        weak = eid in weak_first and seen[eid] == 1
        return {
            "found": True,
            "best_locator": LOCATORS[eid],
            "validated": not weak,
            "element_id": eid,
        }

    return _engine


async def _run(
    elements,
    calls,
    *,
    changed="changed",
    snapshot_effect=None,
    changed_effect=None,
    sessions=None,
    params_for=None,
    weak_first=(),
    status_for=None,
    came_back=False,
    came_back_effect=None,
):
    """Register the action, call it once per id in ``calls``.

    Returns (results, snapshot_mock, interaction_mock, changed_mock, manager).
    ``manager`` records snapshot_html / _do_interaction calls in ONE order; its
    ``came_back_mock`` attribute is the patched ``read_came_back``.
    ``sessions`` maps an element id to the browser session used for that call.
    """
    specs = {e["id"]: e for e in elements}

    async def _interaction(browser_session, active_page, locator_str, element_id, *args, **kwargs):
        performed_actions = args[2]
        action = specs[element_id]["action"]
        status = (status_for or {}).get(element_id, "auto_ok")
        if status == "not_applicable":  # e.g. an input with an empty value: nothing performed
            return "", status, action, ""
        if action in INTERACTING:
            performed_actions.add(element_id)  # the real function does the same
        return "", status, action, ""

    snapshot = AsyncMock(return_value="<html>before</html>", side_effect=snapshot_effect)
    interaction = AsyncMock(side_effect=_interaction)
    read_changed = AsyncMock(return_value=changed, side_effect=changed_effect)
    read_came_back = AsyncMock(return_value=came_back, side_effect=came_back_effect)
    manager = MagicMock()
    manager.attach_mock(snapshot, "snapshot")
    manager.attach_mock(interaction, "interaction")
    manager.came_back_mock = read_came_back

    agent = FakeAgent()
    sessions = sessions or {}
    with (
        patch(
            "browser_service.agent.actions.find_unique_locator_action", new=_make_engine(weak_first)
        ),
        patch("browser_service.agent.registration._do_interaction", new=interaction),
        patch("browser_service.agent.registration.snapshot_html", new=snapshot),
        patch("browser_service.agent.registration.read_changed_since", new=read_changed),
        patch("browser_service.agent.registration.read_came_back", new=read_came_back),
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
    skipped = [r for r in caplog.records if "F1 read-change step skipped" in r.getMessage()]
    assert len(skipped) == 1
    assert "RuntimeError" in skipped[0].getMessage()  # the exception type is named


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
    assert "RuntimeError" in skipped[0].getMessage()  # the exception type is named


@pytest.mark.asyncio
async def test_a_requery_of_the_performed_action_does_not_snapshot_again():
    # elem_1 is found twice (a re-query after it was performed). Its FIRST answer
    # is weak, so the D3 downgrade guard does NOT short-circuit the second query:
    # it reaches the F1 block, where only F1's own `not in _performed_actions`
    # clause stops a second snapshot. _do_interaction's mock adds elem_1 to the
    # performed set, as the real function does. The snapshot mock returns a
    # DIFFERENT page per call, so a second snapshot would replace the kept one.
    results, snapshot, interaction, read_changed, _ = await _run(
        [CLICK, READ],
        ["elem_1", "elem_1", "elem_2"],
        snapshot_effect=["<html>1</html>", "<html>2</html>"],
        weak_first={"elem_1"},
    )

    assert interaction.await_count == 3  # both elem_1 queries and the read got through
    snapshot.assert_awaited_once()
    # the read compares against the page kept before the first (real) action
    read_changed.assert_awaited_once()
    assert read_changed.await_args.args[2] == "<html>1</html>"
    assert results[2].metadata["changed_by_action"] == "elem_1"


FILL = {"id": "elem_1", "action": "input", "value": "shoes"}
NEXT = {"id": "elem_3", "action": "click", "value": ""}
PENDING = {"id": "elem_4", "action": "click", "value": ""}


@pytest.mark.asyncio
async def test_a_requery_after_a_later_unsnapshotted_action_is_not_compared():
    # fill (1), read (2), click (3), click (4) still pending. The read is located after
    # the fill ("same"). The click (3) keeps no page (no read remains), so the saved page
    # must go too: the read located again after it compares with NOTHING.
    results, snapshot, _, read_changed, _ = await _run(
        [FILL, READ, NEXT, PENDING],
        ["elem_1", "elem_2", "elem_3", "elem_2"],
        changed_effect=["same", "changed"],
    )

    snapshot.assert_awaited_once()  # the fill's; nothing was awaited for elem_3
    read_changed.assert_awaited_once()  # the first location only
    assert "changed_by_action" not in results[1].metadata
    assert "changed_by_action" not in results[3].metadata


@pytest.mark.asyncio
async def test_a_failed_snapshot_clears_the_saved_page():
    # A snapshots; B's snapshot returns None (the page could not be read); the read after
    # B must not be compared with the page from before A.
    results, snapshot, _, read_changed, _ = await _run(
        [CLICK, NEXT, READ],
        ["elem_1", "elem_3", "elem_2"],
        snapshot_effect=["<html>pre-A</html>", None],
    )

    assert snapshot.await_count == 2
    read_changed.assert_not_awaited()
    assert "changed_by_action" not in results[2].metadata


@pytest.mark.asyncio
async def test_a_raising_snapshot_clears_the_saved_page(caplog):
    # A snapshots; B's snapshot RAISES (the F1 except path); the read after B must not be
    # compared with the page from before A.
    with caplog.at_level(logging.WARNING, logger="browser_service.agent.registration"):
        results, snapshot, _, read_changed, _ = await _run(
            [CLICK, NEXT, READ],
            ["elem_1", "elem_3", "elem_2"],
            snapshot_effect=["<html>pre-A</html>", RuntimeError("boom")],
        )

    assert snapshot.await_count == 2
    read_changed.assert_not_awaited()
    assert "changed_by_action" not in results[2].metadata
    assert len([r for r in caplog.records if "F1 read-change step skipped" in r.getMessage()]) == 1


EMPTY_INPUT = {"id": "elem_3", "action": "input", "value": ""}


@pytest.mark.asyncio
async def test_an_action_that_performs_nothing_keeps_the_earlier_saved_page():
    # click A (saves page-A); B is an input with an empty value, so _do_interaction
    # performs nothing ("not_applicable"); the read must still compare with page-A.
    results, snapshot, _, read_changed, _ = await _run(
        [CLICK, EMPTY_INPUT, READ],
        ["elem_1", "elem_3", "elem_2"],
        snapshot_effect=["<html>page-A</html>", "<html>page-B</html>"],
        status_for={"elem_3": "not_applicable"},
    )

    assert snapshot.await_count == 2  # B's page was captured, then not kept
    read_changed.assert_awaited_once()
    assert read_changed.await_args.args[2] == "<html>page-A</html>"
    assert results[2].metadata["changed_by_action"] == "elem_1"


@pytest.mark.asyncio
async def test_an_action_that_fails_automation_still_saves_its_page():
    # auto_failed: the agent performs the action natively, so B's page IS the page before it.
    results, _, _, read_changed, _ = await _run(
        [CLICK, EMPTY_INPUT, READ],
        ["elem_1", "elem_3", "elem_2"],
        snapshot_effect=["<html>page-A</html>", "<html>page-B</html>"],
        status_for={"elem_3": "auto_failed"},
    )

    read_changed.assert_awaited_once()
    assert read_changed.await_args.args[2] == "<html>page-B</html>"
    assert results[2].metadata["changed_by_action"] == "elem_3"


# --- came_back (R16): every located read is checked against ALL the pages kept so far ----------

SIGNAL = "(signal: read-came-back)"


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [True, False])
async def test_a_read_is_checked_against_the_kept_page(answer, caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        results, _, _, _, manager = await _run(
            [CLICK, READ], ["elem_1", "elem_2"], came_back=answer
        )

    manager.came_back_mock.assert_awaited_once()
    args = manager.came_back_mock.await_args.args
    assert args[1] == "css=#r"  # the READ's own locator
    assert args[2] == ["<html>before</html>"]
    signals = [r for r in caplog.records if SIGNAL in r.getMessage()]
    if answer:
        assert results[1].metadata["came_back"] is True
        assert len(signals) == 1 and "elem_2" in signals[0].getMessage()
    else:
        assert "came_back" not in results[1].metadata
        assert signals == []
    assert "came_back" not in results[0].metadata  # an action is never checked


@pytest.mark.asyncio
async def test_two_actions_then_a_read_hand_over_both_pages_in_order():
    results, _, _, _, manager = await _run(
        [CLICK, NEXT, READ],
        ["elem_1", "elem_3", "elem_2"],
        snapshot_effect=["<html>A</html>", "<html>B</html>"],
    )

    manager.came_back_mock.assert_awaited_once()
    assert manager.came_back_mock.await_args.args[2] == ["<html>A</html>", "<html>B</html>"]


@pytest.mark.asyncio
async def test_a_read_before_any_action_is_not_checked_for_came_back():
    first_read = {"id": "elem_1", "action": "get_text", "value": ""}
    later_click = {"id": "elem_2", "action": "click", "value": ""}
    results, _, _, _, manager = await _run([first_read, later_click], ["elem_1"], came_back=True)

    manager.came_back_mock.assert_not_awaited()
    assert "came_back" not in results[0].metadata


@pytest.mark.asyncio
async def test_an_action_that_performs_nothing_keeps_no_page_for_came_back():
    results, snapshot, _, _, manager = await _run(
        [CLICK, EMPTY_INPUT, READ],
        ["elem_1", "elem_3", "elem_2"],
        snapshot_effect=["<html>page-A</html>", "<html>page-B</html>"],
        status_for={"elem_3": "not_applicable"},
    )

    assert snapshot.await_count == 2  # B's page was captured, then not kept
    assert manager.came_back_mock.await_args.args[2] == ["<html>page-A</html>"]


@pytest.mark.asyncio
async def test_a_snapshot_that_returned_none_is_a_gap_in_the_list():
    _, _, _, _, manager = await _run(
        [CLICK, NEXT, READ],
        ["elem_1", "elem_3", "elem_2"],
        snapshot_effect=["<html>pre-A</html>", None],
    )

    assert manager.came_back_mock.await_args.args[2] == ["<html>pre-A</html>", None]


@pytest.mark.asyncio
async def test_a_raising_snapshot_is_a_gap_when_the_action_is_still_performed():
    _, _, interaction, _, manager = await _run(
        [CLICK, NEXT, READ],
        ["elem_1", "elem_3", "elem_2"],
        snapshot_effect=["<html>pre-A</html>", RuntimeError("boom")],
    )

    assert interaction.await_count == 3  # the second action still ran
    assert manager.came_back_mock.await_args.args[2] == ["<html>pre-A</html>", None]


@pytest.mark.asyncio
async def test_an_action_after_the_last_read_is_a_gap_for_a_re_queried_read():
    # fill (page kept), read, click (no read remains: no page), click pending, read again.
    _, _, _, _, manager = await _run(
        [FILL, READ, NEXT, PENDING],
        ["elem_1", "elem_2", "elem_3", "elem_2"],
        snapshot_effect=["<html>fill</html>"],
        weak_first={"elem_2"},
    )

    first, second = manager.came_back_mock.await_args_list
    assert first.args[2] == ["<html>fill</html>"]
    assert second.args[2] == ["<html>fill</html>", None]


@pytest.mark.asyncio
async def test_a_requery_of_the_performed_action_keeps_no_second_page():
    _, _, _, _, manager = await _run(
        [CLICK, READ],
        ["elem_1", "elem_1", "elem_2"],
        snapshot_effect=["<html>1</html>", "<html>2</html>"],
        weak_first={"elem_1"},
    )

    assert manager.came_back_mock.await_args.args[2] == ["<html>1</html>"]


@pytest.mark.asyncio
async def test_a_failing_came_back_check_is_flagged_with_one_warning(caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        results, _, _, _, manager = await _run(
            [CLICK, READ], ["elem_1", "elem_2"], came_back_effect=RuntimeError("boom")
        )

    manager.came_back_mock.assert_awaited_once()
    assert results[1].metadata["came_back"] is True  # fail-safe
    assert results[1].metadata["best_locator"] == "css=#r"  # the result is still returned
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "F1" in r.getMessage()  # not the fake page's noise
    ]
    assert len(warnings) == 1
    assert "came-back check skipped" in warnings[0].getMessage()
    assert "RuntimeError" in warnings[0].getMessage() and "boom" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_the_check_does_not_change_the_changed_by_action_mark():
    results, _, _, read_changed, _ = await _run([CLICK, READ], ["elem_1", "elem_2"], came_back=True)

    read_changed.assert_awaited_once()
    assert results[1].metadata["changed_by_action"] == "elem_1"
    assert results[1].metadata["came_back"] is True


@pytest.mark.asyncio
async def test_a_read_flagged_once_stays_flagged_when_it_is_located_again():
    # the second location REPLACES the first result in the workflow's results, so the flag
    # must persist in the per-workflow state even when the second computation says False.
    results, _, _, _, _ = await _run(
        [CLICK, READ],
        ["elem_1", "elem_2", "elem_2"],
        came_back_effect=[True, False],
        weak_first={"elem_2"},
    )

    assert results[1].metadata["came_back"] is True
    assert results[2].metadata["came_back"] is True


@pytest.mark.asyncio
async def test_an_unflagged_read_is_checked_again_when_it_is_located_again():
    results, _, _, _, manager = await _run(
        [CLICK, READ],
        ["elem_1", "elem_2", "elem_2"],
        came_back_effect=[False, True],
        weak_first={"elem_2"},
    )

    assert manager.came_back_mock.await_count == 2
    assert "came_back" not in results[1].metadata
    assert results[2].metadata["came_back"] is True


@pytest.mark.asyncio
async def test_a_read_inside_an_iframe_is_flagged_without_the_check():
    results, _, _, _, manager = await _run(
        [CLICK, READ],
        ["elem_1", "elem_2"],
        sessions={"elem_2": FrameBrowserSession({7: APP_FRAME, 12: CHECKBOX})},
        params_for={"elem_2": {"x": 70, "y": 230, "element_index": 12}},
    )

    manager.came_back_mock.assert_not_awaited()
    assert results[1].metadata["came_back"] is True
