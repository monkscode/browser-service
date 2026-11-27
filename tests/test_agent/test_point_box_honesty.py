"""Point-then-prove D1: an index-less box from the selector map is evidence only when it is the hit.

registration's STEP A takes the smallest INDEXED box around the agent's point. On TodoMVC that
box was the app root (section#root), and it shipped as the checkbox's locator. For a performed
step (IDENTITY_ACTIONS) the box must be the element under the agent's scaled point, or the
nearest control at or above it; otherwise the call goes on with that point and no element data.
A box in another frame than the point is dropped unread. Reads, waits, select steps, indexed
calls and a check that fails keep today's data.

Registration runs through the real Tools registry; the locator engine, Playwright and the three
DOM reads (_hit_at, pointed_target for the box, box_is_hit) are mocked.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from browser_service.agent.registration import register_custom_actions
from browser_service.config import config
from tests.test_agent.test_indexed_read_screen_point import FakeBrowserSession, FakeNode
from tests.test_agent.test_vision_escalation import FakeAgent, _fake_playwright

NOT_FOUND = {"element_id": "elem_1", "found": False, "error": "Could not find unique locator"}
SCALED = (960, 540)  # the raw (500, 500) on the 1920x1080 grid
BOX_CENTRE = (940, 390)
DROPPED = "(signal: point-box-dropped)"
UNJUDGED = "(signal: point-box-kept-unjudged)"
FRAME = 'iframe[id="f"]'
HIT, BOX = object(), object()


def _root():
    node = FakeNode(x=665, y=40, width=550, height=700)  # holds SCALED; centre BOX_CENTRE
    node.node_name = "SECTION"
    return node


def _frame(x, width):
    return SimpleNamespace(
        node_name="IFRAME",
        attributes={"id": "f"},
        absolute_position=SimpleNamespace(x=x, y=0, width=width, height=1080),
    )


async def _call(action, params=None, *, hit=HIT, box=BOX, reads=None, flag=True, nodes=None):
    """One find_unique_locator call; returns the engine's kwargs and the DOM-read mocks."""
    engine = AsyncMock(return_value=dict(NOT_FOUND))
    reads = reads or {}
    mocks = SimpleNamespace(
        hit_at=reads.get("hit_at") or AsyncMock(return_value=hit),
        target=reads.get("target") or AsyncMock(return_value=box),
        same=reads.get("same") or AsyncMock(side_effect=lambda b, h: b is h),
    )
    agent = FakeAgent()
    with (
        patch("browser_service.agent.actions.find_unique_locator_action", new=engine),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
        patch.object(config.locator, "enable_action_fit", flag),
        patch("browser_service.agent.registration._hit_at", new=mocks.hit_at),
        patch("browser_service.agent.registration.pointed_target", new=mocks.target),
        patch("browser_service.agent.registration.box_is_hit", new=mocks.same),
    ):
        assert register_custom_actions(agent, elements=[{"id": "elem_1", "action": action}])
        await agent.tools.registry.execute_action(
            "find_unique_locator",
            {
                "element_id": "elem_1",
                "element_description": "the toggle of Buy milk",
                "x": 500,
                "y": 500,
                **(params or {}),
            },
            browser_session=FakeBrowserSession(nodes or {3: _root()}),
        )
    return engine.await_args.kwargs, mocks


def _kept(kwargs):
    assert kwargs["element_data"]["tagName"] == "section"
    assert (kwargs["x"], kwargs["y"]) == BOX_CENTRE


def _unread(mocks):
    mocks.hit_at.assert_not_awaited()
    mocks.target.assert_not_awaited()


async def test_index_less_click_box_that_is_not_the_hit_is_dropped(caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        kwargs, mocks = await _call("click")

    assert kwargs["element_data"] is None
    assert (kwargs["x"], kwargs["y"]) == SCALED  # the agent's point, not the box centre
    assert kwargs["iframe_context"] is None
    assert kwargs["pointed_by_index"] is False
    # hit-tested at the scaled raw point; the box resolved by its xpath alone
    assert mocks.hit_at.await_args.args[1:] == (None, SCALED)
    assert mocks.target.await_args.args[1] is None
    assert mocks.target.await_args.args[2]["tagName"] == "section"
    assert mocks.target.await_args.args[3:] == (True, None)
    assert [m for m in caplog.messages if m.endswith(DROPPED)] == [
        "   🎯 POINT-BOX DROPPED: the smallest indexed box <section> is not the element under "
        f"the point {SCALED} — the call goes on from the point {DROPPED}"
    ]


async def test_box_that_is_the_hit_is_kept(caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        kwargs, _ = await _call("click", box=HIT)

    _kept(kwargs)
    assert not [m for m in caplog.messages if DROPPED in m]


async def test_no_hit_drops_the_box():
    """Nothing, body, html or an iframe under the point (no error): the box is not the hit."""
    kwargs, mocks = await _call("input", hit=None)

    assert kwargs["element_data"] is None
    assert (kwargs["x"], kwargs["y"]) == SCALED
    mocks.target.assert_not_awaited()


async def test_unresolvable_box_keeps_the_data(caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        kwargs, _ = await _call("check", box=None)

    _kept(kwargs)
    assert not [m for m in caplog.messages if DROPPED in m]
    assert len([m for m in caplog.messages if m.endswith(UNJUDGED)]) == 1


async def test_indexed_call_is_never_checked():
    kwargs, mocks = await _call("click", {"element_index": 3})

    _unread(mocks)
    _kept(kwargs)
    assert kwargs["pointed_by_index"] is True


async def test_stale_index_is_treated_as_no_index(caplog):
    """An element_index the selector map lacks found no node: STEP A's box is checked like any
    index-less box, and the call is not 'by index'."""
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        kwargs, mocks = await _call("click", {"element_index": 7})  # the map holds only [3]

    assert kwargs["element_data"] is None
    assert (kwargs["x"], kwargs["y"]) == SCALED
    assert kwargs["pointed_by_index"] is False
    assert mocks.hit_at.await_args.args[1:] == (None, SCALED)
    assert len([m for m in caplog.messages if m.endswith(DROPPED)]) == 1


async def test_box_kept_for_read_steps():
    kwargs, mocks = await _call("get_text")

    _unread(mocks)
    _kept(kwargs)


@pytest.mark.parametrize(("action", "flag"), [("select", True), ("click", False)])
async def test_box_kept_for_select_steps_and_with_the_flag_off(action, flag):
    kwargs, mocks = await _call(action, flag=flag)

    _unread(mocks)
    _kept(kwargs)


@pytest.mark.parametrize("failing", ["hit_at", "target", "same"])
async def test_check_exception_keeps_data(caplog, failing):
    """A read that raises judges nothing: the hit-test, the box lookup or the comparison."""
    reads = {failing: AsyncMock(side_effect=RuntimeError("frame detached"))}
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        kwargs, _ = await _call("click", reads=reads)

    _kept(kwargs)
    assert not [m for m in caplog.messages if DROPPED in m]
    assert [m for m in caplog.messages if m.endswith(UNJUDGED)] == [
        f"   ⚠️ Point-box check failed, the box is kept: frame detached {UNJUDGED}"
    ]


@pytest.mark.parametrize(
    ("frame", "box_frame", "point_frame", "tag"),
    [
        # box centre (950, 550) left of the frame; SCALED inside it
        pytest.param(_frame(955, 965), "the main document", FRAME, "p", id="point-in-frame"),
        # box centre inside a narrow frame; SCALED right of it (the frame lookup re-picks the box)
        pytest.param(_frame(940, 15), FRAME, "the main document", "p", id="box-in-frame"),
    ],
)
async def test_a_box_in_another_frame_than_the_point_is_dropped_unread(
    caplog, frame, box_frame, point_frame, tag
):
    """A box in another document can never be the hit: dropped without a read, and everything
    after the drop refers to the raw point's frame. The reads would keep it (box is the hit)."""
    box = FakeNode(x=900, y=500, width=100, height=100)  # holds SCALED; centre (950, 550)
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        kwargs, mocks = await _call("click", box=HIT, nodes={3: box, 9: frame})

    _unread(mocks)
    assert kwargs["element_data"] is None
    assert (kwargs["x"], kwargs["y"]) == SCALED
    assert kwargs["iframe_context"] == (FRAME if point_frame == FRAME else None)
    assert [m for m in caplog.messages if m.endswith(DROPPED)] == [
        f"   🎯 POINT-BOX DROPPED: the smallest indexed box <{tag}> is in {box_frame} but the "
        f"point {SCALED} is in {point_frame} — the call goes on from the point {DROPPED}"
    ]
