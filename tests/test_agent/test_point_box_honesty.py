"""Point-then-prove D1: an index-less box from the selector map is evidence only when it is the hit.

registration's STEP A takes the smallest INDEXED box around the agent's point. On TodoMVC that
box was the app root (section#root), and it shipped as the checkbox's locator. For a performed
step (IDENTITY_ACTIONS) the box must be the element under the agent's scaled point, or the
nearest control at or above it; otherwise the call goes on with that point and no element data.
Reads, waits, select steps, indexed calls and a check that fails keep today's data.

Registration runs through the real Tools registry; the locator engine, Playwright and the two
DOM reads (pointed_target, box_is_hit) are mocked.
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
HIT, BOX = object(), object()


def _root():
    node = FakeNode(x=665, y=40, width=550, height=700)  # holds SCALED; centre BOX_CENTRE
    node.node_name = "SECTION"
    return node


async def _call(action, params=None, *, hit=HIT, box=BOX, target=None, flag=True, nodes=None):
    """One find_unique_locator call; returns the engine's kwargs and the pointed_target mock."""
    engine = AsyncMock(return_value=dict(NOT_FOUND))
    if target is None:
        target = AsyncMock(
            side_effect=lambda page, frame, data, by_index, point: hit if point else box
        )
    agent = FakeAgent()
    with (
        patch("browser_service.agent.actions.find_unique_locator_action", new=engine),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
        patch.object(config.locator, "enable_action_fit", flag),
        patch("browser_service.agent.registration.pointed_target", new=target),
        patch(
            "browser_service.agent.registration.box_is_hit",
            new=AsyncMock(side_effect=lambda b, h: b is h),
        ),
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
    return engine.await_args.kwargs, target


def _kept(kwargs):
    assert kwargs["element_data"]["tagName"] == "section"
    assert (kwargs["x"], kwargs["y"]) == BOX_CENTRE


async def test_index_less_click_box_that_is_not_the_hit_is_dropped(caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        kwargs, target = await _call("click")

    assert kwargs["element_data"] is None
    assert (kwargs["x"], kwargs["y"]) == SCALED  # the agent's point, not the box centre
    assert kwargs["iframe_context"] is None
    assert kwargs["pointed_by_index"] is False
    # hit-tested at the scaled raw point; the box resolved by its xpath alone
    assert target.await_args_list[0].args[1:] == (None, None, False, SCALED)
    assert target.await_args_list[1].args[2]["tagName"] == "section"
    assert target.await_args_list[1].args[3:] == (True, None)
    assert len([m for m in caplog.messages if m.endswith(DROPPED)]) == 1


async def test_box_that_is_the_hit_is_kept(caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        kwargs, _ = await _call("click", box=HIT)

    _kept(kwargs)
    assert not [m for m in caplog.messages if DROPPED in m]


async def test_no_hit_drops_the_box():
    kwargs, _ = await _call("input", hit=None)

    assert kwargs["element_data"] is None
    assert (kwargs["x"], kwargs["y"]) == SCALED


async def test_unresolvable_box_keeps_the_data(caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        kwargs, _ = await _call("check", box=None)

    _kept(kwargs)
    assert not [m for m in caplog.messages if DROPPED in m]


async def test_indexed_call_is_never_checked():
    kwargs, target = await _call("click", {"element_index": 3})

    target.assert_not_awaited()
    _kept(kwargs)
    assert kwargs["pointed_by_index"] is True


async def test_box_kept_for_read_steps():
    kwargs, target = await _call("get_text")

    target.assert_not_awaited()
    _kept(kwargs)


@pytest.mark.parametrize(("action", "flag"), [("select", True), ("click", False)])
async def test_box_kept_for_select_steps_and_with_the_flag_off(action, flag):
    kwargs, target = await _call(action, flag=flag)

    target.assert_not_awaited()
    _kept(kwargs)


async def test_check_exception_keeps_data(caplog):
    broken = AsyncMock(side_effect=RuntimeError("frame detached"))
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        kwargs, _ = await _call("click", target=broken)

    _kept(kwargs)
    assert not [m for m in caplog.messages if DROPPED in m]
    assert any("frame detached" in m for m in caplog.messages)


async def test_a_drop_takes_the_frame_at_the_raw_point():
    """Today's frame is detected at the box centre; after a drop everything is the raw point's."""
    box = FakeNode(x=900, y=500, width=100, height=100)  # holds SCALED; centre (950, 550)
    frame = SimpleNamespace(
        node_name="IFRAME",
        attributes={"id": "f"},
        absolute_position=SimpleNamespace(x=955, y=0, width=965, height=1080),
    )
    kwargs, target = await _call("click", nodes={3: box, 9: frame})

    assert kwargs["element_data"] is None
    assert kwargs["iframe_context"] == 'iframe[id="f"]'
    assert target.await_args_list[0].args[1] == 'iframe[id="f"]'
