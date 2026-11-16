"""NLRF #162 wiring: the element spec's ``locate_only`` flag reaches the locator engine (only a literal True counts), and the engine hands it to the read gate."""

from unittest.mock import AsyncMock, patch

import pytest

from browser_service.agent.registration import register_custom_actions
from browser_service.config import config
from tests.test_agent.test_indexed_read_screen_point import (
    RESULT,
    FakeAgent,
    FakeBrowserSession,
    _fake_playwright,
)
from tests.test_agent.test_read_step_identity_wiring import ACCEPT, REJECT, _call, _page


async def _engine_kwargs(spec_extra):
    agent = FakeAgent()
    engine = AsyncMock(return_value=dict(RESULT))
    elements = [{"id": "elem_1", "action": "get_text", "value": "", **spec_extra}]
    with (
        patch("browser_service.agent.actions.find_unique_locator_action", new=engine),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
        patch.object(config.locator, "enable_action_fit", True),
    ):
        assert register_custom_actions(agent, elements=elements) is True
        await agent.tools.registry.execute_action(
            "find_unique_locator",
            {
                "element_id": "elem_1",
                "element_description": "the heading",
                "expected_text": "Dashboard",
                "x": 500,
                "y": 500,
            },
            browser_session=FakeBrowserSession({}),
        )
    return engine.await_args.kwargs


@pytest.mark.parametrize(
    "spec_extra, want",
    [({"locate_only": True}, True), ({}, False), ({"locate_only": "true"}, False)],
)
async def test_the_flag_reaches_the_engine(spec_extra, want):
    assert (await _engine_kwargs(spec_extra))["locate_only"] is want


@pytest.mark.parametrize("facts, path", [(ACCEPT, "candidate"), (REJECT, "cascade")])
async def test_the_flag_reaches_the_gate_on_both_paths(facts, path):
    """The engine hands the flag to the read gate where the agent's candidate is accepted and where the cascade answers."""
    policy = AsyncMock(side_effect=lambda root, result, *a, **k: result)
    await _call(_page(), facts, policy=policy, locate_only=True)
    assert policy.await_count >= 1, path
    assert all(c.kwargs["locate_only"] is True for c in policy.await_args_list), path
