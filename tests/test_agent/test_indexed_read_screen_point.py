"""The read gate's screen point: an indexed call uses its element's DOM box centre (bs #75 E2).

The agent's raw point can be on the pixel scale; registration scales every raw point as a
0-1000 grid point, so (876, 354) became (1681, 382) even though element [7]'s box centre is
(876, 354). An index-less call has no such box and keeps the scaled point.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from browser_service.agent.registration import register_custom_actions
from browser_service.config import config


class FakeNode:
    node_name = "P"
    attributes: dict = {}
    xpath = "/html/body/p"
    parent_node = None

    def __init__(self, x, y, width, height):
        self.absolute_position = SimpleNamespace(x=x, y=y, width=width, height=height)

    def get_all_children_text(self):
        return "Access granted"


class FakeBrowserSession:
    cdp_url = "ws://127.0.0.1:9222/devtools/browser/00000000-0000-0000-0000-000000000000"
    cdp_client = None
    is_cdp_connected = True
    _original_viewport_size = (1920, 1080)
    agent_focus_target_id = None

    def __init__(self, selector_map):
        self._map = selector_map

    async def get_selector_map(self):
        return self._map

    async def get_current_page_url(self):
        return "https://example.test/"


class FakeAgent:
    tools = None


def _fake_playwright():
    fake_page = AsyncMock()
    fake_page.url = "https://example.test/"
    ctx = MagicMock()
    ctx.pages = [fake_page]
    fake_browser = MagicMock()
    fake_browser.contexts = [ctx]
    fake_instance = MagicMock()
    fake_instance.chromium.connect_over_cdp = AsyncMock(return_value=fake_browser)
    starter = MagicMock()
    starter.start = AsyncMock(return_value=fake_instance)
    return MagicMock(return_value=starter)


ELEMENTS = [{"id": "elem_1", "action": "get_text", "value": ""}]
RESULT = {
    "found": True,
    "best_locator": "id=msg",
    "validated": True,
    "count": 1,
    "unique": True,
    "valid": True,
    "validation_method": "playwright",
}


async def _vision_point(params, selector_map):
    agent = FakeAgent()
    engine = AsyncMock(return_value=dict(RESULT))
    with (
        patch("browser_service.agent.actions.find_unique_locator_action", new=engine),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
        patch.object(config.locator, "enable_action_fit", True),
    ):
        assert register_custom_actions(agent, elements=ELEMENTS) is True
        await agent.tools.registry.execute_action(
            "find_unique_locator",
            {
                "element_id": "elem_1",
                "element_description": "the message",
                "expected_text": "Access granted",
                **params,
            },
            browser_session=FakeBrowserSession(selector_map),
        )
    return engine.await_args.kwargs["vision_point"]


async def test_indexed_call_uses_the_box_centre_not_the_scaled_raw_point():
    # raw (876, 354) is a pixel point: scaled as a grid point it would be (1681, 382)
    node = FakeNode(x=826, y=334, width=100, height=40)  # centre (876, 354)
    point = await _vision_point({"x": 876, "y": 354, "element_index": 7}, {7: node})
    assert point == (876, 354)


async def test_indexed_call_without_a_box_keeps_the_scaled_point():
    node = FakeNode(0, 0, 1, 1)
    node.absolute_position = None
    point = await _vision_point({"x": 500, "y": 500, "element_index": 7}, {7: node})
    assert point == (960, 540)


async def test_index_less_call_keeps_the_scaled_point_even_when_step_a_found_a_box():
    node = FakeNode(x=900, y=520, width=140, height=50)  # holds (960, 540); centre (970, 545)
    point = await _vision_point({"x": 500, "y": 500}, {3: node})
    assert point == (960, 540)
