"""
The registered find_unique_locator action hands the step's frame to the
interaction (browser-service #36).

The interaction fallback can only act inside an iframe when it is told which
frame the step is in. The frame is detected in the action closure, from
browser-use's selector map, before the locator search — and it must reach
``_do_interaction`` from there: the locator result does not always carry it
(text-first and description results are prefixed but have no
``iframe_context`` key), and the returned string is not parsed for it.

No browser: the closure runs with a fake session whose selector map holds an
<iframe> and an element whose box lies inside it. The locator engine,
Playwright and ``_do_interaction`` are replaced at their seams; the test reads
what the closure passed to the interaction.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from browser_service.agent.registration import register_custom_actions


class FakePos:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.width, self.height = x, y, w, h


class FakeDomNode:
    """Minimal stand-in for browser-use's EnhancedDOMTreeNode."""

    def __init__(self, node_name, attributes, pos, xpath):
        self.node_name = node_name
        self.attributes = attributes
        self.absolute_position = pos
        self.xpath = xpath
        self.parent_node = None
        self.children_nodes = []
        self.get_meaningful_text_for_llm = lambda: ""
        self.get_all_children_text = lambda: ""


CHECKBOX = FakeDomNode(
    "INPUT", {"type": "checkbox", "name": "remember"}, FakePos(120, 240, 20, 20), "html/body/input"
)
APP_FRAME = FakeDomNode(
    "IFRAME", {"id": "app-frame"}, FakePos(100, 200, 700, 520), "html/body/iframe"
)
# Same checkbox box, but the page's only iframe sits elsewhere.
FAR_FRAME = FakeDomNode(
    "IFRAME", {"id": "ad-frame"}, FakePos(100, 900, 700, 40), "html/body/iframe"
)


class FakeBrowserSession:
    cdp_url = "ws://127.0.0.1:9222/devtools/browser/00000000-0000-0000-0000-000000000000"
    cdp_client = None
    is_cdp_connected = True
    _original_viewport_size = (1920, 1080)
    agent_focus_target_id = None

    def __init__(self, selector_map):
        self._selector_map = selector_map

    async def get_selector_map(self):
        return self._selector_map

    async def get_current_page_url(self):
        return "https://example.org/settings"


class FakeAgent:
    tools = None


def _fake_playwright():
    fake_page = AsyncMock()
    fake_page.url = "https://example.org/settings"
    ctx = MagicMock()
    ctx.pages = [fake_page]
    fake_browser = MagicMock()
    fake_browser.contexts = [ctx]
    fake_instance = MagicMock()
    fake_instance.chromium.connect_over_cdp = AsyncMock(return_value=fake_browser)
    starter = MagicMock()
    starter.start = AsyncMock(return_value=fake_instance)
    return MagicMock(return_value=starter)


async def _frame_handed_to_the_interaction(selector_map, best_locator):
    """Run the registered action once; return the keyword arguments the
    closure passed to ``_do_interaction``."""
    seen: dict = {}

    async def _recording_interaction(
        browser_session,
        active_page,
        locator_str,
        element_id,
        element_index,
        element_specs,
        performed_actions,
        **kwargs,
    ):
        seen.update(kwargs, locator_str=locator_str)
        return "", "auto_ok", "check", ""

    engine = AsyncMock(
        return_value={
            "found": True,
            "best_locator": best_locator,
            "validated": True,
            "count": 1,
            "unique": True,
            "valid": True,
            "validation_method": "playwright",
        }
    )
    agent = FakeAgent()
    with (
        patch("browser_service.agent.actions.find_unique_locator_action", new=engine),
        patch("browser_service.agent.registration._do_interaction", new=_recording_interaction),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
    ):
        elements = [{"id": "elem_1", "action": "check", "value": ""}]
        assert register_custom_actions(agent, elements=elements) is True
        await agent.tools.registry.execute_action(
            "find_unique_locator",
            {
                "x": 70,
                "y": 230,
                "element_id": "elem_1",
                "element_description": "the Remember me checkbox",
                "element_index": 12,
            },
            browser_session=FakeBrowserSession(selector_map),
        )
    assert seen, "the interaction was never reached"
    return seen


async def test_element_inside_an_iframe_passes_its_frame_to_the_interaction():
    seen = await _frame_handed_to_the_interaction(
        {7: APP_FRAME, 12: CHECKBOX}, 'iframe[id="app-frame"] >>> [name="remember"]'
    )

    assert seen["iframe_context"] == 'iframe[id="app-frame"]'
    assert seen["locator_str"] == 'iframe[id="app-frame"] >>> [name="remember"]'


async def test_element_outside_every_iframe_passes_no_frame():
    seen = await _frame_handed_to_the_interaction({7: FAR_FRAME, 12: CHECKBOX}, '[name="remember"]')

    assert seen["iframe_context"] is None
