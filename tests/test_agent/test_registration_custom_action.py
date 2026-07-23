"""Harness tests for the find_unique_locator custom-action closure.

register_custom_actions builds and registers an inner `find_unique_locator`
handler on a browser-use Tools registry. These tests retrieve the raw closure
(unwrapping browser-use's keyword-only normaliser) and drive its
element-data-from-index extraction directly with a fake browser_session,
selector_map and DOM node — no live browser.

The handler cannot open a Playwright connection here (the fake session carries
no CDP endpoint), so after the extraction it degrades to an error ActionResult
rather than raising. That is the contract under test: the DOM extraction runs
and the action fails gracefully. It also exercises the text-content defaulting
that keeps a text-less node (icon button, image) from storing None.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from browser_service.agent.registration import register_custom_actions


def _get_extraction_handler(elements):
    """Register the action on a throwaway agent and return (raw_closure, param_model)."""
    agent = SimpleNamespace(tools=None)
    assert register_custom_actions(agent, elements=elements) is True
    ra = agent.tools.registry.registry.actions["find_unique_locator"]
    return inspect.unwrap(ra.function), ra.param_model


class FakePos:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.width, self.height = x, y, w, h


class FakeDomNode:
    """Minimal stand-in for browser-use's EnhancedDOMTreeNode."""

    def __init__(
        self,
        *,
        node_name="BUTTON",
        attributes=None,
        xpath="/html/body/button",
        text="Submit",
        pos=None,
        with_meaningful=True,
    ):
        self.node_name = node_name
        self.attributes = attributes if attributes is not None else {"id": "submit-btn"}
        self.xpath = xpath
        self.absolute_position = pos if pos is not None else FakePos(900, 500, 120, 80)
        self.parent_node = None
        self.children_nodes = []
        self._text = text
        if with_meaningful:
            self.get_meaningful_text_for_llm = lambda: self._text
        self.get_all_children_text = lambda: self._text or ""


def _session_with_watchdog(selector_map):
    return SimpleNamespace(_dom_watchdog=SimpleNamespace(selector_map=selector_map))


def _is_action_result(res):
    return type(res).__name__ == "ActionResult"


class TestElementIndexExtraction:
    @pytest.mark.asyncio
    async def test_extracts_from_index_then_degrades_without_cdp(self):
        """A provided element_index resolves the DOM node and extracts its data;
        with no CDP endpoint the action returns a graceful error, not a crash."""
        handler, Params = _get_extraction_handler([{"id": "elem_1", "action": "get_text"}])
        session = _session_with_watchdog({5: FakeDomNode()})
        params = Params(
            element_id="elem_1",
            element_description="submit button",
            x=500,
            y=500,
            element_index=5,
        )

        result = await handler(params, session)

        assert _is_action_result(result)
        assert result.error is not None  # Playwright unavailable → graceful failure

    @pytest.mark.asyncio
    async def test_expected_text_and_collection_flags_are_accepted(self):
        """expected_text / is_collection travel through the handler without error."""
        handler, Params = _get_extraction_handler([{"id": "elem_1", "action": "get_text"}])
        session = _session_with_watchdog({5: FakeDomNode()})
        params = Params(
            element_id="elem_1",
            element_description="first row cell",
            x=500,
            y=500,
            element_index=5,
            expected_text="Cierra",
            is_collection=True,
        )

        result = await handler(params, session)

        assert _is_action_result(result)

    @pytest.mark.asyncio
    async def test_textless_node_does_not_crash_extraction(self):
        """get_meaningful_text_for_llm() → None must not enter the payload as None.

        The extraction defaults it to "", so no TypeError is raised while building
        or logging element_data for a text-less element (icon button, image)."""
        handler, Params = _get_extraction_handler([{"id": "elem_1", "action": "get_text"}])
        node = FakeDomNode(text=None, attributes={"id": "icon"})
        session = _session_with_watchdog({7: node})
        params = Params(
            element_id="elem_1",
            element_description="icon button",
            x=400,
            y=300,
            element_index=7,
        )

        result = await handler(params, session)

        assert _is_action_result(result)  # reached here == no TypeError on None text

    @pytest.mark.asyncio
    async def test_children_text_fallback_when_no_meaningful_text_method(self):
        """A node without get_meaningful_text_for_llm falls back to children text."""
        handler, Params = _get_extraction_handler([{"id": "elem_1", "action": "get_text"}])
        node = FakeDomNode(with_meaningful=False, text="Row cell text")
        session = _session_with_watchdog({3: node})
        params = Params(
            element_id="elem_1",
            element_description="a cell",
            x=100,
            y=100,
            element_index=3,
        )

        result = await handler(params, session)

        assert _is_action_result(result)

    @pytest.mark.asyncio
    async def test_missing_index_in_selector_map_still_returns_actionresult(self):
        """An element_index absent from the selector_map degrades, not crashes."""
        handler, Params = _get_extraction_handler([{"id": "elem_1", "action": "get_text"}])
        session = _session_with_watchdog({1: FakeDomNode()})
        params = Params(
            element_id="elem_1",
            element_description="missing",
            x=500,
            y=500,
            element_index=99,
        )

        result = await handler(params, session)

        assert _is_action_result(result)


class TestCoordinateFallbackExtraction:
    """No element_index → STEP A locates the element by its bounding box."""

    @pytest.mark.asyncio
    async def test_finds_element_by_coordinates(self):
        """x=500,y=500 (0-1000 space) scales to (960, 540) at the 1920x1080 default;
        the node whose bbox contains that point is resolved and extracted."""
        handler, Params = _get_extraction_handler([{"id": "elem_1", "action": "get_text"}])
        node = FakeDomNode(pos=FakePos(900, 500, 200, 100))
        session = SimpleNamespace(get_selector_map=AsyncMock(return_value={4: node}))
        params = Params(
            element_id="elem_1",
            element_description="submit button",
            x=500,
            y=500,
            element_index=None,
        )

        result = await handler(params, session)

        assert _is_action_result(result)

    @pytest.mark.asyncio
    async def test_no_element_at_coordinates_degrades(self):
        """A selector_map with nothing under the point degrades gracefully."""
        handler, Params = _get_extraction_handler([{"id": "elem_1", "action": "get_text"}])
        node = FakeDomNode(pos=FakePos(0, 0, 10, 10))  # far from the scaled point
        session = SimpleNamespace(get_selector_map=AsyncMock(return_value={4: node}))
        params = Params(
            element_id="elem_1",
            element_description="submit button",
            x=500,
            y=500,
            element_index=None,
        )

        result = await handler(params, session)

        assert _is_action_result(result)

    @pytest.mark.asyncio
    async def test_empty_selector_map_degrades(self):
        """An empty selector_map degrades gracefully."""
        handler, Params = _get_extraction_handler([{"id": "elem_1", "action": "get_text"}])
        session = SimpleNamespace(get_selector_map=AsyncMock(return_value={}))
        params = Params(
            element_id="elem_1",
            element_description="submit button",
            x=500,
            y=500,
            element_index=None,
        )

        result = await handler(params, session)

        assert _is_action_result(result)
