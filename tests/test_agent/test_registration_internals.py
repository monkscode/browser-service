"""
Unit tests for browser_service.agent.registration — internal helpers.

Purpose: these helpers sit between the vision model's coordinates and the actual
         browser interaction. _find_smallest_containing_element decides WHICH
         element the LLM meant; _dispatch_browser_use_event decides whether the
         fast event-bus path handled the interaction or whether Playwright must
         retry. Both were largely untested, and both have fall-through contracts
         where returning the wrong sentinel means either a double interaction or
         a silently skipped one.

Tests:
  _find_smallest_containing_element: smallest containing box wins, page wrappers
      over 80% of viewport are skipped, skip_tag, zero-area and non-containing
      elements, missing absolute_position, empty inputs
  _get_cdp_url_from_session: the three fallback strategies, pattern rejection,
      and exceptions raised by each strategy
  _strip_rf_select_prefix: RF strategy prefixes stripped, multi-word option text
      left intact
  _wait_for_page_stability: swallows timeouts
  _dispatch_browser_use_event: type/click/select success, the fall-through
      signals for validation errors, failed selects and check/uncheck, and the
      rule that performed_actions is only written on confirmed success
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_service.agent.registration import (
    _extract_cdp_host_port,
    _find_smallest_containing_element,
    _get_cdp_url_from_session,
    _strip_rf_select_prefix,
    _wait_for_page_stability,
)


class Pos:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


def node(x, y, w, h, name="DIV"):
    elem = MagicMock(spec=["absolute_position", "node_name"])
    elem.absolute_position = Pos(x, y, w, h)
    elem.node_name = name
    return elem


VIEWPORT_AREA = 1920 * 1080


class TestFindSmallestContainingElement:
    """Tests for resolving coordinates to the most specific element."""

    def test_smallest_containing_box_wins(self):
        """A button inside a card inside a section must resolve to the button."""
        selector_map = {
            1: node(0, 0, 800, 600),
            2: node(90, 90, 300, 200),
            3: node(95, 95, 50, 20),
        }

        idx, elem = _find_smallest_containing_element(selector_map, (100, 100), VIEWPORT_AREA)

        assert idx == 3
        assert elem is selector_map[3]

    def test_skips_page_wrappers(self):
        """Boxes over 80% of the viewport are <body>-like wrappers, not targets.

        When the model gives wrong coordinates only wrappers match; skipping them
        lets smart_locator's text strategies take over instead of returning a
        locator for the whole page.
        """
        selector_map = {1: node(0, 0, 1920, 1080)}

        assert _find_smallest_containing_element(selector_map, (100, 100), VIEWPORT_AREA) == (
            None,
            None,
        )

    def test_wrapper_skipped_but_smaller_sibling_kept(self):
        """Skipping the wrapper must not discard a legitimate match inside it."""
        selector_map = {1: node(0, 0, 1920, 1080), 2: node(50, 50, 120, 40)}

        idx, _ = _find_smallest_containing_element(selector_map, (100, 60), VIEWPORT_AREA)

        assert idx == 2

    def test_skip_tag_excludes_the_iframe_itself(self):
        """Looking inside an iframe means the contained element, not the frame."""
        selector_map = {1: node(0, 0, 400, 300, name="IFRAME"), 2: node(10, 10, 100, 50)}

        idx, _ = _find_smallest_containing_element(
            selector_map, (50, 30), VIEWPORT_AREA, skip_tag="iframe"
        )

        assert idx == 2

    def test_ignores_non_containing_elements(self):
        """An element the point falls outside of is not a candidate."""
        selector_map = {1: node(500, 500, 100, 100)}

        assert _find_smallest_containing_element(selector_map, (10, 10), VIEWPORT_AREA) == (
            None,
            None,
        )

    def test_ignores_zero_area_elements(self):
        """A collapsed element contains the point mathematically but is not clickable."""
        selector_map = {1: node(100, 100, 0, 0)}

        assert _find_smallest_containing_element(selector_map, (100, 100), VIEWPORT_AREA) == (
            None,
            None,
        )

    def test_ignores_elements_without_position(self):
        """Nodes browser-use could not measure are skipped, not crashed on."""
        unmeasured = MagicMock(spec=["absolute_position"])
        unmeasured.absolute_position = None

        assert _find_smallest_containing_element({1: unmeasured}, (5, 5), VIEWPORT_AREA) == (
            None,
            None,
        )

    def test_empty_inputs(self):
        """No selector map or no coordinates yields no match."""
        assert _find_smallest_containing_element({}, (1, 1), VIEWPORT_AREA) == (None, None)
        assert _find_smallest_containing_element({1: node(0, 0, 5, 5)}, None, VIEWPORT_AREA) == (
            None,
            None,
        )


VALID_CDP = "ws://127.0.0.1:9222/devtools/browser/6f1a-uuid"


class TestGetCdpUrlFromSession:
    """Tests for the CDP URL fallback chain."""

    def test_strategy_1_direct_attribute(self):
        """The common case: browser_session.cdp_url."""
        session = MagicMock(spec=["cdp_url"])
        session.cdp_url = VALID_CDP

        assert _get_cdp_url_from_session(session) == VALID_CDP

    def test_strategy_2_cdp_client_url(self):
        """Falls back to cdp_client.url when cdp_url is absent."""
        session = MagicMock(spec=["cdp_client"])
        session.cdp_client = MagicMock(spec=["url"])
        session.cdp_client.url = VALID_CDP

        assert _get_cdp_url_from_session(session) == VALID_CDP

    def test_strategy_3_attribute_scan(self):
        """Last resort: any public string attribute matching the DevTools pattern."""

        class SessionWithOddAttribute:
            some_endpoint = VALID_CDP

        assert _get_cdp_url_from_session(SessionWithOddAttribute()) == VALID_CDP

    def test_rejects_non_devtools_url(self):
        """A plain http URL is not a CDP endpoint and must not be returned."""
        session = MagicMock(spec=["cdp_url"])
        session.cdp_url = "http://127.0.0.1:9222/"

        assert _get_cdp_url_from_session(session) is None

    def test_raising_cdp_url_property_falls_through(self):
        """A strategy-1 read that raises must not abort the chain.

        The attribute read has to happen inside the try, not in a hasattr()
        guard in front of it: hasattr() performs the read itself and swallows
        only AttributeError, so any other exception escapes before the try is
        entered and strategies 2 and 3 never run.
        """

        class Session:
            @property
            def cdp_url(self):
                raise RuntimeError("session closed")

            other = VALID_CDP

        assert _get_cdp_url_from_session(Session()) == VALID_CDP

    def test_raising_cdp_client_property_falls_through(self):
        """Strategy 2 must survive the same way — this is the live hazard.

        browser-use's BrowserSession.cdp_client asserts on
        _cdp_client_root is not None, so reading it on a session that has been
        reset raises AssertionError, not AttributeError.
        """

        class Session:
            @property
            def cdp_client(self):
                raise AssertionError("CDP client not initialized")

            other = VALID_CDP

        assert _get_cdp_url_from_session(Session()) == VALID_CDP

    def test_missing_attributes_do_fall_through(self):
        """The fall-through that does work: absent attributes, not raising ones."""

        class Session:
            other = VALID_CDP

        assert _get_cdp_url_from_session(Session()) == VALID_CDP

    def test_none_session(self):
        """No session means no URL."""
        assert _get_cdp_url_from_session(None) is None

    def test_nothing_found(self):
        """Exhausting all three strategies returns None rather than raising."""
        assert _get_cdp_url_from_session(MagicMock(spec=[])) is None

    def test_host_port_extraction_for_logging(self):
        """Log lines carry the endpoint without the browser UUID path."""
        assert _extract_cdp_host_port(VALID_CDP) == "ws://127.0.0.1:9222"

    def test_host_port_extraction_passthrough(self):
        """A URL with no /devtools/ segment is returned unchanged."""
        assert _extract_cdp_host_port("ws://host:1234") == "ws://host:1234"


class TestStripRfSelectPrefix:
    """Tests for Robot Framework Select Options By prefix handling."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("label    default", "default"),
            ("value    some_val", "some_val"),
            ("text     My Option", "My Option"),
            ("index    0", "0"),
            ("LABEL    Upper", "Upper"),
        ],
    )
    def test_strips_known_prefixes(self, raw, expected):
        """The RF strategy keyword is metadata, not part of the option text."""
        assert _strip_rf_select_prefix(raw) == expected

    @pytest.mark.parametrize("raw", ["United States", "default", "", "New York City"])
    def test_leaves_plain_values_intact(self, raw):
        """Multi-word option text must not be truncated by a naive first-token split."""
        assert _strip_rf_select_prefix(raw) == raw


class TestWaitForPageStability:
    """Tests for the post-interaction settle."""

    async def test_awaits_domcontentloaded(self):
        """domcontentloaded is enough for the next locator lookup; networkidle is not needed."""
        page = MagicMock()
        page.wait_for_load_state = AsyncMock()

        await _wait_for_page_stability(page)

        page.wait_for_load_state.assert_awaited_once_with("domcontentloaded", timeout=5000)

    async def test_timeout_is_swallowed(self):
        """No navigation occurred is the normal case, not an error."""
        page = MagicMock()
        page.wait_for_load_state = AsyncMock(side_effect=TimeoutError("no navigation"))

        await _wait_for_page_stability(page)


class FakeEvent:
    """An awaitable that also exposes event_result, like browser-use's event handles."""

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def __await__(self):
        async def _noop():
            return None

        return _noop().__await__()

    async def event_result(self, raise_if_any=False, raise_if_none=False):
        if self._error:
            raise self._error
        return self._result


def session_dispatching(event):
    session = MagicMock()
    session.event_bus.dispatch = MagicMock(return_value=event)
    return session


@pytest.fixture
def dispatch():
    """The dispatch helper, with browser-use's event models stubbed out.

    TypeTextEvent and friends are pydantic models that validate `node` against
    EnhancedDOMTreeNode, so a test double is rejected at construction before any
    of the logic under test runs. These tests are about the dispatch and
    fall-through contract, not browser-use's schema, so the models are replaced
    with plain constructors.
    """
    import browser_use.browser.events as events

    from browser_service.agent.registration import _dispatch_browser_use_event

    with (
        patch.object(events, "TypeTextEvent", MagicMock()),
        patch.object(events, "ClickElementEvent", MagicMock()),
        patch.object(events, "SelectDropdownOptionEvent", MagicMock()),
    ):
        yield _dispatch_browser_use_event


class TestDispatchBrowserUseEvent:
    """Tests for the event-bus interaction path and its fall-through contract.

    (None, None) means "Playwright must retry"; anything else means handled.
    performed_actions is written only on confirmed success — a fall-through that
    recorded the action would make the Playwright retry think it was already done.
    """

    async def test_type_success_records_action(self, dispatch):
        """A successful type marks the element done so it is not typed into twice."""
        performed = set()

        note, status = await dispatch(
            session_dispatching(FakeEvent()), MagicMock(), "input", "shoes", "elem_1", performed
        )

        assert status == "auto_ok"
        assert "shoes" in note
        assert performed == {"elem_1"}

    async def test_type_without_value_is_not_applicable(self, dispatch):
        """Nothing to type is not a failure and must not fall through to Playwright."""
        performed = set()

        note, status = await dispatch(
            session_dispatching(FakeEvent()), MagicMock(), "input", "", "elem_1", performed
        )

        assert (note, status) == ("", "not_applicable")
        assert performed == set()

    async def test_click_success_records_action(self, dispatch):
        """A successful click marks the element done."""
        performed = set()

        _, status = await dispatch(
            session_dispatching(FakeEvent()), MagicMock(), "click", "", "elem_2", performed
        )

        assert status == "auto_ok"
        assert performed == {"elem_2"}

    async def test_click_validation_error_falls_through(self, dispatch):
        """browser-use rejects clicks on file inputs and selects; Playwright may still manage."""
        performed = set()
        event = FakeEvent(result={"validation_error": "cannot click <select>"})

        assert await dispatch(
            session_dispatching(event), MagicMock(), "click", "", "elem_3", performed
        ) == (None, None)
        assert performed == set()

    async def test_select_success_records_action(self, dispatch):
        """A confirmed select marks the element done."""
        performed = set()
        event = FakeEvent(result={"success": "true"})

        note, status = await dispatch(
            session_dispatching(event), MagicMock(), "select", "India", "elem_4", performed
        )

        assert status == "auto_ok"
        assert "India" in note
        assert performed == {"elem_4"}

    async def test_select_failure_falls_through_without_recording(self, dispatch):
        """Tom Select and similar widgets fail here; Playwright has wider strategies."""
        performed = set()
        event = FakeEvent(result={"success": "false"})

        assert await dispatch(
            session_dispatching(event), MagicMock(), "select", "India", "elem_5", performed
        ) == (None, None)
        assert performed == set()

    async def test_select_without_value_is_not_applicable(self, dispatch):
        """No option to select is not a failure."""
        performed = set()

        assert await dispatch(
            session_dispatching(FakeEvent()), MagicMock(), "select", "", "elem_6", performed
        ) == ("", "not_applicable")

    @pytest.mark.parametrize("action", ["check", "uncheck"])
    async def test_check_actions_always_fall_through(self, dispatch, action):
        """browser-use has no state-aware check event; a blind click would toggle wrongly."""
        performed = set()

        assert await dispatch(
            session_dispatching(FakeEvent()), MagicMock(), action, "", "elem_7", performed
        ) == (None, None)
        assert performed == set()

    async def test_event_error_falls_through(self, dispatch):
        """An exception in the event path is a retry signal, not a workflow failure."""
        performed = set()
        event = FakeEvent(error=RuntimeError("event bus closed"))

        assert await dispatch(
            session_dispatching(event), MagicMock(), "click", "", "elem_8", performed
        ) == (None, None)
        assert performed == set()
