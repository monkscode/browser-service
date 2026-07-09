"""
Unit tests for helper functions in browser_service.agent.registration.

Purpose: registration.py contains pure helper functions that extract DOM node
         attributes, detect iframe context, and resolve CDP URLs.  These are
         deterministic functions that don't require a live browser — they operate
         on mock objects.  A regression in any of these means locator extraction
         silently fails or CDP connections break.

Tests:
  _extract_dom_node_attributes: full attrs, missing, no attributes attr
  _detect_iframe_context: inside/outside/by-name/ordinal/empty
  _extract_cdp_host_port: standard / no devtools
  _get_cdp_url_from_session: direct / client / search / None / no shared-state side-effects
  _CDP_URL_PATTERN: valid / invalid patterns
"""

import pytest
from unittest.mock import MagicMock, patch


class TestExtractDomNodeAttributes:
    """Tests for _extract_dom_node_attributes."""

    def _get_fn(self):
        from browser_service.agent.registration import _extract_dom_node_attributes
        return _extract_dom_node_attributes

    def test_full_attributes(self):
        """All standard attributes extracted from a DOM node."""
        fn = self._get_fn()
        node = MagicMock()
        node.node_name = "INPUT"
        node.attributes = {
            "id": "search",
            "name": "q",
            "class": "form-input",
            "aria-label": "Search",
            "placeholder": "Type here...",
            "title": "Search box",
            "href": "",
            "role": "searchbox",
            "data-testid": "search-input",
            "type": "text",
            "value": "shoes",
        }
        node.xpath = "/html/body/form/input"

        result = fn(node)
        assert result["tagName"] == "input"
        assert result["id"] == "search"
        assert result["name"] == "q"
        assert result["className"] == "form-input"
        assert result["ariaLabel"] == "Search"
        assert result["placeholder"] == "Type here..."
        assert result["dataTestId"] == "search-input"
        assert result["type"] == "text"
        assert result["xpath"] == "/html/body/form/input"

    def test_missing_attributes_default_empty(self):
        """Missing attributes default to empty string."""
        fn = self._get_fn()
        node = MagicMock()
        node.node_name = "DIV"
        node.attributes = {"id": "main"}
        node.xpath = ""

        result = fn(node)
        assert result["name"] == ""
        assert result["className"] == ""
        assert result["ariaLabel"] == ""

    def test_no_attributes_property(self):
        """Node without 'attributes' attr → all empty."""
        fn = self._get_fn()
        node = MagicMock(spec=[])  # No attributes at all
        node.node_name = "SPAN"

        result = fn(node)
        assert result["tagName"] == "span"
        assert result["id"] == ""

    # --- C3: the source test attribute must be carried with the value ---

    def _make_node(self, attrs):
        node = MagicMock()
        node.node_name = "BUTTON"
        node.attributes = attrs
        node.xpath = ""
        return node

    def test_data_testid_carries_source_attr(self):
        """data-testid present → dataTestAttr names data-testid."""
        result = self._get_fn()(self._make_node({"data-testid": "login-btn"}))
        assert result["dataTestId"] == "login-btn"
        assert result["dataTestAttr"] == "data-testid"

    def test_data_test_only_carries_source_attr(self):
        """C3 regression: element with ONLY data-test must not be reported as
        data-testid — the emitter would build [data-testid=...] which matches
        0 elements and silently loses the hook."""
        result = self._get_fn()(self._make_node({"data-test": "login-btn"}))
        assert result["dataTestId"] == "login-btn"
        assert result["dataTestAttr"] == "data-test"

    def test_both_test_attrs_prefer_data_testid(self):
        result = self._get_fn()(
            self._make_node({"data-testid": "tid", "data-test": "t"})
        )
        assert result["dataTestId"] == "tid"
        assert result["dataTestAttr"] == "data-testid"

    def test_neither_test_attr_defaults(self):
        result = self._get_fn()(self._make_node({"id": "x"}))
        assert result["dataTestId"] == ""
        assert result["dataTestAttr"] == "data-testid"

    # --- Task G (nlrf G7): aria-invalid rides the extraction pipe ---

    def test_aria_invalid_extracted(self):
        """aria-invalid is the ARIA state signal for invalid fields —
        nlrf's state-verification assembler emits a Get Attribute
        assertion when the observed field carries it, covering sites
        that mark errors via ARIA instead of a CSS class."""
        result = self._get_fn()(
            self._make_node({"id": "email", "aria-invalid": "true"})
        )
        assert result["ariaInvalid"] == "true"

    def test_aria_invalid_defaults_empty(self):
        """Fields without the attribute must report empty string, not
        raise — most elements never set aria-invalid."""
        result = self._get_fn()(self._make_node({"id": "email"}))
        assert result["ariaInvalid"] == ""


class TestDetectIframeContext:
    """Tests for _detect_iframe_context."""

    def _get_fn(self):
        from browser_service.agent.registration import _detect_iframe_context
        return _detect_iframe_context

    def _make_iframe_element(self, x, y, w, h, iframe_id="", iframe_name="",
                             iframe_title="", iframe_class=""):
        elem = MagicMock()
        elem.node_name = "iframe"
        pos = MagicMock()
        pos.x = x
        pos.y = y
        pos.width = w
        pos.height = h
        elem.absolute_position = pos
        elem.attributes = {
            "id": iframe_id,
            "name": iframe_name,
            "title": iframe_title,
            "class": iframe_class,
        }
        return elem

    def test_coords_inside_iframe_by_id(self):
        """Coordinates inside an iframe with id → returns iframe[id='...']."""
        fn = self._get_fn()
        iframe = self._make_iframe_element(0, 0, 800, 600, iframe_id="main-frame")
        selector_map = {0: iframe}

        locator, iframe_id = fn(selector_map, (400, 300))
        assert locator is not None
        assert "main-frame" in locator
        assert iframe_id == "main-frame"

    def test_coords_outside_iframe(self):
        """Coordinates outside all iframes → (None, None)."""
        fn = self._get_fn()
        iframe = self._make_iframe_element(0, 0, 100, 100, iframe_id="small")
        selector_map = {0: iframe}

        locator, iframe_id = fn(selector_map, (500, 500))
        assert locator is None
        assert iframe_id is None

    def test_iframe_by_name(self):
        """Iframe with name (no id) → iframe[name='...']."""
        fn = self._get_fn()
        iframe = self._make_iframe_element(0, 0, 800, 600, iframe_name="content")
        selector_map = {0: iframe}

        locator, iframe_id = fn(selector_map, (100, 100))
        assert "name" in locator
        assert "content" in locator

    def test_iframe_ordinal_fallback(self):
        """Iframe without id or name → ordinal-based selector."""
        fn = self._get_fn()
        iframe = self._make_iframe_element(0, 0, 800, 600)
        selector_map = {0: iframe}

        locator, _ = fn(selector_map, (100, 100))
        assert "iframe" in locator
        assert "nth=" in locator

    # --- G6 / Task F: title and class before the ordinal fallback -------
    # The ordinal shifts when an async third-party iframe (ASTPP:
    # #jsd-widget support chat) loads at a different moment between
    # discovery and RF runtime — iframe >> nth=N then points at a
    # DIFFERENT frame. A stable title/class names the frame by what it
    # is instead of where it sits.

    def test_iframe_by_title(self):
        """CKEditor case: no id/name, stable title → iframe[title=...]."""
        fn = self._get_fn()
        editor = self._make_iframe_element(
            0, 0, 800, 600, iframe_title="Rich Text Editor, template",
            iframe_class="cke_wysiwyg_frame cke_reset",
        )
        widget = self._make_iframe_element(
            900, 900, 50, 50, iframe_id="jsd-widget")
        locator, _ = fn({0: editor, 1: widget}, (100, 100))
        assert locator == 'iframe[title="Rich Text Editor, template"]'

    def test_title_quotes_escaped(self):
        fn = self._get_fn()
        iframe = self._make_iframe_element(
            0, 0, 800, 600, iframe_title='He said "hi"')
        locator, _ = fn({0: iframe}, (100, 100))
        assert locator == 'iframe[title="He said \\"hi\\""]'

    def test_dynamic_title_skipped(self):
        """A data-bound title (date) dies next session → not an anchor."""
        fn = self._get_fn()
        iframe = self._make_iframe_element(
            0, 0, 800, 600, iframe_title="Report 2026-07-08")
        locator, _ = fn({0: iframe}, (100, 100))
        assert "nth=" in locator

    def test_duplicate_title_skipped(self):
        """Two iframes sharing a title → frame_locator would be ambiguous."""
        fn = self._get_fn()
        a = self._make_iframe_element(0, 0, 400, 600, iframe_title="Ad")
        b = self._make_iframe_element(500, 0, 400, 600, iframe_title="Ad")
        locator, _ = fn({0: a, 1: b}, (100, 100))
        assert "nth=" in locator

    def test_iframe_by_unique_stable_class(self):
        """No id/name/title → first stable, unique, identifier-shaped class."""
        fn = self._get_fn()
        editor = self._make_iframe_element(
            0, 0, 800, 600, iframe_class="cke_wysiwyg_frame cke_reset")
        widget = self._make_iframe_element(
            900, 900, 50, 50, iframe_class="jsd-frame")
        locator, _ = fn({0: editor, 1: widget}, (100, 100))
        assert locator == "iframe.cke_wysiwyg_frame"

    def test_volatile_class_skipped(self):
        """An init-order counter class (cke_1) is dead next session —
        the scorer must veto it (couples with the cke_\\d+ scorer rule)."""
        fn = self._get_fn()
        iframe = self._make_iframe_element(0, 0, 800, 600,
                                           iframe_class="cke_1")
        locator, _ = fn({0: iframe}, (100, 100))
        assert "nth=" in locator

    def test_volatile_class_skipped_next_class_used(self):
        fn = self._get_fn()
        iframe = self._make_iframe_element(
            0, 0, 800, 600, iframe_class="cke_1 cke_wysiwyg_frame")
        locator, _ = fn({0: iframe}, (100, 100))
        assert locator == "iframe.cke_wysiwyg_frame"

    def test_non_identifier_class_skipped(self):
        """Tailwind-style class (w-1/2) is not a valid bare .class selector."""
        fn = self._get_fn()
        iframe = self._make_iframe_element(0, 0, 800, 600,
                                           iframe_class="w-1/2")
        locator, _ = fn({0: iframe}, (100, 100))
        assert "nth=" in locator

    def test_shared_class_skipped(self):
        """A class carried by another iframe too is ambiguous → ordinal."""
        fn = self._get_fn()
        a = self._make_iframe_element(0, 0, 400, 600,
                                      iframe_class="widget-frame")
        b = self._make_iframe_element(500, 0, 400, 600,
                                      iframe_class="widget-frame extra")
        locator, _ = fn({0: a, 1: b}, (100, 100))
        assert "nth=" in locator

    def test_id_still_wins_over_title(self):
        """Cascade order unchanged at the top: id beats title/class."""
        fn = self._get_fn()
        iframe = self._make_iframe_element(
            0, 0, 800, 600, iframe_id="main-frame",
            iframe_title="Rich Text Editor", iframe_class="cke_wysiwyg_frame")
        locator, _ = fn({0: iframe}, (100, 100))
        assert locator == 'iframe[id="main-frame"]'

    def test_ordinal_counts_prior_iframes(self):
        """The ordinal fallback still counts iframes in selector_map order."""
        fn = self._get_fn()
        first = self._make_iframe_element(900, 900, 50, 50)   # not containing
        second = self._make_iframe_element(0, 0, 800, 600)    # target, bare
        locator, _ = fn({0: first, 1: second}, (100, 100))
        assert locator == "iframe >> nth=1"

    def test_empty_selector_map(self):
        """None or empty selector_map → (None, None)."""
        fn = self._get_fn()
        assert fn(None, (100, 100)) == (None, None)
        assert fn({}, (100, 100)) == (None, None)

    def test_none_coords(self):
        """None coordinates → (None, None)."""
        fn = self._get_fn()
        assert fn({0: MagicMock()}, None) == (None, None)


class TestCdpUrlHelpers:
    """Tests for CDP URL extraction and cache management."""

    def test_extract_cdp_host_port(self):
        """Extracts ws://host:port from full CDP URL."""
        from browser_service.agent.registration import _extract_cdp_host_port
        result = _extract_cdp_host_port("ws://127.0.0.1:9222/devtools/browser/abc-123")
        assert result == "ws://127.0.0.1:9222"

    def test_extract_cdp_host_port_no_devtools(self):
        """URL without /devtools/ returns as-is."""
        from browser_service.agent.registration import _extract_cdp_host_port
        result = _extract_cdp_host_port("ws://localhost:9222")
        assert result == "ws://localhost:9222"

    def test_get_cdp_url_direct(self):
        """Strategy 1: Direct cdp_url attribute."""
        from browser_service.agent.registration import _get_cdp_url_from_session
        session = MagicMock()
        session.cdp_url = "ws://127.0.0.1:9222/devtools/browser/abc"
        result = _get_cdp_url_from_session(session)
        assert result == session.cdp_url

    def test_get_cdp_url_from_client(self):
        """Strategy 2: cdp_client.url attribute."""
        from browser_service.agent.registration import _get_cdp_url_from_session
        session = MagicMock(spec=["cdp_client"])
        session.cdp_url = None  # Force strategy 1 to skip
        del session.cdp_url
        session.cdp_client = MagicMock()
        session.cdp_client.url = "ws://127.0.0.1:9333/devtools/browser/xyz"
        result = _get_cdp_url_from_session(session)
        assert result == "ws://127.0.0.1:9333/devtools/browser/xyz"

    def test_get_cdp_url_does_not_write_shared_cleanup_globals(self):
        """_get_cdp_url_from_session must not mutate module-level globals in cleanup.py.

        In concurrent mode, _store_cdp_port_for_cleanup() had 'last writer wins'
        semantics: any task resolving its CDP URL could overwrite _tracked_cdp_port
        and _tracked_browser_pid in cleanup.py, silently redirecting another task's
        fallback cleanup target to the wrong browser.

        The function is now a pure reader — it returns the URL without side-effects.
        """
        import browser_service.browser.cleanup as cleanup_mod
        from browser_service.agent.registration import _get_cdp_url_from_session

        # Reset global state to a known baseline
        original_port = cleanup_mod._tracked_cdp_port
        original_pid = cleanup_mod._tracked_browser_pid
        cleanup_mod._tracked_cdp_port = "sentinel-port"
        cleanup_mod._tracked_browser_pid = 99999
        try:
            session = MagicMock()
            session.cdp_url = "ws://127.0.0.1:9222/devtools/browser/abc"
            _get_cdp_url_from_session(session)

            # Globals must be untouched
            assert cleanup_mod._tracked_cdp_port == "sentinel-port", (
                "_get_cdp_url_from_session must not write to cleanup._tracked_cdp_port"
            )
            assert cleanup_mod._tracked_browser_pid == 99999, (
                "_get_cdp_url_from_session must not write to cleanup._tracked_browser_pid"
            )
        finally:
            cleanup_mod._tracked_cdp_port = original_port
            cleanup_mod._tracked_browser_pid = original_pid

    def test_get_cdp_url_none_session(self):
        """None session → None."""
        from browser_service.agent.registration import _get_cdp_url_from_session
        assert _get_cdp_url_from_session(None) is None

    def test_cdp_url_pattern_valid(self):
        """CDP URL regex matches valid WebSocket DevTools URLs."""
        from browser_service.agent.registration import _CDP_URL_PATTERN
        assert _CDP_URL_PATTERN.match("ws://127.0.0.1:9222/devtools/browser/abc")
        assert _CDP_URL_PATTERN.match("wss://host:443/devtools/browser/xyz")

    def test_cdp_url_pattern_invalid(self):
        """CDP URL regex rejects non-CDP strings."""
        from browser_service.agent.registration import _CDP_URL_PATTERN
        assert _CDP_URL_PATTERN.match("http://localhost:9222") is None
        assert _CDP_URL_PATTERN.match("not-a-url") is None


class TestNoCacheGlobals:
    """Verify removed module-level globals and shared-state helpers are gone."""

    def test_no_cache_globals(self):
        """Module must not export any of the removed cache globals."""
        import browser_service.agent.registration as reg
        removed = [
            '_playwright_instance_cache',
            '_connected_browser_cache',
            '_cache_cdp_url',
            '_cache_initialized',
            '_cache_lock',
        ]
        for name in removed:
            assert not hasattr(reg, name), f"Removed global still present: {name}"

    def test_cleanup_playwright_cache_removed(self):
        """cleanup_playwright_cache must not exist (was module-cache-specific)."""
        import browser_service.agent.registration as reg
        assert not hasattr(reg, 'cleanup_playwright_cache')

    def test_invalidate_playwright_cache_removed(self):
        """invalidate_playwright_cache must not exist (was module-cache-specific)."""
        import browser_service.agent.registration as reg
        assert not hasattr(reg, 'invalidate_playwright_cache')

    def test_store_cdp_port_for_cleanup_removed(self):
        """_store_cdp_port_for_cleanup must not exist.

        This function was removed because it wrote to module-level globals in
        cleanup.py (_tracked_cdp_port, _tracked_browser_pid). Under concurrent
        load the 'last writer wins', causing one task's cleanup to target another
        task's browser. _get_cdp_url_from_session is now a pure reader.
        """
        import browser_service.agent.registration as reg
        assert not hasattr(reg, '_store_cdp_port_for_cleanup'), (
            "_store_cdp_port_for_cleanup was re-introduced; it writes to shared "
            "module-level globals in cleanup.py which is unsafe for concurrent tasks."
        )
