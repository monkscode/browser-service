"""
K v1 through a REAL browser-use 0.13.7 session (live lane — local only).

The fixture-lane test serves the CDP calls through Playwright's own session.
This one runs them through browser-use's (get_or_create_cdp_session) on a
selector map browser-use built itself — the production path.

Run: pytest tests/test_agent/test_live_read_step_identity.py -m live -v
"""

import asyncio
import contextlib
import functools
import http.server
import threading
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.read_step_identity import read_identity_facts, read_step_verdict
from browser_service.agent.registration import (
    _extract_dom_node_attributes,
    _get_cdp_url_from_session,
    _select_best_page,
)

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "test_locators" / "locator_fixtures"


@pytest.fixture
def fixture_url():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(FIXTURES_DIR))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/read_step_identity.html"
    finally:
        server.shutdown()


def _find(selector_map, tag, attr, value):
    for node in selector_map.values():
        attrs = getattr(node, "attributes", None) or {}
        if node.node_name.lower() == tag and value in (attrs.get(attr) or "").split():
            return node
    raise AssertionError(f"no indexed <{tag}> with {attr} containing {value!r}")


async def test_browser_use_session_u07_accept_and_reverse_reject(fixture_url):
    from browser_use.browser.session import BrowserSession

    session = BrowserSession(
        headless=True, viewport={"width": 1920, "height": 1080}, no_viewport=False
    )
    pw = None
    try:
        await session.start()
        pw = await async_playwright().start()
        await session.navigate_to(fixture_url)
        await asyncio.sleep(1.0)
        await session.get_browser_state_summary()
        selector_map = session._dom_watchdog.selector_map
        browser = await pw.chromium.connect_over_cdp(_get_cdp_url_from_session(session))
        page = _select_best_page(browser)

        link = _find(selector_map, "a", "class", "menu-item")
        element_data = _extract_dom_node_attributes(link)
        assert isinstance(element_data["backendNodeId"], int)

        facts = await read_identity_facts(page, session, element_data, "h6.crumb")
        accept, why = read_step_verdict(facts)
        assert accept is True, why

        save = _find(selector_map, "button", "id", "save")
        facts = await read_identity_facts(
            page, session, _extract_dom_node_attributes(save), "#save-link"
        )
        accept, why = read_step_verdict(facts)
        assert accept is False
        assert "clause 1" in why
    finally:
        if pw is not None:
            with contextlib.suppress(Exception):
                await pw.stop()
        await session.kill()
