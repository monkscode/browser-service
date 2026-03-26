"""
Live integration tests for BrowserSessionManager (Tier 2).

Purpose: Verify browser session lifecycle — creation, status checks,
         and graceful teardown — against a real Playwright/Chromium instance.

Requires:
  - Playwright installed with chromium: playwright install chromium
  - browser-use package installed

Run with:
  pytest tests/test_integration/test_live_session.py -m integration -v
"""

import asyncio
import pytest

pytestmark = pytest.mark.integration


class TestBrowserSessionManagerInit:
    """Unit-level tests for BrowserSessionManager construction — no browser needed."""

    def test_default_init_headless_false(self):
        """Default headless value is False."""
        from browser_service.browser.session import BrowserSessionManager
        mgr = BrowserSessionManager()
        assert mgr.headless is False

    def test_custom_headless_true(self):
        from browser_service.browser.session import BrowserSessionManager
        mgr = BrowserSessionManager(headless=True)
        assert mgr.headless is True

    def test_default_viewport(self):
        """Default viewport is 1920×1080."""
        from browser_service.browser.session import BrowserSessionManager
        mgr = BrowserSessionManager()
        assert mgr.viewport == {"width": 1920, "height": 1080}

    def test_custom_viewport(self):
        from browser_service.browser.session import BrowserSessionManager
        mgr = BrowserSessionManager(viewport={"width": 1280, "height": 720})
        assert mgr.viewport["width"] == 1280

    def test_initial_session_is_none(self):
        """Before create_session(), session attribute is None."""
        from browser_service.browser.session import BrowserSessionManager
        mgr = BrowserSessionManager()
        assert mgr.session is None

    def test_get_session_returns_none_before_start(self):
        from browser_service.browser.session import BrowserSessionManager
        mgr = BrowserSessionManager()
        assert mgr.get_session() is None

    def test_is_active_false_before_start(self):
        from browser_service.browser.session import BrowserSessionManager
        mgr = BrowserSessionManager()
        assert mgr.is_active() is False


class TestBrowserSessionLifecycle:
    """Tests that launch a real Chromium browser — Playwright required."""

    @pytest.mark.asyncio
    async def test_create_session_returns_session_object(self):
        """create_session() returns a session that is not None."""
        from browser_service.browser.session import BrowserSessionManager
        mgr = BrowserSessionManager(headless=True)
        session = None
        try:
            session = await mgr.create_session()
            assert session is not None
            assert mgr.session is not None
            assert mgr.is_active() is True
        finally:
            await mgr.close_session()

    @pytest.mark.asyncio
    async def test_close_session_clears_session(self):
        """After close_session(), session is None and is_active() is False."""
        from browser_service.browser.session import BrowserSessionManager
        mgr = BrowserSessionManager(headless=True)
        await mgr.create_session()
        await mgr.close_session()
        assert mgr.session is None
        assert mgr.is_active() is False

    @pytest.mark.asyncio
    async def test_close_session_idempotent(self):
        """Calling close_session() twice does not raise."""
        from browser_service.browser.session import BrowserSessionManager
        mgr = BrowserSessionManager(headless=True)
        await mgr.create_session()
        await mgr.close_session()
        await mgr.close_session()  # second call — must not raise

    @pytest.mark.asyncio
    async def test_recreate_session_replaces_existing(self):
        """Calling create_session() again closes the old one and opens a new one."""
        from browser_service.browser.session import BrowserSessionManager
        mgr = BrowserSessionManager(headless=True)
        try:
            session1 = await mgr.create_session()
            session2 = await mgr.create_session()
            # sessions are different objects
            assert session1 is not session2
        finally:
            await mgr.close_session()

    @pytest.mark.asyncio
    async def test_get_session_returns_active_session(self):
        """get_session() returns the same object returned by create_session()."""
        from browser_service.browser.session import BrowserSessionManager
        mgr = BrowserSessionManager(headless=True)
        try:
            created = await mgr.create_session()
            retrieved = mgr.get_session()
            assert created is retrieved
        finally:
            await mgr.close_session()
