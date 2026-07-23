"""
Unit tests for browser_service.browser.session — BrowserSessionManager.

Purpose: BrowserSessionManager controls the lifecycle of browser sessions.
         If init defaults are wrong, the browser starts in the wrong mode.
         If has_session/get_session are broken, callers can't check state.

Tests:
  - Default init values (headless=False, viewport=1920x1080)
  - Custom init values
  - has_session false initially
  - get_session returns None initially
  - has_session after manual session assignment
  - create_session builds and starts a BrowserSession with the configured settings
  - create_session closes a pre-existing session first
  - create_session re-raises on failure without leaving a half-built session
  - close_session is a no-op with no session
  - close_session prefers kill(), then close(), then browser.close()
  - close_session swallows teardown errors but always clears the handle
  - is_active mirrors has_session
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_service.browser.session import BrowserSessionManager


class TestBrowserSessionManagerInit:
    """Tests for session manager initialization."""

    def test_default_headless(self):
        """Default headless is False (development-friendly)."""
        mgr = BrowserSessionManager()
        assert mgr.headless is False

    def test_default_viewport(self):
        """Default viewport is 1920x1080."""
        mgr = BrowserSessionManager()
        assert mgr.viewport == {"width": 1920, "height": 1080}

    def test_custom_headless(self):
        """Custom headless=True is stored."""
        mgr = BrowserSessionManager(headless=True)
        assert mgr.headless is True

    def test_custom_viewport(self):
        """Custom viewport is stored."""
        mgr = BrowserSessionManager(viewport={"width": 1280, "height": 720})
        assert mgr.viewport == {"width": 1280, "height": 720}


class TestBrowserSessionManagerState:
    """Tests for session state queries."""

    def test_no_session_initially(self):
        """No session exists before create_session is called."""
        mgr = BrowserSessionManager()
        assert mgr.has_session() is False
        assert mgr.get_session() is None

    def test_has_session_after_manual_set(self):
        """Setting session attribute manually enables has_session."""
        mgr = BrowserSessionManager()
        mgr.session = "mock-session"
        assert mgr.has_session() is True
        assert mgr.get_session() == "mock-session"

    def test_is_active_mirrors_has_session(self):
        """is_active is a semantic alias, not a different check."""
        mgr = BrowserSessionManager()
        assert mgr.is_active() is False
        mgr.session = "mock-session"
        assert mgr.is_active() is True


class TestCreateSession:
    """Tests for create_session."""

    async def test_creates_and_starts_with_configured_settings(self):
        """BrowserSession gets the manager's headless/viewport, and start() is awaited."""
        created = MagicMock()
        created.start = AsyncMock()
        mgr = BrowserSessionManager(headless=True, viewport={"width": 800, "height": 600})

        with patch("browser_use.browser.session.BrowserSession", return_value=created) as cls:
            result = await mgr.create_session()

        cls.assert_called_once_with(headless=True, viewport={"width": 800, "height": 600})
        created.start.assert_awaited_once()
        assert result is created
        assert mgr.get_session() is created

    async def test_closes_existing_session_first(self):
        """A second create_session tears down the first — no orphaned browser."""
        old = MagicMock(spec=["kill"])
        old.kill = AsyncMock()
        new = MagicMock()
        new.start = AsyncMock()

        mgr = BrowserSessionManager()
        mgr.session = old

        with patch("browser_use.browser.session.BrowserSession", return_value=new):
            await mgr.create_session()

        old.kill.assert_awaited_once()
        assert mgr.get_session() is new

    async def test_start_failure_propagates(self):
        """A failing start() re-raises rather than returning a dead session."""
        created = MagicMock()
        created.start = AsyncMock(side_effect=RuntimeError("chrome refused to launch"))
        mgr = BrowserSessionManager()

        with patch("browser_use.browser.session.BrowserSession", return_value=created):
            with pytest.raises(RuntimeError, match="chrome refused to launch"):
                await mgr.create_session()


class TestCloseSession:
    """Tests for close_session teardown paths."""

    async def test_no_session_is_noop(self):
        """Closing with nothing open does not raise."""
        mgr = BrowserSessionManager()
        await mgr.close_session()
        assert mgr.get_session() is None

    async def test_prefers_kill(self):
        """browser-use exposes kill(); it wins when present."""
        session = MagicMock(spec=["kill"])
        session.kill = AsyncMock()
        mgr = BrowserSessionManager()
        mgr.session = session

        await mgr.close_session()

        session.kill.assert_awaited_once()
        assert mgr.get_session() is None

    async def test_falls_back_to_close(self):
        """Without kill(), close() is used."""
        session = MagicMock(spec=["close"])
        session.close = AsyncMock()
        mgr = BrowserSessionManager()
        mgr.session = session

        await mgr.close_session()

        session.close.assert_awaited_once()

    async def test_falls_back_to_browser_close(self):
        """Without kill() or close(), the nested browser handle is closed."""
        session = MagicMock(spec=["browser"])
        session.browser = MagicMock(spec=["close"])
        session.browser.close = AsyncMock()
        mgr = BrowserSessionManager()
        mgr.session = session

        await mgr.close_session()

        session.browser.close.assert_awaited_once()

    async def test_teardown_error_still_clears_handle(self):
        """A raising kill() must not strand the manager holding a dead session."""
        session = MagicMock(spec=["kill"])
        session.kill = AsyncMock(side_effect=RuntimeError("already gone"))
        mgr = BrowserSessionManager()
        mgr.session = session

        await mgr.close_session()

        assert mgr.get_session() is None
        assert mgr.has_session() is False
