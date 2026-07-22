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
"""

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
