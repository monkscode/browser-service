"""
Unit tests for browser_service.browser.cleanup — CDP port/PID tracking and process cleanup.

Purpose: cleanup.py tracks which Chrome process belongs to our service so we can
         kill only that process (not the user's personal Chrome).  A regression
         here means either: (a) we can't clean up → Chrome zombie processes, or
         (b) we kill the wrong Chrome → user loses their browser.

Tests:
  store_cdp_port: valid URL / None / invalid URL
  get/clear stored port and PID
  _get_pid_from_port: mocked netstat output
  count_chrome_processes: mocked tasklist output
"""

import pytest
from unittest.mock import patch, MagicMock


class TestStoreCdpPort:
    """Tests for store_cdp_port — extracting and storing port from CDP URL."""

    def test_valid_cdp_url(self):
        """Extracts port from standard CDP URL and stores it."""
        from browser_service.browser.cleanup import store_cdp_port, get_stored_cdp_port, clear_stored_cdp_port
        # Clean state
        clear_stored_cdp_port()

        with patch("browser_service.browser.cleanup._get_pid_from_port", return_value=12345):
            port = store_cdp_port("ws://127.0.0.1:9222/devtools/browser/abc-123")

        assert port == "9222"
        assert get_stored_cdp_port() == "9222"

        # Clean up
        clear_stored_cdp_port()

    def test_none_url(self):
        """None URL → returns None, stores nothing."""
        from browser_service.browser.cleanup import store_cdp_port, clear_stored_cdp_port
        clear_stored_cdp_port()

        result = store_cdp_port(None)
        assert result is None

    def test_invalid_url(self):
        """URL without port pattern → returns None."""
        from browser_service.browser.cleanup import store_cdp_port, clear_stored_cdp_port
        clear_stored_cdp_port()

        result = store_cdp_port("not-a-cdp-url")
        assert result is None


class TestClearStoredCdpPort:
    """Tests for clearing stored port and PID."""

    def test_clear_resets_both(self):
        """clear_stored_cdp_port resets both port and PID to None."""
        from browser_service.browser import cleanup
        cleanup._tracked_cdp_port = "9222"
        cleanup._tracked_browser_pid = 12345

        cleanup.clear_stored_cdp_port()
        assert cleanup._tracked_cdp_port is None
        assert cleanup._tracked_browser_pid is None


class TestGetPidFromPort:
    """Tests for _get_pid_from_port with mocked subprocess."""

    @patch("browser_service.browser.cleanup.subprocess.run")
    @patch("browser_service.browser.cleanup.sys")
    def test_windows_netstat_finds_pid(self, mock_sys, mock_run):
        """On Windows, parses netstat output to find PID for port."""
        mock_sys.platform = "win32"
        # Simulate netstat -ano output
        mock_run.return_value = MagicMock(
            stdout="  TCP    127.0.0.1:9222    0.0.0.0:0    LISTENING    5678\n",
            returncode=0,
        )

        from browser_service.browser.cleanup import _get_pid_from_port
        pid = _get_pid_from_port("9222")
        assert pid == 5678


class TestCountChromeProcesses:
    """Tests for count_chrome_processes with mocked subprocess."""

    @patch("browser_service.browser.cleanup.subprocess.run")
    @patch("browser_service.browser.cleanup.sys")
    def test_counts_windows_chrome(self, mock_sys, mock_run):
        """On Windows, parses tasklist CSV to count chrome.exe processes."""
        mock_sys.platform.startswith.return_value = True
        mock_run.return_value = MagicMock(
            stdout='"chrome.exe","1234","Console","1","50,000 K"\n"chrome.exe","1235","Console","1","30,000 K"\n',
            returncode=0,
        )

        from browser_service.browser.cleanup import count_chrome_processes
        count, pids = count_chrome_processes()
        assert count == 2
        assert 1234 in pids
        assert 1235 in pids
