"""
Unit tests for browser_service.browser.cleanup — browser PID discovery.

Purpose: cleanup kills a Chrome process by PID. Resolving the wrong PID is not
         a cosmetic bug — it means killing the user's own browser, or worse,
         killing the browser-use service process itself. get_browser_process_id
         walks six fallback strategies in priority order and was entirely
         untested; _get_pid_from_port's column-matching rules exist precisely
         because a naive match once selected the service's own socket.

Companion to test_cleanup.py, which covers CDP port storage and the
cleanup_browser_resources ownership checks.

Tests:
  _get_pid_from_port: short rows, foreign-column match, substring port
                      collision, malformed PID, POSIX lsof, probe failure
  count_chrome_processes: malformed PID column, POSIX pgrep, probe failure
  get_browser_process_id: stored-PID fast path, cdp_url port, stored-port
                          fallback, absent port, netstat error, POSIX lsof,
                          the four Playwright fallbacks, exhaustion, and an
                          unexpected error
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def clean_tracking_state():
    """Reset the module-level CDP port/PID globals around each test."""
    from browser_service.browser import cleanup

    saved = (cleanup._tracked_cdp_port, cleanup._tracked_browser_pid)
    cleanup._tracked_cdp_port = None
    cleanup._tracked_browser_pid = None
    yield cleanup
    cleanup._tracked_cdp_port, cleanup._tracked_browser_pid = saved


class TestGetPidFromPortEdgeCases:
    """Tests for the column-matching rules that stop us killing the wrong process."""

    @patch("browser_service.browser.cleanup.subprocess.run")
    @patch("browser_service.browser.cleanup.sys")
    def test_ignores_short_rows(self, mock_sys, mock_run):
        """Header and blank netstat rows have fewer than 5 columns and are skipped."""
        mock_sys.platform = "win32"
        mock_run.return_value = MagicMock(
            stdout="\nActive Connections\n  Proto  Local  Foreign\n"
            "  TCP    127.0.0.1:9222   0.0.0.0:0   LISTENING   5678\n"
        )

        from browser_service.browser.cleanup import _get_pid_from_port

        assert _get_pid_from_port("9222") == 5678

    @patch("browser_service.browser.cleanup.subprocess.run")
    @patch("browser_service.browser.cleanup.sys")
    def test_ignores_foreign_address_match(self, mock_sys, mock_run):
        """The CDP port in the FOREIGN column is our own client socket, not Chrome.

        Matching it would make cleanup kill the browser-use service itself.
        """
        mock_sys.platform = "win32"
        mock_run.return_value = MagicMock(
            stdout="  TCP    127.0.0.1:55123   127.0.0.1:9222   ESTABLISHED   4242\n"
        )

        from browser_service.browser.cleanup import _get_pid_from_port

        assert _get_pid_from_port("9222") is None

    @patch("browser_service.browser.cleanup.subprocess.run")
    @patch("browser_service.browser.cleanup.sys")
    def test_ignores_substring_port_collision(self, mock_sys, mock_run):
        """Port 9222 must not match a listener on 92220."""
        mock_sys.platform = "win32"
        mock_run.return_value = MagicMock(
            stdout="  TCP    127.0.0.1:92220   0.0.0.0:0   LISTENING   7777\n"
        )

        from browser_service.browser.cleanup import _get_pid_from_port

        assert _get_pid_from_port("9222") is None

    @patch("browser_service.browser.cleanup.subprocess.run")
    @patch("browser_service.browser.cleanup.sys")
    def test_non_numeric_pid_column_is_skipped(self, mock_sys, mock_run):
        """A malformed PID column does not raise."""
        mock_sys.platform = "win32"
        mock_run.return_value = MagicMock(
            stdout="  TCP    127.0.0.1:9222   0.0.0.0:0   LISTENING   not-a-pid\n"
        )

        from browser_service.browser.cleanup import _get_pid_from_port

        assert _get_pid_from_port("9222") is None

    @patch("browser_service.browser.cleanup.subprocess.run")
    @patch("browser_service.browser.cleanup.sys")
    def test_posix_lsof_finds_pid(self, mock_sys, mock_run):
        """On POSIX, lsof -ti reports the listening PID."""
        mock_sys.platform = "linux"
        mock_run.return_value = MagicMock(stdout="4321\n")

        from browser_service.browser.cleanup import _get_pid_from_port

        assert _get_pid_from_port("9222") == 4321
        assert mock_run.call_args[0][0] == ["lsof", "-ti", ":9222"]

    @patch("browser_service.browser.cleanup.subprocess.run")
    @patch("browser_service.browser.cleanup.sys")
    def test_posix_empty_lsof_output(self, mock_sys, mock_run):
        """No listener on the port yields None rather than an exception."""
        mock_sys.platform = "linux"
        mock_run.return_value = MagicMock(stdout="  \n")

        from browser_service.browser.cleanup import _get_pid_from_port

        assert _get_pid_from_port("9222") is None

    @patch("browser_service.browser.cleanup.subprocess.run")
    @patch("browser_service.browser.cleanup.sys")
    def test_subprocess_failure_returns_none(self, mock_sys, mock_run):
        """A timed-out or missing netstat degrades to None, not a crash."""
        mock_sys.platform = "win32"
        mock_run.side_effect = OSError("netstat not found")

        from browser_service.browser.cleanup import _get_pid_from_port

        assert _get_pid_from_port("9222") is None


class TestCountChromeProcessesEdgeCases:
    """Tests for count_chrome_processes beyond the happy Windows path."""

    @patch("browser_service.browser.cleanup.subprocess.run")
    @patch("browser_service.browser.cleanup.sys")
    def test_skips_unparseable_pid_column(self, mock_sys, mock_run):
        """A tasklist row with a non-numeric PID is skipped; others still count."""
        mock_sys.platform = "win32"
        mock_run.return_value = MagicMock(
            stdout='"chrome.exe","N/A","Console","1","0 K"\n'
            '"chrome.exe","1234","Console","1","1 K"\n'
        )

        from browser_service.browser.cleanup import count_chrome_processes

        count, pids = count_chrome_processes()
        assert (count, pids) == (1, [1234])

    @patch("browser_service.browser.cleanup.subprocess.run")
    @patch("browser_service.browser.cleanup.sys")
    def test_posix_pgrep(self, mock_sys, mock_run):
        """On POSIX, pgrep output lines are the PIDs."""
        mock_sys.platform = "linux"
        mock_run.return_value = MagicMock(stdout="111\n222\n333\n")

        from browser_service.browser.cleanup import count_chrome_processes

        count, pids = count_chrome_processes()
        assert (count, pids) == (3, [111, 222, 333])

    @patch("browser_service.browser.cleanup.subprocess.run")
    @patch("browser_service.browser.cleanup.sys")
    def test_failure_returns_sentinel(self, mock_sys, mock_run):
        """A failed probe returns -1 so callers can tell unknown from none."""
        mock_sys.platform = "linux"
        mock_run.side_effect = OSError("pgrep missing")

        from browser_service.browser.cleanup import count_chrome_processes

        assert count_chrome_processes() == (-1, [])


class TestGetBrowserProcessId:
    """Tests for the six-strategy PID resolution ladder, in priority order."""

    def test_stored_pid_short_circuits(self, clean_tracking_state):
        """The stored PID was captured while Chrome was alive, so it wins outright."""
        from browser_service.browser.cleanup import get_browser_process_id

        clean_tracking_state._tracked_browser_pid = 9999

        assert get_browser_process_id(MagicMock()) == 9999

    @patch("browser_service.browser.cleanup.subprocess.run")
    @patch("browser_service.browser.cleanup.sys")
    def test_cdp_url_port_via_netstat(self, mock_sys, mock_run, clean_tracking_state):
        """With no stored PID, the port is parsed from cdp_url and resolved by netstat."""
        from browser_service.browser.cleanup import get_browser_process_id

        mock_sys.platform = "win32"
        mock_run.return_value = MagicMock(
            stdout="  TCP    127.0.0.1:9222   0.0.0.0:0   LISTENING   3131\n"
        )
        session = MagicMock(spec=["cdp_url"])
        session.cdp_url = "http://127.0.0.1:9222/devtools/browser/abc"

        assert get_browser_process_id(session) == 3131

    @patch("browser_service.browser.cleanup.subprocess.run")
    @patch("browser_service.browser.cleanup.sys")
    def test_falls_back_to_stored_port_when_cdp_url_gone(
        self, mock_sys, mock_run, clean_tracking_state
    ):
        """browser-use clears cdp_url before cleanup runs; the stored port covers that."""
        from browser_service.browser.cleanup import get_browser_process_id

        mock_sys.platform = "win32"
        mock_run.return_value = MagicMock(
            stdout="  TCP    127.0.0.1:9333   0.0.0.0:0   LISTENING   4141\n"
        )
        clean_tracking_state._tracked_cdp_port = "9333"
        session = MagicMock(spec=["cdp_url"])
        session.cdp_url = None

        assert get_browser_process_id(session) == 4141

    @patch("browser_service.browser.cleanup.subprocess.run")
    @patch("browser_service.browser.cleanup.sys")
    def test_port_absent_from_netstat(self, mock_sys, mock_run, clean_tracking_state):
        """A browser that already exited leaves no netstat row — None, not a crash."""
        from browser_service.browser.cleanup import get_browser_process_id

        mock_sys.platform = "win32"
        mock_run.return_value = MagicMock(stdout="  TCP  127.0.0.1:80  0.0.0.0:0  LISTENING  1\n")
        session = MagicMock(spec=["cdp_url"])
        session.cdp_url = "http://127.0.0.1:9222/devtools/browser/abc"

        assert get_browser_process_id(session) is None

    @patch("browser_service.browser.cleanup.subprocess.run")
    @patch("browser_service.browser.cleanup.sys")
    def test_netstat_error_is_swallowed(self, mock_sys, mock_run, clean_tracking_state):
        """netstat blowing up must not abort cleanup."""
        from browser_service.browser.cleanup import get_browser_process_id

        mock_sys.platform = "win32"
        mock_run.side_effect = OSError("netstat unavailable")
        session = MagicMock(spec=["cdp_url"])
        session.cdp_url = "http://127.0.0.1:9222/devtools/browser/abc"

        assert get_browser_process_id(session) is None

    @patch("browser_service.browser.cleanup.subprocess.run")
    @patch("browser_service.browser.cleanup.sys")
    def test_posix_lsof_path(self, mock_sys, mock_run, clean_tracking_state):
        """On POSIX the port is resolved with lsof instead of netstat."""
        from browser_service.browser.cleanup import get_browser_process_id

        mock_sys.platform = "linux"
        mock_run.return_value = MagicMock(stdout="5151\n")
        session = MagicMock(spec=["cdp_url"])
        session.cdp_url = "http://127.0.0.1:9222/devtools/browser/abc"

        assert get_browser_process_id(session) == 5151

    @patch("browser_service.browser.cleanup.sys")
    def test_falls_through_to_browser_process(self, mock_sys, clean_tracking_state):
        """Playwright fallback 1: browser._browser_process.pid."""
        from browser_service.browser.cleanup import get_browser_process_id

        mock_sys.platform = "linux"
        session = MagicMock(spec=["cdp_url", "browser"])
        session.cdp_url = None
        session.browser = MagicMock(spec=["_browser_process"])
        session.browser._browser_process.pid = 111

        assert get_browser_process_id(session) == 111

    @patch("browser_service.browser.cleanup.sys")
    def test_falls_through_to_process_attribute(self, mock_sys, clean_tracking_state):
        """Playwright fallback 2: browser.process.pid."""
        from browser_service.browser.cleanup import get_browser_process_id

        mock_sys.platform = "linux"
        session = MagicMock(spec=["cdp_url", "browser"])
        session.cdp_url = None
        session.browser = MagicMock(spec=["process"])
        session.browser.process.pid = 222

        assert get_browser_process_id(session) == 222

    @patch("browser_service.browser.cleanup.sys")
    def test_falls_through_to_impl_browser_process(self, mock_sys, clean_tracking_state):
        """Playwright fallback 3: browser._impl._browser_process.pid."""
        from browser_service.browser.cleanup import get_browser_process_id

        mock_sys.platform = "linux"
        session = MagicMock(spec=["cdp_url", "browser"])
        session.cdp_url = None
        session.browser = MagicMock(spec=["_impl"])
        session.browser._impl = MagicMock(spec=["_browser_process"])
        session.browser._impl._browser_process.pid = 333

        assert get_browser_process_id(session) == 333

    @patch("browser_service.browser.cleanup.sys")
    def test_falls_through_to_context_browser(self, mock_sys, clean_tracking_state):
        """Playwright fallback 4: the first context carrying a browser process."""
        from browser_service.browser.cleanup import get_browser_process_id

        mock_sys.platform = "linux"
        context = MagicMock(spec=["_browser"])
        context._browser = MagicMock(spec=["process"])
        context._browser.process.pid = 444
        session = MagicMock(spec=["cdp_url", "browser"])
        session.cdp_url = None
        session.browser = MagicMock(spec=["contexts"])
        session.browser.contexts = [context]

        assert get_browser_process_id(session) == 444

    @patch("browser_service.browser.cleanup.sys")
    def test_falls_through_to_session_context(self, mock_sys, clean_tracking_state):
        """Playwright fallback 5: session.context.browser.process.pid."""
        from browser_service.browser.cleanup import get_browser_process_id

        mock_sys.platform = "linux"
        session = MagicMock(spec=["cdp_url", "context"])
        session.cdp_url = None
        session.context = MagicMock(spec=["browser"])
        session.context.browser = MagicMock(spec=["process"])
        session.context.browser.process.pid = 555

        assert get_browser_process_id(session) == 555

    @patch("browser_service.browser.cleanup.sys")
    def test_exhausted_strategies_returns_none(self, mock_sys, clean_tracking_state):
        """Nothing resolvable means graceful close only, not an exception."""
        from browser_service.browser.cleanup import get_browser_process_id

        mock_sys.platform = "linux"
        session = MagicMock(spec=["cdp_url"])
        session.cdp_url = None

        assert get_browser_process_id(session) is None

    def test_unexpected_error_returns_none(self, clean_tracking_state):
        """PID discovery is best-effort: an unexpected failure must not propagate."""
        from browser_service.browser.cleanup import get_browser_process_id

        class ExplodingSession:
            @property
            def cdp_url(self):
                raise RuntimeError("session detached")

        assert get_browser_process_id(ExplodingSession()) is None
