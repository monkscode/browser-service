"""
Unit tests for browser_service.browser.cleanup_worker — the standalone
Chrome hard-kill subprocess.

Purpose: the worker's orphan sweep must complete inside the parent's 10s
         wait (cleanup.py).  The old psutil.process_iter sweep opened a
         handle per process on Windows (~13s for ~450 processes), so the
         parent killed the worker on EVERY generation and the sweep never
         ran.  The fix is a single-shot process-table query scoped to the
         session's Chrome PID.

Tests:
  _find_chrome_orphans: Windows CIM CSV parsing / Linux pgrep parsing /
                        empty results / subprocess failure fail-safe
  main: orphans killed by PID, slow per-process iteration never used
"""

import os
import time

import pytest
from unittest.mock import patch, MagicMock


class TestFindChromeOrphansWindows:
    """Windows branch: one Get-CimInstance call, CSV output."""

    @patch("browser_service.browser.cleanup_worker.subprocess.run")
    @patch("browser_service.browser.cleanup_worker.sys")
    def test_returns_only_chrome_children(self, mock_sys, mock_run):
        """Parses CIM CSV and keeps only chrome-named children."""
        mock_sys.platform = "win32"
        mock_run.return_value = MagicMock(
            stdout='"ProcessId","Name"\r\n'
                   '"1111","chrome.exe"\r\n'
                   '"2222","msedge.exe"\r\n'
                   '"3333","chrome.exe"\r\n',
            returncode=0,
        )

        from browser_service.browser.cleanup_worker import _find_chrome_orphans
        pids = _find_chrome_orphans(999, MagicMock())

        assert pids == [1111, 3333]
        # Exactly ONE process-table query — that is the whole point of the fix.
        assert mock_run.call_count == 1
        # The query must be scoped to the session's PID, not the whole table.
        cmd = mock_run.call_args[0][0]
        assert any("ParentProcessId=999" in part for part in cmd)

    @patch("browser_service.browser.cleanup_worker.subprocess.run")
    @patch("browser_service.browser.cleanup_worker.sys")
    def test_no_children_returns_empty(self, mock_sys, mock_run):
        """No child processes → empty stdout → empty list."""
        mock_sys.platform = "win32"
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        from browser_service.browser.cleanup_worker import _find_chrome_orphans
        assert _find_chrome_orphans(999, MagicMock()) == []


class TestFindChromeOrphansLinux:
    """Linux branch: one pgrep -P call."""

    @patch("browser_service.browser.cleanup_worker.subprocess.run")
    @patch("browser_service.browser.cleanup_worker.sys")
    def test_returns_pgrep_pids(self, mock_sys, mock_run):
        """pgrep -P <pid> chrome output lines become the orphan list."""
        mock_sys.platform = "linux"
        mock_run.return_value = MagicMock(stdout="4444\n5555\n", returncode=0)

        from browser_service.browser.cleanup_worker import _find_chrome_orphans
        pids = _find_chrome_orphans(999, MagicMock())

        assert pids == [4444, 5555]
        cmd = mock_run.call_args[0][0]
        assert "-P" in cmd and "999" in cmd

    @patch("browser_service.browser.cleanup_worker.subprocess.run")
    @patch("browser_service.browser.cleanup_worker.sys")
    def test_pgrep_no_match_returns_empty(self, mock_sys, mock_run):
        """pgrep exits rc=1 with empty stdout when nothing matches."""
        mock_sys.platform = "linux"
        mock_run.return_value = MagicMock(stdout="", returncode=1)

        from browser_service.browser.cleanup_worker import _find_chrome_orphans
        assert _find_chrome_orphans(999, MagicMock()) == []


class TestFindChromeOrphansFailSafe:
    """The sweep must never crash the worker — failures return []."""

    @patch("browser_service.browser.cleanup_worker.subprocess.run")
    @patch("browser_service.browser.cleanup_worker.sys")
    def test_subprocess_failure_returns_empty_and_warns(self, mock_sys, mock_run):
        """Query timeout/error → empty list + a warning, never an exception."""
        import subprocess as _sp
        mock_sys.platform = "win32"
        mock_run.side_effect = _sp.TimeoutExpired(cmd="powershell", timeout=8)

        from browser_service.browser.cleanup_worker import _find_chrome_orphans
        logger = MagicMock()
        assert _find_chrome_orphans(999, logger) == []
        assert logger.warning.called

    @patch("browser_service.browser.cleanup_worker.subprocess.run")
    @patch("browser_service.browser.cleanup_worker.sys")
    def test_powershell_nonzero_rc_returns_empty_and_warns(self, mock_sys, mock_run):
        """A powershell failure (rc!=0) must WARN, not masquerade as a clean
        empty sweep — silent-empty is how the safety net dies unnoticed."""
        mock_sys.platform = "win32"
        mock_run.return_value = MagicMock(
            stdout="", stderr="Get-CimInstance : Access denied", returncode=1
        )

        from browser_service.browser.cleanup_worker import _find_chrome_orphans
        logger = MagicMock()
        assert _find_chrome_orphans(999, logger) == []
        assert logger.warning.called

    @patch("browser_service.browser.cleanup_worker.subprocess.run")
    @patch("browser_service.browser.cleanup_worker.sys")
    def test_pgrep_fatal_rc_returns_empty_and_warns(self, mock_sys, mock_run):
        """pgrep rc>=2 is a real failure (rc=1 is just 'no match') — warn."""
        mock_sys.platform = "linux"
        mock_run.return_value = MagicMock(
            stdout="", stderr="pgrep: invalid option", returncode=2
        )

        from browser_service.browser.cleanup_worker import _find_chrome_orphans
        logger = MagicMock()
        assert _find_chrome_orphans(999, logger) == []
        assert logger.warning.called

    @patch("browser_service.browser.cleanup_worker.subprocess.run")
    @patch("browser_service.browser.cleanup_worker.sys")
    def test_pgrep_no_match_does_not_warn(self, mock_sys, mock_run):
        """rc=1 with empty output is pgrep's normal 'nothing matched' — no noise."""
        mock_sys.platform = "linux"
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=1)

        from browser_service.browser.cleanup_worker import _find_chrome_orphans
        logger = MagicMock()
        assert _find_chrome_orphans(999, logger) == []
        assert not logger.warning.called


class TestMainSweepWiring:
    """main() must use the single-shot sweep and kill orphans by PID."""

    def _run_main(self, fake_psutil, orphans):
        """Run main() with argv/sleep/psutil/sweep all controlled."""
        import browser_service.browser.cleanup_worker as worker

        with patch.object(worker.sys, "argv", ["cleanup_worker.py", "999"]), \
             patch.object(worker.time, "sleep"), \
             patch.dict("sys.modules", {"psutil": fake_psutil}), \
             patch.object(worker, "_find_chrome_orphans", return_value=orphans) as mock_sweep, \
             patch.object(worker, "_setup_logging", return_value=MagicMock()):
            worker.main()
        return mock_sweep

    def _fake_psutil(self):
        """psutil stand-in whose Process(999) says 'already dead'."""
        fake = MagicMock()

        class NoSuchProcess(Exception):
            pass

        class AccessDenied(Exception):
            pass

        class TimeoutExpired(Exception):
            pass

        fake.NoSuchProcess = NoSuchProcess
        fake.AccessDenied = AccessDenied
        fake.TimeoutExpired = TimeoutExpired

        created = {}

        def proc_factory(pid):
            if pid == 999:
                raise NoSuchProcess()
            m = MagicMock()
            created[pid] = m
            return m

        fake.Process.side_effect = proc_factory
        fake._created = created
        return fake

    def test_orphans_killed_by_pid(self):
        """Each PID from the sweep gets a psutil.Process(pid).kill()."""
        fake = self._fake_psutil()
        mock_sweep = self._run_main(fake, orphans=[111, 222])

        mock_sweep.assert_called_once()
        assert mock_sweep.call_args[0][0] == 999
        assert fake._created[111].kill.called
        assert fake._created[222].kill.called

    def test_slow_process_iter_never_used(self):
        """The per-process iteration that ate the 10s timeout must be gone."""
        fake = self._fake_psutil()
        self._run_main(fake, orphans=[])

        fake.process_iter.assert_not_called()


class TestSweepLeakedProfiles:
    """Age-gated, capped sweep of leaked browser-use temp profile dirs.

    browser-use 0.12.6 mkdtemps a fresh user_data_dir per session
    (browser-use-user-data-dir-*) and its own teardown never runs because
    our cleanup kills Chrome first — ~55 leaked dirs/day, ~14 MB each
    (416 dirs / 6.1 GB measured 2026-07-15). The worker drains them:
    age-gated so a live session's profile is never touched, capped so a
    single sweep's work is bounded.
    """

    def _make_profile(self, root, name, age_s):
        d = root / name
        d.mkdir()
        (d / "Default").mkdir()
        (d / "Default" / "Cookies").write_text("x")
        old = time.time() - age_s
        os.utime(d, (old, old))
        return d

    def test_deletes_only_old_matching_dirs(self, tmp_path):
        """>24h profile dirs go; young dirs (even 23h), foreign dirs, and
        files stay — the 24h gate is the owner-approved threshold."""
        from browser_service.browser.cleanup_worker import _sweep_leaked_profiles

        old = self._make_profile(tmp_path, "browser-use-user-data-dir-aaa", 90000)
        young = self._make_profile(tmp_path, "browser-use-user-data-dir-bbb", 82800)
        foreign = self._make_profile(tmp_path, "otherapp-user-data-dir-ccc", 90000)
        prefix_file = tmp_path / "browser-use-user-data-dir-ddd"
        prefix_file.write_text("not a dir")
        old_file_time = time.time() - 90000
        os.utime(prefix_file, (old_file_time, old_file_time))

        _sweep_leaked_profiles(MagicMock(), temp_root=str(tmp_path))

        assert not old.exists()
        assert young.exists()
        assert foreign.exists()
        assert prefix_file.exists()

    def test_cap_bounds_work_oldest_first(self, tmp_path):
        """At most max_deletes dirs are attempted per sweep, oldest first."""
        from browser_service.browser.cleanup_worker import _sweep_leaked_profiles

        oldest = self._make_profile(tmp_path, "browser-use-user-data-dir-a", 260000)
        older = self._make_profile(tmp_path, "browser-use-user-data-dir-b", 173000)
        newer = self._make_profile(tmp_path, "browser-use-user-data-dir-c", 90000)

        _sweep_leaked_profiles(
            MagicMock(), temp_root=str(tmp_path), max_deletes=2
        )

        assert not oldest.exists()
        assert not older.exists()
        assert newer.exists()

    def test_deletion_failure_warns_and_continues(self, tmp_path):
        """A locked/undeletable dir must not abort the sweep."""
        import browser_service.browser.cleanup_worker as worker

        first = self._make_profile(tmp_path, "browser-use-user-data-dir-a", 260000)
        second = self._make_profile(tmp_path, "browser-use-user-data-dir-b", 173000)

        # Bind the real function BEFORE patching — the module attribute
        # resolves to the mock at call time.
        real_rmtree = worker.shutil.rmtree

        def rmtree_first_fails(path, *args, **kwargs):
            if str(path) == str(first):
                raise OSError("locked")
            return real_rmtree(path, *args, **kwargs)

        logger = MagicMock()
        with patch.object(worker.shutil, "rmtree", side_effect=rmtree_first_fails):
            worker._sweep_leaked_profiles(logger, temp_root=str(tmp_path))

        assert first.exists()
        assert not second.exists()
        assert logger.warning.called

    def test_missing_root_never_raises(self, tmp_path):
        """The sweep is a safety net, never a crash source."""
        from browser_service.browser.cleanup_worker import _sweep_leaked_profiles

        _sweep_leaked_profiles(
            MagicMock(), temp_root=str(tmp_path / "does-not-exist")
        )

    def test_main_runs_profile_sweep(self):
        """main() sweeps leaked profiles after the kill/orphan phases."""
        import browser_service.browser.cleanup_worker as worker

        fake_psutil = MagicMock()
        fake_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        fake_psutil.AccessDenied = type("AccessDenied", (Exception,), {})
        fake_psutil.TimeoutExpired = type("TimeoutExpired", (Exception,), {})
        fake_psutil.Process.side_effect = fake_psutil.NoSuchProcess()

        with patch.object(worker.sys, "argv", ["cleanup_worker.py", "999"]), \
             patch.object(worker.time, "sleep"), \
             patch.dict("sys.modules", {"psutil": fake_psutil}), \
             patch.object(worker, "_find_chrome_orphans", return_value=[]), \
             patch.object(worker, "_sweep_leaked_profiles") as mock_profile_sweep, \
             patch.object(worker, "_setup_logging", return_value=MagicMock()):
            worker.main()

        mock_profile_sweep.assert_called_once()
