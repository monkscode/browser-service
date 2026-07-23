"""
Standalone cleanup worker — runs as a subprocess.

Kills a Chrome process by PID and its orphaned children.
Called from cleanup.py's cleanup_browser_resources() so that
Chrome hard-kills cannot crash the Flask parent process.

Output is written to the path given by the CLEANUP_WORKER_LOG env var
(set by the parent to point into the mounted logs/ directory).
Falls back to cleanup_worker.log in cwd if the env var is not set.

Usage:
    python /abs/path/to/cleanup_worker.py <browser_pid>
"""

import csv
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time

# Leaked browser-use temp profiles (see _sweep_leaked_profiles).
PROFILE_SWEEP_PREFIX = "browser-use-user-data-dir-"
PROFILE_SWEEP_MAX_AGE_S = 86400  # >24h only (owner-approved gate) — never a live session
PROFILE_SWEEP_MAX_DELETES = 20  # bounded work per sweep; ~55 dirs/day leak vs 1 sweep per task


def _setup_logging() -> logging.Logger:
    """
    Set up logging to a file in the parent process's working directory.
    Falls back to stderr if the log file cannot be opened.
    We deliberately do NOT use DEVNULL — silent failures must be visible.
    """
    log_path = os.environ.get(
        "CLEANUP_WORKER_LOG",
        os.path.join(os.getcwd(), "cleanup_worker.log"),
    )
    handlers = []

    try:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    except OSError:
        handlers.append(logging.StreamHandler(sys.stderr))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s [cleanup_worker] %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def _find_chrome_orphans(browser_pid: int, logger: logging.Logger) -> list:
    """
    Find chrome-named children of browser_pid with ONE process-table query.

    psutil.process_iter(['pid','ppid','name']) opens a handle per process on
    Windows — measured 13.3s for ~456 processes, past the parent's 10s wait
    in cleanup.py, so the worker was killed mid-sweep on every generation.
    A single CIM/pgrep query returns the same PID-scoped set in well under
    the timeout. Kill semantics are unchanged: only chrome-named processes
    whose PPID is this session's Chrome.

    Returns a list of PIDs; empty on any failure (the sweep is a safety net,
    never a crash source).
    """
    try:
        if sys.platform.startswith("win"):
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    f"Get-CimInstance Win32_Process -Filter 'ParentProcessId={int(browser_pid)}' | "
                    "Select-Object ProcessId,Name | ConvertTo-Csv -NoTypeInformation",
                ],
                capture_output=True,
                text=True,
                timeout=8,
            )
            if result.returncode != 0:
                # A failed query must not masquerade as "no orphans found" —
                # that silently disables the safety net (the pre-87fa0f8 bug).
                logger.warning(
                    f"   ⚠️ Orphan sweep query failed (rc={result.returncode}): "
                    f"{(result.stderr or '').strip()[:200]}"
                )
                return []
            pids = []
            for row in csv.DictReader(io.StringIO(result.stdout)):
                name = (row.get("Name") or "").lower()
                pid_str = row.get("ProcessId")
                if pid_str and "chrome" in name:
                    try:
                        pids.append(int(pid_str))
                    except ValueError:
                        pass
            return pids
        else:
            # pgrep matches the pattern against the process name, same
            # "chrome" substring semantics as the old psutil filter.
            result = subprocess.run(
                ["pgrep", "-P", str(int(browser_pid)), "chrome"],
                capture_output=True,
                text=True,
                timeout=8,
            )
            # pgrep rc=1 means "no processes matched" — a normal empty sweep.
            # rc>=2 is a real failure (syntax/fatal) and must be visible.
            if result.returncode not in (0, 1):
                logger.warning(
                    f"   ⚠️ Orphan sweep query failed (rc={result.returncode}): "
                    f"{(result.stderr or '').strip()[:200]}"
                )
                return []
            return [int(p) for p in result.stdout.split() if p.strip().isdigit()]
    except Exception as e:
        logger.warning(f"   ⚠️ Orphan sweep query failed: {e}")
        return []


def _sweep_leaked_profiles(
    logger: logging.Logger,
    temp_root: str = None,
    max_age_s: int = PROFILE_SWEEP_MAX_AGE_S,
    max_deletes: int = PROFILE_SWEEP_MAX_DELETES,
) -> None:
    """
    Delete leaked browser-use temp profile dirs, oldest first.

    browser-use 0.12.6 defaults user_data_dir=None → mkdtemp per session
    (browser-use-user-data-dir-*), and its own teardown never runs because
    cleanup.py hard-kills Chrome first — ~55 leaked dirs/day at ~14 MB each
    (416 dirs / 6.1 GB measured 2026-07-15).

    Safety: age-gated on the dir's own mtime (a live session's profile is
    being written and stays young; after Chrome dies the mtime freezes), and
    capped per sweep so one worker invocation does bounded work. Deletion
    attempts count toward the cap — a locked dir warns and the sweep moves
    on. Like the orphan sweep, this is a safety net, never a crash source.
    """
    try:
        root = temp_root or tempfile.gettempdir()
        now = time.time()
        candidates = []
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.name.startswith(PROFILE_SWEEP_PREFIX):
                    continue
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    mtime = entry.stat(follow_symlinks=False).st_mtime
                except OSError:
                    continue
                if now - mtime > max_age_s:
                    candidates.append((mtime, entry.path))

        candidates.sort()
        deleted = 0
        failed = 0
        for _mtime, path in candidates[:max_deletes]:
            try:
                shutil.rmtree(path)
                deleted += 1
            except OSError as e:
                failed += 1
                logger.warning(f"   ⚠️ Could not delete leaked profile {path}: {e}")

        remaining = len(candidates) - deleted - failed
        if candidates:
            logger.info(
                f"   🧹 Leaked profile sweep: {deleted} deleted, {failed} failed, "
                f"{remaining} eligible left for future sweeps"
            )
        else:
            logger.info(f"   ✅ No leaked browser-use profiles older than {max_age_s}s found")
    except Exception as e:
        logger.warning(f"   ⚠️ Leaked profile sweep failed: {e}")


def main():
    logger = _setup_logging()

    if len(sys.argv) < 2:
        logger.error("Usage: cleanup_worker <browser_pid>")
        sys.exit(1)

    try:
        browser_pid = int(sys.argv[1])
    except ValueError:
        logger.error(f"Invalid PID argument: {sys.argv[1]!r}")
        sys.exit(1)

    logger.info(f"🧹 Cleanup worker started for Chrome PID {browser_pid}")

    try:
        import psutil
    except ImportError:
        logger.warning("psutil not available — cannot force kill Chrome process")
        logger.info("🧹 Cleanup worker complete (no psutil)")
        return

    # Small grace period — give session.kill() / graceful close a moment to finish
    # before we check whether the process is still alive.
    time.sleep(1)

    # -----------------------------------------------------------------------
    # Kill parent Chrome process if still alive
    # -----------------------------------------------------------------------
    try:
        proc = psutil.Process(browser_pid)
        if proc.is_running():
            logger.info(f"   ⚠️ Chrome PID {browser_pid} still running — sending SIGKILL")
            proc.kill()
            try:
                proc.wait(timeout=5)
                logger.info(f"   ✅ Chrome PID {browser_pid} confirmed dead")
            except psutil.TimeoutExpired:
                logger.warning(f"   ⚠️ Chrome PID {browser_pid} did not die within 5s after SIGKILL")
        else:
            logger.info(f"   ✅ Chrome PID {browser_pid} not running (already terminated)")
    except psutil.NoSuchProcess:
        logger.info(f"   ✅ Chrome PID {browser_pid} already terminated (NoSuchProcess)")
    except psutil.AccessDenied as e:
        logger.warning(f"   ⚠️ Access denied killing Chrome PID {browser_pid}: {e}")
    except Exception as e:
        logger.warning(f"   ⚠️ Unexpected error killing Chrome PID {browser_pid}: {e}")

    # -----------------------------------------------------------------------
    # Kill orphaned Chrome children by PPID
    # Scoped to this session's parent PID only — safe for multi-session use.
    # On Windows, PPID is preserved in the process table even after the parent
    # exits, so orphaned children are still findable by their recorded PPID.
    # -----------------------------------------------------------------------
    try:
        orphans = _find_chrome_orphans(browser_pid, logger)
        if orphans:
            logger.info(
                f"   ⚠️ Found {len(orphans)} orphaned Chrome child(ren) with PPID {browser_pid}"
            )
            for orphan_pid in orphans:
                try:
                    psutil.Process(orphan_pid).kill()
                    logger.info(f"   ✅ Killed orphaned Chrome PID {orphan_pid}")
                except psutil.NoSuchProcess:
                    pass  # Already gone — not an error
                except psutil.AccessDenied as e:
                    logger.warning(f"   ⚠️ Access denied killing orphan PID {orphan_pid}: {e}")
        else:
            logger.info(f"   ✅ No orphaned Chrome children found for PPID {browser_pid}")
    except Exception as e:
        logger.warning(f"   ⚠️ Error during orphan cleanup: {e}")

    # -----------------------------------------------------------------------
    # Drain leaked browser-use temp profiles (age-gated, capped per sweep).
    # Runs last: this session's Chrome is dead by now, so nothing we could
    # delete here is still being written by us.
    # -----------------------------------------------------------------------
    _sweep_leaked_profiles(logger)

    logger.info("🧹 Cleanup worker complete")


if __name__ == "__main__":
    main()
