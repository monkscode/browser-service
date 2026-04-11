"""
Browser cleanup utilities for browser service.

This module provides utilities for cleaning up browser resources and terminating
browser processes. It implements a multi-strategy approach to ensure reliable
cleanup across different browser automation scenarios.

Cleanup Strategy:
    1. Track browser process ID before closing
    2. Close connected browser gracefully
    3. Stop Playwright instance
    4. Close browser session
    5. Force kill tracked Chrome process if still running (Windows only)

The cleanup is designed to only terminate the Chrome process that was started
by the browser service, not the user's personal Chrome instances.
"""

import sys
import re
import logging
import subprocess
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


def count_chrome_processes() -> Tuple[int, List[int]]:
    """
    Count running Chrome processes and return their PIDs.
    
    Returns:
        Tuple of (count, list of PIDs)
    """
    try:
        if sys.platform.startswith('win'):
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq chrome.exe', '/FO', 'CSV', '/NH'],
                capture_output=True,
                text=True,
                timeout=5
            )
            pids = []
            for line in result.stdout.strip().split('\n'):
                if line and 'chrome.exe' in line.lower():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1].strip('"'))
                            pids.append(pid)
                        except ValueError:
                            pass
            return len(pids), pids
        else:
            result = subprocess.run(
                ['pgrep', '-f', 'chrome'],
                capture_output=True,
                text=True,
                timeout=5
            )
            pids = [int(p) for p in result.stdout.strip().split('\n') if p.strip()]
            return len(pids), pids
    except Exception as e:
        logger.warning(f"Error counting Chrome processes: {e}")
        return -1, []


# Module-level storage for CDP port and PID - set this EARLY when browser starts
_tracked_cdp_port: Optional[str] = None
_tracked_browser_pid: Optional[int] = None


def _get_pid_from_port(port: str) -> Optional[int]:
    """
    Get the browser PID from the CDP port using netstat/lsof.
    This is called IMMEDIATELY when we have the port, while Chrome is still running.
    
    Args:
        port: The CDP port number as a string
        
    Returns:
        The browser process PID, or None if not found
    """
    try:
        if sys.platform.startswith('win'):
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Check for both LISTENING and ESTABLISHED states
            for line in result.stdout.split('\n'):
                if f':{port}' in line:
                    if 'LISTENING' in line or 'ESTABLISHED' in line:
                        parts = line.split()
                        if parts:
                            pid = int(parts[-1])
                            state = 'LISTENING' if 'LISTENING' in line else 'ESTABLISHED'
                            logger.debug(f"   Found PID {pid} for port {port} ({state})")
                            return pid
        else:
            # On Linux/Mac, use lsof
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.stdout.strip():
                pid = int(result.stdout.strip().split('\n')[0])
                return pid
    except Exception as e:
        logger.debug(f"Error getting PID from port {port}: {e}")
    
    return None


def store_cdp_port(cdp_url: str) -> Optional[str]:
    """
    Store CDP port AND browser PID from URL for later cleanup use.
    Call this EARLY when CDP URL is first available and Chrome is running.
    
    This captures the PID immediately while the browser is active, before
    graceful shutdown closes the port.
    
    Args:
        cdp_url: CDP URL like ws://127.0.0.1:12345/devtools/browser/...
        
    Returns:
        The extracted port, or None if extraction failed
    """
    global _tracked_cdp_port, _tracked_browser_pid
    
    if not cdp_url:
        return None
    
    # Extract port from CDP URL
    match = re.search(r':(\d+)/', cdp_url)
    if not match:
        return None
    
    port = match.group(1)
    _tracked_cdp_port = port
    
    # CRITICAL: Get PID immediately while Chrome is still running
    pid = _get_pid_from_port(port)
    if pid:
        _tracked_browser_pid = pid
        logger.info(f"   📍 Stored CDP port {port} and PID {pid} for cleanup")
    else:
        logger.warning(f"   ⚠️ Stored CDP port {port} but could not get PID (browser may not be listening yet)")
    
    return port


def get_stored_cdp_port() -> Optional[str]:
    """Get the stored CDP port."""
    return _tracked_cdp_port


def get_stored_browser_pid() -> Optional[int]:
    """Get the stored browser PID."""
    return _tracked_browser_pid


def clear_stored_cdp_port():
    """Clear the stored CDP port and PID (call after cleanup)."""
    global _tracked_cdp_port, _tracked_browser_pid
    _tracked_cdp_port = None
    _tracked_browser_pid = None


def capture_session_pid(session) -> Optional[int]:
    """
    Capture the browser PID from a live browser session's CDP URL.

    Call this IMMEDIATELY after session.start() returns, while Chrome is
    guaranteed to be running. The CDP URL becomes unavailable after browser-use
    fires on_BrowserStopEvent (which kills the parent Chrome before agent.run()
    returns), so early capture is the only reliable moment.

    Args:
        session: Live browser session (BrowserSession from browser-use)

    Returns:
        Chrome browser process PID, or None if capture failed
    """
    try:
        cdp_url = getattr(session, 'cdp_url', None)
        if not cdp_url:
            return None
        match = re.search(r':(\d+)/', cdp_url)
        if not match:
            return None
        port = match.group(1)
        return _get_pid_from_port(port)
    except Exception as e:
        logger.debug(f"Error capturing session PID: {e}")
        return None



def get_browser_process_id(session) -> Optional[int]:
    """
    Extract the Chrome process ID from the browser session.

    This function attempts multiple strategies to find the browser process ID:
    0. Use stored PID (captured early when CDP port was stored) - FAST PATH
    1. Extract from CDP URL and use netstat (PRIMARY - works with browser-use)
    2. Check session.browser._browser_process.pid (Playwright direct)
    3. Check session.browser.process.pid (Playwright direct)
    4. Check session.browser._impl._browser_process.pid (Playwright internal)
    5. Check browser contexts for process info
    6. Check session.context.browser.process.pid

    Args:
        session: Browser session object (typically from browser-use)

    Returns:
        Process ID (PID) of the Chrome browser process, or None if not found
    """
    try:
        # FAST PATH: Use stored PID if available (captured when CDP port was stored)
        # This is reliable because the PID was captured while Chrome was running
        stored_pid = get_stored_browser_pid()
        if stored_pid:
            logger.info(f"   📍 Using STORED browser PID: {stored_pid}")
            return stored_pid
        
        # PRIMARY METHOD: Try to get from CDP endpoint (works with browser-use)
        # browser-use doesn't expose the Playwright browser object, only CDP URL
        logger.info(f"   🔍 Checking session for cdp_url...")
        has_cdp_url = hasattr(session, 'cdp_url')
        cdp_url_value = getattr(session, 'cdp_url', None) if has_cdp_url else None
        
        # Try to extract port from current cdp_url
        port = None
        if cdp_url_value:
            logger.info(f"   🔍 cdp_url value: {cdp_url_value[:60]}...")
            match = re.search(r':(\d+)/', cdp_url_value)
            if match:
                port = match.group(1)
        
        # FALLBACK: Use stored port if current cdp_url is None
        if not port:
            stored_port = get_stored_cdp_port()
            if stored_port:
                port = stored_port
                logger.info(f"   📍 Using STORED CDP port: {port} (session cdp_url was None)")
            else:
                logger.info(f"   🔍 cdp_url is None/Empty and no stored port available")
        
        if port:
            logger.info(f"   📍 Found CDP port: {port}, searching for Chrome PID...")

            # On Windows, use netstat to find PID listening on this port
            if sys.platform.startswith('win'):
                try:
                    result = subprocess.run(
                        ['netstat', '-ano'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )

                    # Check for both LISTENING and ESTABLISHED states
                    # Chrome might be connected (ESTABLISHED) rather than just listening
                    for line in result.stdout.split('\n'):
                        if f':{port}' in line:
                            if 'LISTENING' in line or 'ESTABLISHED' in line:
                                parts = line.split()
                                if parts:
                                    pid = int(parts[-1])
                                    state = 'LISTENING' if 'LISTENING' in line else 'ESTABLISHED'
                                    logger.info(f"   📍 Found browser PID via CDP port {port} ({state}): {pid}")
                                    return pid
                    
                    # Log if port not found
                    logger.info(f"   ⚠️ Port {port} not found in netstat output (browser may have closed)")
                except Exception as e:
                    logger.warning(f"   ⚠️ Error using netstat: {e}")
            else:
                # On Linux/Mac, use lsof
                try:
                    result = subprocess.run(
                        ['lsof', '-ti', f':{port}'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.stdout.strip():
                        pid = int(result.stdout.strip().split('\n')[0])
                        logger.info(f"   📍 Found browser PID via CDP port {port}: {pid}")
                        return pid
                except Exception as e:
                    logger.debug(f"   Error using lsof: {e}")

        # FALLBACK METHODS: Try to get PID from session.browser (for direct Playwright usage)
        if hasattr(session, 'browser') and session.browser:
            browser = session.browser

            # Method 1: Check for _browser_process attribute (Playwright)
            if hasattr(browser, '_browser_process') and browser._browser_process:
                pid = browser._browser_process.pid
                if pid:
                    logger.info(f"   📍 Found browser PID via _browser_process: {pid}")
                    return pid

            # Method 2: Check for process attribute (Playwright)
            if hasattr(browser, 'process') and browser.process:
                pid = browser.process.pid
                if pid:
                    logger.info(f"   📍 Found browser PID via process: {pid}")
                    return pid

            # Method 3: Check _impl.process (Playwright internal)
            if hasattr(browser, '_impl') and hasattr(browser._impl, '_browser_process'):
                pid = browser._impl._browser_process.pid
                if pid:
                    logger.info(f"   📍 Found browser PID via _impl._browser_process: {pid}")
                    return pid

            # Method 4: Check contexts for process info
            if hasattr(browser, 'contexts') and browser.contexts:
                for context in browser.contexts:
                    if hasattr(context, '_browser') and hasattr(context._browser, 'process'):
                        pid = context._browser.process.pid
                        if pid:
                            logger.info(f"   📍 Found browser PID via context: {pid}")
                            return pid

        # Try session.context
        if hasattr(session, 'context') and session.context:
            if hasattr(session.context, 'browser') and hasattr(session.context.browser, 'process'):
                pid = session.context.browser.process.pid
                if pid:
                    logger.info(f"   📍 Found browser PID via session.context: {pid}")
                    return pid

        logger.warning("   ⚠️ Could not determine browser PID - graceful close only")
        return None

    except Exception as e:
        logger.warning(f"   ⚠️ Error getting browser PID: {e}")
        return None


async def cleanup_browser_resources(
    session=None,
    connected_browser=None,
    playwright_instance=None,
    browser_pid: Optional[int] = None
) -> None:
    """
    Simple and robust browser cleanup.

    This function performs a comprehensive cleanup of browser resources:
    1. Resolves browser PID (uses provided value, or derives from session as fallback)
    2. Closes connected browser gracefully
    3. Stops Playwright instance
    4. Closes browser session
    5. Kills the parent Chrome process if still alive, then kills any orphaned
       child processes by PPID (psutil-based, cross-platform)

    The cleanup only kills Chrome processes belonging to this specific session.
    User's personal Chrome instances and other sessions are never affected because
    the PPID filter is scoped to the exact PID of this session's parent process.

    Args:
        session: Browser session object to clean up
        connected_browser: Connected browser instance to close
        playwright_instance: Playwright instance to stop
        browser_pid: PID of the Chrome browser process started for this session.
                     Pass this from workflow.py (captured right after session.start())
                     for reliable early capture. Falls back to session-derived detection
                     if not provided.

    Returns:
        None
    """
    logger.info("🧹 Starting browser cleanup...")
    
    # Count Chrome processes BEFORE cleanup
    before_count, before_pids = count_chrome_processes()
    logger.info(f"   📊 Chrome processes BEFORE cleanup: {before_count}")

    # Resolve browser PID: use the value provided by the caller (captured early,
    # while Chrome was guaranteed alive) or fall back to session-derived detection.
    if browser_pid is not None:
        logger.info(f"   📍 Using provided browser PID: {browser_pid}")
    elif session:
        browser_pid = get_browser_process_id(session)
        if browser_pid:
            logger.warning(
                f"   ⚠️ Using fallback PID detection (derived from session): {browser_pid}. "
                f"This may be inaccurate during concurrent task execution."
            )
        else:
            logger.debug("   ⚠️ Could not track browser PID - will use graceful close only")

    # Step 1: Close connected browser
    if connected_browser:
        try:
            await connected_browser.close()
            logger.info("   ✅ Connected browser closed")
        except Exception:
            pass

    # Step 2: Stop playwright instance
    if playwright_instance:
        try:
            await playwright_instance.stop()
            logger.info("   ✅ Playwright stopped")
        except Exception:
            pass

    # Step 3: Close session gracefully
    # browser-use uses kill() method for cleanup
    if session:
        try:
            if hasattr(session, 'kill'):
                await session.kill()
            elif hasattr(session, 'close'):
                await session.close()
            elif hasattr(session, 'browser') and session.browser:
                await session.browser.close()
            logger.info("   ✅ Session closed")
        except Exception:
            pass

    # Step 4: Hard-kill Chrome in a subprocess and AWAIT its result.
    # CRITICAL: proc.kill() on Windows can crash the calling process (Flask) when called
    # from within the async event loop. Running the hard kill in a completely separate OS
    # process protects Flask:
    #   - If the subprocess crashes → only it dies, Flask and tasks_dict survive
    #   - Flask can continue serving /query requests with the result already stored
    #
    # We use asyncio.create_subprocess_exec (instead of a detached Popen) so that
    # we can await the subprocess result and get verified confirmation that the kill
    # actually ran — while still keeping the kill isolated in a separate OS process.
    #
    # IMPORTANT: We use the absolute file path to cleanup_worker.py (same directory as
    # this file) rather than `-m module.path` because sys.path is NOT inherited by
    # subprocesses — only environment variables are. Absolute path works unconditionally.
    if browser_pid:
        try:
            import asyncio as _asyncio
            import os as _os
            _worker_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "cleanup_worker.py")
            # Pass the log path so cleanup_worker writes into the mounted logs/ dir
            _log_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "logs")
            _log_path = _os.path.normpath(_os.path.join(_log_dir, "cleanup_worker.log"))
            _env = {**_os.environ, "CLEANUP_WORKER_LOG": _log_path}

            kwargs = {}
            if sys.platform.startswith("win"):
                # CREATE_NO_WINDOW prevents a console flash on Windows
                kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

            proc = await _asyncio.create_subprocess_exec(
                sys.executable, _worker_path, str(browser_pid),
                stdout=_asyncio.subprocess.DEVNULL,
                stderr=_asyncio.subprocess.DEVNULL,
                env=_env,
                **kwargs,
            )
            logger.info(
                f"   🚀 Chrome hard-kill subprocess started (browser PID {browser_pid}), awaiting result..."
            )
            try:
                await _asyncio.wait_for(proc.wait(), timeout=10.0)
                if proc.returncode == 0:
                    logger.info(f"   ✅ Cleanup worker finished successfully (rc=0)")
                else:
                    logger.warning(f"   ⚠️ Cleanup worker exited with rc={proc.returncode}")
            except _asyncio.TimeoutError:
                logger.warning("   ⚠️ Cleanup worker timed out after 10s — killing it")
                proc.kill()
                try:
                    # Reap the process so the OS cleans up the handle.
                    # Process was just SIGKILL'd so this returns almost immediately.
                    await _asyncio.wait_for(proc.wait(), timeout=2.0)
                except Exception:
                    pass  # Best-effort reap; process is already signalled dead
        except Exception as e:
            logger.warning(f"   ⚠️ Could not spawn cleanup subprocess: {e}")
    else:
        logger.warning("   ⚠️ No browser PID tracked — cannot force kill (graceful close only)")

    # Count Chrome processes AFTER cleanup (subprocess kill has now completed).
    # Report the delta vs before so accumulating orphans are immediately visible.
    # Delta is only meaningful when both counts succeeded (count >= 0).
    after_count, after_pids = count_chrome_processes()
    logger.info(f"   📊 Chrome processes AFTER cleanup: {after_count}")
    if before_count >= 0 and after_count >= 0:
        still_alive = set(before_pids) & set(after_pids)
        new_orphans = set(after_pids) - set(before_pids)
        if still_alive:
            logger.warning(f"   ⚠️ {len(still_alive)} pre-existing Chrome PID(s) still alive: {sorted(still_alive)}")
        if new_orphans:
            logger.error(f"   ❌ {len(new_orphans)} new orphaned Chrome PID(s) appeared: {sorted(new_orphans)}")
        if not still_alive and not new_orphans:
            logger.info("   ✅ No unexpected Chrome processes remain")

    # Only clear global CDP tracking state when the stored PID belongs to this task.
    # These globals are shared across concurrent tasks; clearing unconditionally can
    # wipe state that another task still needs as a fallback (e.g. when browser_pid
    # was not captured early and get_browser_process_id() relies on _tracked_browser_pid).
    stored_pid = get_stored_browser_pid()
    if stored_pid is not None and browser_pid is not None and browser_pid == stored_pid:
        clear_stored_cdp_port()
        logger.debug("   Cleared global CDP port tracking (owned by this task)")
    elif stored_pid is None:
        logger.debug("   Global CDP port tracking already clear")
    else:
        logger.debug(
            f"   Skipping global CDP clear — stored PID {stored_pid} "
            f"does not match this task's PID {browser_pid}"
        )

    logger.info("🧹 Cleanup complete")
