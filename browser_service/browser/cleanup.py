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
    playwright_instance=None
) -> None:
    """
    Simple and robust browser cleanup.

    This function performs a comprehensive cleanup of browser resources:
    1. Tracks the browser process ID before closing
    2. Closes connected browser gracefully
    3. Stops Playwright instance
    4. Closes browser session
    5. Force kills the tracked Chrome process if still running (Windows only)

    The cleanup only kills the Chrome process that was started by this service,
    not the user's personal Chrome instances.

    Args:
        session: Browser session object to clean up
        connected_browser: Connected browser instance to close
        playwright_instance: Playwright instance to stop

    Returns:
        None
    """
    logger.info("🧹 Starting browser cleanup...")
    
    # Count Chrome processes BEFORE cleanup
    before_count, before_pids = count_chrome_processes()
    logger.info(f"   📊 Chrome processes BEFORE cleanup: {before_count}")

    # Get the PID of OUR Chrome process before closing
    browser_pid = None
    if session:
        browser_pid = get_browser_process_id(session)
        if browser_pid:
            logger.info(f"   📍 Tracked browser PID: {browser_pid}")
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
    # browser-use 0.9.x uses kill() method for cleanup
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

    # Step 4: Force kill ONLY our Chrome process if it's still running
    if browser_pid:
        if sys.platform.startswith('win'):
            try:
                import subprocess

                # Check if our specific Chrome process is still running
                result = subprocess.run(
                    ['tasklist', '/FI', f'PID eq {browser_pid}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if 'chrome.exe' in result.stdout.lower():
                    logger.warning(f"   ⚠️ Chrome process {browser_pid} still running after graceful close")
                    logger.info(f"   🔨 Force killing Chrome PID {browser_pid} and its children...")

                    # Kill only our specific Chrome process and its children
                    # /T flag kills the process tree (parent + all children)
                    subprocess.run(
                        ['taskkill', '/F', '/PID', str(browser_pid), '/T'],
                        capture_output=True,
                        timeout=5
                    )

                    # Wait a moment for processes to terminate
                    import time
                    time.sleep(0.5)

                    # Verify it's gone
                    result = subprocess.run(
                        ['tasklist', '/FI', f'PID eq {browser_pid}'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )

                    if 'chrome.exe' not in result.stdout.lower():
                        logger.info(f"   ✅ Chrome process {browser_pid} and children terminated")
                    else:
                        logger.warning(f"   ⚠️ Chrome process {browser_pid} may still be running")
                else:
                    logger.info(f"   ✅ Chrome process {browser_pid} already terminated")

            except Exception as e:
                logger.warning(f"   ⚠️ Error during force kill: {e}")
        else:
            # Linux/Mac cleanup
            try:
                import subprocess
                
                # Try to kill the process group
                logger.info(f"   🔨 Terminating Chrome PID {browser_pid}...")
                subprocess.run(['kill', '-TERM', str(browser_pid)], timeout=5)
                
                # Wait a moment
                import time
                time.sleep(0.5)
                
                # Check if still running, force kill if needed
                try:
                    subprocess.run(['kill', '-0', str(browser_pid)], timeout=5, check=True)
                    # Still running, force kill
                    logger.warning(f"   ⚠️ Chrome process {browser_pid} still running, force killing...")
                    subprocess.run(['kill', '-KILL', str(browser_pid)], timeout=5)
                    logger.info(f"   ✅ Chrome process {browser_pid} force killed")
                except subprocess.CalledProcessError:
                    # Process doesn't exist anymore
                    logger.info(f"   ✅ Chrome process {browser_pid} terminated")
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Error during force kill: {e}")
    else:
        logger.warning("   ⚠️ No browser PID tracked - cannot force kill (graceful close only)")

    # Count Chrome processes AFTER cleanup
    after_count, after_pids = count_chrome_processes()
    logger.info(f"   📊 Chrome processes AFTER cleanup: {after_count}")
    
    # Compare before and after
    if before_count > 0 and after_count >= 0:
        diff = before_count - after_count
        if diff > 0:
            logger.info(f"   ✅ Cleanup reduced Chrome processes by {diff} (from {before_count} to {after_count})")
        elif diff == 0:
            logger.info(f"   ℹ️ Chrome process count unchanged ({before_count})")
        else:
            logger.warning(f"   ⚠️ Chrome processes increased by {-diff} (from {before_count} to {after_count})")

    # Clear stored CDP port for next session
    clear_stored_cdp_port()
    
    logger.info("🧹 Cleanup complete")
