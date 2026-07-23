"""
Browser management module for browser service.

This module handles browser lifecycle management including:
- Browser session creation and management
- Browser resource cleanup and process termination
- Browser process tracking and monitoring

Modules:
    session: Browser session lifecycle management
    cleanup: Browser cleanup utilities and process termination
"""

from .cleanup import capture_session_pid, cleanup_browser_resources, get_browser_process_id
from .session import BrowserSessionManager

__all__ = [
    "get_browser_process_id",
    "cleanup_browser_resources",
    "capture_session_pid",
    "BrowserSessionManager",
]
