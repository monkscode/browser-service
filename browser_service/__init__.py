"""
Browser Service Package

This package contains modular components for the browser automation service,
extracted from the monolithic browser_use_service.py file.

Modules:
    - config: Configuration management (BatchConfig, LocatorConfig, LLMConfig)
    - utils: Shared utility functions (JSON parsing, metrics, logging)
    - browser: Browser lifecycle and resource management
    - locators: Locator generation, validation, and extraction
    - prompts: Prompt building for AI agents
    - agent: Custom action definitions and registration
    - tasks: Task processing and workflow execution
    - api: HTTP API endpoints and request handling

Usage:
    # Import global config instance
    from browser_service import config

    # Import specific components
    from browser_service.config import BrowserServiceConfig, BatchConfig, LocatorConfig, LLMConfig
    from browser_service.browser import BrowserSessionManager, cleanup_browser_resources
    from browser_service.locators import validate_locator_playwright
    from browser_service.prompts import build_workflow_prompt, build_system_prompt
    from browser_service.agent import find_unique_locator_action, register_custom_actions
    from browser_service.tasks import process_workflow_task, TaskProcessor
    from browser_service.api import register_routes
    from browser_service.utils import setup_logging, record_workflow_metrics
"""

import logging
import logging.handlers
import os
import sys

# ---------------------------------------------------------------------------
# Structured Logging — configures structlog with JSON or console output.
# LOG_FORMAT=console → human-readable; unset → JSON (production default).
# Must run before any logging calls and before setup_logging() in the entry
# point is imported, so this block owns the root logger configuration.
# ---------------------------------------------------------------------------
_LOG_FILE = os.path.join(os.environ.get("BROWSER_USE_LOG_DIR", "logs"), "browser_use.log")
_LOG_MAX_BYTES = 50 * 1024 * 1024  # 50 MB
_LOG_BACKUP_COUNT = 7  # 7 backups = 350 MB max

try:
    import structlog

    _shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    structlog.configure(
        processors=_shared_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    _log_format = os.environ.get("LOG_FORMAT", "").lower()
    _renderer = (
        structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
        if _log_format == "console"
        else structlog.processors.JSONRenderer()
    )
    _formatter = structlog.stdlib.ProcessorFormatter(
        processor=_renderer,
        foreign_pre_chain=_shared_processors[:-1],  # exclude wrap_for_formatter
    )

    # Ensure UTF-8 on Windows without replacing sys.stdout: reconfigure() mutates
    # the existing object in-place (Python 3.7+), preserving references held by
    # test frameworks (pytest capsys), log handlers, or embedded hosts.
    if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (ValueError, AttributeError):
            pass

    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setFormatter(_formatter)

    _root = logging.getLogger()
    _root.handlers.clear()
    _root.addHandler(_console_handler)
    _root.setLevel(logging.INFO)

    try:
        os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)
        _file_handler: logging.Handler = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        _file_handler.setFormatter(_formatter)
        _root.addHandler(_file_handler)
    except OSError as _e:
        # Cannot create log dir or open file — stdout-only, warning is JSON-formatted
        logging.getLogger(__name__).warning(
            "Cannot open %s, falling back to stdout only: %s", _LOG_FILE, _e
        )

    # Suppress noisy third-party loggers
    for _noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

except ImportError:
    # structlog not installed — fall back to stdlib default format
    logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# LLM Observability — only when OTLP endpoint is explicitly configured.
# Default is none: the NL-to-RF service owns trace storage (SQLite backend).
# Set OBSERVABILITY_BACKEND=otlp and OTLP_ENDPOINT=http://tempo:4318 when
# running with the Grafana stack (docker-compose.grafana.yml).
# ---------------------------------------------------------------------------
_obs_backend = os.environ.get("OBSERVABILITY_BACKEND", "none").lower()
_otlp_endpoint = os.environ.get("OTLP_ENDPOINT", "").strip()

if _obs_backend != "none" and _otlp_endpoint:
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from traceloop.sdk import Traceloop

        os.environ.setdefault("TRACELOOP_TRACE_CONTENT", "true")
        Traceloop.init(
            app_name="mark1-browser-service",
            exporter=OTLPSpanExporter(endpoint=f"{_otlp_endpoint.rstrip('/')}/v1/traces"),
            traceloop_sync_enabled=False,
        )
        logging.getLogger(__name__).info(
            "[OBSERVABILITY] OpenLLMetry initialized — endpoint=%s", _otlp_endpoint
        )
    except ImportError:
        pass  # traceloop-sdk not installed — no tracing
    except Exception as e:
        logging.getLogger(__name__).warning("[OBSERVABILITY] Init failed (non-fatal): %s", e)

# Single source of truth for the version: the installed package metadata
# (pyproject [project].version). Falls back when running from an uninstalled
# source checkout.
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("browser-service")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# Import key components for convenient access
from browser_service.config import (
    BatchConfig,
    BrowserServiceConfig,
    LLMConfig,
    LocatorConfig,
    config,
)

__all__ = [
    # Version
    "__version__",
    # Configuration
    "config",
    "BrowserServiceConfig",
    "BatchConfig",
    "LocatorConfig",
    "LLMConfig",
]
