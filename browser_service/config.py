"""
Configuration management for Browser Service.

This module centralizes all configuration settings including:
- Batch processing configuration
- Locator extraction configuration
- LLM configuration (Gemini Developer API and Vertex AI)
- Feature flags

All configuration values can be overridden via environment variables.

LLM provider selection:
  MODEL_PROVIDER=gemini  → Gemini Developer API (GEMINI_API_KEY required)
  MODEL_PROVIDER=vertex  → Vertex AI via service account JSON key
                           (VERTEXAI_CREDENTIALS, VERTEXAI_PROJECT, VERTEXAI_LOCATION required)
  MODEL_PROVIDER=local   → not supported; browser-service requires a Google vision model

Vertex AI credentials are loaded once at startup in BrowserServiceConfig.__init__()
and cached as config.llm.vertexai_credentials. workflow.py reads the cached object
directly — no file I/O occurs per request.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Batch processing configuration settings."""

    # Stop agent after N steps
    max_agent_steps: int = 15

    # Retry element N times before skipping
    max_retries_per_element: int = 2

    # Max time per element (seconds)
    element_timeout: int = 120


@dataclass
class LocatorConfig:
    """Locator extraction configuration settings."""

    # Content-based search (finds element by visible text)
    # Try content search N times
    content_based_retries: int = 7

    # Coordinate-based search (finds element by screen position)
    # Higher count for coordinate fallback
    coordinate_based_retries: int = 7

    # Element type fallback (finds first visible element of type)
    # Last resort
    element_type_retries: int = 5

    # Coordinate offset attempts (try nearby coordinates if first fails)
    # Try N different offsets
    coordinate_offset_attempts: int = 7

    # Budget (seconds) for one find_unique_locator call — the whole strategy
    # cascade shares it. On expiry the finder returns a structured
    # found=False result instead of hanging the agent step.
    custom_action_timeout: int = 5

    # Coordinate offsets to try (pixels)
    coordinate_offsets: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"x": 100, "y": 0, "reason": "escape sidebar/left panel"},
            {"x": 200, "y": 0, "reason": "escape wide sidebar"},
            {"x": 50, "y": 0, "reason": "slight right adjustment"},
            {"x": 0, "y": 20, "reason": "move down slightly"},
            {"x": 100, "y": 20, "reason": "diagonal adjustment"},
        ]
    )


@dataclass
class LLMConfig:
    """LLM (Language Model) configuration settings."""

    model_provider: str = "vertex"  # "gemini" | "vertex" | "local"
    google_api_key: str = ""
    google_model: str = "gemini-2.5-flash"

    # Vertex AI fields — only meaningful when model_provider == "vertex"
    vertexai_project: str = ""
    vertexai_location: str = ""
    vertexai_credentials_path: str = ""
    # Loaded once at startup from vertexai_credentials_path.
    # google-auth handles OAuth2 token refresh on this object automatically.
    vertexai_credentials: object | None = None


class BrowserServiceConfig:
    """
    Centralized configuration for Browser Service.

    Loads configuration from environment variables with sensible defaults.
    Provides validation to ensure configuration is valid before service starts.
    """

    def __init__(self):
        """Initialize configuration from environment variables."""

        # Collect non-numeric env var errors during __init__ so validate() can report
        # them as friendly messages instead of letting int() raise ValueError and abort
        # the import before validation runs.
        self._parse_errors: list[str] = []

        # Batch processing configuration
        self.batch = BatchConfig(
            max_agent_steps=self._int_env("MAX_AGENT_STEPS", 15),
            max_retries_per_element=self._int_env("MAX_RETRIES_PER_ELEMENT", 2),
            element_timeout=self._int_env("ELEMENT_TIMEOUT", 120),
        )

        # Locator extraction configuration
        self.locator = LocatorConfig(
            content_based_retries=self._int_env("CONTENT_BASED_RETRIES", 7),
            coordinate_based_retries=self._int_env("COORDINATE_BASED_RETRIES", 7),
            element_type_retries=self._int_env("ELEMENT_TYPE_RETRIES", 5),
            coordinate_offset_attempts=self._int_env("COORDINATE_OFFSET_ATTEMPTS", 7),
            custom_action_timeout=self._int_env("CUSTOM_ACTION_TIMEOUT", 5),
        )

        # LLM configuration
        # Determine model provider: env var → NL repo settings → default "vertex"
        model_provider = os.getenv("MODEL_PROVIDER", "")
        if not model_provider:
            try:
                from src.backend.core.config import settings

                model_provider = settings.MODEL_PROVIDER or "vertex"
            except (ImportError, AttributeError):
                model_provider = "vertex"
        model_provider = model_provider.lower()

        # Gemini API key (only needed for model_provider == "gemini")
        google_api_key = os.getenv("GEMINI_API_KEY", "")
        if not google_api_key:
            try:
                from src.backend.core.config import settings

                google_api_key = settings.GEMINI_API_KEY or ""
            except (ImportError, AttributeError):
                pass

        # Vertex AI configuration (only needed for model_provider == "vertex")
        vertexai_project = os.getenv("VERTEXAI_PROJECT", "")
        vertexai_location = os.getenv("VERTEXAI_LOCATION", "")
        vertexai_credentials_path = os.getenv("VERTEXAI_CREDENTIALS", "")
        if not vertexai_project or not vertexai_location:
            try:
                from src.backend.core.config import settings

                vertexai_project = (
                    vertexai_project or getattr(settings, "VERTEXAI_PROJECT", "") or ""
                )
                vertexai_location = (
                    vertexai_location or getattr(settings, "VERTEXAI_LOCATION", "") or ""
                )
                # VERTEXAI_CREDENTIALS is not a Settings field — it is a raw env var loaded
                # into os.environ by load_dotenv("src/backend/.env"). os.getenv() above already
                # captures it; no settings fallback is needed.
            except (ImportError, AttributeError):
                pass

        self.llm = LLMConfig(
            model_provider=model_provider,
            google_api_key=google_api_key,
            google_model=self._get_google_model(),
            vertexai_project=vertexai_project,
            vertexai_location=vertexai_location,
            vertexai_credentials_path=vertexai_credentials_path,
        )

        # Load Vertex AI credentials once at startup and cache on the config singleton.
        # workflow.py reads config.llm.vertexai_credentials directly — no file I/O per request.
        if model_provider == "vertex" and vertexai_credentials_path:
            try:
                from google.oauth2 import service_account

                self.llm.vertexai_credentials = (
                    service_account.Credentials.from_service_account_file(
                        vertexai_credentials_path,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                )
                logger.info(f"🔑 Vertex AI credentials loaded from: {vertexai_credentials_path}")
            except Exception as e:
                # validate() will surface this clearly. Log now so the error is visible
                # in startup output before validate() is called.
                logger.error(f"❌ Failed to load Vertex AI credentials: {e}")

        # Robot Framework library type. Browser Library (Playwright) is the only
        # supported target: the locator engine emits Playwright-only syntax
        # (role=, text=, >>> iframe piercing), so any other value would produce
        # valid-looking tests that fail on every step. Fail fast at construction —
        # validate() is not called on the production startup path.
        self.robot_library = os.getenv("ROBOT_LIBRARY", "browser")
        if self.robot_library == "selenium":
            raise ValueError(
                "ROBOT_LIBRARY=selenium is no longer supported; this service "
                "generates Browser Library (Playwright) locators only. Remove "
                "the setting or set ROBOT_LIBRARY=browser."
            )
        if self.robot_library != "browser":
            raise ValueError(f"ROBOT_LIBRARY must be 'browser', got '{self.robot_library}'")

        # Browser headless mode
        # When true: Browser runs without UI (faster, for CI/CD)
        # When false: Browser UI visible (for debugging/development)
        self.headless = os.getenv("BROWSER_HEADLESS", "true").lower() == "true"

        # Agent vision mode (Task 28 A1-INLINE):
        #   auto — vision-off; the screenshot attaches to the next LLM call only
        #          when find_unique_locator fails validation (in-run escalation)
        #   on   — full vision on every step (escape hatch)
        # Any other value would silently fork bench vs production behavior, so
        # construction fails fast — same contract as ROBOT_LIBRARY above.
        self.agent_vision_mode = os.getenv("AGENT_VISION_MODE", "auto").lower()
        if self.agent_vision_mode not in ("auto", "on"):
            raise ValueError(
                f"AGENT_VISION_MODE must be 'auto' or 'on', got '{self.agent_vision_mode}'"
            )

        # Feature flags
        self.enable_custom_actions = os.getenv("ENABLE_CUSTOM_ACTIONS", "true").lower() == "true"

        # Concurrency configuration
        # Controls how many browser automation tasks can run simultaneously.
        # Each task spawns a headless Chrome instance (~250MB RAM).
        # Default 10 supports 5-10 concurrent users on a 32GB machine.
        self.max_concurrent_tasks = self._int_env("MAX_CONCURRENT_TASKS", 10)

    def _int_env(self, name: str, default: int) -> int:
        """Parse an integer environment variable.

        Returns the default value on non-numeric input and records a parse error
        so validate() can surface a clear message rather than crashing the import.
        """
        raw = os.getenv(name, str(default))
        try:
            return int(raw)
        except ValueError:
            self._parse_errors.append(f"{name} must be an integer, got '{raw}'")
            return default

    def _get_google_model(self) -> str:
        """
        Get Google model name from environment or settings.
        Strips any provider prefix so ChatGoogle receives a bare model name.

        Examples:
            "gemini/gemini-2.5-flash"    → "gemini-2.5-flash"
            "vertex_ai/gemini-2.5-flash" → "gemini-2.5-flash"
            "gemini-2.5-flash"           → "gemini-2.5-flash"
        """
        try:
            from src.backend.core.config import settings

            model = settings.ONLINE_MODEL if hasattr(settings, "ONLINE_MODEL") else None
            if model:
                return model.split("/", 1)[-1] if "/" in model else model
        except ImportError:
            pass

        model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
        return model.split("/", 1)[-1] if "/" in model else model

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of error messages. Empty list if configuration is valid.
        """
        errors = list(
            getattr(self, "_parse_errors", [])
        )  # include any non-numeric parse errors from __init__

        # Validate LLM configuration (provider-specific)
        if self.llm.model_provider == "gemini":
            if not self.llm.google_api_key:
                errors.append("GEMINI_API_KEY environment variable is not set")
        elif self.llm.model_provider == "vertex":
            if not self.llm.vertexai_credentials_path:
                errors.append("VERTEXAI_CREDENTIALS environment variable is not set")
            elif not os.path.exists(self.llm.vertexai_credentials_path):
                errors.append(
                    f"Vertex AI credentials file not found: {self.llm.vertexai_credentials_path}"
                )
            elif self.llm.vertexai_credentials is None:
                errors.append(
                    f"Vertex AI credentials failed to load from: {self.llm.vertexai_credentials_path} "
                    "(check logs for details)"
                )
            if not self.llm.vertexai_project:
                errors.append("VERTEXAI_PROJECT environment variable is not set")
            if not self.llm.vertexai_location:
                errors.append("VERTEXAI_LOCATION environment variable is not set")
        elif self.llm.model_provider == "local":
            # browser-service requires a Google vision model — local/Ollama is not supported.
            # Raising an error here gives a clear startup message rather than a cryptic
            # ChatGoogle failure at the first workflow request.
            errors.append(
                "MODEL_PROVIDER=local is not supported by browser-service. "
                "Browser-service requires a Google vision model (gemini or vertex)."
            )
        else:
            errors.append(
                f"MODEL_PROVIDER must be 'gemini', 'vertex', or 'local', "
                f"got '{self.llm.model_provider}'"
            )

        # Validate batch configuration
        if self.batch.max_agent_steps < 1:
            errors.append(f"MAX_AGENT_STEPS must be >= 1, got {self.batch.max_agent_steps}")

        if self.batch.max_retries_per_element < 0:
            errors.append(
                f"MAX_RETRIES_PER_ELEMENT must be >= 0, got {self.batch.max_retries_per_element}"
            )

        if self.batch.element_timeout < 1:
            errors.append(f"ELEMENT_TIMEOUT must be >= 1, got {self.batch.element_timeout}")

        # Validate locator configuration
        if self.locator.content_based_retries < 0:
            errors.append(
                f"CONTENT_BASED_RETRIES must be >= 0, got {self.locator.content_based_retries}"
            )

        if self.locator.coordinate_based_retries < 0:
            errors.append(
                f"COORDINATE_BASED_RETRIES must be >= 0, got {self.locator.coordinate_based_retries}"
            )

        if self.locator.element_type_retries < 0:
            errors.append(
                f"ELEMENT_TYPE_RETRIES must be >= 0, got {self.locator.element_type_retries}"
            )

        if self.locator.coordinate_offset_attempts < 0:
            errors.append(
                f"COORDINATE_OFFSET_ATTEMPTS must be >= 0, got {self.locator.coordinate_offset_attempts}"
            )

        if self.locator.custom_action_timeout < 1:
            errors.append(
                f"CUSTOM_ACTION_TIMEOUT must be >= 1, got {self.locator.custom_action_timeout}"
            )

        # Validate concurrency configuration
        if self.max_concurrent_tasks < 1:
            errors.append(f"MAX_CONCURRENT_TASKS must be >= 1, got {self.max_concurrent_tasks}")

        return errors


# Global configuration instance
config = BrowserServiceConfig()
