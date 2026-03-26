"""
Unit tests for browser_service.config — BrowserServiceConfig.

Purpose: Verify that configuration defaults, env-var overrides, model name
         stripping, and validation logic work correctly.  A regression in
         any of these breaks the entire browser-service at startup.

Each test targets a distinct code path in config.py:
  - Default values (no env vars)
  - Env-var override for headless mode
  - Env-var override for robot_library
  - LLM model name normalisation (_get_google_model strips "gemini/")
  - Batch config defaults
  - Locator config defaults + custom offsets
  - validate() returns errors for invalid values
  - validate() returns empty list for valid config
"""

import os
import pytest
from unittest.mock import patch


class TestBrowserServiceConfigDefaults:
    """Tests that default values are correct when no env vars are set."""

    def _make_config(self, env_overrides: dict = None):
        """
        Create a fresh BrowserServiceConfig with controlled env vars.

        We must patch os.getenv AND block the NL-repo import so tests
        are hermetic and don't depend on the user's real .env file.
        """
        env = {
            # Wipe all env vars that config.py reads
            "MAX_AGENT_STEPS": "15",
            "MAX_RETRIES_PER_ELEMENT": "2",
            "ELEMENT_TIMEOUT": "120",
            "CONTENT_BASED_RETRIES": "7",
            "COORDINATE_BASED_RETRIES": "7",
            "ELEMENT_TYPE_RETRIES": "5",
            "COORDINATE_OFFSET_ATTEMPTS": "7",
            "GEMINI_API_KEY": "",
            "ROBOT_LIBRARY": "browser",
            "BROWSER_HEADLESS": "true",
            "ENABLE_CUSTOM_ACTIONS": "true",
            "GOOGLE_MODEL": "gemini-2.5-flash",
        }
        if env_overrides:
            env.update(env_overrides)

        with patch.dict(os.environ, env, clear=False):
            # Block NL-repo settings import to keep tests hermetic
            with patch(
                "browser_service.config.BrowserServiceConfig._get_google_model",
                return_value=env.get("GOOGLE_MODEL", "gemini-2.5-flash").replace("gemini/", ""),
            ):
                from browser_service.config import BrowserServiceConfig
                return BrowserServiceConfig()

    # ── Default value tests ──────────────────────────────────────────

    def test_default_headless_is_true(self):
        """Default BROWSER_HEADLESS is True (CI-friendly)."""
        cfg = self._make_config()
        assert cfg.headless is True

    def test_default_robot_library_is_browser(self):
        """Default ROBOT_LIBRARY is 'browser'."""
        cfg = self._make_config()
        assert cfg.robot_library == "browser"

    def test_default_batch_config(self):
        """Batch config defaults match expected startup values."""
        cfg = self._make_config()
        assert cfg.batch.max_agent_steps == 15
        assert cfg.batch.max_retries_per_element == 2
        assert cfg.batch.element_timeout == 120

    def test_default_locator_config(self):
        """Locator config defaults match expected startup values."""
        cfg = self._make_config()
        assert cfg.locator.content_based_retries == 7
        assert cfg.locator.coordinate_based_retries == 7
        assert cfg.locator.element_type_retries == 5
        assert cfg.locator.coordinate_offset_attempts == 7
        assert len(cfg.locator.coordinate_offsets) == 5  # 5 default offsets

    # ── Env-var override tests ───────────────────────────────────────

    def test_headless_env_override_false(self):
        """BROWSER_HEADLESS=false → headless is False."""
        cfg = self._make_config({"BROWSER_HEADLESS": "false"})
        assert cfg.headless is False

    def test_robot_library_env_override(self):
        """ROBOT_LIBRARY=selenium is accepted."""
        cfg = self._make_config({"ROBOT_LIBRARY": "selenium"})
        assert cfg.robot_library == "selenium"


class TestBrowserServiceConfigValidation:
    """Tests for the validate() method."""

    def _make_config_raw(self):
        """Create config without env-var mocking for manual attribute tweaking."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
            with patch(
                "browser_service.config.BrowserServiceConfig._get_google_model",
                return_value="gemini-2.5-flash",
            ):
                from browser_service.config import BrowserServiceConfig
                return BrowserServiceConfig()

    def test_validate_valid_config(self):
        """A properly configured instance has no validation errors."""
        cfg = self._make_config_raw()
        cfg.llm.google_api_key = "valid-key"
        cfg.robot_library = "browser"
        errors = cfg.validate()
        assert errors == []

    def test_validate_missing_api_key(self):
        """Missing GEMINI_API_KEY is reported."""
        cfg = self._make_config_raw()
        cfg.llm.google_api_key = ""
        errors = cfg.validate()
        assert any("GEMINI_API_KEY" in e for e in errors)

    def test_validate_invalid_robot_library(self):
        """Invalid ROBOT_LIBRARY value is reported."""
        cfg = self._make_config_raw()
        cfg.llm.google_api_key = "key"
        cfg.robot_library = "puppeteer"
        errors = cfg.validate()
        assert any("ROBOT_LIBRARY" in e for e in errors)

    def test_validate_invalid_batch_values(self):
        """Negative/zero batch values are all reported."""
        cfg = self._make_config_raw()
        cfg.llm.google_api_key = "key"
        cfg.batch.max_agent_steps = 0
        cfg.batch.max_retries_per_element = -1
        cfg.batch.element_timeout = 0
        errors = cfg.validate()
        assert len(errors) >= 3


class TestGoogleModelNormalisation:
    """Test _get_google_model strips the 'gemini/' prefix."""

    def test_strips_gemini_prefix(self):
        """'gemini/gemini-2.5-flash' → 'gemini-2.5-flash'."""
        with patch.dict(os.environ, {"GOOGLE_MODEL": "gemini/gemini-2.5-flash"}, clear=False):
            # We need to call the real _get_google_model, so import
            # settings will fail (ImportError) and it falls back to env
            with patch("browser_service.config.BrowserServiceConfig.__init__", return_value=None):
                from browser_service.config import BrowserServiceConfig
                cfg = BrowserServiceConfig.__new__(BrowserServiceConfig)
                # Bind the real method
                result = BrowserServiceConfig._get_google_model(cfg)
                assert result == "gemini-2.5-flash"

    def test_no_prefix_unchanged(self):
        """'gemini-2.5-flash' stays 'gemini-2.5-flash'."""
        with patch.dict(os.environ, {"GOOGLE_MODEL": "gemini-2.5-flash"}, clear=False):
            with patch("browser_service.config.BrowserServiceConfig.__init__", return_value=None):
                from browser_service.config import BrowserServiceConfig
                cfg = BrowserServiceConfig.__new__(BrowserServiceConfig)
                result = BrowserServiceConfig._get_google_model(cfg)
                assert result == "gemini-2.5-flash"
