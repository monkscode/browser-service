"""
Unit tests for browser_service.config — BrowserServiceConfig.

Purpose: Verify that configuration defaults, env-var overrides, model name
         stripping, and validation logic work correctly.  A regression in
         any of these breaks the entire browser-service at startup.

Each test targets a distinct code path in config.py:
  - Default values (no env vars)
  - Env-var override for headless mode
  - ROBOT_LIBRARY: only 'browser' accepted; 'selenium' rejected at construction
  - LLM model name normalisation (_get_google_model strips "gemini/")
  - Batch config defaults
  - Locator config defaults + custom offsets
  - validate() returns errors for invalid values
  - validate() returns empty list for valid config
"""

import os
import sys
from unittest.mock import patch

import pytest


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
            "MAX_CONCURRENT_TASKS": "10",
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

    def test_robot_library_selenium_rejected_at_startup(self):
        """ROBOT_LIBRARY=selenium fails construction with a migration message."""
        with pytest.raises(ValueError, match="no longer supported"):
            self._make_config({"ROBOT_LIBRARY": "selenium"})

    def test_robot_library_unknown_rejected_at_startup(self):
        """Any non-'browser' ROBOT_LIBRARY value fails construction."""
        with pytest.raises(ValueError, match="ROBOT_LIBRARY"):
            self._make_config({"ROBOT_LIBRARY": "puppeteer"})

    def test_default_max_concurrent_tasks(self):
        """Default MAX_CONCURRENT_TASKS is 10."""
        cfg = self._make_config()
        assert cfg.max_concurrent_tasks == 10

    def test_max_concurrent_tasks_env_override(self):
        """MAX_CONCURRENT_TASKS env var is read and stored as int."""
        cfg = self._make_config({"MAX_CONCURRENT_TASKS": "5"})
        assert cfg.max_concurrent_tasks == 5


class TestAgentVisionMode:
    """AGENT_VISION_MODE (Task 28 A1-INLINE): 'auto' default, 'on' escape hatch.

    'auto' = vision-off with in-run failure-triggered screenshot escalation;
    'on'   = full vision on every step (escape hatch).
    Anything else forks bench vs production behavior silently, so construction
    fails fast — same contract as ROBOT_LIBRARY.
    """

    def _make_config(self, vision_mode: str | None):
        env = {"GEMINI_API_KEY": "test-key", "MODEL_PROVIDER": "gemini"}
        if vision_mode is not None:
            env["AGENT_VISION_MODE"] = vision_mode
        with patch.dict(os.environ, env, clear=False):
            if vision_mode is None:
                os.environ.pop("AGENT_VISION_MODE", None)
            with patch(
                "browser_service.config.BrowserServiceConfig._get_google_model",
                return_value="gemini-2.5-flash",
            ):
                from browser_service.config import BrowserServiceConfig

                return BrowserServiceConfig()

    def test_default_is_auto(self):
        """No env var → agent_vision_mode defaults to 'auto' (vision-off + escalation)."""
        cfg = self._make_config(None)
        assert cfg.agent_vision_mode == "auto"

    def test_on_escape_hatch(self):
        """AGENT_VISION_MODE=on → full vision."""
        cfg = self._make_config("on")
        assert cfg.agent_vision_mode == "on"

    def test_case_insensitive(self):
        """Value is normalised to lowercase."""
        cfg = self._make_config("ON")
        assert cfg.agent_vision_mode == "on"

    def test_invalid_value_rejected_at_startup(self):
        """Unknown mode fails construction with a clear message."""
        with pytest.raises(ValueError, match="AGENT_VISION_MODE"):
            self._make_config("sometimes")

    def test_off_is_not_a_mode(self):
        """'off' is not accepted — vision-off IS 'auto' (escalation stays armed)."""
        with pytest.raises(ValueError, match="AGENT_VISION_MODE"):
            self._make_config("off")


class TestBrowserServiceConfigValidation:
    """Tests for the validate() method."""

    def _make_config_raw(self):
        """Create config without env-var mocking for manual attribute tweaking."""
        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-key", "MODEL_PROVIDER": "gemini"}, clear=False
        ):
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
        errors = cfg.validate()
        assert errors == []

    def test_validate_missing_api_key(self):
        """Missing GEMINI_API_KEY is reported."""
        cfg = self._make_config_raw()
        cfg.llm.google_api_key = ""
        errors = cfg.validate()
        assert any("GEMINI_API_KEY" in e for e in errors)

    def test_validate_invalid_batch_values(self):
        """Negative/zero batch values are all reported."""
        cfg = self._make_config_raw()
        cfg.llm.google_api_key = "key"
        cfg.batch.max_agent_steps = 0
        cfg.batch.max_retries_per_element = -1
        cfg.batch.element_timeout = 0
        errors = cfg.validate()
        assert len(errors) >= 3

    def test_validate_max_concurrent_tasks_zero(self):
        """MAX_CONCURRENT_TASKS=0 is rejected."""
        cfg = self._make_config_raw()
        cfg.llm.google_api_key = "key"
        cfg.max_concurrent_tasks = 0
        errors = cfg.validate()
        assert any("MAX_CONCURRENT_TASKS" in e for e in errors)

    def test_validate_max_concurrent_tasks_negative(self):
        """MAX_CONCURRENT_TASKS=-1 is rejected."""
        cfg = self._make_config_raw()
        cfg.llm.google_api_key = "key"
        cfg.max_concurrent_tasks = -1
        errors = cfg.validate()
        assert any("MAX_CONCURRENT_TASKS" in e for e in errors)

    def test_validate_max_concurrent_tasks_one_is_valid(self):
        """MAX_CONCURRENT_TASKS=1 (single-task mode) is valid."""
        cfg = self._make_config_raw()
        cfg.llm.google_api_key = "key"
        cfg.max_concurrent_tasks = 1
        errors = cfg.validate()
        assert not any("MAX_CONCURRENT_TASKS" in e for e in errors)


class TestGoogleModelNormalisation:
    """Test _get_google_model strips the 'gemini/' prefix."""

    def test_strips_gemini_prefix(self):
        """'gemini/gemini-2.5-flash' → 'gemini-2.5-flash'."""
        with patch.dict(os.environ, {"GOOGLE_MODEL": "gemini/gemini-2.5-flash"}, clear=False):
            # Stub the settings module to None so the import raises ImportError and
            # _get_google_model falls back to GOOGLE_MODEL env (the path under test).
            # In the combined repo src.backend.core.config IS importable, so without
            # this stub the real settings (ONLINE_MODEL) would shadow the env path.
            with (
                patch.dict(sys.modules, {"src.backend.core.config": None}),
                patch("browser_service.config.BrowserServiceConfig.__init__", return_value=None),
            ):
                from browser_service.config import BrowserServiceConfig

                cfg = BrowserServiceConfig.__new__(BrowserServiceConfig)
                # Bind the real method
                result = BrowserServiceConfig._get_google_model(cfg)
                assert result == "gemini-2.5-flash"

    def test_no_prefix_unchanged(self):
        """'gemini-2.5-flash' stays 'gemini-2.5-flash'."""
        with patch.dict(os.environ, {"GOOGLE_MODEL": "gemini-2.5-flash"}, clear=False):
            with (
                patch.dict(sys.modules, {"src.backend.core.config": None}),
                patch("browser_service.config.BrowserServiceConfig.__init__", return_value=None),
            ):
                from browser_service.config import BrowserServiceConfig

                cfg = BrowserServiceConfig.__new__(BrowserServiceConfig)
                result = BrowserServiceConfig._get_google_model(cfg)
                assert result == "gemini-2.5-flash"

    def test_strips_vertex_ai_prefix(self):
        """'vertex_ai/gemini-2.5-flash' → 'gemini-2.5-flash'."""
        with patch.dict(os.environ, {"GOOGLE_MODEL": "vertex_ai/gemini-2.5-flash"}, clear=False):
            with (
                patch.dict(sys.modules, {"src.backend.core.config": None}),
                patch("browser_service.config.BrowserServiceConfig.__init__", return_value=None),
            ):
                from browser_service.config import BrowserServiceConfig

                cfg = BrowserServiceConfig.__new__(BrowserServiceConfig)
                result = BrowserServiceConfig._get_google_model(cfg)
                assert result == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Vertex AI provider tests
# ---------------------------------------------------------------------------


class TestVertexAIProviderConfig:
    """Tests for MODEL_PROVIDER=vertex path in BrowserServiceConfig."""

    def _make_vertex_config(self, extra_env=None, mock_credentials=True):
        """
        Build a BrowserServiceConfig with vertex provider env vars.

        mock_credentials=True patches google.oauth2 so no real file is needed.
        Returns (cfg, mock_load) when mock_credentials=True, else just cfg.
        """
        env = {
            "MODEL_PROVIDER": "vertex",
            "VERTEXAI_PROJECT": "my-gcp-project",
            "VERTEXAI_LOCATION": "asia-south1",
            "VERTEXAI_CREDENTIALS": "/fake/credentials.json",
            "GEMINI_API_KEY": "",
            "GOOGLE_MODEL": "gemini-2.5-flash",
        }
        if extra_env:
            env.update(extra_env)

        sentinel = object()  # unique object to assert identity

        with patch.dict(os.environ, env, clear=False):
            with patch(
                "browser_service.config.BrowserServiceConfig._get_google_model",
                return_value="gemini-2.5-flash",
            ):
                with patch(
                    "browser_service.config.BrowserServiceConfig.__init__.__globals__",
                    {},
                    create=True,
                ):
                    # Block NL-repo settings import
                    pass
                if mock_credentials:
                    with patch(
                        "browser_service.config.service_account"
                        if False
                        else "google.oauth2.service_account.Credentials.from_service_account_file",
                        return_value=sentinel,
                    ) as mock_load:
                        from browser_service.config import BrowserServiceConfig

                        with patch(
                            "google.oauth2.service_account.Credentials.from_service_account_file",
                            return_value=sentinel,
                        ) as mock_load2:
                            cfg = BrowserServiceConfig()
                            return cfg, mock_load2
                else:
                    from browser_service.config import BrowserServiceConfig

                    cfg = BrowserServiceConfig()
                    return cfg

    def _make_vertex_config_simple(self, extra_env=None):
        """Simpler helper: patches entire __init__ credential block."""
        env = {
            "MODEL_PROVIDER": "vertex",
            "VERTEXAI_PROJECT": "my-gcp-project",
            "VERTEXAI_LOCATION": "asia-south1",
            "VERTEXAI_CREDENTIALS": "/fake/credentials.json",
            "GEMINI_API_KEY": "",
            "GOOGLE_MODEL": "gemini-2.5-flash",
        }
        if extra_env:
            env.update(extra_env)

        fake_creds = object()

        with patch.dict(os.environ, env, clear=False):
            with patch(
                "browser_service.config.BrowserServiceConfig._get_google_model",
                return_value="gemini-2.5-flash",
            ):
                with patch(
                    "google.oauth2.service_account.Credentials.from_service_account_file",
                    return_value=fake_creds,
                ) as mock_load:
                    from browser_service.config import BrowserServiceConfig

                    cfg = BrowserServiceConfig()
                    return cfg, mock_load, fake_creds

    def test_model_provider_set_to_vertex(self):
        """MODEL_PROVIDER=vertex is stored on config.llm.model_provider."""
        cfg, _, _ = self._make_vertex_config_simple()
        assert cfg.llm.model_provider == "vertex"

    def test_vertex_fields_populated(self):
        """VERTEXAI_PROJECT, VERTEXAI_LOCATION, VERTEXAI_CREDENTIALS are stored."""
        cfg, _, _ = self._make_vertex_config_simple()
        assert cfg.llm.vertexai_project == "my-gcp-project"
        assert cfg.llm.vertexai_location == "asia-south1"
        assert cfg.llm.vertexai_credentials_path == "/fake/credentials.json"

    def test_credentials_object_loaded_at_startup(self):
        """Credentials.from_service_account_file is called once during __init__."""
        cfg, mock_load, fake_creds = self._make_vertex_config_simple()
        mock_load.assert_called_once_with(
            "/fake/credentials.json",
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        assert cfg.llm.vertexai_credentials is fake_creds

    def test_credentials_not_loaded_for_gemini_provider(self):
        """Credentials loading is skipped when model_provider=gemini."""
        env = {
            "MODEL_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key",
            "GOOGLE_MODEL": "gemini-2.5-flash",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch(
                "browser_service.config.BrowserServiceConfig._get_google_model",
                return_value="gemini-2.5-flash",
            ):
                with patch(
                    "google.oauth2.service_account.Credentials.from_service_account_file"
                ) as mock_load:
                    from browser_service.config import BrowserServiceConfig

                    BrowserServiceConfig()
                    mock_load.assert_not_called()

    def test_credentials_remain_none_when_path_empty(self):
        """If VERTEXAI_CREDENTIALS is unset, vertexai_credentials stays None."""
        env = {
            "MODEL_PROVIDER": "vertex",
            "VERTEXAI_PROJECT": "proj",
            "VERTEXAI_LOCATION": "us-central1",
            "VERTEXAI_CREDENTIALS": "",
            "GOOGLE_MODEL": "gemini-2.5-flash",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch(
                "browser_service.config.BrowserServiceConfig._get_google_model",
                return_value="gemini-2.5-flash",
            ):
                from browser_service.config import BrowserServiceConfig

                cfg = BrowserServiceConfig()
        assert cfg.llm.vertexai_credentials is None

    def test_credentials_none_on_load_failure(self):
        """If from_service_account_file raises, vertexai_credentials stays None."""
        env = {
            "MODEL_PROVIDER": "vertex",
            "VERTEXAI_PROJECT": "proj",
            "VERTEXAI_LOCATION": "us-central1",
            "VERTEXAI_CREDENTIALS": "/fake/bad.json",
            "GOOGLE_MODEL": "gemini-2.5-flash",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch(
                "browser_service.config.BrowserServiceConfig._get_google_model",
                return_value="gemini-2.5-flash",
            ):
                with patch(
                    "google.oauth2.service_account.Credentials.from_service_account_file",
                    side_effect=ValueError("bad JSON"),
                ):
                    from browser_service.config import BrowserServiceConfig

                    cfg = BrowserServiceConfig()
        assert cfg.llm.vertexai_credentials is None


class TestVertexAIValidation:
    """Tests for validate() provider-specific checks (vertex branch)."""

    def _base_vertex_cfg(self):
        """Return a BrowserServiceConfig with all vertex fields set to valid values."""
        from browser_service.config import BrowserServiceConfig, LLMConfig

        cfg = BrowserServiceConfig.__new__(BrowserServiceConfig)
        cfg.llm = LLMConfig(
            model_provider="vertex",
            vertexai_project="my-project",
            vertexai_location="asia-south1",
            vertexai_credentials_path="/fake/creds.json",
            vertexai_credentials=object(),  # non-None = successfully loaded
        )
        from browser_service.config import BatchConfig, LocatorConfig

        cfg.batch = BatchConfig()
        cfg.locator = LocatorConfig()
        cfg.headless = True
        cfg.enable_custom_actions = True
        cfg.max_concurrent_tasks = 10
        return cfg

    def test_validate_vertex_all_valid(self):
        """All vertex fields set + credentials loaded → no errors."""
        cfg = self._base_vertex_cfg()
        with patch("os.path.exists", return_value=True):
            errors = cfg.validate()
        assert errors == [], errors

    def test_validate_vertex_missing_credentials_env(self):
        cfg = self._base_vertex_cfg()
        cfg.llm.vertexai_credentials_path = ""
        errors = cfg.validate()
        assert any("VERTEXAI_CREDENTIALS" in e for e in errors)

    def test_validate_vertex_credentials_file_not_found(self):
        cfg = self._base_vertex_cfg()
        cfg.llm.vertexai_credentials_path = "/nonexistent/path.json"
        cfg.llm.vertexai_credentials = None
        with patch("os.path.exists", return_value=False):
            errors = cfg.validate()
        assert any("not found" in e for e in errors)

    def test_validate_vertex_credentials_failed_to_load(self):
        """File exists on disk but credentials object is None (e.g. malformed JSON)."""
        cfg = self._base_vertex_cfg()
        cfg.llm.vertexai_credentials = None  # load failed
        with patch("os.path.exists", return_value=True):
            errors = cfg.validate()
        assert any("failed to load" in e for e in errors)

    def test_validate_vertex_missing_project(self):
        cfg = self._base_vertex_cfg()
        cfg.llm.vertexai_project = ""
        with patch("os.path.exists", return_value=True):
            errors = cfg.validate()
        assert any("VERTEXAI_PROJECT" in e for e in errors)

    def test_validate_vertex_missing_location(self):
        cfg = self._base_vertex_cfg()
        cfg.llm.vertexai_location = ""
        with patch("os.path.exists", return_value=True):
            errors = cfg.validate()
        assert any("VERTEXAI_LOCATION" in e for e in errors)

    def test_validate_local_provider_returns_error(self):
        """MODEL_PROVIDER=local is explicitly rejected — not a silent pass."""
        cfg = self._base_vertex_cfg()
        cfg.llm.model_provider = "local"
        errors = cfg.validate()
        assert any("local" in e and "not supported" in e for e in errors)

    def test_validate_unknown_provider_returns_error(self):
        """An unrecognised provider value returns a clear error."""
        cfg = self._base_vertex_cfg()
        cfg.llm.model_provider = "openai"
        errors = cfg.validate()
        assert any("openai" in e for e in errors)


class TestCustomActionTimeoutConfig:
    """
    CUSTOM_ACTION_TIMEOUT is a browser-service env var (Task 4 / D5).

    Before this knob existed, the locator-finder budget was resolved by
    importing nlrf settings with a silent fallback to a hard-coded 5 —
    setting the env var in browser-service changed nothing.
    """

    def _make_config(self, monkeypatch, value=None):
        if value is None:
            monkeypatch.delenv("CUSTOM_ACTION_TIMEOUT", raising=False)
        else:
            monkeypatch.setenv("CUSTOM_ACTION_TIMEOUT", value)
        with patch(
            "browser_service.config.BrowserServiceConfig._get_google_model",
            return_value="gemini-2.5-flash",
        ):
            from browser_service.config import BrowserServiceConfig

            return BrowserServiceConfig()

    def test_default_is_5_when_unset(self, monkeypatch):
        """No env var → 5 seconds (the historical hard-coded value)."""
        cfg = self._make_config(monkeypatch)
        assert cfg.locator.custom_action_timeout == 5

    def test_env_var_overrides_default(self, monkeypatch):
        """CUSTOM_ACTION_TIMEOUT=30 → 30. The knob must actually work."""
        cfg = self._make_config(monkeypatch, "30")
        assert cfg.locator.custom_action_timeout == 30

    def test_non_numeric_falls_back_and_records_error(self, monkeypatch):
        """Garbage input keeps the default and surfaces a parse error via validate()."""
        cfg = self._make_config(monkeypatch, "abc")
        assert cfg.locator.custom_action_timeout == 5
        assert any("CUSTOM_ACTION_TIMEOUT" in e for e in cfg._parse_errors)

    def test_non_positive_rejected_by_validate(self, monkeypatch):
        """0 or negative budgets are configuration errors, matching ELEMENT_TIMEOUT."""
        cfg = self._make_config(monkeypatch, "0")
        errors = cfg.validate()
        assert any("CUSTOM_ACTION_TIMEOUT" in e for e in errors)
