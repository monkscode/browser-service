"""
Mocked integration tests for browser_service.api.routes — full HTTP endpoint tests.

Purpose: These test the Flask routes end-to-end through the test client,
         with the task_processor mocked.  They verify HTTP status codes,
         response JSON shapes, and route dispatch logic.

Tests:
  GET /           → 200, has service info
  GET /health     → 200, has status: healthy
  GET /probe      → 200, has status: alive
  POST /workflow  → 202 (valid), 400 (invalid/missing), 429 (busy)
  POST /batch     → alias for /workflow
  GET /query/<id> → 200 (found), 404 (not found)
  GET /tasks      → 200, list of tasks
"""

import logging
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask


@pytest.fixture
def app_and_processor():
    """Create a Flask app with routes registered and a mocked processor."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    mock_processor = MagicMock()
    mock_processor.tasks = {}
    mock_processor.get_tasks_dict.return_value = {}
    mock_processor.count_active_tasks.return_value = 0
    mock_processor.tasks_submitted_count.return_value = 0
    mock_processor.list_tasks.return_value = []
    mock_processor.get_task_status.return_value = None

    with (
        patch("browser_service.api.routes.config") as mock_config,
        patch("browser_service.api.routes._nl_settings") as mock_settings,
        patch("browser_service.api.routes.process_workflow_task"),
    ):
        mock_config.llm.google_api_key = "test-key"
        mock_config.max_concurrent_tasks = 10
        mock_settings.ENABLE_CUSTOM_ACTIONS = True

        from browser_service.api.routes import register_routes

        register_routes(app, mock_processor)

    return app, mock_processor


@pytest.fixture
def client(app_and_processor):
    app, _ = app_and_processor
    with app.test_client() as c:
        yield c


@pytest.fixture
def processor(app_and_processor):
    _, proc = app_and_processor
    return proc


class TestStaticEndpoints:
    """Tests for static/informational endpoints."""

    def test_root_returns_service_info(self, client):
        """GET / returns service name and endpoints list."""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "service" in data
        assert "endpoints" in data

    def test_health_returns_healthy(self, client):
        """GET /health returns status: healthy."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"

    def test_probe_returns_alive(self, client):
        """GET /probe returns status: alive."""
        resp = client.get("/probe")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "alive"


class TestWorkflowSubmit:
    """Tests for POST /workflow endpoint."""

    def test_valid_workflow_submit(self, client, processor):
        """Valid workflow request returns 202 with task_id."""
        resp = client.post(
            "/workflow",
            json={
                "elements": [{"id": "e1", "description": "button", "action": "click"}],
                "url": "https://example.com",
                "user_query": "click the button",
            },
        )
        # 202 exactly. `in [200, 202]` cannot tell an accepted-async submit from a
        # synchronous completion, which is the only thing this status communicates.
        assert resp.status_code == 202
        data = resp.get_json()
        assert "task_id" in data

    def test_invalid_request_returns_400(self, client):
        """Request without elements returns 400."""
        resp = client.post("/workflow", json={"url": "https://example.com"})
        assert resp.status_code == 400

    def test_missing_json_returns_400(self, client):
        """Non-JSON request returns 400."""
        resp = client.post("/workflow", data="not json", content_type="text/plain")
        assert resp.status_code == 400

    def test_busy_returns_429_when_at_capacity(self, client, processor):
        """When try_submit_task returns False (at capacity), returns 429."""
        from browser_service.config import config as real_config

        limit = real_config.max_concurrent_tasks
        processor.try_submit_task.return_value = False
        processor.count_active_tasks.return_value = limit  # used in 429 response body
        resp = client.post(
            "/workflow",
            json={
                "elements": [{"id": "e1", "description": "btn", "action": "click"}],
                "url": "https://example.com",
            },
        )
        assert resp.status_code == 429

    def test_busy_response_contains_capacity_fields(self, client, processor):
        """429 response body includes active_tasks and max_tasks fields."""
        from browser_service.config import config as real_config

        limit = real_config.max_concurrent_tasks
        processor.try_submit_task.return_value = False
        processor.count_active_tasks.return_value = limit
        resp = client.post(
            "/workflow",
            json={
                "elements": [{"id": "e1", "description": "btn", "action": "click"}],
                "url": "https://example.com",
            },
        )
        assert resp.status_code == 429
        data = resp.get_json()
        assert data["active_tasks"] == limit
        assert data["max_tasks"] == limit

    def test_below_capacity_is_accepted(self, client, processor):
        """When try_submit_task returns True (under capacity), request is accepted."""
        processor.try_submit_task.return_value = True
        resp = client.post(
            "/workflow",
            json={
                "elements": [{"id": "e1", "description": "btn", "action": "click"}],
                "url": "https://example.com",
            },
        )
        assert resp.status_code == 202

    def test_batch_alias_works(self, client, processor):
        """POST /batch works as alias for /workflow — same status, same body shape."""
        processor.count_active_tasks.return_value = 0
        resp = client.post(
            "/batch",
            json={
                "elements": [{"id": "e1", "description": "btn", "action": "click"}],
                "url": "https://example.com",
            },
        )
        assert resp.status_code == 202
        assert "task_id" in resp.get_json()


class TestQueryEndpoint:
    """Tests for GET /query/<task_id>."""

    def test_query_completed_task(self, client, processor):
        """A completed task returns 200 AND carries its results."""
        processor.get_task_status.return_value = {
            "task_id": "t1",
            "status": "completed",
            "results": {"locator_mapping": {"e1": {"best_locator": "id=x"}}},
        }
        resp = client.get("/query/t1")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "completed"
        assert data["results"] == {"locator_mapping": {"e1": {"best_locator": "id=x"}}}

    @pytest.mark.parametrize("status", ["processing", "running"])
    def test_query_in_flight_task_returns_202_without_results(self, client, processor, status):
        """In-flight tasks return 202 and must NOT leak partial results.

        Both branches are pinned: the client polls on 202, so a branch that
        returned 200 would end the poll on an unfinished task, and one that
        included results would hand back a partial locator_mapping.
        """
        processor.get_task_status.return_value = {
            "task_id": "t1",
            "status": status,
            "results": {"locator_mapping": {"e1": "partial"}},
        }
        resp = client.get("/query/t1")
        assert resp.status_code == 202
        data = resp.get_json()
        assert data["status"] == status
        assert "results" not in data

    def test_query_unrecognised_status_returns_200_without_results(self, client, processor):
        """A status outside the known set falls to the terminal 200 branch."""
        processor.get_task_status.return_value = {
            "task_id": "t1",
            "status": "failed",
            "results": {"locator_mapping": {}},
        }
        resp = client.get("/query/t1")
        assert resp.status_code == 200
        assert "results" not in resp.get_json()

    def test_query_unknown_task(self, client, processor):
        """Query unknown task_id returns 404."""
        processor.get_task_status.return_value = None
        resp = client.get("/query/nonexistent")
        assert resp.status_code == 404

    def test_query_processor_exception_returns_500(self, client, processor):
        """An unexpected processor failure surfaces as 500, not a stack trace."""
        processor.get_task_status.side_effect = RuntimeError("processor exploded")
        resp = client.get("/query/t1")
        assert resp.status_code == 500


class TestTasksEndpoint:
    """Tests for GET /tasks."""

    def test_list_tasks(self, client, processor):
        """GET /tasks returns list of all tasks."""
        processor.list_tasks.return_value = [
            {"task_id": "t1", "status": "completed", "objective": "Test"},
        ]
        resp = client.get("/tasks")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Health endpoint — provider-aware fields
# ---------------------------------------------------------------------------


class TestHealthProviderFields:
    """
    Tests that /health returns model_provider and the correct google_api_configured
    value for each provider, using config attributes rather than the global singleton.
    """

    @contextmanager
    def _make_app_with_config(self, model_provider, credentials_obj, api_key=""):
        """
        Context manager that yields a Flask test client with config.llm patched
        for the given provider.

        The patch must stay active while requests are made: Flask route handlers
        look up `config` from the module globals at call time, so the mock must
        remain in place throughout the test.  Using a context manager guarantees
        the patch is alive for the duration of the `with` block.
        """
        from unittest.mock import MagicMock, patch

        from flask import Flask

        app = Flask(__name__)
        app.config["TESTING"] = True

        mock_processor = MagicMock()
        mock_processor.get_tasks_dict.return_value = {}
        mock_processor.count_active_tasks.return_value = 0
        mock_processor.tasks_submitted_count.return_value = 0

        llm_cfg = MagicMock()
        llm_cfg.model_provider = model_provider
        llm_cfg.google_api_key = api_key
        llm_cfg.vertexai_credentials = credentials_obj

        mock_cfg = MagicMock()
        mock_cfg.llm = llm_cfg
        mock_cfg.max_concurrent_tasks = 10
        mock_cfg.headless = True

        with (
            patch("browser_service.api.routes.config", mock_cfg),
            patch("browser_service.api.routes._nl_settings", None),
            patch("browser_service.api.routes.process_workflow_task"),
        ):
            from browser_service.api.routes import register_routes

            register_routes(app, mock_processor)
            with app.test_client() as client:
                yield client

    def test_health_vertex_credentials_loaded(self):
        """Vertex provider with loaded credentials: google_api_configured=true."""
        with self._make_app_with_config("vertex", credentials_obj=object()) as client:
            resp = client.get("/health")
        data = resp.get_json()
        assert data["model_provider"] == "vertex"
        assert data["google_api_configured"] is True

    def test_health_vertex_credentials_not_loaded(self):
        """Vertex provider with credentials=None: google_api_configured=false."""
        with self._make_app_with_config("vertex", credentials_obj=None) as client:
            resp = client.get("/health")
        data = resp.get_json()
        assert data["model_provider"] == "vertex"
        assert data["google_api_configured"] is False

    def test_health_gemini_with_api_key(self):
        """Gemini provider with a valid API key: google_api_configured=true."""
        with self._make_app_with_config(
            "gemini", credentials_obj=None, api_key="real-key"
        ) as client:
            resp = client.get("/health")
        data = resp.get_json()
        assert data["model_provider"] == "gemini"
        assert data["google_api_configured"] is True

    def test_health_gemini_no_api_key(self):
        """Gemini provider with empty API key: google_api_configured=false."""
        with self._make_app_with_config("gemini", credentials_obj=None, api_key="") as client:
            resp = client.get("/health")
        data = resp.get_json()
        assert data["model_provider"] == "gemini"
        assert data["google_api_configured"] is False

    def test_health_local_provider(self):
        """local provider: google_api_configured=false (unsupported)."""
        with self._make_app_with_config("local", credentials_obj=None) as client:
            resp = client.get("/health")
        data = resp.get_json()
        assert data["model_provider"] == "local"
        assert data["google_api_configured"] is False

    def test_health_model_provider_key_always_present(self):
        """model_provider key is always present in /health response."""
        for provider in ("gemini", "vertex", "local"):
            with self._make_app_with_config(provider, credentials_obj=None) as client:
                resp = client.get("/health")
            data = resp.get_json()
            assert "model_provider" in data, f"model_provider missing for provider={provider}"


class TestHealthCapacityFields:
    """Tests for capacity fields in the /health response."""

    def test_idle_health_shows_zero_active_tasks(self, client, processor):
        """No active tasks → active_tasks=0, available_slots=max_tasks."""
        processor.count_active_tasks.return_value = 0
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["active_tasks"] == 0
        assert data["max_tasks"] == 10
        assert data["available_slots"] == 10

    def test_under_load_shows_correct_available_slots(self, client, processor):
        """3 active tasks → available_slots = max_tasks - 3."""
        processor.count_active_tasks.return_value = 3
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["active_tasks"] == 3
        assert data["available_slots"] == 10 - 3

    def test_at_capacity_shows_zero_available_slots(self, client, processor):
        """active_tasks == max_tasks → available_slots=0."""
        processor.count_active_tasks.return_value = 10
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["active_tasks"] == 10
        assert data["available_slots"] == 0

    def test_health_includes_tasks_submitted(self, client, processor):
        """tasks_submitted is monotonically cumulative — does not decrease after TTL eviction."""
        processor.count_active_tasks.return_value = 0
        processor.tasks_submitted_count.return_value = 3
        resp = client.get("/health")
        data = resp.get_json()
        assert data["tasks_submitted"] == 3


class TestHealthHeadlessField:
    """
    /health reports the effective headless flag (Task 4 rider).

    The nlrf bench preflight pins BROWSER_HEADLESS=true for guardrail runs;
    without this field it can only warn "not reported" instead of verifying.
    """

    @contextmanager
    def _make_app_with_headless(self, headless):
        app = Flask(__name__)
        app.config["TESTING"] = True

        mock_processor = MagicMock()
        mock_processor.get_tasks_dict.return_value = {}
        mock_processor.count_active_tasks.return_value = 0
        mock_processor.tasks_submitted_count.return_value = 0

        mock_cfg = MagicMock()
        mock_cfg.max_concurrent_tasks = 10
        mock_cfg.headless = headless
        mock_cfg.llm.model_provider = "vertex"
        mock_cfg.llm.vertexai_credentials = object()

        with (
            patch("browser_service.api.routes.config", mock_cfg),
            patch("browser_service.api.routes._nl_settings", None),
            patch("browser_service.api.routes.process_workflow_task"),
        ):
            from browser_service.api.routes import register_routes

            register_routes(app, mock_processor)
            with app.test_client() as client:
                yield client

    def test_health_reports_headless_true(self):
        with self._make_app_with_headless(True) as client:
            resp = client.get("/health")
        assert resp.get_json()["headless"] is True

    def test_health_reports_headless_false(self):
        with self._make_app_with_headless(False) as client:
            resp = client.get("/health")
        assert resp.get_json()["headless"] is False


class TestErrorResponsesDoNotLeakInternals:
    """A 500 body must not carry the exception text.

    Exception messages routinely embed absolute paths, CDP endpoints, config
    values and upstream API payloads. The detail belongs in the server log —
    which every one of these handlers already writes with exc_info=True — not
    in a response any caller can read.
    """

    SECRET = "postgresql://user:hunter2@10.0.0.4:5432/internal"

    def _assert_scrubbed(self, resp, caplog):
        assert resp.status_code == 500
        body = resp.get_data(as_text=True)
        assert self.SECRET not in body, f"exception text leaked into 500 body: {body}"
        assert resp.get_json()["message"] == "Internal server error"
        assert self.SECRET in caplog.text, "detail must still reach the server log"

    def test_workflow_submit_500_is_scrubbed(self, client, processor, caplog):
        processor.try_submit_task.side_effect = RuntimeError(self.SECRET)
        with caplog.at_level(logging.ERROR, logger="browser_service.api.routes"):
            resp = client.post(
                "/workflow",
                json={
                    "elements": [{"id": "e1", "description": "button", "action": "click"}],
                    "url": "https://example.com",
                    "user_query": "click the button",
                },
            )
        self._assert_scrubbed(resp, caplog)

    def test_query_500_is_scrubbed(self, client, processor, caplog):
        processor.get_task_status.side_effect = RuntimeError(self.SECRET)
        with caplog.at_level(logging.ERROR, logger="browser_service.api.routes"):
            resp = client.get("/query/t1")
        self._assert_scrubbed(resp, caplog)

    def test_list_tasks_500_is_scrubbed(self, client, processor, caplog):
        processor.list_tasks.side_effect = RuntimeError(self.SECRET)
        with caplog.at_level(logging.ERROR, logger="browser_service.api.routes"):
            resp = client.get("/tasks")
        self._assert_scrubbed(resp, caplog)

    def test_unhandled_exception_handler_omits_error_field(self, caplog):
        """The app-wide 500 handler returns no `error` key and logs the cause.

        TESTING/PROPAGATE_EXCEPTIONS must be off, otherwise Flask re-raises past
        the handler and this asserts nothing.
        """
        app = Flask(__name__)
        app.config["PROPAGATE_EXCEPTIONS"] = False

        mock_processor = MagicMock()
        with (
            patch("browser_service.api.routes.config") as mock_config,
            patch("browser_service.api.routes._nl_settings", None),
            patch("browser_service.api.routes.process_workflow_task"),
        ):
            mock_config.max_concurrent_tasks = 10
            from browser_service.api.routes import register_routes

            register_routes(app, mock_processor)

        @app.route("/boom")
        def boom():
            raise RuntimeError(self.SECRET)

        with caplog.at_level(logging.ERROR, logger="browser_service.api.routes"):
            resp = app.test_client().get("/boom")

        assert resp.status_code == 500
        data = resp.get_json()
        assert "error" not in data, f"500 handler still exposes an error field: {data}"
        assert data["message"] == "Internal server error"
        assert self.SECRET in caplog.text, "the 500 handler must log the cause"


class TestLogSanitization:
    """Caller-controlled values must not be able to forge log entries (CWE-117).

    Two different reachability stories, and the distinction is the point:

    - task_id is a PATH segment. Werkzeug strips CR/LF before the view runs
      (verified: "aaa\\r\\nWARNING x" arrives as "aaaWARNING x") and the value
      must match a server-generated UUID key to reach the log line at all.
      Forging through it is not reachable; sanitizing it is defence in depth
      and the fix for the SonarCloud finding.
    - parent_workflow_id, url and user_query are JSON BODY fields. Nothing
      strips them — validate_workflow_request does no newline checking — so a
      newline in the body reached the log record verbatim and forged an entry.
      That WAS reachable, in one request, and the end-to-end tests below drive
      it through the Flask client to prove it stays closed.
    """

    FORGED = "abc\r\n2026-01-01 CRITICAL forged entry"

    def _assert_no_record_spans_lines(self, caplog):
        offenders = [r.getMessage() for r in caplog.records if "\n" in r.getMessage()]
        assert not offenders, f"log record carries a newline from caller input: {offenders}"

    def test_body_parent_workflow_id_cannot_forge_a_log_entry(self, client, caplog):
        caplog.set_level(logging.INFO, logger="browser_service.api.routes")
        client.post(
            "/workflow",
            json={
                "elements": [{"id": "e1", "description": "d"}],
                "url": "https://example.com",
                "user_query": "click",
                "parent_workflow_id": self.FORGED,
            },
        )
        assert "forged entry" in caplog.text, "the value must still be logged, just flattened"
        self._assert_no_record_spans_lines(caplog)

    def test_body_url_and_query_cannot_forge_a_log_entry(self, client, caplog):
        caplog.set_level(logging.INFO, logger="browser_service.api.routes")
        client.post(
            "/workflow",
            json={
                "elements": [{"id": "e1", "description": "d"}],
                "url": f"https://example.com/{self.FORGED}",
                "user_query": self.FORGED,
            },
        )
        self._assert_no_record_spans_lines(caplog)

    def test_long_url_is_not_truncated_to_the_default_cap(self, client, caplog):
        """Sanitizing must not cost debuggability.

        URLs routinely run past the 80-char default (ASTPP query strings), and
        the submission log line is the one place the target URL is recorded.
        Flattening is the requirement; truncating to 80 is not.
        """
        caplog.set_level(logging.INFO, logger="browser_service.api.routes")
        long_url = "https://example.com/" + "a" * 150
        client.post(
            "/workflow",
            json={
                "elements": [{"id": "e1", "description": "d"}],
                "url": long_url,
                "user_query": "click",
            },
        )
        assert long_url in caplog.text

    def test_crlf_in_logged_value_is_neutralised(self):
        from browser_service.api.routes import _sanitize_for_log

        forged = "abc\r\nINFO Task deadbeef query completed: True"
        out = _sanitize_for_log(forged)
        assert "\n" not in out
        assert "\r" not in out
        assert out.startswith("abc")

    def test_long_value_is_capped(self):
        from browser_service.api.routes import _sanitize_for_log

        assert len(_sanitize_for_log("x" * 500)) <= 80

    def test_ordinary_uuid_passes_through_unchanged(self):
        from browser_service.api.routes import _sanitize_for_log

        tid = "51aa8eaa-6637-464e-b078-89af2d65191f"
        assert _sanitize_for_log(tid) == tid
