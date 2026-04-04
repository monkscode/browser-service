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

import pytest
from contextlib import contextmanager
from unittest.mock import patch, MagicMock
from flask import Flask


@pytest.fixture
def app_and_processor():
    """Create a Flask app with routes registered and a mocked processor."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    mock_processor = MagicMock()
    mock_processor.tasks = {}
    mock_processor.get_tasks_dict.return_value = {}
    mock_processor.list_tasks.return_value = []
    mock_processor.get_task_status.return_value = None

    with patch("browser_service.api.routes.config") as mock_config, \
         patch("browser_service.api.routes._nl_settings") as mock_settings, \
         patch("browser_service.api.routes.process_workflow_task"):
        mock_config.llm.google_api_key = "test-key"
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
        resp = client.post("/workflow", json={
            "elements": [{"id": "e1", "description": "button", "action": "click"}],
            "url": "https://example.com",
            "user_query": "click the button",
        })
        # Should be 200 or 202 — accept either
        assert resp.status_code in [200, 202]
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

    def test_busy_returns_429(self, client, processor):
        """When a task is already processing, returns 429."""
        processor.get_tasks_dict.return_value = {
            "existing": {"status": "processing"}
        }
        resp = client.post("/workflow", json={
            "elements": [{"id": "e1", "description": "btn", "action": "click"}],
            "url": "https://example.com",
        })
        assert resp.status_code == 429

    def test_batch_alias_works(self, client, processor):
        """POST /batch works as alias for /workflow."""
        processor.get_tasks_dict.return_value = {}
        resp = client.post("/batch", json={
            "elements": [{"id": "e1", "description": "btn", "action": "click"}],
            "url": "https://example.com",
        })
        assert resp.status_code in [200, 202]


class TestQueryEndpoint:
    """Tests for GET /query/<task_id>."""

    def test_query_completed_task(self, client, processor):
        """Query a completed task returns result."""
        processor.get_task_status.return_value = {
            "task_id": "t1",
            "status": "completed",
            "results": {"locator_mapping": {}},
        }
        resp = client.get("/query/t1")
        assert resp.status_code == 200

    def test_query_unknown_task(self, client, processor):
        """Query unknown task_id returns 404."""
        processor.get_task_status.return_value = None
        resp = client.get("/query/nonexistent")
        assert resp.status_code == 404


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
        from flask import Flask
        from unittest.mock import MagicMock, patch

        app = Flask(__name__)
        app.config["TESTING"] = True

        mock_processor = MagicMock()
        mock_processor.get_tasks_dict.return_value = {}

        llm_cfg = MagicMock()
        llm_cfg.model_provider = model_provider
        llm_cfg.google_api_key = api_key
        llm_cfg.vertexai_credentials = credentials_obj

        mock_cfg = MagicMock()
        mock_cfg.llm = llm_cfg

        with patch("browser_service.api.routes.config", mock_cfg), \
             patch("browser_service.api.routes._nl_settings", None), \
             patch("browser_service.api.routes.process_workflow_task"):
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
        with self._make_app_with_config("gemini", credentials_obj=None, api_key="real-key") as client:
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
