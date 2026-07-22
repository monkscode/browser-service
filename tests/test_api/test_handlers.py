"""
Unit tests for browser_service.api.handlers — request validation and response formatting.

Purpose: The handlers module is the contract between the Flask API and internal services.
         validate_workflow_request decides what gets accepted or rejected (takes Flask Request).
         format_task_response / format_error_response / format_task_list_response decide
         what the client sees.  A regression here means valid requests are rejected or
         invalid requests slip through.

Tests:
  validate_workflow_request: valid, not-JSON, no elements, empty elements, optional defaults
  format_task_response: completed/running/truncated-objective
  format_error_response: message-only / with additional data
  format_task_list_response: multiple tasks with active count
"""

import time
from unittest.mock import MagicMock

import pytest

from browser_service.api.handlers import (
    format_error_response,
    format_task_list_response,
    format_task_response,
    validate_workflow_request,
)


def _make_flask_request(json_data=None, is_json=True):
    """Create a mock Flask Request with json data."""
    req = MagicMock()
    req.is_json = is_json
    req.get_json.return_value = json_data
    return req


class TestValidateWorkflowRequest:
    """Tests for request validation logic."""

    def test_valid_request(self):
        """Minimum valid request with elements and url."""
        req = _make_flask_request(
            {
                "elements": [{"id": "e1", "description": "btn", "action": "click"}],
                "url": "https://example.com",
            }
        )
        is_valid, error, data = validate_workflow_request(req)
        assert is_valid is True
        assert error is None
        assert data["elements"][0]["id"] == "e1"

    def test_not_json_rejected(self):
        """Non-JSON Content-Type is rejected."""
        req = _make_flask_request(is_json=False)
        is_valid, error, data = validate_workflow_request(req)
        assert is_valid is False
        assert "JSON" in error

    def test_empty_json_rejected(self):
        """Empty/None JSON body is rejected."""
        req = _make_flask_request(json_data=None)
        is_valid, error, data = validate_workflow_request(req)
        assert is_valid is False

    def test_missing_elements_rejected(self):
        """Request without 'elements' key is rejected."""
        req = _make_flask_request({"url": "https://example.com"})
        is_valid, error, data = validate_workflow_request(req)
        assert is_valid is False

    def test_empty_elements_rejected(self):
        """Empty elements list is rejected — nothing to process."""
        req = _make_flask_request({"elements": [], "url": "https://x.com"})
        is_valid, error, data = validate_workflow_request(req)
        assert is_valid is False

    def test_optional_fields_default(self):
        """Missing optional fields (user_query, session_config) get defaults."""
        req = _make_flask_request(
            {
                "elements": [{"id": "e1", "description": "btn", "action": "click"}],
                "url": "https://example.com",
            }
        )
        is_valid, _, data = validate_workflow_request(req)
        assert is_valid is True
        assert data["user_query"] == ""
        assert data["session_config"] == {}


class TestFormatTaskResponse:
    """Tests for task response formatting."""

    def test_completed_task(self):
        """Completed task includes results and total_time."""
        now = time.time()
        task_data = {
            "task_id": "t1",
            "status": "completed",
            "objective": "Find locators",
            "created_at": now - 10,
            "started_at": now - 8,
            "completed_at": now,
            "message": "Done",
            "results": {"locator_mapping": {"e1": {"best_locator": "id=x"}}},
        }
        resp = format_task_response(task_data)
        assert resp["status"] == "completed"
        assert resp["task_id"] == "t1"
        assert "results" in resp
        assert resp["total_time"] == pytest.approx(10, abs=1)

    def test_running_task(self):
        """Running task includes running_time."""
        task_data = {
            "task_id": "t2",
            "status": "running",
            "objective": "Processing",
            "created_at": time.time() - 5,
            "started_at": time.time() - 3,
        }
        resp = format_task_response(task_data)
        assert resp["status"] == "running"
        assert "running_time" in resp

    def test_truncated_objective(self):
        """Long objective is truncated to specified length."""
        task_data = {
            "task_id": "t3",
            "status": "processing",
            "objective": "A" * 500,
            "created_at": time.time(),
        }
        resp = format_task_response(task_data, truncate_objective=100)
        assert len(resp["objective"]) == 100


class TestFormatErrorResponse:
    """Tests for error response formatting."""

    def test_error_with_message(self):
        """Error response includes status and message."""
        resp, code = format_error_response("Something broke", 500)
        assert resp["status"] == "error"
        assert resp["message"] == "Something broke"
        assert code == 500

    def test_error_with_additional_data(self):
        """Error response merges additional_data."""
        resp, code = format_error_response("Fail", 400, additional_data={"field": "elements"})
        assert resp["field"] == "elements"
        assert code == 400


class TestFormatTaskListResponse:
    """Tests for task list formatting."""

    def test_list_multiple_tasks(self):
        """Multiple tasks with active count."""
        tasks = [
            {
                "task_id": "t1",
                "status": "completed",
                "objective": "Find",
                "created_at": 1.0,
                "completed_at": 2.0,
                "results": {"success": True},
            },
            {"task_id": "t2", "status": "processing", "objective": "Running", "created_at": 3.0},
        ]
        resp = format_task_list_response(tasks)
        assert resp["total_tasks"] == 2
        assert resp["active_tasks"] == 1
        assert len(resp["tasks"]) == 2
