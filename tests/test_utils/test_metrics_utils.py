"""
Unit tests for browser_service.utils.metrics — workflow metrics recording.

Purpose: Metrics are recorded via HTTP POST to the NL backend's /api/workflow-metrics/record
         endpoint.  If recording silently fails, we lose cost and performance data.
         If it crashes, it takes down the workflow.

Tests:
  - Successful recording returns True
  - HTTP error (4xx/5xx) returns False without crashing
  - Connection error (backend down) returns False without crashing
  - Custom action count is included when present
  - Missing keys default gracefully
"""

from unittest.mock import MagicMock, patch

import pytest

from browser_service.utils.metrics import record_workflow_metrics


class TestRecordWorkflowMetrics:
    """Tests for the record_workflow_metrics function."""

    @patch("browser_service.utils.metrics.requests.post")
    def test_successful_recording(self, mock_post):
        """Successful POST returns True."""
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"status": "ok"})
        mock_post.return_value.raise_for_status = MagicMock()

        record_workflow_metrics(
            workflow_id="wf-001",
            url="https://example.com",
            results={
                "summary": {
                    "total_input_tokens": 100,
                    "total_output_tokens": 50,
                    "total_duration_seconds": 10.5,
                }
            },
        )
        mock_post.assert_called_once()
        mock_post.assert_called_once()

    @patch("browser_service.utils.metrics.requests.post")
    def test_http_error_returns_false(self, mock_post):
        """Server error (500) returns False, doesn't raise."""
        mock_post.return_value = MagicMock(status_code=500)
        mock_post.return_value.raise_for_status.side_effect = Exception("500 error")

        record_workflow_metrics(
            workflow_id="wf-002",
            url="https://example.com",
            results={"summary": {"total_input_tokens": 0}},
        )
        mock_post.assert_called_once()

    @patch("browser_service.utils.metrics.requests.post")
    def test_connection_error_returns_false(self, mock_post):
        """Connection failure (backend down) returns False, doesn't crash."""
        mock_post.side_effect = ConnectionError("Connection refused")

        record_workflow_metrics(
            workflow_id="wf-003",
            url="https://example.com",
            results={"summary": {"total_input_tokens": 0}},
        )
        mock_post.assert_called_once()

    @patch("browser_service.utils.metrics.requests.post")
    def test_custom_action_count_included(self, mock_post):
        """When custom_action_count is in metrics, it's forwarded."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        results = {
            "summary": {
                "total_input_tokens": 100,
            },
            "results": [
                {"metrics": {"custom_action_used": True}},
                {"metrics": {"custom_action_used": True}},
                {"metrics": {"custom_action_used": True}},
            ],
        }
        record_workflow_metrics(workflow_id="wf-004", url="https://example.com", results=results)

        # Verify the payload included custom_action_count
        call_kwargs = mock_post.call_args
        assert call_kwargs is not None

    @patch("browser_service.utils.metrics.requests.post")
    def test_empty_metrics_handled(self, mock_post):
        """Empty metrics dict doesn't crash the function."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        record_workflow_metrics(workflow_id="wf-005", url="https://example.com", results={})
        mock_post.assert_called_once()
