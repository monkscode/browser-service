"""
Unit tests for browser_service.utils.metrics — workflow metrics recording.

Purpose: Metrics are recorded via HTTP POST to the NL backend's /api/workflow-metrics/record
         endpoint.  If recording silently fails, we lose cost and performance data.
         If it crashes, it takes down the workflow.

`record_workflow_metrics` returns None on every path — success and failure alike —
so there is no return value to assert on. What IS observable, and what the NL
backend actually depends on, is the POST: the endpoint URL, the payload keys and
their values, and the guarantee that no transport failure escapes. These tests
assert that, not merely that requests.post was reached.

Tests:
  - Successful recording posts the full payload to the right endpoint
  - Summary fields are mapped onto the payload under the backend's key names
  - Cost is read from the key the workflow actually emits (`actual_cost`)
  - Per-element cost is derived, and a zero-element workflow does not divide by zero
  - Missing summary keys fall back to documented defaults, not KeyError
  - custom_action_usage_count counts only results with custom_action_used truthy
  - HTTP error (4xx/5xx) is swallowed — the workflow keeps running
  - Connection error (backend down) is swallowed
  - backend_port / session_id are honoured
"""

from unittest.mock import MagicMock, patch

import pytest

from browser_service.utils.metrics import record_workflow_metrics


def _ok_response():
    resp = MagicMock(status_code=200)
    resp.raise_for_status = MagicMock()
    return resp


def _posted(mock_post):
    """The JSON payload of the single POST that was made."""
    mock_post.assert_called_once()
    return mock_post.call_args.kwargs["json"]


class TestRecordWorkflowMetrics:
    """Tests for the record_workflow_metrics function."""

    @patch("browser_service.utils.metrics.requests.post")
    def test_posts_to_metrics_endpoint(self, mock_post):
        """The POST goes to the backend's record endpoint on the given port."""
        mock_post.return_value = _ok_response()

        record_workflow_metrics(
            workflow_id="wf-001",
            url="https://example.com",
            results={"summary": {}},
            backend_port=9123,
        )

        mock_post.assert_called_once()
        assert mock_post.call_args.args[0] == "http://localhost:9123/api/workflow-metrics/record"
        assert mock_post.call_args.kwargs["timeout"] == 5
        assert mock_post.call_args.kwargs["headers"] == {"Content-Type": "application/json"}

    @patch("browser_service.utils.metrics.requests.post")
    def test_summary_fields_mapped_into_payload(self, mock_post):
        """Every summary field the backend reads is forwarded under its own key.

        The rename (summary["successful"] → payload["successful_elements"]) is the
        part that silently breaks: a typo on either side loses the column without
        raising anywhere.
        """
        mock_post.return_value = _ok_response()

        record_workflow_metrics(
            workflow_id="wf-002",
            url="https://example.com/products",
            session_id="sess-42",
            results={
                "execution_time": 45.2,
                "summary": {
                    "total_elements": 5,
                    "successful": 4,
                    "failed": 1,
                    "success_rate": 0.8,
                    "total_llm_calls": 12,
                    "avg_llm_calls_per_element": 2.4,
                    "actual_cost": 0.07,
                    "custom_actions_enabled": True,
                    "element_approach_metrics": [{"elem_1": "candidate"}],
                },
            },
        )

        payload = _posted(mock_post)
        assert payload == {
            "workflow_id": "wf-002",
            "total_elements": 5,
            "successful_elements": 4,
            "failed_elements": 1,
            "success_rate": 0.8,
            "total_llm_calls": 12,
            "avg_llm_calls_per_element": 2.4,
            "total_cost": 0.07,
            "avg_cost_per_element": 0.07 / 5,
            "custom_actions_enabled": True,
            "custom_action_usage_count": 0,
            "execution_time": 45.2,
            "url": "https://example.com/products",
            "session_id": "sess-42",
            "element_approach_metrics": [{"elem_1": "candidate"}],
        }

    @patch("browser_service.utils.metrics.requests.post")
    def test_missing_summary_keys_use_defaults(self, mock_post):
        """An empty results dict yields zeroed defaults, not a KeyError."""
        mock_post.return_value = _ok_response()

        record_workflow_metrics(workflow_id="wf-003", url="https://example.com", results={})

        payload = _posted(mock_post)
        assert payload["total_elements"] == 0
        assert payload["successful_elements"] == 0
        assert payload["failed_elements"] == 0
        assert payload["success_rate"] == 0.0
        assert payload["total_cost"] == 0.0
        assert payload["custom_actions_enabled"] is False
        assert payload["custom_action_usage_count"] == 0
        assert payload["execution_time"] == 0
        assert payload["session_id"] is None
        assert payload["element_approach_metrics"] == []

    @patch("browser_service.utils.metrics.requests.post")
    def test_cost_is_read_from_the_key_the_workflow_emits(self, mock_post):
        """The summary key carrying cost is `actual_cost`, not `estimated_total_cost`.

        This is the exact shape run_unified_workflow builds (workflow.py, the
        "Cost tracking metrics" block of its summary). Reading any other key
        silently persisted total_cost=0.0 for every browser-service workflow —
        88/88 rows in the backend's workflow_metrics table before this fix.
        """
        mock_post.return_value = _ok_response()

        record_workflow_metrics(
            workflow_id="wf-cost",
            url="https://example.com",
            results={
                "summary": {
                    "total_elements": 4,
                    "successful": 4,
                    "failed": 0,
                    "success_rate": 1.0,
                    "total_llm_calls": 8,
                    "avg_llm_calls_per_element": 2.0,
                    "custom_actions_enabled": True,
                    "total_tokens": 311865,
                    "input_tokens": 300000,
                    "output_tokens": 11865,
                    "cached_tokens": 0,
                    "actual_cost": 0.0512,
                    "element_approach_metrics": [],
                }
            },
        )

        payload = _posted(mock_post)
        assert payload["total_cost"] == 0.0512
        assert payload["avg_cost_per_element"] == 0.0512 / 4

    @patch("browser_service.utils.metrics.requests.post")
    def test_stale_estimated_keys_are_not_read(self, mock_post):
        """Guards the regression: `estimated_*` keys no producer emits stay dead.

        A summary carrying only the old key names must yield 0.0 rather than
        appearing to work — otherwise reintroducing the mismatch looks correct.
        """
        mock_post.return_value = _ok_response()

        record_workflow_metrics(
            workflow_id="wf-stale",
            url="https://example.com",
            results={
                "summary": {
                    "total_elements": 2,
                    "estimated_total_cost": 0.09,
                    "estimated_cost_per_element": 0.045,
                }
            },
        )

        payload = _posted(mock_post)
        assert payload["total_cost"] == 0.0
        assert payload["avg_cost_per_element"] == 0.0

    @patch("browser_service.utils.metrics.requests.post")
    def test_zero_elements_does_not_divide_by_zero(self, mock_post):
        """The failure-path summary reports total_elements=0; deriving the
        per-element average must not raise inside the try block and lose the
        whole POST."""
        mock_post.return_value = _ok_response()

        record_workflow_metrics(
            workflow_id="wf-empty",
            url="https://example.com",
            results={"summary": {"total_elements": 0, "actual_cost": 0.03}},
        )

        payload = _posted(mock_post)
        assert payload["total_cost"] == 0.03
        assert payload["avg_cost_per_element"] == 0.0

    @patch("browser_service.utils.metrics.requests.post")
    def test_custom_action_count_counts_only_used(self, mock_post):
        """Only elements whose metrics say custom_action_used are counted.

        The mixed list is deliberate: a count that returned len(results) — or one
        that ignored the flag — would pass a list where every entry used it.
        """
        mock_post.return_value = _ok_response()

        record_workflow_metrics(
            workflow_id="wf-004",
            url="https://example.com",
            results={
                "summary": {},
                "results": [
                    {"metrics": {"custom_action_used": True}},
                    {"metrics": {"custom_action_used": False}},
                    {"metrics": {}},
                    {},
                    {"metrics": {"custom_action_used": True}},
                ],
            },
        )

        assert _posted(mock_post)["custom_action_usage_count"] == 2

    @pytest.mark.parametrize(
        "failure",
        [
            pytest.param({"side_effect": ConnectionError("Connection refused")}, id="backend-down"),
            pytest.param({"side_effect": ValueError("bad payload")}, id="unexpected-error"),
        ],
    )
    @patch("browser_service.utils.metrics.requests.post")
    def test_transport_failure_never_propagates(self, mock_post, failure):
        """Metrics recording must not take down the workflow it is measuring."""
        mock_post.configure_mock(**failure)

        record_workflow_metrics(
            workflow_id="wf-005",
            url="https://example.com",
            results={"summary": {"total_elements": 1}},
        )

        mock_post.assert_called_once()

    @patch("browser_service.utils.metrics.requests.post")
    def test_http_error_status_is_swallowed(self, mock_post):
        """A 500 from the backend is logged, not raised."""
        mock_post.return_value = MagicMock(status_code=500, text="boom")

        record_workflow_metrics(
            workflow_id="wf-006",
            url="https://example.com",
            results={"summary": {"total_elements": 1}},
        )

        assert _posted(mock_post)["workflow_id"] == "wf-006"
