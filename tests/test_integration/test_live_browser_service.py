"""
Live integration tests for browser-service (Tier 2).

Purpose: Verify that the browser-service Flask app is running and responding
         to health/status endpoints.  These tests only hit lightweight endpoints
         ($0 cost — no LLM calls, no browser automation).

Requires: Browser-service running on localhost:4999
          Start with: python run.py

Tests:
  - /health returns status: healthy
  - /probe returns status: alive
  - / returns service info
  - /tasks returns task list (empty when fresh)
"""

import pytest
import requests

# All live tests require the service to be running
pytestmark = pytest.mark.integration

SERVICE_URL = "http://localhost:4999"


class TestLiveBrowserService:
    """Live health-check tests for running browser-service."""

    def test_health_endpoint(self):
        """GET /health returns healthy status."""
        resp = requests.get(f"{SERVICE_URL}/health", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "active_tasks" in data

    def test_probe_endpoint(self):
        """GET /probe returns alive status."""
        resp = requests.get(f"{SERVICE_URL}/probe", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "alive"

    def test_root_endpoint(self):
        """GET / returns service information."""
        resp = requests.get(f"{SERVICE_URL}/", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert "service" in data
        assert data["status"] == "running"

    def test_tasks_list_empty(self):
        """GET /tasks returns empty list on fresh service."""
        resp = requests.get(f"{SERVICE_URL}/tasks", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        assert isinstance(data["tasks"], list)
