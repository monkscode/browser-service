"""
Shared pytest fixtures for browser-service test suite.

Provides:
- Flask test client with mocked task processor
- Reusable sample data (elements, locator results)
- Mock Playwright page objects
"""

import pytest
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor


# ---------------------------------------------------------------------------
# Flask app fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flask_app():
    """Create a Flask test app with mocked dependencies."""
    # Patch the global task_processor before importing routes
    with patch("browser_service.api.routes.task_processor") as mock_processor:
        from browser_service.api.routes import app

        app.config["TESTING"] = True
        mock_processor.tasks = {}
        mock_processor.get_task_status.return_value = None
        mock_processor.list_tasks.return_value = []
        yield app, mock_processor


@pytest.fixture
def client(flask_app):
    """Flask test client for HTTP assertions."""
    app, _ = flask_app
    with app.test_client() as c:
        yield c


@pytest.fixture
def mock_processor(flask_app):
    """Access to the mocked TaskProcessor behind the Flask app."""
    _, processor = flask_app
    return processor


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_elements():
    """Reusable list of element dicts as sent by NL backend."""
    return [
        {
            "id": "elem_1",
            "description": "search input field in header",
            "action": "input",
            "value": "shoes",
        },
        {
            "id": "elem_2",
            "description": "submit button in main form",
            "action": "click",
        },
    ]


@pytest.fixture
def sample_element_attributes():
    """Typical element attributes dict returned by extraction."""
    return {
        "tagName": "input",
        "id": "search-box",
        "name": "q",
        "className": "search-input form-control",
        "ariaLabel": "Search products",
        "placeholder": "Search...",
        "title": "",
        "href": "",
        "role": "searchbox",
        "dataTestId": "search-input",
        "type": "text",
        "value": "",
        "xpath": "/html/body/div/header/form/input",
    }


@pytest.fixture
def sample_locator_result():
    """Sample successful locator result from browser-use workflow."""
    return {
        "success": True,
        "locator_mapping": {
            "elem_1": {
                "best_locator": "id=search-box",
                "found": True,
                "element_info": {"tagName": "input", "id": "search-box"},
                "all_locators": ["id=search-box", "name=q"],
            }
        },
        "summary": {"total_elements": 1, "successful": 1, "failed": 0},
    }


# ---------------------------------------------------------------------------
# Mock Playwright page
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_page():
    """Mock Playwright page object with common methods stubbed."""
    page = MagicMock()
    page.url = "https://example.com"

    # query_selector_all returns a list of mock elements
    mock_element = MagicMock()
    mock_element.text_content.return_value = "Example"
    page.query_selector_all.return_value = [mock_element]
    page.locator.return_value.count.return_value = 1

    return page


# ---------------------------------------------------------------------------
# Task processor (standalone, without Flask)
# ---------------------------------------------------------------------------

@pytest.fixture
def task_processor():
    """Standalone TaskProcessor instance for unit tests."""
    from browser_service.tasks.processor import TaskProcessor

    executor = ThreadPoolExecutor(max_workers=1)
    processor = TaskProcessor(executor=executor)
    yield processor
    executor.shutdown(wait=False)
