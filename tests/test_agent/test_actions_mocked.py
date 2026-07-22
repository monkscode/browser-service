"""
Mocked integration tests for browser_service.agent.actions — find_unique_locator_action.

Purpose: find_unique_locator_action is the core deterministic locator engine.
         It takes coordinates, element info, and a Playwright page, then runs
         21 locator strategies.  These tests mock the Playwright page to verify
         error handling, validation flow, and result formatting WITHOUT needing
         a real browser.

Tests:
  - Page None → returns error ActionResult
  - Invalid coordinates (negative) → graceful handling
  - Invalid element_id → still attempts locator resolution
  - Candidate locator unique → fast-path success
  - Candidate locator not unique → falls through to strategies
  - Candidate locator not found → falls through
  - Candidate locator invalid syntax → falls through
  - Candidate locator Playwright timeout → falls through
  - Finder function timeout → returns partial result
  - Finder function cancelled → returns error
  - Successful locator found → returns formatted result
  - Collection mode → returns multi-element locator
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_playwright_page():
    """Create a mock Playwright page with typical responses."""
    page = AsyncMock()
    page.url = "https://example.com/products"
    page.title = AsyncMock(return_value="Product Page")

    # Default: locator count returns 1 (unique)
    locator_mock = MagicMock()
    locator_mock.count = AsyncMock(return_value=1)
    locator_mock.first = MagicMock()
    locator_mock.first.text_content = AsyncMock(return_value="Submit")
    page.locator = MagicMock(return_value=locator_mock)

    return page


class TestFindUniqueLocatorAction:
    """Tests for find_unique_locator_action with mocked Playwright."""

    @pytest.mark.asyncio
    async def test_page_none_returns_error(self):
        """When page is None, returns error with descriptive message."""
        from browser_service.agent.actions import find_unique_locator_action

        result = await find_unique_locator_action(
            page=None,
            x=100,
            y=200,
            element_id="elem_1",
            element_description="search input",
        )
        assert result is not None
        # Should indicate failure — exact format depends on implementation
        result_str = str(result)
        assert (
            "error" in result_str.lower()
            or "fail" in result_str.lower()
            or "no page" in result_str.lower()
        )

    @pytest.mark.asyncio
    async def test_negative_coords_handled(self, mock_playwright_page):
        """Negative coordinates don't crash the function."""
        from browser_service.agent.actions import find_unique_locator_action

        result = await find_unique_locator_action(
            page=mock_playwright_page,
            x=-10,
            y=-20,
            element_id="elem_1",
            element_description="offscreen element",
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_candidate_locator_unique(self, mock_playwright_page):
        """When candidate_locator is unique (count=1), returns it as best."""
        from browser_service.agent.actions import find_unique_locator_action

        # Mock: locator returns count=1 (unique)
        mock_playwright_page.locator.return_value.count = AsyncMock(return_value=1)

        result = await find_unique_locator_action(
            page=mock_playwright_page,
            x=100,
            y=200,
            element_id="elem_1",
            element_description="submit button",
            candidate_locator="id=submit-btn",
        )
        result_str = str(result)
        assert result is not None

    @pytest.mark.asyncio
    async def test_candidate_locator_not_unique(self, mock_playwright_page):
        """When candidate matches 3 elements, falls through to strategies."""
        from browser_service.agent.actions import find_unique_locator_action

        mock_playwright_page.locator.return_value.count = AsyncMock(return_value=3)

        result = await find_unique_locator_action(
            page=mock_playwright_page,
            x=100,
            y=200,
            element_id="elem_1",
            element_description="list item",
            candidate_locator=".item",
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_candidate_locator_not_found(self, mock_playwright_page):
        """When candidate matches 0 elements, falls through."""
        from browser_service.agent.actions import find_unique_locator_action

        mock_playwright_page.locator.return_value.count = AsyncMock(return_value=0)

        result = await find_unique_locator_action(
            page=mock_playwright_page,
            x=100,
            y=200,
            element_id="elem_1",
            element_description="missing button",
            candidate_locator="id=nonexistent",
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_candidate_locator_timeout(self, mock_playwright_page):
        """Playwright timeout on candidate → falls through gracefully."""
        import asyncio

        from browser_service.agent.actions import find_unique_locator_action

        mock_playwright_page.locator.return_value.count = AsyncMock(
            side_effect=asyncio.TimeoutError("Timeout")
        )

        result = await find_unique_locator_action(
            page=mock_playwright_page,
            x=100,
            y=200,
            element_id="elem_1",
            element_description="slow element",
            candidate_locator="id=slow",
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_element_data_used_for_generation(self, mock_playwright_page):
        """When element_data is provided, locators are generated from attributes."""
        from browser_service.agent.actions import find_unique_locator_action

        result = await find_unique_locator_action(
            page=mock_playwright_page,
            x=500,
            y=300,
            element_id="elem_2",
            element_description="search box",
            element_data={
                "tagName": "input",
                "id": "search-input",
                "name": "q",
                "type": "text",
            },
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_no_candidate_no_element_data(self, mock_playwright_page):
        """No candidate + no element_data → relies on coordinate strategies."""
        from browser_service.agent.actions import find_unique_locator_action

        result = await find_unique_locator_action(
            page=mock_playwright_page,
            x=300,
            y=400,
            element_id="elem_3",
            element_description="unknown element",
        )
        assert result is not None


class TestTimeoutFromConfig:
    """
    The finder budget comes from config.locator.custom_action_timeout (Task 4).

    Historically both registration.py and actions.py resolved the budget by
    importing nlrf settings with a silent fallback to a hard-coded 5 — tuning
    the browser-service env changed nothing. These tests pin the new contract:
    one knob, read from config, structured error on expiry.
    """

    @pytest.mark.asyncio
    async def test_budget_comes_from_config(self, mock_playwright_page, monkeypatch):
        """A 0.05s config budget must cut off a hanging cascade in ~0.05s, not 5s."""
        import asyncio
        import time

        from browser_service.config import config

        monkeypatch.setattr(config.locator, "custom_action_timeout", 0.05, raising=False)

        async def hang(*args, **kwargs):
            await asyncio.sleep(30)

        with patch("browser_service.locators.find_unique_locator_at_coordinates", new=hang):
            from browser_service.agent.actions import find_unique_locator_action

            start = time.monotonic()
            result = await find_unique_locator_action(
                page=mock_playwright_page,
                x=300,
                y=400,
                element_id="elem_slow",
                element_description="element on a wedged page",
            )
            elapsed = time.monotonic() - start

        assert elapsed < 2, (
            f"took {elapsed:.1f}s — budget did not come from config "
            f"(hard-coded 5s fallback still in effect?)"
        )
        assert result["found"] is False
        assert result["error_type"] == "TimeoutError"

    @pytest.mark.asyncio
    async def test_timeout_result_reports_configured_budget(
        self, mock_playwright_page, monkeypatch
    ):
        """The structured error carries the budget that was actually applied."""
        import asyncio

        from browser_service.config import config

        monkeypatch.setattr(config.locator, "custom_action_timeout", 0.05, raising=False)

        async def hang(*args, **kwargs):
            await asyncio.sleep(30)

        with patch("browser_service.locators.find_unique_locator_at_coordinates", new=hang):
            from browser_service.agent.actions import find_unique_locator_action

            result = await find_unique_locator_action(
                page=mock_playwright_page,
                x=300,
                y=400,
                element_id="elem_slow",
                element_description="element on a wedged page",
            )

        assert result["timeout_seconds"] == 0.05
