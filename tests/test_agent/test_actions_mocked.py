"""
Mocked integration tests for browser_service.agent.actions — find_unique_locator_action.

Purpose: find_unique_locator_action is the core deterministic locator engine.
         It takes coordinates, element info, and a Playwright page, then runs
         21 locator strategies.  These tests mock the Playwright page to verify
         error handling, validation flow, and result formatting WITHOUT needing
         a real browser.

Every test here asserts the OBSERVABLE contract — the returned dict's keys, or
whether the 21-strategy cascade was delegated to — never merely that a result
came back. `find_unique_locator_action` always returns a dict, so
`assert result is not None` is unconditionally true and cannot fail; those
assertions were removed.

The cascade (`find_unique_locator_at_coordinates`) is patched with a sentinel in
the candidate-path tests. Two reasons: it isolates the candidate accept/reject
decision, which is what these tests are about, and it stops the real cascade
from traversing an AsyncMock page — which produced garbage work and a stream of
"coroutine was never awaited" RuntimeWarnings that no assertion could see.

Tests:
  - Page None → PageObjectError, cascade never reached
  - Negative / non-numeric coordinates → InvalidCoordinatesError, cascade not called
  - Empty element_id → InvalidElementIdError
  - Candidate locator unique → fast-path accept, cascade SKIPPED
  - Candidate not unique / not found / blank / Playwright timeout → delegates to cascade
  - element_data and candidate-less calls → forwarded to the cascade intact
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Sentinel returned by the patched cascade. Distinct from any candidate-path
# result, so "did we fall through?" is answerable by identity, not by guesswork.
CASCADE_RESULT = {
    "element_id": "sentinel",
    "found": True,
    "best_locator": "css=.from-cascade",
    "source": "cascade",
}


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


@pytest.fixture
def cascade():
    """Patch the 21-strategy cascade with a recording sentinel.

    actions.py imports it from `browser_service.locators` at call time, so the
    patch target is the package attribute, not a module-level binding.
    """
    spy = AsyncMock(return_value=CASCADE_RESULT)
    with patch("browser_service.locators.find_unique_locator_at_coordinates", new=spy):
        yield spy


class TestFindUniqueLocatorAction:
    """Tests for find_unique_locator_action with mocked Playwright."""

    @pytest.mark.asyncio
    async def test_page_none_returns_error(self, cascade):
        """page=None is rejected up front with a typed error, before any work."""
        from browser_service.agent.actions import find_unique_locator_action

        result = await find_unique_locator_action(
            page=None,
            x=100,
            y=200,
            element_id="elem_1",
            element_description="search input",
        )
        assert result["found"] is False
        assert result["error_type"] == "PageObjectError"
        assert result["element_id"] == "elem_1"
        assert result["validated"] is False
        cascade.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "x, y",
        [
            pytest.param(-10, -20, id="both-negative"),
            pytest.param(-10, 20, id="x-only"),
            pytest.param(10, -20, id="y-only"),
        ],
    )
    async def test_negative_coords_rejected(self, mock_playwright_page, cascade, x, y):
        """Negative coordinates are rejected — not silently passed to the cascade.

        Each axis is exercised alone. With both negative, an `and` guard reads the
        same as the correct `or`, so a both-negative case cannot pin the operator.
        """
        from browser_service.agent.actions import find_unique_locator_action

        result = await find_unique_locator_action(
            page=mock_playwright_page,
            x=x,
            y=y,
            element_id="elem_1",
            element_description="offscreen element",
        )
        assert result["found"] is False
        assert result["error_type"] == "InvalidCoordinatesError"
        assert result["coordinates"] == {"x": x, "y": y}
        cascade.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_numeric_coords_rejected(self, mock_playwright_page, cascade):
        """A non-numeric coordinate on EITHER axis is rejected.

        The guard is `not isinstance(x) or not isinstance(y)`; parametrising both
        axes is what pins the `or` — an `and` would let a half-bad pair through.
        """
        from browser_service.agent.actions import find_unique_locator_action

        for bad_x, bad_y in (("100", 200), (100, None)):
            result = await find_unique_locator_action(
                page=mock_playwright_page,
                x=bad_x,
                y=bad_y,
                element_id="elem_1",
                element_description="element with junk coords",
            )
            assert result["found"] is False, f"x={bad_x!r} y={bad_y!r} was not rejected"
            assert result["error_type"] == "InvalidCoordinatesError"
        cascade.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_element_id_rejected(self, mock_playwright_page, cascade):
        """An empty element_id is rejected — results would be unattributable."""
        from browser_service.agent.actions import find_unique_locator_action

        result = await find_unique_locator_action(
            page=mock_playwright_page,
            x=100,
            y=200,
            element_id="",
            element_description="nameless element",
        )
        assert result["found"] is False
        assert result["error_type"] == "InvalidElementIdError"
        cascade.assert_not_called()

    @pytest.mark.asyncio
    async def test_candidate_locator_unique(self, mock_playwright_page, cascade):
        """count==1 → accept the candidate verbatim and SKIP the cascade."""
        from browser_service.agent.actions import find_unique_locator_action

        mock_playwright_page.locator.return_value.count = AsyncMock(return_value=1)

        result = await find_unique_locator_action(
            page=mock_playwright_page,
            x=100,
            y=200,
            element_id="elem_1",
            element_description="submit button",
            candidate_locator="id=submit-btn",
        )

        assert result["found"] is True
        assert result["best_locator"] == "id=submit-btn"
        assert result["element_id"] == "elem_1"
        assert result["stability"] == "stable"

        # The whole point of the fast path: the 21 strategies are not run.
        cascade.assert_not_called()

        # all_locators carries the candidate provenance nlrf reads downstream.
        assert len(result["all_locators"]) == 1
        entry = result["all_locators"][0]
        assert entry["type"] == "candidate"
        assert entry["priority"] == 0
        assert entry["count"] == 1
        assert entry["unique"] is True
        assert entry["valid"] is True
        assert entry["validated"] is True
        assert entry["validation_method"] == "playwright"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "count, candidate",
        [
            pytest.param(3, ".item", id="not-unique"),
            pytest.param(0, "id=nonexistent", id="not-found"),
            pytest.param(2, "#dup", id="exactly-two"),
        ],
    )
    async def test_non_unique_candidate_delegates_to_cascade(
        self, mock_playwright_page, cascade, count, candidate
    ):
        """Any count other than exactly 1 must fall through to the 21 strategies.

        count==2 is included deliberately: a `count >= 1` or `count != 0` accept
        gate would pass the 0-and-3 cases and still be wrong.
        """
        from browser_service.agent.actions import find_unique_locator_action

        mock_playwright_page.locator.return_value.count = AsyncMock(return_value=count)

        result = await find_unique_locator_action(
            page=mock_playwright_page,
            x=100,
            y=200,
            element_id="elem_1",
            element_description="ambiguous element",
            candidate_locator=candidate,
        )

        cascade.assert_called_once()
        assert result == CASCADE_RESULT

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "candidate",
        [pytest.param("", id="empty"), pytest.param("   ", id="whitespace")],
    )
    async def test_blank_candidate_delegates_to_cascade(
        self, mock_playwright_page, cascade, candidate
    ):
        """A blank candidate is not a locator — never validated, never accepted."""
        from browser_service.agent.actions import find_unique_locator_action

        result = await find_unique_locator_action(
            page=mock_playwright_page,
            x=100,
            y=200,
            element_id="elem_1",
            element_description="element with a blank candidate",
            candidate_locator=candidate,
        )

        cascade.assert_called_once()
        assert result == CASCADE_RESULT
        # A blank string must not reach Playwright at all.
        mock_playwright_page.locator.assert_not_called()

    @pytest.mark.asyncio
    async def test_candidate_locator_timeout(self, mock_playwright_page, cascade):
        """Playwright timeout while counting the candidate → fall through, don't raise."""
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

        cascade.assert_called_once()
        assert result == CASCADE_RESULT

    @pytest.mark.asyncio
    async def test_element_data_forwarded_to_cascade(self, mock_playwright_page, cascade):
        """element_data reaches the cascade intact — dropping it silently degrades locators."""
        from browser_service.agent.actions import find_unique_locator_action

        element_data = {
            "tagName": "input",
            "id": "search-input",
            "name": "q",
            "type": "text",
        }

        result = await find_unique_locator_action(
            page=mock_playwright_page,
            x=500,
            y=300,
            element_id="elem_2",
            element_description="search box",
            element_data=element_data,
        )

        cascade.assert_called_once()
        kwargs = cascade.call_args.kwargs
        assert kwargs["element_data"] == element_data
        assert kwargs["element_id"] == "elem_2"
        assert kwargs["x"] == 500
        assert kwargs["y"] == 300
        assert result == CASCADE_RESULT

    @pytest.mark.asyncio
    async def test_no_candidate_no_element_data(self, mock_playwright_page, cascade):
        """No candidate + no element_data → straight to the coordinate cascade."""
        from browser_service.agent.actions import find_unique_locator_action

        result = await find_unique_locator_action(
            page=mock_playwright_page,
            x=300,
            y=400,
            element_id="elem_3",
            element_description="unknown element",
        )

        cascade.assert_called_once()
        kwargs = cascade.call_args.kwargs
        assert kwargs["element_data"] is None
        assert kwargs["search_context"] is mock_playwright_page
        assert result == CASCADE_RESULT


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
