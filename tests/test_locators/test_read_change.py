"""read_change: inputs that must never reach the browser, and errors that must degrade to "unknown"."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from browser_service.locators.read_change import read_changed_since, snapshot_html


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "locator,html",
    [
        ("css=#x", None),
        ("css=#x", ""),
        ("", "<p>x</p>"),
        ('iframe[id="f"] >>> id=x', "<p>x</p>"),  # frame hop: main-frame HTML cannot answer
    ],
)
async def test_unanswerable_inputs_are_unknown_without_touching_the_page(locator, html):
    page = MagicMock()
    assert await read_changed_since(page, locator, html) == "unknown"
    page.locator.assert_not_called()


@pytest.mark.asyncio
async def test_a_page_error_is_unknown_not_raised():
    page = MagicMock()
    page.locator.return_value.first.element_handle = AsyncMock(side_effect=RuntimeError("detached"))
    assert await read_changed_since(page, "css=#x", "<p id='x'>a</p>") == "unknown"


@pytest.mark.asyncio
async def test_snapshot_failure_is_none():
    page = MagicMock()
    page.content = AsyncMock(side_effect=RuntimeError("closed"))
    assert await snapshot_html(page) is None
