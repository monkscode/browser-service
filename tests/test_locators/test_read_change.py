"""read_change: inputs that must never reach the browser, and errors that must degrade to "unknown"."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from playwright.async_api import Error as PlaywrightError

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


_LOGGER = "browser_service.locators.read_change"


@pytest.mark.asyncio
async def test_playwright_error_in_a_read_is_info_and_unknown(caplog):
    page = MagicMock()
    page.locator.return_value.first.element_handle = AsyncMock(side_effect=PlaywrightError("gone"))
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert await read_changed_since(page, "css=#x", "<p id='x'>a</p>") == "unknown"
    assert [r.levelno for r in caplog.records if r.name == _LOGGER] == [logging.INFO]


@pytest.mark.asyncio
async def test_timeout_in_a_read_is_info_and_unknown(caplog):
    page = MagicMock()
    page.locator.return_value.first.element_handle = AsyncMock(side_effect=TimeoutError())
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert await read_changed_since(page, "css=#x", "<p id='x'>a</p>") == "unknown"
    assert [r.levelno for r in caplog.records if r.name == _LOGGER] == [logging.INFO]


@pytest.mark.asyncio
async def test_unexpected_error_in_a_read_is_one_warning_with_message_and_unknown(caplog):
    page = MagicMock()
    page.locator.return_value.first.element_handle = AsyncMock(side_effect=TypeError("boom"))
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert await read_changed_since(page, "css=#x", "<p id='x'>a</p>") == "unknown"
    records = [r for r in caplog.records if r.name == _LOGGER]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert "TypeError" in records[0].getMessage() and "boom" in records[0].getMessage()


@pytest.mark.asyncio
async def test_playwright_error_in_a_snapshot_is_info_and_none(caplog):
    page = MagicMock()
    page.content = AsyncMock(side_effect=PlaywrightError("closed"))
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert await snapshot_html(page) is None
    assert [r.levelno for r in caplog.records if r.name == _LOGGER] == [logging.INFO]


@pytest.mark.asyncio
async def test_unexpected_error_in_a_snapshot_is_one_warning_with_message_and_none(caplog):
    page = MagicMock()
    page.content = AsyncMock(side_effect=TypeError("boom"))
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert await snapshot_html(page) is None
    records = [r for r in caplog.records if r.name == _LOGGER]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert "TypeError" in records[0].getMessage() and "boom" in records[0].getMessage()
