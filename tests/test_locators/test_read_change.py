"""read_change: inputs that must never reach the browser, and errors that must degrade to "unknown"."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from playwright.async_api import Error as PlaywrightError

from browser_service.locators.read_change import (
    came_back_from_statuses,
    read_came_back,
    read_changed_since,
    snapshot_html,
)


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


# ---- came_back (R16): did the read's value come back to an earlier value? ----------------------
# A status per kept page: "same" = the read's text there equals its live text; "changed",
# "absent", "ambiguous" = it differs there. came_back = some page i is "same" and a LATER page
# differs. Anything else the engine could say ("unknown": not comparable) is fail-safe True.


@pytest.mark.parametrize(
    "statuses,expected",
    [
        ([], False),
        (["same", "changed"], True),  # [0, 1]; live 0
        (["changed", "same"], False),  # [0, 1]; live 1
        (["changed", "changed"], False),  # [Page 1, Page 1]; live Page 2
        (["same", "changed", "same"], True),  # [0, 1, 0]; live 0 (Z1)
        (["changed", "same"], False),  # Z2: [0, 1]; live 1
        (["absent", "same"], False),  # [absent, 0]; live 0
        (["same", "absent"], True),  # [0, absent]; live 0
        (["changed", "changed"], False),  # [Samsung, Sony]; live MacBook
        (["same", "ambiguous"], True),  # ambiguous later counts as different
        (["ambiguous", "same"], False),  # ambiguous earlier is not an earlier value
        (["absent", "ambiguous", "changed"], False),
        (["same", "same"], False),  # never different after the match
        (["unknown"], True),  # not comparable (a <noscript> kept match)
        (["changed", "unknown", "changed"], True),
        (["same", "gap"], True),  # a status the engine never returns for a page: fail-safe
    ],
)
def test_came_back_from_statuses_truth_table(statuses, expected):
    assert came_back_from_statuses(statuses) is expected


@pytest.mark.asyncio
async def test_no_kept_page_is_not_came_back_and_never_touches_the_page():
    page = MagicMock()
    assert await read_came_back(page, "css=#x", []) is False
    page.locator.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "locator,htmls",
    [
        ("css=#x", ["<p>a</p>", None]),  # a gap in the kept pages
        ("css=#x", [None]),
        ("css=#x", ["<p>a</p>", ""]),  # an empty page is a gap too
        ("", ["<p>a</p>"]),
        ('iframe[id="f"] >>> id=x', ["<p>a</p>"]),  # frame hop
    ],
)
async def test_unanswerable_came_back_inputs_are_true_without_touching_the_page(locator, htmls):
    page = MagicMock()
    assert await read_came_back(page, locator, htmls) is True
    page.locator.assert_not_called()


@pytest.mark.asyncio
async def test_an_element_handle_of_none_is_came_back():
    page = MagicMock()
    page.locator.return_value.first.element_handle = AsyncMock(return_value=None)
    assert await read_came_back(page, "css=#x", ["<p id='x'>a</p>"]) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["unknown", None, ["same"], ["same", "gap"]])
async def test_an_engine_answer_that_is_not_one_status_per_page_is_came_back(answer):
    page = MagicMock()
    page.locator.return_value.first.element_handle = AsyncMock(return_value=object())
    page.evaluate = AsyncMock(return_value=answer)
    assert await read_came_back(page, "css=#x", ["<p>a</p>", "<p>b</p>"]) is True


@pytest.mark.asyncio
async def test_the_engine_statuses_go_through_the_formula():
    page = MagicMock()
    page.locator.return_value.first.element_handle = AsyncMock(return_value=object())
    page.evaluate = AsyncMock(return_value=["same", "changed"])
    assert await read_came_back(page, "css=#x", ["<p>a</p>", "<p>b</p>"]) is True
    page.evaluate = AsyncMock(return_value=["changed", "same"])
    assert await read_came_back(page, "css=#x", ["<p>a</p>", "<p>b</p>"]) is False
    # ONE evaluate call carries every kept page
    assert page.evaluate.await_count == 1
    assert page.evaluate.await_args.args[1][0] == ["<p>a</p>", "<p>b</p>"]


@pytest.mark.asyncio
async def test_a_page_error_in_came_back_is_true_not_raised():
    page = MagicMock()
    page.locator.return_value.first.element_handle = AsyncMock(side_effect=RuntimeError("detached"))
    assert await read_came_back(page, "css=#x", ["<p id='x'>a</p>"]) is True


@pytest.mark.asyncio
async def test_playwright_error_in_came_back_is_info_and_true(caplog):
    page = MagicMock()
    page.locator.return_value.first.element_handle = AsyncMock(side_effect=PlaywrightError("gone"))
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert await read_came_back(page, "css=#x", ["<p id='x'>a</p>"]) is True
    assert [r.levelno for r in caplog.records if r.name == _LOGGER] == [logging.INFO]


@pytest.mark.asyncio
async def test_timeout_in_came_back_is_info_and_true(caplog):
    page = MagicMock()
    page.locator.return_value.first.element_handle = AsyncMock(side_effect=TimeoutError())
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert await read_came_back(page, "css=#x", ["<p id='x'>a</p>"]) is True
    assert [r.levelno for r in caplog.records if r.name == _LOGGER] == [logging.INFO]


@pytest.mark.asyncio
async def test_a_slow_evaluation_times_out_to_true(monkeypatch):
    import asyncio

    monkeypatch.setattr("browser_service.locators.read_change.EVAL_TIMEOUT_S", 0.01)
    page = MagicMock()
    page.locator.return_value.first.element_handle = AsyncMock(return_value=object())

    async def _slow(*_a, **_k):
        await asyncio.sleep(1)

    page.evaluate = _slow
    assert await read_came_back(page, "css=#x", ["<p>a</p>"]) is True


@pytest.mark.asyncio
async def test_unexpected_error_in_came_back_is_one_warning_with_message_and_true(caplog):
    page = MagicMock()
    page.locator.return_value.first.element_handle = AsyncMock(side_effect=TypeError("boom"))
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        assert await read_came_back(page, "css=#x", ["<p id='x'>a</p>"]) is True
    records = [r for r in caplog.records if r.name == _LOGGER]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert "TypeError" in records[0].getMessage() and "boom" in records[0].getMessage()
