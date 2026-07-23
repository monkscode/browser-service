"""
Unit tests for browser_service.locators.dom_probe.

These tests mock Playwright's page.evaluate() because the probe's logic
that we care about for unit testing lives in the Python wrapper:
input validation, result-shape normalization, defensive error handling.
The JS-side walk is exercised end-to-end in the integration fixtures
under tests/test_locators/test_locator_fixtures.py.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from browser_service.locators.dom_probe import probe_specialized_type


def _make_page(evaluate_return=None, evaluate_raises=None):
    page = MagicMock()
    if evaluate_raises is not None:
        page.evaluate = AsyncMock(side_effect=evaluate_raises)
    else:
        page.evaluate = AsyncMock(return_value=evaluate_return)
    return page


# ----------------------------------------------------------------------
# Input validation — the probe never raises; it returns _UNCONFIRMED.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsupported_type_returns_unconfirmed_without_calling_js():
    page = _make_page()
    result = await probe_specialized_type(page, suspected_type="button", coords=(100, 100))
    assert result == {
        "confirmed": False,
        "framework": "",
        "signals": [],
        "anchor_xpath": "",
        "anchor_tag": "",
    }
    page.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_no_coords_or_xpath_returns_unconfirmed_without_calling_js():
    page = _make_page()
    result = await probe_specialized_type(page, suspected_type="dropdown")
    assert result["confirmed"] is False
    page.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_coords_alone_invokes_js():
    page = _make_page(
        evaluate_return={
            "confirmed": True,
            "framework": "tom-select",
            "signals": ["self:class:ts-control"],
            "anchor_xpath": "/html/body/div[1]/div[2]",
            "anchor_tag": "div",
        }
    )
    result = await probe_specialized_type(
        page,
        suspected_type="dropdown",
        coords=(450.0, 320.0),
    )
    assert result["confirmed"] is True
    assert result["framework"] == "tom-select"
    assert result["anchor_tag"] == "div"
    page.evaluate.assert_awaited_once()
    # Verify the args structure passed to evaluate
    call_args = page.evaluate.call_args
    args_dict = call_args[0][1]
    assert args_dict["coords"] == {"x": 450.0, "y": 320.0}
    assert args_dict["suspectedType"] == "dropdown"
    assert args_dict["candidateXPath"] is None
    assert isinstance(args_dict["frameworkPatterns"], list)
    assert len(args_dict["frameworkPatterns"]) > 0  # framework table populated


@pytest.mark.asyncio
async def test_xpath_alone_invokes_js():
    page = _make_page(
        evaluate_return={
            "confirmed": True,
            "framework": "",
            "signals": ["self:role:combobox"],
            "anchor_xpath": "/html/body/div",
            "anchor_tag": "div",
        }
    )
    result = await probe_specialized_type(
        page,
        suspected_type="dropdown",
        candidate_xpath="//div[@id='rate_group']",
    )
    assert result["confirmed"] is True
    args_dict = page.evaluate.call_args[0][1]
    assert args_dict["coords"] is None
    assert args_dict["candidateXPath"] == "//div[@id='rate_group']"


# ----------------------------------------------------------------------
# Result normalization — defensive against JS shape drift.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_result_keys_always_present_even_when_js_returns_partial():
    # JS returns only `confirmed`; wrapper fills defaults for the rest.
    page = _make_page(evaluate_return={"confirmed": False})
    result = await probe_specialized_type(
        page,
        suspected_type="checkbox",
        coords=(0, 0),
    )
    assert result == {
        "confirmed": False,
        "framework": "",
        "signals": [],
        "anchor_xpath": "",
        "anchor_tag": "",
    }


@pytest.mark.asyncio
async def test_result_signals_coerced_to_list_when_js_returns_none():
    page = _make_page(
        evaluate_return={
            "confirmed": True,
            "framework": None,
            "signals": None,
            "anchor_xpath": None,
            "anchor_tag": None,
        }
    )
    result = await probe_specialized_type(
        page,
        suspected_type="radio",
        coords=(50, 50),
    )
    assert result["signals"] == []
    assert result["framework"] == ""
    assert result["anchor_xpath"] == ""
    assert result["anchor_tag"] == ""


@pytest.mark.asyncio
async def test_non_dict_js_return_yields_unconfirmed():
    page = _make_page(evaluate_return="oops not a dict")
    result = await probe_specialized_type(
        page,
        suspected_type="dropdown",
        coords=(10, 10),
    )
    assert result["confirmed"] is False
    assert result["signals"] == []


# ----------------------------------------------------------------------
# Error handling — never raises; returns _UNCONFIRMED with a logged warning.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_js_eval_exception_returns_unconfirmed():
    page = _make_page(evaluate_raises=RuntimeError("page crashed"))
    result = await probe_specialized_type(
        page,
        suspected_type="dropdown",
        coords=(10, 10),
    )
    assert result["confirmed"] is False
    assert result["signals"] == []


@pytest.mark.asyncio
async def test_js_timeout_does_not_propagate():
    page = _make_page(evaluate_raises=TimeoutError("evaluate timed out"))
    result = await probe_specialized_type(
        page,
        suspected_type="collection",
        candidate_xpath="//table[1]",
    )
    assert result["confirmed"] is False


# ----------------------------------------------------------------------
# Coord type coercion — JS expects floats; ints from caller are coerced.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_int_coords_coerced_to_float_for_js():
    page = _make_page(evaluate_return={"confirmed": False})
    await probe_specialized_type(
        page,
        suspected_type="dropdown",
        coords=(100, 200),
    )
    args = page.evaluate.call_args[0][1]
    assert args["coords"]["x"] == 100.0
    assert isinstance(args["coords"]["x"], float)
    assert args["coords"]["y"] == 200.0
    assert isinstance(args["coords"]["y"], float)


# ----------------------------------------------------------------------
# Framework patterns table — kept in sync with classifier.
# ----------------------------------------------------------------------


def test_framework_patterns_match_classifier_table():
    # The probe should expose every framework the classifier knows about.
    # Catches drift if a new framework is added to one but not the other.
    import json

    from browser_service.locators.classifier import _DROPDOWN_FRAMEWORK_PATTERNS
    from browser_service.locators.dom_probe import _FRAMEWORK_PATTERNS_JSON

    probe_table = json.loads(_FRAMEWORK_PATTERNS_JSON)
    classifier_names = {name for name, _ in _DROPDOWN_FRAMEWORK_PATTERNS}
    probe_names = {entry[0] for entry in probe_table}
    assert classifier_names == probe_names


# ----------------------------------------------------------------------
# Confirmation flow — the dispatcher consumes these signals.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirmation_propagates_anchor_xpath_for_handler_retargeting():
    # Bug 3 pattern: coords land on outer .row, probe finds .ts-wrapper
    # at an ancestor's sibling. The handler can re-anchor using
    # anchor_xpath rather than the original (wrong) element_data.
    page = _make_page(
        evaluate_return={
            "confirmed": True,
            "framework": "tom-select",
            "signals": [
                "ancestor[2].sib:class:ts-wrapper",
                "ancestor[2].sib.descendant:class:ts-control",
            ],
            "anchor_xpath": "/html/body/div/form/div[3]/div[2]/div[1]",
            "anchor_tag": "div",
        }
    )
    result = await probe_specialized_type(
        page,
        suspected_type="dropdown",
        coords=(450, 320),
        candidate_xpath="//div[contains(@class,'row')]",
    )
    assert result["confirmed"] is True
    assert result["framework"] == "tom-select"
    assert "ancestor[2].sib:class:ts-wrapper" in result["signals"]
    assert result["anchor_xpath"].startswith("/html/body/")
