"""
Unit tests for browser_service.locators.handlers.checkbox.

Async coverage with mocks for the dispatcher contract:
  - find_locator routes native / custom / toggle frameworks correctly
  - element_data id / name+value / name anchors validate via locator.count()
  - Always-fallback: returns None when nothing matches

Plus pure invocation of find_checkbox_or_radio_by_label preserved as a
re-export to keep the legacy text-first call site (smart_locator.py L910)
working unchanged.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from browser_service.locators.classifier import ElementTypeInfo
from browser_service.locators.handlers.checkbox import (
    find_checkbox_or_radio_by_label,
    find_locator,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _mock_search_context(locator_counts: dict[str, int]):
    """Build a search_context whose .locator(sel).count() returns the
    integer keyed by `sel`. Unrecognized selectors return 0.

    Each `.first.get_attribute(...)` call returns ``None`` by default;
    individual tests can override via the dict the caller manipulates."""
    ctx = MagicMock()

    def _locator(sel):
        loc = MagicMock()
        loc.count = AsyncMock(return_value=locator_counts.get(sel, 0))
        # ``.first.get_attribute`` is awaited in the legacy helper; tests
        # that don't exercise that path can ignore it.
        first = MagicMock()
        first.get_attribute = AsyncMock(return_value=None)
        loc.first = first
        return loc

    ctx.locator = _locator
    return ctx


def _info(framework: str, primary_type: str = "checkbox", confidence: str = "high"):
    return ElementTypeInfo(
        primary_type=primary_type,
        framework=framework,
        confidence=confidence,
        signals=[f"role={framework}", "tier:0"],
    )


# ======================================================================
# Strategy 1 — element_data attribute anchors
# ======================================================================


class TestStrategy1AttributeAnchors:
    """Native (and custom) widgets with stable attributes win immediately."""

    @pytest.mark.asyncio
    async def test_id_anchored_native_checkbox(self):
        ctx = _mock_search_context({"id=newsletter": 1})
        result = await find_locator(
            page=ctx,
            element_data={"tagName": "input", "type": "checkbox", "id": "newsletter"},
            type_info=_info("native"),
            element_id="elem_chk",
            element_description="Subscribe to newsletter checkbox",
            expected_text="Subscribe to newsletter",
            search_context=ctx,
            iframe_context=None,
            confirmed_coords=None,
        )
        assert result is not None
        assert result["best_locator"] == "id=newsletter"
        assert result["element_type"] == "checkbox"
        assert result["framework"] == "native"
        assert result["classifier_confidence"] == "high"

    @pytest.mark.asyncio
    async def test_name_value_anchored_radio(self):
        ctx = _mock_search_context({'[name="plan"][value="pro"]': 1})
        result = await find_locator(
            page=ctx,
            element_data={
                "tagName": "input", "type": "radio",
                "name": "plan", "value": "pro",
            },
            type_info=_info("native", primary_type="radio"),
            element_id="elem_radio",
            element_description="Pro plan radio",
            expected_text="Pro",
            search_context=ctx,
            iframe_context=None,
            confirmed_coords=None,
        )
        assert result is not None
        assert result["best_locator"] == '[name="plan"][value="pro"]'
        assert result["element_type"] == "radio"

    @pytest.mark.asyncio
    async def test_name_only_anchored_when_value_missing(self):
        ctx = _mock_search_context({'[name="remember"]': 1})
        result = await find_locator(
            page=ctx,
            element_data={"tagName": "input", "type": "checkbox", "name": "remember"},
            type_info=_info("native"),
            element_id="elem",
            element_description="Remember me",
            expected_text="Remember me",
            search_context=ctx,
            iframe_context=None,
            confirmed_coords=None,
        )
        assert result is not None
        assert result["best_locator"] == '[name="remember"]'

    @pytest.mark.asyncio
    async def test_id_takes_priority_over_name(self):
        # Both unique — id should win since it's tried first.
        ctx = _mock_search_context({
            "id=terms": 1,
            '[name="terms"]': 1,
        })
        result = await find_locator(
            page=ctx,
            element_data={"tagName": "input", "type": "checkbox",
                          "id": "terms", "name": "terms"},
            type_info=_info("native"),
            element_id="elem",
            element_description="Accept terms",
            expected_text="Accept terms",
            search_context=ctx,
            iframe_context=None,
            confirmed_coords=None,
        )
        assert result["best_locator"] == "id=terms"

    @pytest.mark.asyncio
    async def test_attr_skipped_when_count_not_one(self):
        # id matches 0 elements (e.g., dynamically removed) — handler
        # falls through to the next candidate.
        ctx = _mock_search_context({
            "id=missing": 0,
            '[name="plan"][value="basic"]': 1,
        })
        result = await find_locator(
            page=ctx,
            element_data={
                "tagName": "input", "type": "radio", "id": "missing",
                "name": "plan", "value": "basic",
            },
            type_info=_info("native", primary_type="radio"),
            element_id="elem",
            element_description="Basic plan",
            expected_text="Basic",
            search_context=ctx,
            iframe_context=None,
            confirmed_coords=None,
        )
        assert result is not None
        assert result["best_locator"] == '[name="plan"][value="basic"]'


# ======================================================================
# Strategy 3 — custom widgets (role=checkbox / radio / switch)
# ======================================================================


class TestStrategy3CustomWidget:
    @pytest.mark.asyncio
    async def test_custom_checkbox_aria_label_anchor(self):
        ctx = _mock_search_context({
            '[role="checkbox"][aria-label="Enable analytics"]': 1,
        })
        result = await find_locator(
            page=ctx,
            element_data={"tagName": "span", "role": "checkbox"},
            type_info=_info("custom"),
            element_id="elem",
            element_description="Enable analytics",
            expected_text="Enable analytics",
            search_context=ctx,
            iframe_context=None,
            confirmed_coords=None,
        )
        assert result is not None
        assert (
            result["best_locator"]
            == '[role="checkbox"][aria-label="Enable analytics"]'
        )
        assert result["element_type"] == "checkbox"
        assert result["framework"] == "custom"

    @pytest.mark.asyncio
    async def test_toggle_role_switch_anchor(self):
        ctx = _mock_search_context({
            '[role="switch"][aria-label="Email notifications"]': 1,
        })
        result = await find_locator(
            page=ctx,
            element_data={"tagName": "span", "role": "switch"},
            type_info=_info("toggle"),
            element_id="elem",
            element_description="Email notifications toggle",
            expected_text="Email notifications",
            search_context=ctx,
            iframe_context=None,
            confirmed_coords=None,
        )
        assert result is not None
        assert "role=\"switch\"" in result["best_locator"]
        assert result["element_type"] == "checkbox"  # toggle maps under checkbox primary_type

    @pytest.mark.asyncio
    async def test_custom_radio_falls_back_to_role_name(self):
        # First candidate (aria-label) misses; second (role=...[name=...]) hits.
        ctx = _mock_search_context({
            'role=radio[name="Light theme"]': 1,
        })
        result = await find_locator(
            page=ctx,
            element_data={"tagName": "span", "role": "radio"},
            type_info=_info("custom", primary_type="radio"),
            element_id="elem",
            element_description="Light theme",
            expected_text="Light theme",
            search_context=ctx,
            iframe_context=None,
            confirmed_coords=None,
        )
        assert result is not None
        assert result["best_locator"] == 'role=radio[name="Light theme"]'


# ======================================================================
# Always-fallback contract
# ======================================================================


class TestAlwaysFallback:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_attrs_no_label(self):
        ctx = _mock_search_context({})
        result = await find_locator(
            page=ctx,
            element_data={"tagName": "input", "type": "checkbox"},
            type_info=_info("native"),
            element_id="elem",
            element_description="",
            expected_text="",
            search_context=ctx,
            iframe_context=None,
            confirmed_coords=None,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_locator_count_validation_fails(self):
        # All Strategy 1 candidates miss; no label-based search will run
        # because find_checkbox_or_radio_by_label requires real Playwright
        # (we mock it away). Verify graceful None.
        ctx = _mock_search_context({})  # everything returns 0
        # Use an unknown framework so Strategy 3 short-circuits too.
        info = ElementTypeInfo(
            primary_type="checkbox", framework="unknown",
            confidence="low", signals=[],
        )
        result = await find_locator(
            page=ctx,
            element_data={"tagName": "input", "type": "checkbox", "id": "missing"},
            type_info=info,
            element_id="elem",
            element_description="x",
            expected_text="x",
            search_context=ctx,
            iframe_context=None,
            confirmed_coords=None,
        )
        assert result is None


# ======================================================================
# Re-export — find_checkbox_or_radio_by_label is callable
# ======================================================================


class TestLegacyHelperReExport:
    @pytest.mark.asyncio
    async def test_returns_none_for_empty_label(self):
        ctx = _mock_search_context({})
        assert await find_checkbox_or_radio_by_label(ctx, "") is None
        assert await find_checkbox_or_radio_by_label(ctx, None) is None

    def test_re_exported_from_handlers_checkbox(self):
        """The legacy import path in smart_locator.py must still resolve."""
        from browser_service.locators.smart_locator import (
            _find_checkbox_or_radio_by_label,
        )
        assert _find_checkbox_or_radio_by_label is find_checkbox_or_radio_by_label
