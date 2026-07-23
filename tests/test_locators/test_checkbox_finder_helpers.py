"""Unit tests for the checkbox finder/proxy helpers extracted during the
cognitive-complexity refactor.

These cover the branches the real-DOM gate tests do not reach:
_proxy_candidates (pure), and the nested-input / adjacent-input resolvers
driven by a locator stub — no browser.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from browser_service.locators.handlers.checkbox import (
    _proxy_candidates,
    _resolve_adjacent_input,
    _resolve_nested_input,
)


class _FinderPage:
    """locator(sel) → object with .count() and .first.get_attribute()."""

    def __init__(self, spec: dict):
        # spec: selector -> {"count": int, "attrs": {name: value}}
        self._spec = spec

    def locator(self, selector):
        s = self._spec.get(selector, {})
        node = MagicMock()
        node.count = AsyncMock(return_value=s.get("count", 0))
        first = MagicMock()

        async def get_attr(name, _attrs=s.get("attrs", {})):
            return _attrs.get(name)

        first.get_attribute = get_attr
        node.first = first
        return node


class TestProxyCandidates:
    def test_all_three_candidate_kinds_when_probe_supports_them(self):
        probe = {
            "id": "agree",
            "hasLabelFor": True,
            "hasWrappingLabel": True,
            "siblingTag": "span",
        }
        kinds = [kind for _, kind in _proxy_candidates("id=agree", probe)]
        assert kinds == ["label-for", "wrapping-label", "adjacent-sibling"]

    def test_plain_css_locator_supplies_input_css_when_no_id(self):
        # No id, but the input_locator is a plain CSS selector → usable as input_css,
        # so the wrapping-label candidate is still offered.
        probe = {"id": "", "hasWrappingLabel": True, "siblingTag": "label"}
        cands = _proxy_candidates("input.toggle", probe)
        selectors = [sel for sel, _ in cands]
        assert any("label:has(input.toggle)" == s for s in selectors)

    def test_non_css_locator_without_id_yields_no_css_candidates(self):
        # xpath locator + no id → no input_css, so only nothing (no label-for either)
        probe = {"id": "", "hasWrappingLabel": True, "siblingTag": "span"}
        assert _proxy_candidates("xpath=//input", probe) == []


class TestResolveNestedInput:
    @pytest.mark.asyncio
    async def test_nested_checkbox_without_id_or_name_falls_back(self):
        label = 'label:text-is("Agree")'
        page = _FinderPage({f'{label} >> input[type="checkbox"]': {"count": 1, "attrs": {}}})
        result = await _resolve_nested_input(page, label)
        assert result == {
            "locator": f'{label} >> input[type="checkbox"]',
            "element_type": "checkbox",
        }

    @pytest.mark.asyncio
    async def test_nested_radio_by_name_and_value(self):
        label = 'label:text-is("Card")'
        page = _FinderPage(
            {
                f'{label} >> input[type="checkbox"]': {"count": 0},
                f'{label} >> input[type="radio"]': {
                    "count": 1,
                    "attrs": {"name": "pay", "value": "card"},
                },
            }
        )
        result = await _resolve_nested_input(page, label)
        assert result == {
            "locator": 'input[type="radio"][name="pay"][value="card"]',
            "element_type": "radio",
        }

    @pytest.mark.asyncio
    async def test_nested_radio_without_attributes_falls_back(self):
        label = 'label:text-is("Pick")'
        page = _FinderPage(
            {
                f'{label} >> input[type="checkbox"]': {"count": 0},
                f'{label} >> input[type="radio"]': {"count": 1, "attrs": {}},
            }
        )
        result = await _resolve_nested_input(page, label)
        assert result == {
            "locator": f'{label} >> input[type="radio"]',
            "element_type": "radio",
        }

    @pytest.mark.asyncio
    async def test_nested_radio_by_name_only(self):
        label = 'label:text-is("Card")'
        page = _FinderPage(
            {
                f'{label} >> input[type="checkbox"]': {"count": 0},
                f'{label} >> input[type="radio"]': {"count": 1, "attrs": {"name": "pay"}},
            }
        )
        result = await _resolve_nested_input(page, label)
        assert result == {
            "locator": 'input[type="radio"][name="pay"]',
            "element_type": "radio",
        }


class TestResolveAdjacentInput:
    @pytest.mark.asyncio
    async def test_adjacent_checkbox_by_id(self):
        text = "Subscribe"
        pattern = f'input[type="checkbox"]:left-of(:text("{text}"):visible)'
        page = _FinderPage({pattern: {"count": 1, "attrs": {"type": "checkbox", "id": "sub"}}})
        result = await _resolve_adjacent_input(page, text)
        assert result == {"locator": "id=sub", "element_type": "checkbox"}

    @pytest.mark.asyncio
    async def test_adjacent_input_by_name_and_value(self):
        text = "Yes"
        pattern = f'input[type="radio"]:left-of(:text("{text}"):visible)'
        page = _FinderPage(
            {
                f'input[type="checkbox"]:left-of(:text("{text}"):visible)': {"count": 0},
                pattern: {
                    "count": 1,
                    "attrs": {"type": "radio", "name": "confirm", "value": "yes"},
                },
            }
        )
        result = await _resolve_adjacent_input(page, text)
        assert result == {
            "locator": 'input[type="radio"][name="confirm"][value="yes"]',
            "element_type": "radio",
        }

    @pytest.mark.asyncio
    async def test_no_adjacent_match_returns_none(self):
        page = _FinderPage({})
        assert await _resolve_adjacent_input(page, "Nothing") is None
