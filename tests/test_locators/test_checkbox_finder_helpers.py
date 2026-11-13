"""Unit tests for the checkbox finder/proxy helpers extracted during the
cognitive-complexity refactor.

These cover the branches the real-DOM gate tests do not reach:
_proxy_candidates (pure), and the nested-input / adjacent-input resolvers
driven by a locator stub — no browser.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from browser_service.locators.handlers.checkbox import (
    _counted,
    _proxy_candidates,
    _resolve_adjacent_input,
    _resolve_label_for,
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
                'input[type="radio"][name="pay"][value="card"]': {"count": 1},
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
                'input[type="radio"][name="pay"]': {"count": 1},
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
        page = _FinderPage(
            {
                pattern: {"count": 1, "attrs": {"type": "checkbox", "id": "sub"}},
                "id=sub": {"count": 1},
            }
        )
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
                'input[type="radio"][name="confirm"][value="yes"]': {"count": 1},
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


class _CountPage:
    """locator(sel).count() with a scripted result per selector; records every call."""

    def __init__(self, counts: dict):
        self._counts = counts
        self.calls: list[str] = []

    def locator(self, selector):
        self.calls.append(selector)
        node = MagicMock()
        result = self._counts.get(selector, 0)
        if isinstance(result, Exception):
            node.count = AsyncMock(side_effect=result)
        else:
            node.count = AsyncMock(return_value=result)
        return node


class TestCounted:
    """bs #64: a string ships only when it was counted to exactly one element here."""

    @pytest.mark.asyncio
    async def test_preferred_returned_at_count_one(self):
        page = _CountPage({"id=rm": 1})
        assert await _counted(page, "id=rm", "label >> input") == "id=rm"

    @pytest.mark.asyncio
    async def test_counted_returned_at_count_two(self):
        page = _CountPage({"id=rm": 2})
        assert await _counted(page, "id=rm", "label >> input") == "label >> input"

    @pytest.mark.asyncio
    async def test_counted_returned_at_count_zero(self):
        page = _CountPage({})
        assert await _counted(page, "id=rm", "label >> input") == "label >> input"

    @pytest.mark.asyncio
    async def test_counted_returned_when_count_raises(self):
        page = _CountPage({"id=rm": RuntimeError("detached")})
        assert await _counted(page, "id=rm", "label >> input") == "label >> input"

    @pytest.mark.asyncio
    async def test_no_count_when_preferred_is_the_counted_string(self):
        page = _CountPage({})
        assert await _counted(page, "label >> input", "label >> input") == "label >> input"
        assert page.calls == []


class TestFinderShipsOnlyCountedStrings:
    @pytest.mark.asyncio
    async def test_nested_checkbox_with_duplicate_id_keeps_the_counted_string(self):
        label = 'label:text-is("Remember me")'
        nested = f'{label} >> input[type="checkbox"]'
        page = _FinderPage({nested: {"count": 1, "attrs": {"id": "rm"}}, "id=rm": {"count": 2}})
        result = await _resolve_nested_input(page, label)
        assert result == {"locator": nested, "element_type": "checkbox"}

    @pytest.mark.asyncio
    async def test_adjacent_input_with_shared_name_keeps_the_counted_string(self):
        text = "Remember me"
        pattern = f':text("{text}") >> xpath=preceding-sibling::input[@type="checkbox"]'
        page = _FinderPage(
            {
                pattern: {"count": 1, "attrs": {"type": "checkbox", "name": "prefs"}},
                'input[type="checkbox"][name="prefs"]': {"count": 2},
            }
        )
        result = await _resolve_adjacent_input(page, text)
        assert result == {"locator": pattern, "element_type": "checkbox"}

    @pytest.mark.asyncio
    async def test_label_for_with_duplicate_id_ships_the_attribute_form(self):
        page = _FinderPage(
            {
                'input[id="rm"]': {"count": 1, "attrs": {"type": "checkbox"}},
                "id=rm": {"count": 2},
            }
        )
        result = await _resolve_label_for(page, "rm")
        assert result == {"locator": 'input[id="rm"]', "element_type": "checkbox"}
