"""
q05 guard (ii) + label= emission repair in the accessibility tree search
(_find_element_via_accessibility_tree, STEP 2.5c).

Guard (ii) — recorded failure (2 of 3 corrupt-expected_text q05 runs):
with expected_text="Email *" (the LABEL's text) and the candidate dead,
STEP 2.5c's get_by_text strategies matched the bare LABEL node and
emitted text="Email *" as the locator for an INPUT-field description.
A fill/Get Classes target must never be a bare text node: when the
description names a field/input, a unique get_by_text match is accepted
only if it resolves to a form control (INPUT/TEXTAREA/SELECT or
contenteditable). Rejected matches fall through to get_by_label, the
mechanism that resolves label associations to the control itself.
Signal: text-node-rejected-for-field-description

label= repair — live-verified 2026-07-17: 'label="..."' is NOT a valid
locator engine at runtime (Playwright: Unknown engine "label"; zero
lifetime emissions in production logs). get_by_label successes now
resolve the matched control to a runtime-valid locator (id= first, then
css tag[name=]), or return None — never the broken label= string.
Signal: label-resolved-to-control
"""

import logging

import pytest

from browser_service.locators.smart_locator import (
    _find_element_via_accessibility_tree,
)

FIELD_DESC = "Email input field in the customer add form"
CLICK_TEXT_DESC = "the 'Customer' menu entry text"
LABEL_TEXT = "Email *"


class FakeMatch:
    """Stand-in for a get_by_* Locator: count() + first.evaluate()."""

    def __init__(self, count: int, tag: str = 'LABEL', ce: bool = False,
                 el_id: str = '', name: str = ''):
        self._count = count
        self._tag = tag
        self._ce = ce
        self._id = el_id
        self._name = name

    async def count(self) -> int:
        return self._count

    @property
    def first(self) -> 'FakeMatch':
        return self

    async def evaluate(self, js: str):
        return {
            'tag': self._tag,
            'isContentEditable': self._ce,
            'id': self._id,
            'name': self._name,
        }


class FakeTreePage:
    """Page stand-in for the tree search: role search finds nothing,
    text/label searches configurable, locator() for the resolved-control
    uniqueness check."""

    def __init__(self, text_match: FakeMatch = None,
                 label_match: FakeMatch = None,
                 unique_selectors: set = None):
        self._text = text_match or FakeMatch(0)
        self._label = label_match or FakeMatch(0)
        self._unique = unique_selectors or set()
        self.probed = []

    def get_by_role(self, role, name=None, exact=None):
        return FakeMatch(0)

    def get_by_text(self, text, exact=None):
        return self._text

    def get_by_label(self, text, exact=None):
        return self._label

    def locator(self, selector: str):
        self.probed.append(selector)
        return FakeMatch(1 if selector in self._unique else 0)


@pytest.mark.asyncio
async def test_field_description_rejects_bare_text_node(caplog):
    """The q05 shape: text search finds the LABEL — reject it and resolve
    through get_by_label to the control's own attributes."""
    page = FakeTreePage(
        text_match=FakeMatch(1, tag='LABEL'),
        label_match=FakeMatch(1, tag='INPUT', el_id='', name='email'),
        unique_selectors={'input[name="email"]'},
    )
    with caplog.at_level(logging.INFO):
        result = await _find_element_via_accessibility_tree(
            page, LABEL_TEXT, FIELD_DESC
        )
    assert 'text-node-rejected-for-field-description' in caplog.text
    assert result is not None
    assert result['locator'] == 'input[name="email"]'
    assert 'label=' not in result['locator']
    assert 'text=' not in result['locator']


@pytest.mark.asyncio
async def test_non_field_description_keeps_text_emission():
    """Regression pin: clicking visible text is exactly what get_by_text
    is for — descriptions without field/input wording are untouched."""
    page = FakeTreePage(text_match=FakeMatch(1, tag='SPAN'))
    result = await _find_element_via_accessibility_tree(
        page, 'Customer', CLICK_TEXT_DESC
    )
    assert result is not None
    assert result['locator'] == 'text="Customer"'


@pytest.mark.asyncio
async def test_field_description_accepts_form_control_text_match():
    """getByText can match an input by its value — a form-control match
    for a field description is legitimate and stays accepted."""
    page = FakeTreePage(text_match=FakeMatch(1, tag='INPUT'))
    result = await _find_element_via_accessibility_tree(
        page, LABEL_TEXT, FIELD_DESC
    )
    assert result is not None
    assert result['locator'] == f'text="{LABEL_TEXT}"'


@pytest.mark.asyncio
async def test_label_match_resolves_to_id_locator(caplog):
    """get_by_label success emits the control's id — never label=."""
    page = FakeTreePage(
        label_match=FakeMatch(1, tag='INPUT', el_id='email_field'),
        unique_selectors={'id=email_field'},
    )
    with caplog.at_level(logging.INFO):
        result = await _find_element_via_accessibility_tree(
            page, LABEL_TEXT, FIELD_DESC
        )
    assert result is not None
    assert result['locator'] == 'id=email_field'
    assert 'label-resolved-to-control' in caplog.text


@pytest.mark.asyncio
async def test_label_match_without_id_or_name_returns_none():
    """No runtime-valid resolution -> None. The broken label= string is
    never emitted (Unknown engine \"label\" at runtime)."""
    page = FakeTreePage(
        label_match=FakeMatch(1, tag='INPUT', el_id='', name=''),
    )
    result = await _find_element_via_accessibility_tree(
        page, LABEL_TEXT, FIELD_DESC
    )
    assert result is None
