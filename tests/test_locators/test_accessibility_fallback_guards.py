"""
Dialog-clobber guards D1 + D2 (discovered 2026-07-16, ASTPP gate r2:
q02 r3 wf 99b7c4b8, q04 r1 wf 975435dd; env-triggered, engine-real).

The chain: slow site -> the agent re-queried the already-validated Sign
In button with the announcement modal covering it -> elementFromPoint
found the DIALOG -> the accessibility fallback logged "Semantic
mismatch: expected 'Sign In'" and ACCEPTED anyway (D1) -> the dialog's
MULTILINE accessible name went into role=dialog[name="..."] unsanitized
(D2) -> the .robot variable kept only line 1 (unclosed quote) -> runtime
InvalidSelectorError, invisible to dryrun.

D1: a DETECTED semantic mismatch rejects the accessibility fallback
    result — fall through to the coordinate-based approach (Task 7
    precedent: never return-something-over-fail on a detected
    contradiction). Signal: accessibility-mismatch-rejected
D2: accessible names are whitespace-normalized before locator
    construction (Playwright's role-engine name matching normalizes
    whitespace itself, so the normalized form matches whenever the raw
    form does — and newlines can never reach the .robot variable).
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from browser_service.agent.actions import find_unique_locator_action
from browser_service.locators.smart_locator import (
    _find_element_via_accessibility,
)

DIALOG_NAME_RAW = "Smarter Support Starts Here…\nAutomate replies, cut costs"
DIALOG_NAME_NORM = "Smarter Support Starts Here… Automate replies, cut costs"
SIGNIN_DESC = "Sign In button in the login form"
X, Y = 640, 400


# ======================================================================
# D2 — accessible-name whitespace normalization at locator build time
# ======================================================================


class D2FakeLocator:
    def __init__(self, count: int):
        self._count = count

    async def count(self) -> int:
        return self._count


class D2FakeCtx:
    """search_context stand-in: evaluate() answers the elementFromPoint
    JS with a dialog whose accessible name is multiline; locator()
    reports count=1 for any role=dialog[name=...] form."""

    def __init__(self):
        self.probed = []

    async def evaluate(self, js, arg=None):
        return {
            'role': 'dialog',
            'accessibleName': DIALOG_NAME_RAW,
            'tagName': 'dialog',
            'isCollection': False,
            'id': None,
            'className': 'announcement-modal',
        }

    def locator(self, selector: str):
        self.probed.append(selector)
        return D2FakeLocator(1 if selector.startswith('role=dialog[name=') else 0)


@pytest.mark.asyncio
async def test_accessible_name_is_whitespace_normalized():
    """No newline may survive into the locator string — the .robot
    variable line truncates at it and the selector dies at runtime."""
    ctx = D2FakeCtx()
    result = await _find_element_via_accessibility(
        page=ctx, x=X, y=Y,
        element_description="announcement",  # no table keywords
        expected_text=None,                   # skip 2.5a
        search_context=ctx,
    )
    assert result is not None
    assert '\n' not in result['locator']
    assert result['locator'] == f'role=dialog[name="{DIALOG_NAME_NORM}"]'


# ======================================================================
# D1 — detected mismatch rejects the accessibility fallback result
# ======================================================================


class LenientLocator:
    def __init__(self, count: int = 0):
        self._count = count

    async def count(self) -> int:
        return self._count

    async def bounding_box(self):
        return None

    def nth(self, i):
        return self

    async def evaluate(self, js):
        return {}


class LenientPage:
    url = 'https://sujal.astppbilling.org/'

    def locator(self, selector: str):
        return LenientLocator(0)

    async def evaluate(self, *args, **kwargs):
        return None


def _dialog_fallback_result(name: str) -> dict:
    return {
        'locator': f'role=dialog[name="{name}"]',
        'count': 1,
        'unique': True,
        'role': 'dialog',
        'accessible_name': name,
        'element_type': 'dialog',
        'strategy': 'accessibility_role',
    }


class TestD1MismatchReject:

    @pytest.mark.asyncio
    async def test_detected_mismatch_is_rejected(self, caplog):
        """The clobber replay: expected 'Sign In', fallback found the
        dialog — the mismatched result must NOT be returned."""
        dialog = _dialog_fallback_result(DIALOG_NAME_NORM)
        with patch(
            'browser_service.locators.smart_locator.'
            '_find_element_via_accessibility',
            new=AsyncMock(return_value=dialog),
        ):
            with caplog.at_level(logging.INFO):
                result = await find_unique_locator_action(
                    x=X, y=Y,
                    element_id='elem_3',
                    element_description=SIGNIN_DESC,
                    expected_text='Sign In',
                    candidate_locator=None,
                    element_data=None,
                    page=LenientPage(),
                    is_collection=False,
                )
        assert 'accessibility-mismatch-rejected' in caplog.text
        assert result.get('best_locator') != dialog['locator']

    @pytest.mark.asyncio
    async def test_compatible_name_still_accepted(self):
        """Regression pin: containment match ('Sign In' in 'Sign In
        button') keeps today's accept."""
        ok = _dialog_fallback_result('Sign In button')
        ok['role'] = 'button'
        ok['locator'] = 'role=button[name="Sign In button"]'
        with patch(
            'browser_service.locators.smart_locator.'
            '_find_element_via_accessibility',
            new=AsyncMock(return_value=ok),
        ):
            result = await find_unique_locator_action(
                x=X, y=Y,
                element_id='elem_3',
                element_description=SIGNIN_DESC,
                expected_text='Sign In',
                candidate_locator=None,
                element_data=None,
                page=LenientPage(),
                is_collection=False,
            )
        assert result['found'] is True
        assert result['best_locator'] == 'role=button[name="Sign In button"]'
        assert result['semantic_match'] is True

    @pytest.mark.asyncio
    async def test_no_expected_text_accept_unchanged(self):
        """No expected_text -> nothing to contradict -> accept stands
        (paste-mode and icon-only flows unchanged)."""
        dialog = _dialog_fallback_result(DIALOG_NAME_NORM)
        with patch(
            'browser_service.locators.smart_locator.'
            '_find_element_via_accessibility',
            new=AsyncMock(return_value=dialog),
        ):
            result = await find_unique_locator_action(
                x=X, y=Y,
                element_id='elem_3',
                element_description=SIGNIN_DESC,
                expected_text=None,
                candidate_locator=None,
                element_data=None,
                page=LenientPage(),
                is_collection=False,
            )
        assert result['found'] is True
        assert result['best_locator'] == dialog['locator']
