"""
K v1 wiring: when find_unique_locator_action consults the read-step identity
check, and what it does with the verdict. Fast lane — the live read is patched;
its behaviour on a real DOM is pinned by test_read_step_identity_dom.py.
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from browser_service.agent.actions import find_unique_locator_action
from browser_service.agent.read_step_identity import ReadIdentityFacts

U07_CANDIDATE = "css=h6.oxd-topbar-header-breadcrumb-module"

U07_ELEMENT_DATA = {
    "tagName": "a",
    "id": "",
    "className": "oxd-main-menu-item active",
    "textContent": "Dashboard",
    "xpath": "html/body/div/div[1]/div[1]/aside/nav/div[2]/ul/li[8]/a",
    "backendNodeId": 230,
}

U07_RESOLVED_H6 = {
    "id": "",
    "tagName": "h6",
    "textContent": "Dashboard",
    "className": "oxd-text oxd-text--h6 oxd-topbar-header-breadcrumb-module",
    "ariaInvalid": "",
    "parentClassName": "",
    "name": "",
    "dataTestId": "",
    "role": "",
    "type": "",
}

SIDEBAR_CASCADE_RESULT = {
    "element_id": "elem_4",
    "found": True,
    "best_locator": 'text="Dashboard" >> nth=0',
    "all_locators": [{"type": "text", "locator": 'text="Dashboard" >> nth=0'}],
    "element_info": {"tagName": "a"},
    "approach_metrics": {"locator_approach": "element_data", "fallback_depth": 1},
}

ACCEPT = ReadIdentityFacts(indexed_ok=True, candidate_tag="h6", candidate_centre=(1070.0, 32.0))
REJECT = ReadIdentityFacts(indexed_ok=True, is_indexed=True, candidate_tag="h6")


class FakeLocator:
    def __init__(self, count, resolved=None):
        self._count = count
        self._resolved = resolved or {}

    async def count(self):
        return self._count

    async def bounding_box(self):
        return None

    def nth(self, i):
        return FakeLocator(1, self._resolved)

    async def evaluate(self, js, arg=None, *, timeout=None):
        text = self._resolved.get("textContent", "")
        if "labelledbyText" in js:  # semantic validation probe
            return {
                "textContent": text,
                "textContentLength": len(text),
                "innerText": text,
                "placeholder": "",
                "ariaLabel": "",
                "value": "",
                "labelText": "",
                "labelledbyText": "",
                "tagName": self._resolved.get("tagName", ""),
                "isContentEditable": False,
            }
        return dict(self._resolved)


class FakePage:
    url = "https://opensource-demo.orangehrmlive.com/web/index.php/dashboard/index"

    def __init__(self, locators):
        self._locators = locators

    def locator(self, selector):
        return self._locators.get(selector, FakeLocator(0))

    def frame_locator(self, selector):
        return self

    async def evaluate(self, *args, **kwargs):
        return None


def _page(resolved=U07_RESOLVED_H6, count=1):
    return FakePage({U07_CANDIDATE: FakeLocator(count, resolved)})


async def _call(page, facts, **overrides):
    kwargs = dict(
        x=250,
        y=420,
        element_id="elem_4",
        element_description="the Dashboard heading at the top of the page",
        expected_text="Dashboard",
        candidate_locator=U07_CANDIDATE,
        element_data=dict(U07_ELEMENT_DATA),
        page=page,
        browser_session=object(),
        action="get_text",
        vision_point=(1072, 30),
    )
    kwargs.update(overrides)
    k_read = AsyncMock(return_value=facts)
    with (
        patch(
            "browser_service.locators.find_unique_locator_at_coordinates",
            return_value=dict(SIDEBAR_CASCADE_RESULT),
        ),
        patch("browser_service.agent.actions.read_identity_facts", k_read),
        patch("browser_service.agent.actions.check_action_fit", AsyncMock(return_value="")),
        patch("browser_service.agent.actions._indexed_element_misfit", AsyncMock(return_value="")),
        patch(
            "browser_service.agent.actions.apply_read_target_policy",
            AsyncMock(side_effect=lambda root, result, *a, **k: result),
        ),
    ):
        result = await find_unique_locator_action(**kwargs)
    return result, k_read


async def test_read_step_accept_keeps_the_candidate(caplog):
    with caplog.at_level(logging.INFO):
        result, k_read = await _call(_page(), ACCEPT)
    k_read.assert_awaited_once()
    assert result["best_locator"] == U07_CANDIDATE
    assert result["all_locators"][0]["type"] == "candidate"
    assert result["element_info"]["tagName"] == "h6"
    assert "read-step-identity-accept" in caplog.text
    assert "(1072, 30)" in caplog.text  # vision point before replacement
    assert "(250, 420)" in caplog.text  # the point bs actually used
    assert "1070.0" in caplog.text  # the candidate's centre
    accept_line = [ln for ln in caplog.text.splitlines() if "read-step-identity-accept" in ln][0]
    assert "element_id=elem_4" in accept_line
    assert "k_ms=" in accept_line


async def test_read_step_reject_verdict_keeps_the_reject(caplog):
    with caplog.at_level(logging.INFO):
        result, _ = await _call(_page(), REJECT)
    assert result["best_locator"] == SIDEBAR_CASCADE_RESULT["best_locator"]
    assert "read-step-identity-keeps-reject" in caplog.text
    assert "candidate-resolves-to-different-element" in caplog.text
    reject_line = [
        ln for ln in caplog.text.splitlines() if "read-step-identity-keeps-reject" in ln
    ][0]
    assert "element_id=elem_4" in reject_line
    assert "k_ms=" in reject_line


async def test_failed_read_keeps_the_reject():
    result, _ = await _call(_page(), None)
    assert result["best_locator"] == SIDEBAR_CASCADE_RESULT["best_locator"]


@pytest.mark.parametrize("action", ["click", "select", "input", None])
async def test_non_read_steps_never_consult_k(action):
    """Click/select/input act through the index in discovery (registration.py:976-988);
    None = ENABLE_ACTION_FIT=false."""
    result, k_read = await _call(_page(), ACCEPT, action=action)
    k_read.assert_not_awaited()
    assert result["best_locator"] == SIDEBAR_CASCADE_RESULT["best_locator"]


async def test_get_attribute_is_a_read_step():
    result, k_read = await _call(_page(), ACCEPT, action="get_attribute")
    k_read.assert_awaited_once()
    assert result["best_locator"] == U07_CANDIDATE


async def test_iframe_calls_never_consult_k():
    result, k_read = await _call(_page(), ACCEPT, iframe_context='iframe[id="main"]')
    k_read.assert_not_awaited()


async def test_no_browser_session_never_consults_k():
    result, k_read = await _call(_page(), ACCEPT, browser_session=None)
    k_read.assert_not_awaited()
    assert result["best_locator"] == SIDEBAR_CASCADE_RESULT["best_locator"]


async def test_identity_match_never_consults_k():
    same_tag = dict(U07_RESOLVED_H6, tagName="a")
    result, k_read = await _call(_page(resolved=same_tag), ACCEPT)
    k_read.assert_not_awaited()
    assert result["best_locator"] == U07_CANDIDATE


async def test_k_accept_still_faces_the_semantic_check():
    """K clears only the identity reason; a text mismatch still falls through."""
    buzz = dict(U07_RESOLVED_H6, textContent="Buzz")
    result, k_read = await _call(_page(resolved=buzz), ACCEPT)
    k_read.assert_awaited_once()
    assert result["best_locator"] == SIDEBAR_CASCADE_RESULT["best_locator"]


async def test_collection_path_never_consults_k():
    result, k_read = await _call(_page(count=3), ACCEPT, is_collection=True)
    k_read.assert_not_awaited()
