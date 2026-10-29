"""bs #72: a find_unique_locator call with no element evidence is refused up front.

Mocked page, no Chromium; the real-DOM proof is test_no_evidence_refused_live.py.

Evidence is an element the index or the point lookup found (a non-empty
``element_data``, filled in by registration.py before the call) or an
``expected_text`` with a non-blank character. Without either the call names no
element, so it is refused before the page is touched: no candidate is counted,
no frame is entered, the cascade does not run (one exception: a frame container as the
only element data passes this check and is refused by the cascade itself, before any step).
A row anchor, ``is_collection`` or a candidate locator alone are NOT evidence (owner:
collections get no exception).
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_service.agent.actions import find_unique_locator_action
from browser_service.locators.smart_locator import find_unique_locator_at_coordinates

CASCADE_RESULT = {"element_id": "elem_1", "found": True, "best_locator": "css=.from-cascade"}

ERROR = (
    "No element evidence: no element_index matched an element and no expected_text was given, "
    "so nothing was checked. Send the element's index, or its visible text as expected_text."
)
FRAME = 'iframe[id="editor-frame"]'


@pytest.fixture
def page():
    """A page that matches nothing; the refusal tests assert it is never touched."""
    locator = MagicMock()
    locator.count = AsyncMock(return_value=0)
    root = MagicMock()
    root.locator = MagicMock(return_value=locator)
    pg = MagicMock()
    pg.url = "https://example.com/app"
    pg.locator = MagicMock(return_value=locator)
    pg.frame_locator = MagicMock(return_value=root)
    return pg


@pytest.fixture
def cascade():
    spy = AsyncMock(return_value=dict(CASCADE_RESULT))
    with patch("browser_service.locators.find_unique_locator_at_coordinates", new=spy):
        yield spy


async def _call(page, **kwargs):
    args = {
        "x": 120,
        "y": 340,
        "element_id": "elem_1",
        "element_description": "the status message",
    }
    args.update(kwargs)
    return await find_unique_locator_action(page=page, **args)


REFUSED = [
    pytest.param({}, id="no-data-no-text"),
    pytest.param({"expected_text": "   "}, id="blank-text"),
    pytest.param({"element_data": {}}, id="empty-element-data"),
    pytest.param({"row_anchor_text": "John"}, id="row-anchor-alone"),
    pytest.param({"is_collection": True}, id="collection-alone"),
    pytest.param({"candidate_locator": "id=status"}, id="candidate-alone"),
]


@pytest.mark.parametrize("kwargs", REFUSED)
async def test_call_without_evidence_is_refused_and_the_page_untouched(page, cascade, kwargs):
    result = await _call(page, **kwargs)

    assert result["found"] is False
    assert result["error_type"] == "NoElementEvidence"
    assert result["error"] == ERROR
    assert result["evidence"] == "none"
    assert result["coordinates"] == {"x": 120, "y": 340}
    assert "best_locator" not in result
    page.locator.assert_not_called()
    page.frame_locator.assert_not_called()
    cascade.assert_not_called()


async def test_the_refusal_is_logged_as_one_signal_line(page, cascade, caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.actions"):
        await _call(page, candidate_locator="id=status")

    lines = [m for m in caplog.messages if "signal: no-evidence-refused" in m]
    assert lines == [
        "   ⛔ NO EVIDENCE: elem_1 'the status message' at (120, 340), candidate='id=status' "
        "— no element data and no expected text; refused (signal: no-evidence-refused)"
    ]


async def test_expected_text_is_evidence_and_the_candidate_is_counted(page, cascade):
    result = await _call(page, expected_text="Submit", candidate_locator="id=submit")

    page.locator.assert_called_with("id=submit")  # got past the rule: the candidate was counted
    cascade.assert_called_once()  # count 0, so the cascade carries on
    assert "evidence" not in result


async def test_expected_text_alone_reaches_the_cascade(page, cascade):
    result = await _call(page, expected_text="Submit")

    cascade.assert_called_once()
    assert "evidence" not in result


async def test_element_data_is_evidence(page, cascade):
    result = await _call(page, element_data={"tagName": "button", "id": "submit"})

    cascade.assert_called_once()
    assert cascade.call_args.kwargs["element_data"] == {"tagName": "button", "id": "submit"}
    assert "evidence" not in result


async def test_element_data_found_in_a_frame_is_evidence(page, cascade):
    """The in-frame lookup fills element_data too; such a call must not be refused."""
    result = await _call(
        page,
        element_data={"tagName": "p", "id": ""},
        iframe_context=FRAME,
        candidate_locator="p[contenteditable='true']",
    )

    page.frame_locator.assert_called_with(FRAME)  # the candidate was counted inside the frame
    cascade.assert_called_once()
    assert "evidence" not in result


# --- the frame container: evidence at the action's top check, not in the cascade (review I1) ---
# The point lookup can hand over the <iframe> node itself. The candidate block may use it (the
# identity check compares against it), so the top check lets the call through; the cascade drops
# it and, with no text, refuses before any step ships what lies under the point.

CONTAINER = {"tagName": "iframe", "id": "editor-frame"}


async def _cascade_call(page, **kwargs):
    args = {
        "x": 120,
        "y": 340,
        "element_id": "elem_1",
        "element_description": "the status message",
        "search_context": page.frame_locator(FRAME),
        "iframe_context": FRAME,
    }
    args.update(kwargs)
    return await find_unique_locator_at_coordinates(page=page, **args)


async def test_the_action_lets_the_frame_container_through_and_the_cascade_refuses_it(page):
    result = await _call(page, element_data=CONTAINER, iframe_context=FRAME)

    assert result["found"] is False
    assert result["error_type"] == "NoElementEvidence"
    assert result["error"] == ERROR
    assert result["evidence"] == "none"
    page.locator.assert_not_called()  # no step ran against the page


async def test_the_cascade_refuses_a_dropped_frame_container_without_text(page, caplog):
    root = page.frame_locator.return_value
    with caplog.at_level(logging.INFO, logger="browser_service.locators.smart_locator"):
        result = await _cascade_call(page, element_data=CONTAINER)

    assert result["found"] is False
    assert result["error_type"] == "NoElementEvidence"
    assert result["error"] == ERROR
    assert result["evidence"] == "none"
    assert result["coordinates"] == {"x": 120, "y": 340}
    assert "approach_metrics" not in result
    page.locator.assert_not_called()
    root.locator.assert_not_called()
    assert [m for m in caplog.messages if "signal: no-evidence-refused" in m] == [
        "   ⛔ NO EVIDENCE: elem_1 'the status message' at (120, 340) — no element data and no "
        "expected text after the iframe reset; refused (signal: no-evidence-refused)"
    ]


@pytest.mark.parametrize("text", [None, "", "   "])
async def test_blank_text_is_no_text_for_the_cascade_rule(page, text):
    result = await _cascade_call(page, element_data=CONTAINER, expected_text=text)
    assert result["evidence"] == "none"


async def test_a_dropped_frame_container_with_text_is_not_refused(page, caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.locators.smart_locator"):
        result = await _cascade_call(page, element_data=CONTAINER, expected_text="Synced")

    assert "evidence" not in result
    assert [m for m in caplog.messages if "signal: no-evidence-refused" in m] == []


async def test_an_iframe_node_without_a_frame_is_not_dropped_and_not_refused(page):
    step0 = AsyncMock(return_value={"found": True, "best_locator": "css=iframe#editor-frame"})
    with patch(
        "browser_service.locators.smart_locator._generate_locators_from_element_data", new=step0
    ):
        result = await _cascade_call(page, element_data=CONTAINER, iframe_context=None)

    step0.assert_awaited_once()
    assert result["found"] is True
    assert "evidence" not in result


async def test_a_parked_candidate_survives_the_cascade_refusal():
    """E2: a demoted candidate is the last resort; a no-evidence cascade answer is a not-found
    answer, so the candidate is returned as before."""
    from tests.test_agent.test_action_fit_actions import (
        CANDIDATE,
        SEARCH_TEXT,
        FakeLocator,
        FakePage,
    )

    refusal = {
        "found": False,
        "error": ERROR,
        "error_type": "NoElementEvidence",
        "evidence": "none",
    }
    cascade = AsyncMock(return_value=refusal)
    with (
        patch("browser_service.locators.find_unique_locator_at_coordinates", new=cascade),
        patch(
            "browser_service.agent.actions.check_action_fit",
            new=AsyncMock(return_value="<div> cannot be filled"),
        ),
    ):
        out = await find_unique_locator_action(
            x=100,
            y=100,
            element_id="elem_1",
            element_description="the search input box",
            expected_text=SEARCH_TEXT,
            candidate_locator=CANDIDATE,
            page=FakePage({CANDIDATE: FakeLocator(1)}),
            action="input",
        )

    assert out["best_locator"] == CANDIDATE
    assert out["action_fit"] == "misfit"


async def test_fit_mode_refusal_is_logged_once_and_the_cascade_runs_once(page, caplog):
    """Fit mode (an action is given) re-runs today's cascade when nothing was found; a refusal
    is not a miss, so it must not run again (the signal is counted)."""
    runs = []

    async def cascade(**kwargs):
        runs.append(kwargs["action"])
        return await find_unique_locator_at_coordinates(**kwargs)

    with (
        patch("browser_service.locators.find_unique_locator_at_coordinates", new=cascade),
        caplog.at_level(logging.INFO),
    ):
        result = await _call(page, element_data=CONTAINER, iframe_context=FRAME, action="get_text")

    assert result["evidence"] == "none"
    assert runs == ["get_text"]
    assert len([m for m in caplog.messages if "signal: no-evidence-refused" in m]) == 1
