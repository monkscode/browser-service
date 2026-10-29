"""bs #72: a find_unique_locator call with no element evidence is refused up front.

Mocked page, no Chromium; the real-DOM proof is test_no_evidence_refused_live.py.

Evidence is an element the index or the point lookup found (a non-empty
``element_data``, filled in by registration.py before the call) or an
``expected_text`` with a non-blank character. Without either the call names no
element, so it is refused before the page is touched: no candidate is counted,
no frame is entered, the cascade does not run. A row anchor, ``is_collection`` or
a candidate locator alone are NOT evidence (owner: collections get no exception).
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_service.agent.actions import find_unique_locator_action

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
