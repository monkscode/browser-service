"""bs #86 — an explicit is_collection=True that ends in ONE unique element is "not found", not a one-element answer.

Real failure (bench q10, books.toscrape.com, run b9a6f92f-4938-4689-a7ae-8148697d3442): "book item containers in the
main product grid", is_collection=True, no text. The cascade shipped `xpath=//li[1]/article/div[1]/a` (unique) and
the generated test counted 1 of 20. Owner decision 2026-11-19: a True flag ships a group, or not found.

Unit tests of the helper that gates the cascade's final result (the cascade itself is covered by the integration
tests in tests/test_locators/test_collection_indexed_walk.py).
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_service.agent.actions import (
    _refuse_one_element_for_group,
    find_unique_locator_action,
)

DESCRIPTION = "book item containers in the main product grid"


def _plain():
    return {
        "element_id": "elem_2",
        "description": DESCRIPTION,
        "found": True,
        "best_locator": "xpath=//li[1]/article/div[1]/a",
        "unique": True,
        "count": 1,
        "element_type": "link",
        "coordinates": {"x": 10, "y": 20},
    }


def test_true_plain_unique_is_refused(caplog):
    with caplog.at_level(logging.INFO):
        out = _refuse_one_element_for_group(_plain(), True, None, DESCRIPTION)
    assert out["found"] is False
    assert out["error_type"] == "CollectionNotFound"
    assert (
        out["error"]
        == "is_collection was true but no locator matching the group was found; a one-element "
        "locator was refused"
    )
    assert out["refused_locator"] == "xpath=//li[1]/article/div[1]/a"
    assert out["element_id"] == "elem_2"
    assert out["description"] == DESCRIPTION
    assert out["validated"] is False
    assert out["count"] == 0
    assert out["unique"] is False
    assert out["valid"] is False
    assert "⛔ COLLECTION NOT FOUND for elem_2" in caplog.text
    assert "signal: collection-not-found" in caplog.text


def test_true_collection_answer_is_kept():
    result = {**_plain(), "element_type": "collection", "unique": False, "count": 4}
    assert _refuse_one_element_for_group(result, True, None, DESCRIPTION) is result


def test_true_unique_false_is_kept():
    result = {**_plain(), "unique": False, "count": 4}
    assert _refuse_one_element_for_group(result, True, None, DESCRIPTION) is result


@pytest.mark.parametrize("kind", ["dropdown", "checkbox", "radio", "file-upload", "date-picker"])
def test_true_specialised_type_is_kept(kind):
    result = {**_plain(), "element_type": kind}
    assert _refuse_one_element_for_group(result, True, None, DESCRIPTION) is result


def test_true_with_explicit_anchor_is_kept():
    result = _plain()
    assert _refuse_one_element_for_group(result, True, "Smith", DESCRIPTION) is result


def test_true_with_derived_anchor_is_kept():
    result = _plain()
    description = "the edit link in the row containing 'Smith'"
    assert _refuse_one_element_for_group(result, True, None, description) is result


@pytest.mark.parametrize("flag", [False, None])
def test_false_or_absent_flag_is_kept(flag):
    result = _plain()
    assert _refuse_one_element_for_group(result, flag, None, DESCRIPTION) is result


def test_not_found_result_is_kept():
    result = {**_plain(), "found": False}
    assert _refuse_one_element_for_group(result, True, None, DESCRIPTION) is result


# --- the wiring: find_unique_locator_action applies the helper to the cascade's final result ---

CASCADE_RESULT = {
    "element_id": "elem_2",
    "found": True,
    "best_locator": "css=.from-cascade",
    "unique": True,
    "count": 1,
    "element_type": "link",
}


@pytest.fixture
def page():
    shipped = MagicMock()
    shipped.count = AsyncMock(return_value=1)
    pg = MagicMock()
    pg.url = "https://example.com/app"
    pg.locator = MagicMock(return_value=shipped)
    return pg


@pytest.fixture
def cascade():
    spy = AsyncMock(return_value=dict(CASCADE_RESULT))
    with patch("browser_service.locators.find_unique_locator_at_coordinates", new=spy):
        yield spy


async def _action(page, **kwargs):
    return await find_unique_locator_action(
        page=page,
        x=120,
        y=340,
        element_id="elem_2",
        element_description=DESCRIPTION,
        element_data={"tagName": "a", "className": ""},
        **kwargs,
    )


async def test_action_refuses_a_unique_cascade_answer_for_true(page, cascade):
    result = await _action(page, is_collection=True)
    cascade.assert_called_once()
    assert result["found"] is False
    assert result["error_type"] == "CollectionNotFound"
    assert result["refused_locator"] == "css=.from-cascade"


async def test_action_passes_the_unique_cascade_answer_without_the_flag(page, cascade):
    """Control: is_collection=None leaves the cascade's unique answer alone."""
    result = await _action(page, is_collection=None)
    assert result["found"] is True
    assert result["best_locator"] == "css=.from-cascade"
