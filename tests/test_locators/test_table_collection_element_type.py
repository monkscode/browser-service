"""The accessibility table detector produces a COLLECTION, not a bespoke
"table-rows" element_type.

``_find_table_via_accessibility`` returns a locator matching ALL data rows
(``count=N``, ``unique=False``) — it is structurally a collection producer,
and the ONLY live site that stamped ``element_type="table-rows"``. Nothing
routed on that string:

  - browser-service's own re-ranker (workflow.py) exempts only
    ``element_type == "collection"`` from the unique-locator filter, so the
    multi-row (``unique=False``) locator was dropped from scoring;
  - the validation gate (workflow.py) treats only ``"collection"`` as
    count>1-expected, so the row locator was flagged "not unique";
  - the NL-to-RF assembler routes its FOR-loop block on
    ``element_type == "collection"`` (``_needs_loop``), so no loop was
    generated → a single-element keyword on a many-match locator raises a
    strict-mode violation at run time, invisible to ``--dryrun``.

Emitting ``"collection"`` — the canonical vocabulary the classifier already
uses for ``<tr>``/``<li>`` — fixes all three at once.

Fast fake-context unit tests (no chromium), mirroring
test_accessibility_fallback_guards.py.
"""

import pytest

from browser_service.locators.smart_locator import (
    _find_element_via_accessibility,
    _find_table_via_accessibility,
)


class _FakeRowCount:
    def __init__(self, count: int):
        self._count = count

    async def count(self) -> int:
        return self._count


class _FakeTableCtx:
    """search_context stand-in: ``evaluate()`` answers the elementFromPoint
    JS with a data table; ``locator()`` reports the row count."""

    def __init__(self, row_count: int = 3):
        self._row_count = row_count

    async def evaluate(self, js, arg=None):
        return {
            "found": True,
            "role": "table",
            "ariaLabel": "Customers",
            "totalRows": self._row_count + 1,  # + header row
            "dataRowCount": self._row_count,
            "visibleRowCount": self._row_count,
            "hadRowElement": True,
            "tag": "table",
            "rowTexts": ["Alice Smith", "Bob Jones", "Carol White"][: self._row_count],
            "rowSelector": "tbody tr",
            "selectorType": "html-table",
        }

    def locator(self, selector: str):
        return _FakeRowCount(self._row_count)


@pytest.mark.asyncio
async def test_table_detector_stamps_collection():
    """The producer itself: a detected data table is a 'collection'."""
    ctx = _FakeTableCtx(row_count=3)
    result = await _find_table_via_accessibility(ctx, x=100.0, y=200.0, expected_text=None)

    assert result is not None
    # The fix: was "table-rows" (routed nowhere) → "collection" (routes the
    # FOR-loop block and is exempt from the unique-locator filter).
    assert result["element_type"] == "collection"
    assert result["unique"] is False
    assert result["count"] == 3


@pytest.mark.asyncio
async def test_accessibility_entry_routes_table_to_collection():
    """End-to-end through the accessibility entry point: a table-keyword
    description with no expected_text reaches the table detector, and the
    returned result carries element_type='collection'."""
    ctx = _FakeTableCtx(row_count=3)
    result = await _find_element_via_accessibility(
        page=ctx,
        x=100.0,
        y=200.0,
        element_description="all data rows in the results table",
        expected_text=None,  # skip STRATEGY 2.5a (coordinate-independent role)
        search_context=ctx,
    )

    assert result is not None
    assert result["element_type"] == "collection"
    assert result["unique"] is False
    assert result["count"] == 3
