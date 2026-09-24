"""
D v1 wiring: every found result that leaves find_unique_locator_action carries
the resolved tag. smart_locator has 11 approach_metrics returns
(**_approach_metrics_base); all of them reach the caller through ONE call site
(_run_cascade), so the stamp sits there, plus the two candidate accepts that
return early.
"""

import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import browser_service.agent.actions as actions_module
import browser_service.locators.smart_locator as smart_locator_module
from browser_service.agent.actions import find_unique_locator_action

INDEXED = {"tagName": "a", "id": "", "textContent": "Dashboard", "xpath": "html/body/a"}

# (locator_approach, found) for the 11 smart_locator returns on 62f640b:
# 6 found (element_data, collection, text_first, semantic, accessibility,
# coordinate_fallback@5300) and 5 failures (coordinate_fallback).
APPROACHES = [
    ("element_data", True),
    ("collection", True),
    ("text_first", True),
    ("semantic", True),
    ("accessibility", True),
    ("coordinate_fallback", True),
    ("coordinate_fallback", False),
    ("coordinate_fallback", False),
    ("coordinate_fallback", False),
    ("coordinate_fallback", False),
    ("coordinate_fallback", False),
]


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
        if "labelledbyText" in js:
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
    url = "https://example.test/"

    def __init__(self, locators):
        self._locators = locators

    def locator(self, selector):
        return self._locators.get(selector, FakeLocator(0))

    def frame_locator(self, selector):
        return self

    async def evaluate(self, *args, **kwargs):
        return None


def _cascade_result(approach, found):
    return {
        "element_id": "elem_1",
        "found": found,
        "best_locator": "css=h6.crumb" if found else None,
        "all_locators": [],
        "element_info": {"tagName": "a"},
        "approach_metrics": {
            "element_tag": "a",
            "locator_approach": approach,
            "fallback_depth": 7,
            "success": found,
        },
    }


async def _cascade_call(cascade_result):
    page = FakePage({"css=h6.crumb": FakeLocator(1, {"tagName": "h6"})})
    with patch(
        "browser_service.locators.find_unique_locator_at_coordinates",
        return_value=dict(cascade_result),
    ):
        return await find_unique_locator_action(
            x=10,
            y=10,
            element_id="elem_1",
            element_description="the heading",
            element_data=dict(INDEXED),
            page=page,
        )


@pytest.mark.parametrize("approach, found", APPROACHES)
async def test_every_cascade_return_is_stamped(approach, found):
    result = await _cascade_call(_cascade_result(approach, found))
    m = result["approach_metrics"]
    assert m["indexed_tag"] == "a"
    if found:
        assert (m["element_tag"], m["element_tag_source"]) == ("h6", "resolved")
    else:
        assert (m["element_tag"], m["element_tag_source"]) == ("a", "indexed")


def test_the_population_is_the_one_the_test_names():
    """If smart_locator gains or loses an approach_metrics return, or actions gains a
    second cascade call site, this fails, and APPROACHES / the stamp sites are revisited."""
    smart_src = Path(smart_locator_module.__file__).read_text(encoding="utf-8")
    actions_src = Path(actions_module.__file__).read_text(encoding="utf-8")
    assert smart_src.count("**_approach_metrics_base") == 11
    assert len(re.findall(r"find_unique_locator_at_coordinates\(", actions_src)) == 1


async def test_unique_candidate_accept_is_stamped():
    resolved = {"tagName": "h6", "textContent": "Dashboard", "id": ""}
    page = FakePage({"css=h6.crumb": FakeLocator(1, resolved)})
    result = await find_unique_locator_action(
        x=10,
        y=10,
        element_id="elem_1",
        element_description="the heading",
        expected_text="Dashboard",
        candidate_locator="css=h6.crumb",
        element_data={"tagName": "h6", "id": "", "textContent": "Dashboard"},
        page=page,
    )
    assert result["all_locators"][0]["type"] == "candidate"
    m = result["approach_metrics"]
    assert (m["element_tag"], m["element_tag_source"], m["indexed_tag"]) == ("h6", "resolved", "h6")


async def test_collection_candidate_accept_is_stamped():
    resolved = {"tagName": "li", "textContent": "Book", "id": ""}
    page = FakePage({"css=ol.row > li": FakeLocator(20, resolved)})
    result = await find_unique_locator_action(
        x=10,
        y=10,
        element_id="elem_2",
        element_description="all books",
        candidate_locator="css=ol.row > li",
        is_collection=True,
        element_data={"tagName": "li", "id": "", "textContent": "Book"},
        page=page,
    )
    assert result["element_type"] == "collection"
    m = result["approach_metrics"]
    assert (m["element_tag"], m["element_tag_source"]) == ("li", "resolved")


async def test_stamp_changes_nothing_but_approach_metrics():
    """best_locator, semantic_match and element_info are identical with and without D."""
    stamped = await _cascade_call(_cascade_result("text_first", True))
    with patch("browser_service.agent.actions._stamp_resolved_tag", AsyncMock(return_value=None)):
        plain = await _cascade_call(_cascade_result("text_first", True))
    for key in ("best_locator", "semantic_match", "element_info", "found", "all_locators"):
        assert stamped.get(key) == plain.get(key), key
