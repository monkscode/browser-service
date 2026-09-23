"""
D v1: element_approach_metrics[].element_tag reports the node the RETURNED
locator resolves to. It used to mean "the indexed node's tag" on cascade results,
so a pass on the wrong element (u07: indexed <a>, returned <a>, golden <h6>) and
an honest pass looked the same. New fields only; best_locator, semantic_match
and element_info are never touched.
"""

from browser_service.agent.actions import _stamp_resolved_tag


class FakeLocator:
    def __init__(self, resolved=None, raises=False):
        self._resolved = resolved or {}
        self._raises = raises

    def nth(self, i):
        return self

    async def evaluate(self, js, arg=None, *, timeout=None):
        if self._raises:
            raise RuntimeError("Element is not attached to the DOM")
        return dict(self._resolved)


class FakeRoot:
    def __init__(self, locators):
        self._locators = locators
        self.asked = []

    def locator(self, selector):
        self.asked.append(selector)
        return self._locators.get(selector, FakeLocator(raises=True))


def _result(found=True, best="css=h6.crumb", metrics=None):
    return {
        "found": found,
        "best_locator": best,
        "element_info": {"tagName": "a"},
        "approach_metrics": dict(
            metrics or {"element_tag": "a", "locator_approach": "element_data"}
        ),
    }


async def test_found_result_gets_the_resolved_tag():
    root = FakeRoot({"css=h6.crumb": FakeLocator({"tagName": "h6"})})
    result = _result()
    await _stamp_resolved_tag(result, root, {"tagName": "a"}, None)
    m = result["approach_metrics"]
    assert (m["element_tag"], m["element_tag_source"], m["indexed_tag"]) == ("h6", "resolved", "a")


async def test_unreadable_found_result_is_unknown():
    """Review Focus 5: unreadable -> "" and 'unknown', never a raise."""
    root = FakeRoot({})
    result = _result(best="xpath=//not[")
    await _stamp_resolved_tag(result, root, {"tagName": "a"}, None)
    m = result["approach_metrics"]
    assert (m["element_tag"], m["element_tag_source"]) == ("", "unknown")


async def test_not_found_result_keeps_the_indexed_tag():
    root = FakeRoot({})
    result = _result(found=False)
    await _stamp_resolved_tag(result, root, {"tagName": "a"}, None)
    m = result["approach_metrics"]
    assert (m["element_tag"], m["element_tag_source"], m["indexed_tag"]) == ("a", "indexed", "a")
    assert root.asked == []


async def test_no_metrics_dict_is_left_alone():
    root = FakeRoot({})
    result = {"found": True, "best_locator": "css=h6.crumb"}
    await _stamp_resolved_tag(result, root, {"tagName": "a"}, None)
    assert "approach_metrics" not in result
    assert root.asked == []


async def test_iframe_prefix_is_stripped():
    """Review Focus 5: '<iframe> >>> x' is Browser-library syntax; read x through the frame."""
    root = FakeRoot({"#go": FakeLocator({"tagName": "button"})})
    result = _result(best='iframe[id="main"] >>> #go')
    await _stamp_resolved_tag(result, root, {"tagName": "iframe"}, 'iframe[id="main"]')
    assert root.asked == ["#go"]
    assert result["approach_metrics"]["element_tag"] == "button"


async def test_no_element_data_gives_an_empty_indexed_tag():
    root = FakeRoot({"css=h6.crumb": FakeLocator({"tagName": "h6"})})
    result = _result()
    await _stamp_resolved_tag(result, root, None, None)
    assert result["approach_metrics"]["indexed_tag"] == ""


async def test_only_approach_metrics_change():
    root = FakeRoot({"css=h6.crumb": FakeLocator({"tagName": "h6"})})
    result = _result()
    before = {k: v for k, v in result.items() if k != "approach_metrics"}
    await _stamp_resolved_tag(result, root, {"tagName": "a"}, None)
    assert {k: v for k, v in result.items() if k != "approach_metrics"} == before
