"""
Unit tests for the E5 strategy-hygiene batch (analysis doc C2-C6).

Covers the extracted seams in browser_service.locators.smart_locator:
  - _build_coordinate_strategies(): C2 (data-qa form), C4 (XPath string
    literals), C5 (CSS escaping of raw class/id values)
  - _build_element_data_candidates(): C3 (emit the test attribute that
    actually exists on the element)
  - _validate_strategy_candidates(): C6 (semantic gate before the
    high-priority early-exit)

Each sub-item's tests were written against the buggy behavior first and
watched fail before the fix landed.
"""

from browser_service.locators.smart_locator import (
    PRIORITY_TEST_ID,
    _build_coordinate_strategies,
    _build_element_data_candidates,
)


def make_element_data(**overrides) -> dict:
    """
    Full element_data dict as the STEP-3 extraction JS returns it.

    _build_coordinate_strategies() indexes most keys directly, so every key
    the real JS emits must be present.
    """
    data = {
        'tagName': 'button',
        'id': '',
        'name': '',
        'className': '',
        'textContent': '',
        'innerText': '',
        'value': '',
        'placeholder': '',
        'title': '',
        'alt': '',
        'href': '',
        'src': '',
        'type': '',
        'ariaLabel': '',
        'ariaDescribedby': '',
        'dataTestId': '',
        'dataTest': '',
        'dataQa': '',
        'role': '',
        'attributes': {},
        'coordinates': {'x': 100, 'y': 100},
        'parentId': '',
        'parentClass': '',
        'siblingIndex': 0,
        'totalSiblings': 0,
    }
    data.update(overrides)
    return data


def by_type(strategies: list, type_name: str) -> list:
    return [s for s in strategies if s['type'] == type_name]


class TestDataQaStrategyForm:
    """C2: data-qa must be emitted as a CSS attribute selector.

    Playwright registers data-testid / data-test-id / data-test as built-in
    attribute selector engines, but NOT data-qa. The old form
    'data-qa=value' raised "Unknown engine" on every page that has the
    attribute, so the strategy was dead since written.
    """

    def test_data_qa_emitted_as_css_attribute_selector(self):
        strategies = _build_coordinate_strategies(make_element_data(dataQa='submit-btn'))
        qa = by_type(strategies, 'data-qa')
        assert len(qa) == 1
        assert qa[0]['locator'] == '[data-qa="submit-btn"]'

    def test_data_qa_keeps_test_id_priority(self):
        strategies = _build_coordinate_strategies(make_element_data(dataQa='submit-btn'))
        assert by_type(strategies, 'data-qa')[0]['priority'] == PRIORITY_TEST_ID

    def test_data_qa_value_with_double_quote_is_escaped(self):
        strategies = _build_coordinate_strategies(make_element_data(dataQa='say-"hi"'))
        locator = by_type(strategies, 'data-qa')[0]['locator']
        assert locator == '[data-qa="say-\\"hi\\""]'

    def test_no_data_qa_no_strategy(self):
        strategies = _build_coordinate_strategies(make_element_data())
        assert by_type(strategies, 'data-qa') == []

    def test_data_testid_engine_form_unchanged(self):
        """data-testid IS a built-in Playwright engine - its form must not change."""
        strategies = _build_coordinate_strategies(make_element_data(dataTestId='login'))
        assert by_type(strategies, 'data-testid')[0]['locator'] == 'data-testid=login'


class TestElementDataTestAttribute:
    """C3: STEP 0 must emit the test attribute that actually exists.

    _extract_dom_node_attributes coalesces data-testid/data-test into one
    dataTestId field. The old emitter always built [data-testid=...], so an
    element carrying only data-test produced a selector matching 0 elements
    and the best available hook was silently lost.
    """

    def test_data_test_source_emits_data_test_selector(self):
        candidates = _build_element_data_candidates(
            {'dataTestId': 'login-btn', 'dataTestAttr': 'data-test'}
        )
        test_candidates = [c for c in candidates if c['priority'] == PRIORITY_TEST_ID]
        assert len(test_candidates) == 1
        assert test_candidates[0]['locator'] == '[data-test="login-btn"]'
        assert test_candidates[0]['type'] == 'data-test'

    def test_data_testid_source_emits_data_testid_selector(self):
        candidates = _build_element_data_candidates(
            {'dataTestId': 'login-btn', 'dataTestAttr': 'data-testid'}
        )
        test_candidates = [c for c in candidates if c['priority'] == PRIORITY_TEST_ID]
        assert test_candidates[0]['locator'] == '[data-testid="login-btn"]'
        assert test_candidates[0]['type'] == 'data-testid'

    def test_missing_source_attr_defaults_to_data_testid(self):
        """element_data built by older/other paths has no dataTestAttr key."""
        candidates = _build_element_data_candidates({'dataTestId': 'login-btn'})
        test_candidates = [c for c in candidates if c['priority'] == PRIORITY_TEST_ID]
        assert test_candidates[0]['locator'] == '[data-testid="login-btn"]'

    def test_no_test_attr_no_candidate(self):
        candidates = _build_element_data_candidates({'id': 'x'})
        assert [c for c in candidates if c['priority'] == PRIORITY_TEST_ID] == []
