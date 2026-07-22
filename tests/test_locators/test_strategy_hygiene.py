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
    _validate_strategy_candidates,
)


def make_element_data(**overrides) -> dict:
    """
    Full element_data dict as the STEP-3 extraction JS returns it.

    _build_coordinate_strategies() indexes most keys directly, so every key
    the real JS emits must be present.
    """
    data = {
        "tagName": "button",
        "id": "",
        "name": "",
        "className": "",
        "textContent": "",
        "innerText": "",
        "value": "",
        "placeholder": "",
        "title": "",
        "alt": "",
        "href": "",
        "src": "",
        "type": "",
        "ariaLabel": "",
        "ariaDescribedby": "",
        "dataTestId": "",
        "dataTest": "",
        "dataQa": "",
        "role": "",
        "attributes": {},
        "coordinates": {"x": 100, "y": 100},
        "parentId": "",
        "parentClass": "",
        "siblingIndex": 0,
        "totalSiblings": 0,
    }
    data.update(overrides)
    return data


def by_type(strategies: list, type_name: str) -> list:
    return [s for s in strategies if s["type"] == type_name]


class TestNameStrategyBrowserForm:
    """E8: browser-only — the name strategy always emits the Playwright
    attribute-selector form. The SeleniumLibrary `name=` prefix branch was
    deleted with dual-library support (Task 11); Browser Library has no
    name= engine, so the prefix form never worked in browser mode anyway.
    """

    def test_name_emitted_as_css_attribute_selector(self):
        strategies = _build_coordinate_strategies(make_element_data(name="username"))
        name = by_type(strategies, "name")
        assert len(name) == 1
        assert name[0]["locator"] == '[name="username"]'

    def test_name_value_with_double_quote_is_escaped(self):
        strategies = _build_coordinate_strategies(make_element_data(name='say-"hi"'))
        assert by_type(strategies, "name")[0]["locator"] == '[name="say-\\"hi\\""]'

    def test_no_name_no_strategy(self):
        strategies = _build_coordinate_strategies(make_element_data())
        assert by_type(strategies, "name") == []


class TestDataQaStrategyForm:
    """C2: data-qa must be emitted as a CSS attribute selector.

    Playwright registers data-testid / data-test-id / data-test as built-in
    attribute selector engines, but NOT data-qa. The old form
    'data-qa=value' raised "Unknown engine" on every page that has the
    attribute, so the strategy was dead since written.
    """

    def test_data_qa_emitted_as_css_attribute_selector(self):
        strategies = _build_coordinate_strategies(make_element_data(dataQa="submit-btn"))
        qa = by_type(strategies, "data-qa")
        assert len(qa) == 1
        assert qa[0]["locator"] == '[data-qa="submit-btn"]'

    def test_data_qa_keeps_test_id_priority(self):
        strategies = _build_coordinate_strategies(make_element_data(dataQa="submit-btn"))
        assert by_type(strategies, "data-qa")[0]["priority"] == PRIORITY_TEST_ID

    def test_data_qa_value_with_double_quote_is_escaped(self):
        strategies = _build_coordinate_strategies(make_element_data(dataQa='say-"hi"'))
        locator = by_type(strategies, "data-qa")[0]["locator"]
        assert locator == '[data-qa="say-\\"hi\\""]'

    def test_no_data_qa_no_strategy(self):
        strategies = _build_coordinate_strategies(make_element_data())
        assert by_type(strategies, "data-qa") == []

    def test_data_testid_engine_form_unchanged(self):
        """data-testid IS a built-in Playwright engine - its form must not change."""
        strategies = _build_coordinate_strategies(make_element_data(dataTestId="login"))
        assert by_type(strategies, "data-testid")[0]["locator"] == "data-testid=login"


class TestElementDataTestAttribute:
    """C3: STEP 0 must emit the test attribute that actually exists.

    _extract_dom_node_attributes coalesces data-testid/data-test into one
    dataTestId field. The old emitter always built [data-testid=...], so an
    element carrying only data-test produced a selector matching 0 elements
    and the best available hook was silently lost.
    """

    def test_data_test_source_emits_data_test_selector(self):
        candidates = _build_element_data_candidates(
            {"dataTestId": "login-btn", "dataTestAttr": "data-test"}
        )
        test_candidates = [c for c in candidates if c["priority"] == PRIORITY_TEST_ID]
        assert len(test_candidates) == 1
        assert test_candidates[0]["locator"] == '[data-test="login-btn"]'
        assert test_candidates[0]["type"] == "data-test"

    def test_data_testid_source_emits_data_testid_selector(self):
        candidates = _build_element_data_candidates(
            {"dataTestId": "login-btn", "dataTestAttr": "data-testid"}
        )
        test_candidates = [c for c in candidates if c["priority"] == PRIORITY_TEST_ID]
        assert test_candidates[0]["locator"] == '[data-testid="login-btn"]'
        assert test_candidates[0]["type"] == "data-testid"

    def test_missing_source_attr_defaults_to_data_testid(self):
        """element_data built by older/other paths has no dataTestAttr key."""
        candidates = _build_element_data_candidates({"dataTestId": "login-btn"})
        test_candidates = [c for c in candidates if c["priority"] == PRIORITY_TEST_ID]
        assert test_candidates[0]["locator"] == '[data-testid="login-btn"]'

    def test_no_test_attr_no_candidate(self):
        candidates = _build_element_data_candidates({"id": "x"})
        assert [c for c in candidates if c["priority"] == PRIORITY_TEST_ID] == []


class TestParentContextCssStep0:
    """The Priority-5.5 parent-context CSS fallback must fire for the STEP-0
    payload. ``_extract_dom_node_attributes`` — the only producer that reaches
    ``_build_element_data_candidates`` — carries the parent's classes as
    ``parentClassName`` and emits no ``parentId``/``parentClass``. The block
    read only ``parentClass``, so it was dead for every STEP-0 element that
    lacked an id/name of its own.
    """

    def test_parent_class_css_from_step0_parent_class_name(self):
        candidates = _build_element_data_candidates(
            {
                "tagName": "input",
                "type": "text",
                "id": "",
                "name": "",
                "parentClassName": "search-box wrapper",
            }
        )
        parent_css = [c for c in candidates if c["type"] == "parent-class-css"]
        assert len(parent_css) == 1
        assert parent_css[0]["locator"] == '.search-box input[type="text"]'

    def test_legacy_parent_class_key_still_read(self):
        """Coordinate-shape payloads carry ``parentClass`` — keep them working."""
        candidates = _build_element_data_candidates(
            {
                "tagName": "input",
                "id": "",
                "name": "",
                "parentClass": "legacy-wrap",
            }
        )
        parent_css = [c for c in candidates if c["type"] == "parent-class-css"]
        assert len(parent_css) == 1
        assert parent_css[0]["locator"] == ".legacy-wrap input"

    def test_no_parent_context_no_candidate(self):
        candidates = _build_element_data_candidates(
            {
                "tagName": "input",
                "id": "",
                "name": "",
            }
        )
        assert [c for c in candidates if c["type"] == "parent-class-css"] == []


class TestXPathStringLiterals:
    """C4: XPath text values must be real XPath 1.0 string literals.

    The old code did text.replace("'", "\\'") — XPath 1.0 has no backslash
    escaping, so any apostrophe made the expression a syntax error and the
    strategy died. _xpath_string_literal picks quoting (or concat()) that is
    always valid.
    """

    def test_xpath_text_with_apostrophe(self):
        strategies = _build_coordinate_strategies(make_element_data(innerText="It's a deal"))
        locator = by_type(strategies, "xpath-text")[0]["locator"]
        assert locator == 'xpath=//button[contains(text(), "It\'s a deal")]'

    def test_xpath_title_with_apostrophe(self):
        strategies = _build_coordinate_strategies(make_element_data(title="Bob's data"))
        locator = by_type(strategies, "xpath-title")[0]["locator"]
        assert locator == 'xpath=//button[@title="Bob\'s data"]'

    def test_xpath_multi_attr_with_apostrophe(self):
        strategies = _build_coordinate_strategies(
            make_element_data(className="btn", innerText="It's here")
        )
        locator = by_type(strategies, "xpath-multi-attr")[0]["locator"]
        assert locator == (
            "xpath=//button[contains(@class, 'btn') and contains(text(), \"It's here\")]"
        )

    def test_no_backslash_escaping_anywhere(self):
        strategies = _build_coordinate_strategies(
            make_element_data(className="btn", innerText="It's", title="Bob's")
        )
        for s in strategies:
            assert "\\'" not in s["locator"], s

    def test_both_quote_types_use_concat(self):
        strategies = _build_coordinate_strategies(
            make_element_data(innerText='He said "don\'t" now')
        )
        locator = by_type(strategies, "xpath-text")[0]["locator"]
        assert "concat(" in locator
        assert "\\'" not in locator

    def test_plain_text_keeps_single_quotes(self):
        strategies = _build_coordinate_strategies(make_element_data(innerText="Submit Order"))
        locator = by_type(strategies, "xpath-text")[0]["locator"]
        assert locator == "xpath=//button[contains(text(), 'Submit Order')]"

    def test_xpath_text_truncates_raw_text_to_50(self):
        strategies = _build_coordinate_strategies(make_element_data(innerText="x" * 60))
        locator = by_type(strategies, "xpath-text")[0]["locator"]
        assert locator == f"xpath=//button[contains(text(), '{'x' * 50}')]"

    def test_xpath_multi_attr_truncates_raw_text_to_30(self):
        strategies = _build_coordinate_strategies(
            make_element_data(className="btn", innerText="y" * 40)
        )
        locator = by_type(strategies, "xpath-multi-attr")[0]["locator"]
        assert f"contains(text(), '{'y' * 30}')" in locator


class TestCssEscaping:
    """C5: raw class/id values must be CSS-escaped in the STEP-3 CSS strategies.

    Tailwind-style classes (w-1/2, md:flex, p-1.5) contain characters with
    CSS meta meaning. Unescaped they made Strategies 11/12/13 invalid
    selectors that threw on every page using such classes. Only those three
    strategies interpolate raw values into CSS - the XPath class strategies
    quote the value as a plain string and must NOT be CSS-escaped.
    """

    def test_css_class_escapes_tailwind_slash(self):
        strategies = _build_coordinate_strategies(make_element_data(className="w-1/2 p-4"))
        assert by_type(strategies, "css-class")[0]["locator"] == "button.w-1\\/2"

    def test_css_parent_id_escapes_id_and_class(self):
        strategies = _build_coordinate_strategies(
            make_element_data(parentId="form:main", className="w-1/2 p-4")
        )
        locator = by_type(strategies, "css-parent-id")[0]["locator"]
        assert locator == "#form\\:main button.w-1\\/2"

    def test_css_nth_child_escapes_parent_class(self):
        strategies = _build_coordinate_strategies(
            make_element_data(parentClass="grid-2/3 wrap", siblingIndex=2)
        )
        locator = by_type(strategies, "css-nth-child")[0]["locator"]
        assert locator == ".grid-2\\/3 > button:nth-child(2)"

    def test_unescapable_parent_id_skips_css_strategy_only(self):
        """Whitespace inside an id cannot be CSS-escaped -> skip Strategy 11.

        The XPath parent-id strategy quotes the id as a string and survives.
        """
        strategies = _build_coordinate_strategies(
            make_element_data(parentId="my id", className="btn")
        )
        assert by_type(strategies, "css-parent-id") == []
        assert len(by_type(strategies, "xpath-parent-id")) == 1

    def test_plain_values_unchanged(self):
        strategies = _build_coordinate_strategies(
            make_element_data(parentId="formMain", className="btn primary")
        )
        assert by_type(strategies, "css-parent-id")[0]["locator"] == "#formMain button.btn"
        assert by_type(strategies, "css-class")[0]["locator"] == "button.btn"

    def test_xpath_class_strategies_keep_raw_slash(self):
        strategies = _build_coordinate_strategies(
            make_element_data(className="w-1/2 p-4", parentClass="grid-2/3 x", siblingIndex=2)
        )
        for type_name in (
            "xpath-parent-class-position",
            "xpath-class-position",
            "xpath-first-of-class",
        ):
            locator = by_type(strategies, type_name)[0]["locator"]
            assert "\\" not in locator, locator


class FakeLocator:
    """Stands in for a Playwright Locator: count() plus the element-info
    evaluate() that validate_semantic_match's fallback path runs."""

    def __init__(self, count: int, text: str = ""):
        self._count = count
        self._text = text

    async def count(self) -> int:
        return self._count

    async def evaluate(self, js: str) -> dict:
        return {
            "textContent": self._text,
            "textContentLength": len(self._text),
            "innerText": self._text,
            "placeholder": "",
            "ariaLabel": "",
            "value": "",
        }


class FakeSearchContext:
    """Maps selector string -> FakeLocator, like page.locator(sel)."""

    def __init__(self, locators: dict):
        self._locators = locators

    def locator(self, selector: str) -> FakeLocator:
        return self._locators[selector]


def make_strategy_pair() -> list:
    """A priority-1 id candidate plus a priority-6 text candidate, both of
    which the fake DOM will report as unique."""
    return [
        {"type": "id", "locator": "id=save", "priority": 1, "strategy": "Native ID attribute"},
        {
            "type": "text",
            "locator": 'text="Submit order"',
            "priority": 6,
            "strategy": "Visible text content",
        },
    ]


class TestSemanticGateBeforeEarlyExit:
    """C6: the priority<=3 early-exit must not trust a unique-but-wrong id.

    The old loop broke on the first unique high-priority candidate without
    looking at its text. When coordinates land on the wrong element, that
    left Step 5 with a single semantically-wrong locator -> found=false,
    even though a lower-priority strategy pointed at the right element.
    """

    async def test_wrong_text_id_does_not_break_early(self):
        ctx = FakeSearchContext(
            {
                "id=save": FakeLocator(1, "Save draft"),
                'text="Submit order"': FakeLocator(1, "Submit order"),
            }
        )
        result = await _validate_strategy_candidates(
            ctx, make_strategy_pair(), expected_text="Submit order"
        )
        assert len(result) == 2
        assert all(r["validated"] and r["unique"] for r in result)

    async def test_vetoed_id_stays_valid_for_step_5(self):
        """The gate only controls the break - Step 5 stays authoritative,
        so the wrong-text id must still be recorded as unique/valid."""
        ctx = FakeSearchContext(
            {
                "id=save": FakeLocator(1, "Save draft"),
                'text="Submit order"': FakeLocator(1, "Submit order"),
            }
        )
        result = await _validate_strategy_candidates(
            ctx, make_strategy_pair(), expected_text="Submit order"
        )
        id_entry = [r for r in result if r["type"] == "id"][0]
        assert id_entry["unique"] and id_entry["valid"]

    async def test_matching_text_id_keeps_early_exit(self):
        ctx = FakeSearchContext(
            {
                "id=save": FakeLocator(1, "Submit order"),
                'text="Submit order"': FakeLocator(1, "Submit order"),
            }
        )
        result = await _validate_strategy_candidates(
            ctx, make_strategy_pair(), expected_text="Submit order"
        )
        assert len(result) == 1
        assert result[0]["type"] == "id"

    async def test_no_expected_text_keeps_early_exit(self):
        ctx = FakeSearchContext(
            {
                "id=save": FakeLocator(1, "Save draft"),
                'text="Submit order"': FakeLocator(1, "Submit order"),
            }
        )
        result = await _validate_strategy_candidates(ctx, make_strategy_pair())
        assert len(result) == 1
        assert result[0]["type"] == "id"
