"""
Unit tests for the Day 02 validate_semantic_match rewrite.

Covers:
  - Fast-path rejection: candidate locator is unique but resolves to the wrong element
  - Bare input (empty expected_text): new function does NOT silently return True when
    haystack is non-empty (old short-circuit removed)
  - Icon/aria-only match: ax_node.name surface matches expected_text → True
  - Probe 18 carve-out: SVG-icon-only interactive elements with empty haystack are accepted
  - Legacy fallback (page + locator): evaluate-based path used when node is None

These tests exercise validate_semantic_match directly (unit) and the Change C
fast-path integration via find_unique_locator_action (integration/mocked).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_node(
    tag="input",
    ax_name="",
    meaningful_text="",
    placeholder="",
    aria_label="",
    value="",
    node_value="",
):
    """Build a minimal EnhancedDOMTreeNode mock."""
    node = MagicMock()
    node.tag_name = tag
    node.node_name = tag
    node.node_value = node_value
    node.attributes = {
        "placeholder": placeholder,
        "aria-label": aria_label,
        "value": value,
    }
    node.get_meaningful_text_for_llm = MagicMock(return_value=meaningful_text)

    ax = MagicMock()
    ax.name = ax_name
    node.ax_node = ax

    return node


def _make_page_with_evaluate(evaluate_result):
    """Build a mock Playwright page whose locator().evaluate() returns evaluate_result."""
    locator_mock = AsyncMock()
    locator_mock.count = AsyncMock(return_value=1)
    locator_mock.evaluate = AsyncMock(return_value=evaluate_result)

    page = MagicMock()
    page.locator = MagicMock(return_value=locator_mock)
    return page


# ─────────────────────────────────────────────────────────────────────────────
# Test: no `if not expected_text: return True` short-circuit
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyExpectedText:
    """Bare-input case: expected_text is absent. Old code silently returned True."""

    @pytest.mark.asyncio
    async def test_empty_expected_text_non_empty_haystack_does_not_return_true(self):
        """
        When expected_text is '' and the node has meaningful text, the new function
        must NOT return (True, '').  Probe 16 case: bare login-email input whose
        get_meaningful_text_for_llm() returns 'Enter your login email'.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        node = _make_node(
            tag="input",
            meaningful_text="Enter your login email",
        )

        is_match, observed = await validate_semantic_match(node, "")

        # Old behaviour was (True, "") — new behaviour must be different.
        assert not is_match, (
            "Expected (False, ...) when expected_text is empty but haystack "
            "is non-empty — the old short-circuit has been removed."
        )

    @pytest.mark.asyncio
    async def test_empty_expected_text_empty_haystack_is_accepted(self):
        """
        Truly anonymous element: empty expected_text AND empty haystack.
        The function should accept it (the only case the old short-circuit protected).
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        node = _make_node(tag="div", ax_name="", meaningful_text="")

        is_match, _ = await validate_semantic_match(node, "")
        assert is_match, "Anonymous element (empty haystack) with empty expected_text should be accepted"


# ─────────────────────────────────────────────────────────────────────────────
# Test: icon / aria-only buttons (probe 16 case)
# ─────────────────────────────────────────────────────────────────────────────

class TestIconAriaOnlyButton:
    """ax_node.name is the sole semantic surface — probe 16 icon case."""

    @pytest.mark.asyncio
    async def test_ax_name_matches_expected_text(self):
        """
        Icon-only submit button: ax_node.name == 'Submit the form'.
        expected_text='Submit the form' → should match.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        node = _make_node(
            tag="button",
            ax_name="Submit the form",
            meaningful_text="Submit the form",
        )

        is_match, _ = await validate_semantic_match(node, "Submit the form")
        assert is_match

    @pytest.mark.asyncio
    async def test_ax_name_does_not_match_wrong_sibling(self):
        """
        Wrong sibling: ax_node.name == 'Log in'.
        expected_text='Register' → should not match.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        node = _make_node(
            tag="button",
            ax_name="Log in",
            meaningful_text="Log in",
        )

        is_match, _ = await validate_semantic_match(node, "Register")
        assert not is_match


# ─────────────────────────────────────────────────────────────────────────────
# Test: probe 18 carve-out (SVG icon buttons with empty haystacks)
# ─────────────────────────────────────────────────────────────────────────────

class TestProbe18Carveout:
    """
    demoqa.com/webtables pattern: SVG-only icon button, no text/aria-label/placeholder.
    When haystack is empty AND tag is interactive, accept rather than reject.
    """

    @pytest.mark.asyncio
    async def test_empty_haystack_interactive_tag_is_accepted(self):
        """SVG-icon-only button with no semantic surface: carve-out accepts it."""
        from browser_service.locators.smart_locator import validate_semantic_match

        node = _make_node(tag="button", ax_name="", meaningful_text="")

        is_match, _ = await validate_semantic_match(node, "delete row")
        assert is_match, (
            "Probe 18 carve-out: nameless SVG-icon button should be accepted, "
            "not silently rejected, to avoid regressions on admin dashboards."
        )

    @pytest.mark.asyncio
    async def test_empty_haystack_non_interactive_tag_is_rejected(self):
        """
        A <div> with no semantic surface: NOT interactive, no carve-out.
        expected_text provided → should return False.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        node = _make_node(tag="div", ax_name="", meaningful_text="")

        is_match, _ = await validate_semantic_match(node, "delete row")
        assert not is_match


# ─────────────────────────────────────────────────────────────────────────────
# Test: legacy fallback (page + locator, no node)
# ─────────────────────────────────────────────────────────────────────────────

class TestLegacyFallback:
    """
    When node=None, the function falls back to a single page.locator(sel).evaluate()
    call. Verifies the fast-path rejection scenario from probe 06.
    """

    @pytest.mark.asyncio
    async def test_wrong_element_via_evaluate_returns_false(self):
        """
        Probe 06 scenario: #search-input resolves to placeholder 'Search the site',
        but expected_text is 'Search products'.
        Legacy path should return False — the hole is closed.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        page = _make_page_with_evaluate({
            "tag": "input",
            "textContent": "",
            "innerText": "",
            "placeholder": "Search the site",
            "ariaLabel": "",
            "value": "",
        })

        is_match, observed = await validate_semantic_match(
            None, "Search products", page=page, locator="#search-input"
        )
        assert not is_match
        assert "search the site" in observed.lower() or observed == ""

    @pytest.mark.asyncio
    async def test_correct_element_via_evaluate_returns_true(self):
        """
        Same probe 06 page: #product-search has placeholder 'Search products'.
        expected_text='Search products' → should match.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        page = _make_page_with_evaluate({
            "tag": "input",
            "textContent": "",
            "innerText": "",
            "placeholder": "Search products",
            "ariaLabel": "",
            "value": "",
        })

        is_match, _ = await validate_semantic_match(
            None, "Search products", page=page, locator="#product-search"
        )
        assert is_match


# ─────────────────────────────────────────────────────────────────────────────
# Test: Change C fast-path integration — semantic mismatch falls through
# ─────────────────────────────────────────────────────────────────────────────

class TestChangeCFastPath:
    """
    Verify that the actions.py count==1 path now calls validate_semantic_match
    before returning, and falls through to the smart-locator path on mismatch.
    """

    @pytest.mark.asyncio
    async def test_semantic_mismatch_falls_through_to_smart_locator(self):
        """
        Probe 06: candidate '#search-input' is unique (count==1) but resolves to the
        wrong element.  With Change C, the fast-path must NOT return it; instead it
        falls through to find_unique_locator_at_coordinates.
        After fallthrough, smart_locator returns found=False (no real browser),
        but the candidate locator must NOT appear as best_locator.
        """
        from browser_service.agent.actions import find_unique_locator_action

        # Mock page: count=1 (unique), evaluate returns the WRONG element's text
        locator_mock = AsyncMock()
        locator_mock.count = AsyncMock(return_value=1)
        locator_mock.evaluate = AsyncMock(return_value={
            "tag": "input",
            "textContent": "",
            "innerText": "",
            "placeholder": "Search the site",  # wrong element
            "ariaLabel": "",
            "value": "",
        })

        page = MagicMock()
        page.url = "https://example.com"
        page.locator = MagicMock(return_value=locator_mock)

        # Patch find_unique_locator_at_coordinates where actions.py imports it from.
        # actions.py does: from browser_service.locators import find_unique_locator_at_coordinates
        with patch(
            "browser_service.locators.find_unique_locator_at_coordinates",
            new=AsyncMock(return_value={"found": False, "element_id": "elem_1",
                                        "description": "search input",
                                        "error": "Semantic mismatch in test"})
        ):
            result = await find_unique_locator_action(
                x=100, y=200,
                element_id="elem_1",
                element_description="search products input",
                expected_text="Search products",
                candidate_locator="#search-input",
                page=page,
            )

        # The wrong candidate must NOT be the best_locator
        assert result.get("best_locator") != "#search-input", (
            "The semantically-wrong candidate '#search-input' must not be returned "
            "as best_locator after Change C closes the fast-path hole."
        )

    @pytest.mark.asyncio
    async def test_no_expected_text_accepts_unique_candidate(self):
        """
        When expected_text is absent, the fast-path still accepts a unique candidate
        without calling validate_semantic_match (guard: if expected_text:).
        """
        from browser_service.agent.actions import find_unique_locator_action

        locator_mock = AsyncMock()
        locator_mock.count = AsyncMock(return_value=1)

        page = MagicMock()
        page.url = "https://example.com"
        page.locator = MagicMock(return_value=locator_mock)

        result = await find_unique_locator_action(
            x=100, y=200,
            element_id="elem_1",
            element_description="some button",
            expected_text=None,
            candidate_locator="#submit-btn",
            page=page,
        )

        assert result.get("found") is True
        assert result.get("best_locator") == "#submit-btn"

    @pytest.mark.asyncio
    async def test_semantic_match_returns_candidate_directly(self):
        """
        When the candidate resolves to the correct element (semantic match),
        the fast-path returns it without falling through.
        """
        from browser_service.agent.actions import find_unique_locator_action

        locator_mock = AsyncMock()
        locator_mock.count = AsyncMock(return_value=1)
        locator_mock.evaluate = AsyncMock(return_value={
            "tag": "input",
            "textContent": "",
            "innerText": "",
            "placeholder": "Search products",
            "ariaLabel": "",
            "value": "",
        })

        page = MagicMock()
        page.url = "https://example.com"
        page.locator = MagicMock(return_value=locator_mock)

        result = await find_unique_locator_action(
            x=100, y=200,
            element_id="elem_1",
            element_description="product search input",
            expected_text="Search products",
            candidate_locator="#product-search",
            page=page,
        )

        assert result.get("found") is True
        assert result.get("best_locator") == "#product-search"


# ─────────────────────────────────────────────────────────────────────────────
# Test: word-level soft match — per-field, not combined haystack
# ─────────────────────────────────────────────────────────────────────────────

class TestWordLevelSoftMatch:
    """
    Verifies that word-level matching requires all significant words to appear
    in the same individual field, not scattered across different fields.
    """

    @pytest.mark.asyncio
    async def test_words_in_same_field_match(self):
        """
        "email login" reversed against "Please enter your login email": not a
        substring, but both words are in the meaningful field → True.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        node = _make_node(meaningful_text="Please enter your login email")

        is_match, _ = await validate_semantic_match(node, "email login")
        assert is_match

    @pytest.mark.asyncio
    async def test_words_split_across_fields_rejected(self):
        """
        "delete" in meaningful, "row" in aria-label.
        expected="delete row": no single field contains both words → False.
        Old combined-haystack code would have matched this incorrectly.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        node = _make_node(meaningful_text="delete account", aria_label="row 5")

        is_match, _ = await validate_semantic_match(node, "delete row")
        assert not is_match

    @pytest.mark.asyncio
    async def test_partial_words_rejected(self):
        """
        Only one of the required words appears in any field → False.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        node = _make_node(meaningful_text="Delete this item")

        is_match, _ = await validate_semantic_match(node, "delete row")
        assert not is_match

    @pytest.mark.asyncio
    async def test_two_char_words_included(self):
        """
        "so do" — both words are exactly 2 chars, kept by the >= 2 filter.
        "so do" is not a substring of "I do so agree" (reversed order fails exact check),
        but word-level match finds both "so" and "do" present in the field → True.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        node = _make_node(meaningful_text="I do so agree")

        is_match, _ = await validate_semantic_match(node, "so do")
        assert is_match


# ─────────────────────────────────────────────────────────────────────────────
# Test: container rejection
# ─────────────────────────────────────────────────────────────────────────────

class TestContainerRejection:
    """
    Verifies that elements whose text surface far exceeds the search term are
    rejected as layout containers, even when the expected text is present.
    """

    @pytest.mark.asyncio
    async def test_primary_path_large_meaningful_rejected(self):
        """
        Primary path: meaningful = 607 chars containing "Submit".
        threshold = max(6*40=240, 500) = 500. 607 > 500 → rejected despite match.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        large_text = "Submit " + "a" * 600  # 607 chars
        node = _make_node(meaningful_text=large_text)

        is_match, _ = await validate_semantic_match(node, "Submit")
        assert not is_match

    @pytest.mark.asyncio
    async def test_legacy_path_textcontentlength_triggers_rejection(self):
        """
        Legacy path: textContent is short (contains "Submit") but textContentLength=800
        reveals the full DOM length. Container check fires on 800 > 500 → rejected.
        Without textContentLength the substring match would return True — this test
        confirms it is textContentLength, not the sliced textContent, doing the work.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        page = _make_page_with_evaluate({
            "tag": "div",
            "textContent": "Submit",        # sliced view contains expected text
            "textContentLength": 800,       # full length reveals container
            "innerText": "",
            "placeholder": "",
            "ariaLabel": "",
            "value": "",
        })

        is_match, _ = await validate_semantic_match(
            None, "Submit", page=page, locator="div.content"
        )
        assert not is_match

    @pytest.mark.asyncio
    async def test_legacy_path_without_textcontentlength_not_rejected(self):
        """
        Legacy path: no textContentLength key in evaluate result (e.g. old mock or
        element where JS returns partial data). Fallback uses len(textContent) which
        is ≤ 500. Threshold is ≥ 500, so container check never fires. Substring
        match proceeds normally → True.
        This is the backward-compat guarantee: missing the field is safe, not broken.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        page = _make_page_with_evaluate({
            "tag": "input",
            "textContent": "",
            "innerText": "",
            "placeholder": "Submit",
            "ariaLabel": "",
            "value": "",
            # no textContentLength key
        })

        is_match, _ = await validate_semantic_match(
            None, "Submit", page=page, locator="#submit-btn"
        )
        assert is_match

    @pytest.mark.asyncio
    async def test_primary_path_normal_size_accepted(self):
        """
        Primary path: meaningful = "Submit" (6 chars). 6 > 500 = False.
        Container check does not fire; substring match succeeds → True.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        node = _make_node(meaningful_text="Submit")

        is_match, _ = await validate_semantic_match(node, "Submit")
        assert is_match

    @pytest.mark.asyncio
    async def test_legacy_path_normal_size_accepted(self):
        """
        Legacy path: textContentLength=100. 100 > 500 = False.
        Substring match on textContent succeeds → True.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        page = _make_page_with_evaluate({
            "tag": "button",
            "textContent": "Submit products",
            "textContentLength": 100,
            "innerText": "Submit products",
            "placeholder": "",
            "ariaLabel": "",
            "value": "",
        })

        is_match, _ = await validate_semantic_match(
            None, "Submit", page=page, locator="#btn"
        )
        assert is_match

    @pytest.mark.asyncio
    async def test_boundary_at_threshold_not_rejected(self):
        """
        textContentLength=500, expected="Submit" → threshold=max(240,500)=500.
        500 > 500 is False (strict greater-than), so container check does not fire.
        Substring match succeeds → True.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        page = _make_page_with_evaluate({
            "tag": "div",
            "textContent": "Submit more products",
            "textContentLength": 500,
            "innerText": "Submit more products",
            "placeholder": "",
            "ariaLabel": "",
            "value": "",
        })

        is_match, _ = await validate_semantic_match(
            None, "Submit", page=page, locator="div.btn"
        )
        assert is_match


# ─────────────────────────────────────────────────────────────────────────────
# Test: CDP fallback node fast path gate
# ─────────────────────────────────────────────────────────────────────────────

class TestCDPFallbackFastPathGate:
    """
    get_dom_element_at_coordinates returns two node types:
      - Cache-hit:     children_nodes is a list (possibly empty for leaf elements)
      - CDP fallback:  children_nodes is None, ax_node is None

    The fast path in find_unique_locator_at_coordinates only uses the node-based
    validate_semantic_match when children_nodes is not None.  A CDP fallback node
    has no AX data and no children text — validate_semantic_match would see an
    empty haystack and fire the Probe 18 carve-out, accepting ANY interactive
    element regardless of expected_text.  The gate prevents that.
    """

    @pytest.mark.asyncio
    async def test_cdp_fallback_node_probe18_danger(self):
        """
        Calling validate_semantic_match directly with a CDP fallback node
        (children_nodes=None, ax_node=None, no DOM attributes) returns (True, '')
        via Probe 18 even when expected_text does not match anything.
        This documents WHY the fast path must gate on children_nodes is not None.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        cdp_fallback = MagicMock()
        cdp_fallback.tag_name = "button"
        cdp_fallback.node_name = "button"
        cdp_fallback.node_value = ""
        cdp_fallback.attributes = {}
        cdp_fallback.ax_node = None
        cdp_fallback.children_nodes = None
        cdp_fallback.get_meaningful_text_for_llm = MagicMock(return_value="")

        is_match, observed = await validate_semantic_match(cdp_fallback, "Delete account")

        # Probe 18 fires: empty haystack + interactive tag → accepts any text.
        # This is the false-positive the fast path gate prevents.
        assert is_match is True
        assert observed == ""

    @pytest.mark.asyncio
    async def test_cache_hit_leaf_node_validates_correctly(self):
        """
        A cache-hit leaf node (children_nodes=[], e.g. <input>) has DOM attributes
        populated.  validate_semantic_match can correctly accept or reject based on them.
        """
        from browser_service.locators.smart_locator import validate_semantic_match

        cache_hit = _make_node(tag="input", placeholder="Search products")
        cache_hit.children_nodes = []  # leaf node — no children, but cache-hit

        is_match, _ = await validate_semantic_match(cache_hit, "Search products")
        assert is_match

        is_match, _ = await validate_semantic_match(cache_hit, "Delete account")
        assert not is_match

    def test_fast_path_gate_condition(self):
        """
        The gate condition used in find_unique_locator_at_coordinates Step 5 is
        `resolved_node is not None and resolved_node.children_nodes is not None`.
        Verify it correctly separates CDP fallback nodes from cache-hit nodes.
        """
        def _uses_fast_path(node):
            return node is not None and node.children_nodes is not None

        cdp_fallback = MagicMock()
        cdp_fallback.children_nodes = None

        cache_hit_leaf = MagicMock()
        cache_hit_leaf.children_nodes = []

        cache_hit_parent = MagicMock()
        cache_hit_parent.children_nodes = [MagicMock()]

        assert not _uses_fast_path(None)
        assert not _uses_fast_path(cdp_fallback), "CDP fallback must use slow path"
        assert _uses_fast_path(cache_hit_leaf), "Leaf cache node must use fast path"
        assert _uses_fast_path(cache_hit_parent), "Parent cache node must use fast path"
