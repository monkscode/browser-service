"""
Unit tests for browser_service/locators/stability.py (Task 10, E1).

The scorer classifies locator raw material by how likely it is to survive
a fresh browser session:

- "stable"     — hand-authored ids/names/classes, test attributes.
- "volatile"   — session-generated values (framework id counters, UUIDs,
                 hashes, timestamp suffixes) that change on the next load.
- "positional" — locators that encode today's DOM order (nth=, nth-child,
                 numeric XPath predicates, ordinal group indexes).

Evidence anchors (2026-07-06 log/bench audit):
- id=880667900 (GitHub repo database id) passed bench q01 3/3 in fresh RF
  sessions -> bare all-digit values MUST stay "stable" (narrowed digit rule).
- #dt-search-0 (DataTables, 24+ emissions, never failed) -> short ordinal
  suffixes on word ids MUST stay "stable".
- tomselect-N (35 log hits, the original production incident) -> volatile.
"""

import pytest

from browser_service.locators.stability import (
    POSITIONAL,
    STABLE,
    VOLATILE,
    classify_locator,
    is_dynamic_text,
    is_positional_locator,
    score_stability,
    stability_rank,
)


class TestScoreStabilityVolatile:
    """Framework-generated and machine-shaped values are volatile."""

    @pytest.mark.parametrize(
        "value",
        [
            # Framework id counters (analysis-doc table + the Tom Select incident)
            "ext-gen1042",  # ExtJS
            "ext-comp-1009",  # ExtJS component
            "ember472",  # Ember
            "mat-input-5",  # Angular Material
            "mat-select-12",
            "mat-checkbox-3",
            "select2-country-x7-container",  # Select2 (random middle segment)
            "gwt-uid-23",  # GWT
            "tomselect-2",  # Tom Select (the original incident)
            "tomselect-1-ts-control",
            "cke_10",  # CKEditor 4 toolbar init-order counter (G6)
            "cke_75",
            "radix-:r1:",  # Radix UI
            ":r0:",  # React useId
            ":r2a:",
            # Timestamp / counter suffixes glued to a word prefix
            "field-1749283746",
            "session_1699999999",
            "input-00012345",
            # UUID and hash shapes
            "a3f8b2c1-9d4e-4f6a-8b2c-1d9e4f6a8b2c",
            "5f4dcc3b5aa765d61d8327deb882cf99",
        ],
    )
    def test_volatile_id(self, value):
        assert score_stability("id", value) == VOLATILE

    def test_applies_to_name_attribute(self):
        assert score_stability("name", "field_1749283746") == VOLATILE

    def test_applies_to_class_attribute(self):
        assert score_stability("class", "ember472") == VOLATILE

    def test_applies_to_data_testid(self):
        assert (
            score_stability(
                "data-testid",
                "a3f8b2c1-9d4e-4f6a-8b2c-1d9e4f6a8b2c",
            )
            == VOLATILE
        )


class TestScoreStabilityStable:
    """Hand-authored values stay stable — including the narrowed digit rule."""

    @pytest.mark.parametrize(
        "value",
        [
            # Top real-traffic ids from the 2026-07-06 log audit
            "username",
            "save_button",
            "password",
            "first_name",
            "searchBox",
            "login-button",
            "user-name",
            "table1",
            "non_cli_pricelist_id",
            "APjFqb",  # Google search box — short opaque, no digits
            # Narrowed digit rule: bare all-digit ids are database ids, not
            # session counters. q01 (id=880667900) passes 3/3 in fresh sessions.
            "880667900",
            "42",
            # Short ordinal suffixes are deterministic per page (DataTables etc.)
            "dt-search-0",
            "tab-1",
            "step-2",
            "item3",
            # CKEditor name-derived values are deterministic, only the bare
            # counter shape (cke_10) is session-volatile (G6)
            "cke_wysiwyg_frame",
            "cke_editor1",
            # Digits, but not a suffix-counter shape
            "q4-2025-report",
        ],
    )
    def test_stable_id(self, value):
        assert score_stability("id", value) == STABLE

    def test_empty_value_is_stable(self):
        # Nothing to judge; callers never build candidates from empty values.
        assert score_stability("id", "") == STABLE

    def test_hand_authored_name(self):
        assert score_stability("name", "username") == STABLE


class TestIsPositionalLocator:
    @pytest.mark.parametrize(
        "locator",
        [
            'text="Home" >> nth=0',
            'text="Solutions" >> visible=true >> nth=0',
            "css=div.card >> nth=3",
            "li:nth-child(2)",
            "div:nth-of-type(4) > span",
            "xpath=(//div[contains(@class,'ts-wrapper')])[2]",  # Tom Select ordinal
            "xpath=//div[1]/i",  # structural xpath
            "xpath=//div[4]/div/div/button",
            "//div[2]/button[2]",
            "iframe >> nth=0 >>> #username",  # iframe ordinal hop
        ],
    )
    def test_positional(self, locator):
        assert is_positional_locator(locator) is True

    @pytest.mark.parametrize(
        "locator",
        [
            "#username",
            "id=save_button",
            '[name="q"]',
            'text="OEM Solution"',
            '[data-testid="submit"]',
            'xpath=//div[@id="main"]//button',  # attribute predicates only
            "xpath=//input[@name='user']",
            ".products > div.product-line",  # scoped collection, no ordinals
        ],
    )
    def test_not_positional(self, locator):
        assert is_positional_locator(locator) is False


class TestIsDynamicText:
    @pytest.mark.parametrize(
        "text",
        [
            "Cart (3 items)",
            "Inbox (12)",
            "Notifications (99+)",
            "Last updated 2026-07-06",
            "Updated 07/06/2026",
            "12:45",
            "3",
            "1,234.56",
            "$4.99",
        ],
    )
    def test_dynamic(self, text):
        assert is_dynamic_text(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "OEM Solution",
            "Add to cart",
            "Solutions",
            "Save",
            "Q1 2025 report",  # digits, but not a counter/date/price shape
            "Page 2 heading",  # embedded digit without dynamic shape
            "",
        ],
    )
    def test_static(self, text):
        assert is_dynamic_text(text) is False


class TestClassifyLocator:
    """Whole-locator classification for payload assembly sites."""

    def test_positional_wins_over_everything(self):
        # nth= on a text locator: position is the dominant fragility.
        assert classify_locator('text="Home" >> nth=0') == POSITIONAL

    @pytest.mark.parametrize(
        "locator",
        [
            "#ext-gen1042",
            "id=ext-gen1042",
            '[id="tomselect-3"]',
            "css=#ember472 .item",
        ],
    )
    def test_volatile_embedded_id(self, locator):
        assert classify_locator(locator) == VOLATILE

    def test_dynamic_text_locator_is_volatile(self):
        assert classify_locator('text="Cart (3 items)"') == VOLATILE

    @pytest.mark.parametrize(
        "locator",
        [
            "#username",
            "id=save_button",
            '[name="q"]',
            'text="OEM Solution"',
            "#880667900",  # narrowed digit rule flows through
            "id=880667900",
        ],
    )
    def test_stable(self, locator):
        assert classify_locator(locator) == STABLE


class TestStabilityRank:
    def test_ordering_for_sort_keys(self):
        assert stability_rank(STABLE) < stability_rank(VOLATILE)
        assert stability_rank(VOLATILE) < stability_rank(POSITIONAL)

    def test_unknown_tier_sorts_last(self):
        assert stability_rank("garbage") > stability_rank(POSITIONAL)
