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

import time
from unittest.mock import patch

import pytest

from browser_service.config import config
from browser_service.locators.stability import (
    LONG_TEXT_CHARS,
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

    @pytest.mark.parametrize(
        ("locator", "expected"),
        [
            ("xpath=//a[@title='Item [3]']", False),
            ('button[class~="[&:nth-child(2)]:block"]', False),
            ("button[class~='[&:nth-child(3)]:p-2']", False),
            ('text="Step >> nth=0 of 3"', False),
            ("text=it's >> nth=0", True),  # unbalanced quote: the nth stays visible
            ('text="a' + chr(92) + '"b" >> nth=0', True),  # an agent \" form
            ("xpath=//a[contains(text(), concat('it', \"'\", 's'))][2]", True),
            ('iframe >> nth=1 >>> text="Pay"', True),  # an ordinal frame stays positional
            ("xpath=(//li)[3]", True),
            ('li:nth-child(2) >> text="x"', True),
            ('[aria-label="Pay now"]', False),
            ("text=a >> nth=0", True),
        ],
    )
    def test_quoted_text_is_not_a_position(self, locator, expected):
        assert is_positional_locator(locator) is expected


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


@pytest.fixture
def action_fit_on():
    with patch.object(config.locator, "enable_action_fit", True):
        yield


@pytest.fixture
def action_fit_off():
    with patch.object(config.locator, "enable_action_fit", False):
        yield


E2C_COUNTER_IDS = [
    "react-select-3-option-1",  # u11: resolved to an aria-disabled option at runtime
    "react-select-3-input",
    "mui-42",
    "headlessui-menu-button-1",
    "downshift-0-item-3",
]


class TestExtendedFrameworkIds:
    @pytest.mark.parametrize("value", E2C_COUNTER_IDS)
    def test_stable_with_flag_off(self, value, action_fit_off):
        assert score_stability("id", value) == STABLE

    @pytest.mark.parametrize("value", E2C_COUNTER_IDS)
    def test_volatile_with_flag_on(self, value, action_fit_on):
        assert score_stability("id", value) == VOLATILE

    @pytest.mark.parametrize(
        "value", ["react-select-color-option-1", "mui-component-select-age", "downshift-menu"]
    )
    def test_hand_authored_neighbours_stay_stable(self, value, action_fit_on):
        assert score_stability("id", value) == STABLE

    def test_probe_locator_is_volatile(self, action_fit_on):
        assert classify_locator("id=react-select-3-option-1") == VOLATILE


class TestLongText:
    LONG = "Portronics Toad 23 Wireless Optical Mouse with 2.4GHz"  # 53 chars, u01

    def test_static_with_flag_off(self, action_fit_off):
        assert is_dynamic_text(self.LONG) is False

    def test_dynamic_with_flag_on(self, action_fit_on):
        assert is_dynamic_text(self.LONG) is True

    def test_boundary_is_strictly_longer_than_40(self, action_fit_on):
        assert is_dynamic_text("x" * LONG_TEXT_CHARS) is False
        assert is_dynamic_text("x" * (LONG_TEXT_CHARS + 1)) is True

    def test_aria_label_locator(self, action_fit_on):
        assert classify_locator(f'[aria-label="{self.LONG}"]') == VOLATILE

    def test_title_contains_locator(self, action_fit_on):
        # u02 r2 shipped this locator for a read step
        loc = "a[title*='Sports Sneaker Running And Outdoor Walking Shoes']"
        assert classify_locator(loc) == VOLATILE

    def test_aria_label_not_scanned_with_flag_off(self, action_fit_off):
        assert classify_locator('[aria-label="Cart (3 items)"]') == STABLE

    def test_short_label_stays_stable(self, action_fit_on):
        assert classify_locator('[aria-label="Search"]') == STABLE


@pytest.mark.parametrize(
    "name, want",
    [
        ("data-v-7ba5bd90", True),  # Vue scoped-style hash
        ("data-v-03eb1f1f", True),
        ("data-v-app", False),  # Vue's mount marker, not a hash
        ("data-v-7ba5bd9", False),  # 7 hex
        ("data-v-7BA5BD90", False),  # Vue emits lowercase
        ("data-id", False),
        ("data-testid", False),
        ("", False),
    ],
)
def test_is_volatile_attr_name(name, want):
    from browser_service.locators.stability import is_volatile_attr_name

    assert is_volatile_attr_name(name) is want


# bs #58: framework counters in ANY segment of a value (JSF naming containers, UI5 view prefixes, ...).
SEGMENT_VOLATILE = [
    "j_idt45",
    "mainForm:j_idt45",
    "form:tbl:0:j_idt12",
    "j_id12",
    "j_id_7z",
    "j_id__v_0",
    "__button0",
    "__xmlview0--saveBtn",
    "container-app---__component0---main--__button3",
    "myApp---main--item-__clone3",
    "__BVID__23",
    "rc_select_0",
    "rc-tabs-1-tab-2",
    "pr_id_3",
    "pn_id_5",
    "pv_id_7",
    "el-id-1024-12",
    "cdk-overlay-0",
    "cdk-dialog-2",
    "ui-id-5",
    ":R1:",
    ":R1:-label",
    "«r1»",
    "_R_17d_",
    "_r_7d_",
    "row-8f3a9c1d2e4f5a6b7c",
    "item_0123456789ab",
]

SEGMENT_STABLE = [
    "880667900",
    "dt-search-0",
    "mat-icon-button",
    "embermail",
    "myApp---main--saveBtn",
    "saveBtn",
    "j_username",
    "j_password",
    "j_idt",
    "rc-select",
    "ui-id",
    "el-id-foo",
    "cdk-column-col1",
    "card__title2",
    "table1",
    "item-1",
    "step-2",
    "q1",
    "__next",
    "__section",
    "abc-123def",
    "color-ffffff",
    "deadbeefcafe",
    "mainForm:results:save",  # hand-written JSF segments: why the React colon form is start-only
    "form:row1:name",
    "app:region:panel",
    "cdk-overlay-container",
]


class TestSegmentFrameworkIds:
    @pytest.mark.parametrize("value", SEGMENT_VOLATILE)
    def test_volatile_with_flag_on(self, value, action_fit_on):
        assert score_stability("id", value) == VOLATILE

    @pytest.mark.parametrize("value", SEGMENT_STABLE)
    def test_stable_with_flag_on(self, value, action_fit_on):
        assert score_stability("id", value) == STABLE

    @pytest.mark.parametrize("value", ["j_idt45", "__BVID__23"])
    def test_counter_in_name_is_volatile_too(self, value, action_fit_on):
        assert score_stability("name", value) == VOLATILE

    @pytest.mark.parametrize("value", ["j_idt45", "mainForm:j_idt45"])
    def test_flag_off_keeps_todays_verdict(self, value, action_fit_off):
        assert score_stability("id", value) == STABLE

    def test_segments_false_keeps_todays_verdict(self, action_fit_on):
        # the read gate (read_target._is_volatile_token) passes segments=False so its tokens do not change
        assert score_stability("id", "j_idt45", segments=False) == STABLE
        assert score_stability("id", "react-select-3-input", segments=False) == VOLATILE

    @pytest.mark.parametrize(
        "value",
        ["__a" + "1" * 30000 + "a", ":__a" * 7500, "-cdk-overlay" * 2500, "-cdk-menu" * 3333],
    )
    def test_hostile_input_is_linear(self, value, action_fit_on):
        start = time.perf_counter()
        score_stability("id", value)
        assert time.perf_counter() - start < 0.5

    @pytest.mark.parametrize(
        ("value", "want"),
        [
            ("__section2", VOLATILE),  # look-alike of UI5: accepted over-flag
            ("user_pr_id_3", VOLATILE),  # "_" counts as a segment start
            ("j_id-123", STABLE),  # Mojarra useFaceletsID with a negative hash is not matched
            ("field-:r2:", STABLE),  # React id after a prefix: the colon form is start-only
        ],
    )
    def test_stated_limits(self, value, want, action_fit_on):
        assert score_stability("id", value) == want

    def test_escaped_hash_locator_is_a_stated_limit(self, action_fit_on):
        # item 2 of the evaluation: the escaped "#mainForm\:j_idt45" form is not unescaped before scoring
        assert classify_locator(r"#mainForm\:j_idt45") == STABLE
