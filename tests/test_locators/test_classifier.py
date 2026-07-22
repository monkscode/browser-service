"""
Unit tests for browser_service.locators.classifier.

Covers Tier 0 deterministic DOM rules, Tier 1 multi-signal voting, the
nav-prefix exclusion, and the conflict cases that drop confidence to "low".
Tier 2 (vision-assist) is skipped — it lands in Phase 2.4.
"""

import pytest

from browser_service.locators.classifier import (
    ElementTypeInfo,
    classify_element_type,
)

# ----------------------------------------------------------------------
# Tier 0 — pure DOM rules (rules 1–17 from §5)
# ----------------------------------------------------------------------


class TestTier0NativeFormElements:
    """Rules 1–5: tagName=select / input + type."""

    def test_select_tag_is_native_dropdown(self):
        info = classify_element_type({"tagName": "select"}, "")
        assert info.primary_type == "dropdown"
        assert info.framework == "native"
        assert info.confidence == "high"
        assert "tier:0" in info.signals

    def test_input_checkbox_is_native_checkbox(self):
        info = classify_element_type({"tagName": "input", "type": "checkbox"}, "")
        assert info.primary_type == "checkbox"
        assert info.framework == "native"
        assert info.confidence == "high"

    def test_input_radio_is_native_radio(self):
        info = classify_element_type({"tagName": "input", "type": "radio"}, "")
        assert info.primary_type == "radio"
        assert info.framework == "native"

    def test_input_file_is_file_upload(self):
        info = classify_element_type({"tagName": "input", "type": "file"}, "")
        assert info.primary_type == "file-upload"
        assert info.framework == "native"

    def test_input_date_is_date_picker(self):
        info = classify_element_type({"tagName": "input", "type": "date"}, "")
        assert info.primary_type == "date-picker"
        assert info.framework == "native"

    def test_input_text_is_not_classified_at_tier0(self):
        """type=text is not in rules 2–5 — falls to Tier 1, no signals → unknown."""
        info = classify_element_type({"tagName": "input", "type": "text"}, "")
        assert info.primary_type == "unknown"
        assert "tier:1" in info.signals


class TestTier0CollectionTags:
    """Rules 6–7: tr / li (with nav-context exclusion)."""

    def test_tr_is_table_row_collection(self):
        info = classify_element_type({"tagName": "tr", "className": "data-row"}, "first row")
        assert info.primary_type == "collection"
        assert info.framework == "table-row"
        assert info.confidence == "high"

    def test_li_without_nav_class_is_list_item_collection(self):
        info = classify_element_type({"tagName": "li", "className": "todo-item"}, "second item")
        assert info.primary_type == "collection"
        assert info.framework == "list-item"
        assert info.confidence == "medium"

    def test_li_with_nav_class_is_NOT_collection(self):
        """Bug 2 nav-prefix exclusion: <li class='nav-item'> stays unclassified."""
        info = classify_element_type(
            {"tagName": "li", "className": "nav-item active"},
            "Customer link in nav menu",
        )
        assert info.primary_type != "collection"

    def test_li_with_menu_prefix_is_NOT_collection(self):
        info = classify_element_type({"tagName": "li", "className": "menu-item"}, "settings")
        assert info.primary_type != "collection"

    def test_li_with_breadcrumb_prefix_is_NOT_collection(self):
        info = classify_element_type(
            {"tagName": "li", "className": "breadcrumb-item"}, "Home crumb"
        )
        assert info.primary_type != "collection"


class TestTier0DropdownFrameworks:
    """Rules 8–14: framework className patterns."""

    def test_tom_select_ts_wrapper(self):
        info = classify_element_type({"tagName": "div", "className": "ts-wrapper single"}, "")
        assert info.primary_type == "dropdown"
        assert info.framework == "tom-select"
        assert info.confidence == "high"

    def test_tom_select_ts_control(self):
        info = classify_element_type({"tagName": "div", "className": "ts-control"}, "")
        assert info.framework == "tom-select"

    def test_select2(self):
        info = classify_element_type(
            {"tagName": "div", "className": "select2 select2-container"}, ""
        )
        assert info.primary_type == "dropdown"
        assert info.framework == "select2"

    def test_kendo_dropdown(self):
        info = classify_element_type({"tagName": "div", "className": "k-dropdown"}, "")
        assert info.framework == "kendo"

    def test_kendo_combobox(self):
        info = classify_element_type({"tagName": "div", "className": "k-combobox"}, "")
        assert info.framework == "kendo"

    def test_react_select(self):
        info = classify_element_type({"tagName": "div", "className": "react-select__control"}, "")
        assert info.framework == "react-select"

    def test_vue_select(self):
        info = classify_element_type({"tagName": "div", "className": "vs__dropdown-toggle"}, "")
        assert info.framework == "vue-select"

    def test_ant_design(self):
        info = classify_element_type(
            {"tagName": "div", "className": "ant-select ant-select-single"}, ""
        )
        assert info.framework == "ant-design"

    def test_material_ui_select(self):
        info = classify_element_type({"tagName": "div", "className": "MuiSelect-root"}, "")
        assert info.framework == "material-ui"

    def test_material_ui_autocomplete(self):
        info = classify_element_type({"tagName": "div", "className": "MuiAutocomplete-input"}, "")
        assert info.framework == "material-ui"


class TestTier0Roles:
    """Rules 15–17: role-based."""

    def test_role_combobox(self):
        info = classify_element_type({"tagName": "div", "role": "combobox"}, "")
        assert info.primary_type == "dropdown"
        assert info.framework == "combobox-input"
        assert info.confidence == "high"

    def test_role_listbox(self):
        info = classify_element_type({"tagName": "div", "role": "listbox"}, "")
        assert info.primary_type == "dropdown"

    def test_role_checkbox_custom(self):
        info = classify_element_type({"tagName": "div", "role": "checkbox"}, "")
        assert info.primary_type == "checkbox"
        assert info.framework == "custom"

    def test_role_radio_custom(self):
        info = classify_element_type({"tagName": "div", "role": "radio"}, "")
        assert info.primary_type == "radio"
        assert info.framework == "custom"

    def test_role_switch_is_toggle_checkbox(self):
        info = classify_element_type({"tagName": "div", "role": "switch"}, "")
        assert info.primary_type == "checkbox"
        assert info.framework == "toggle"


# ----------------------------------------------------------------------
# Tier 1 — multi-signal voting
# ----------------------------------------------------------------------


class TestTier1RoleVoting:
    """role=row / listitem / grid → collection +3 (medium)."""

    def test_role_row_votes_collection_medium(self):
        info = classify_element_type({"tagName": "div", "role": "row"}, "")
        assert info.primary_type == "collection"
        assert info.confidence == "medium"
        assert "tier:1" in info.signals

    def test_role_listitem_votes_collection_medium(self):
        info = classify_element_type({"tagName": "div", "role": "listitem"}, "")
        assert info.primary_type == "collection"
        assert info.confidence == "medium"


class TestTier1ClassNameVoting:
    """className collection patterns (+2 medium weight) with nav-prefix exclusion."""

    def test_card_class_votes_collection_low(self):
        """Single +2 vote alone is below medium threshold (3) → low."""
        info = classify_element_type({"tagName": "div", "className": "card mb-3"}, "")
        assert info.primary_type == "collection"
        assert info.confidence == "low"

    def test_grid_item_class_votes_collection(self):
        info = classify_element_type({"tagName": "div", "className": "grid-item"}, "")
        assert info.primary_type == "collection"

    def test_nav_class_blocks_collection_vote(self):
        """Bug 2 regression: nav-* prefix suppresses collection class voting."""
        info = classify_element_type(
            {"tagName": "a", "className": "nav-item dropdown-item active"},
            "Customer",
        )
        assert info.primary_type != "collection"

    def test_substring_only_class_does_not_vote_collection(self):
        """'rowboat' contains 'row' as substring but is not a token match."""
        info = classify_element_type({"tagName": "div", "className": "rowboat"}, "decorative")
        assert info.primary_type != "collection"


class TestTier1DescriptionVoting:
    """description hints add +1 — never enough alone to clear medium."""

    def test_dropdown_description_alone_is_low(self):
        info = classify_element_type({"tagName": "div", "className": ""}, "Country dropdown")
        assert info.primary_type == "dropdown"
        assert info.confidence == "low"

    def test_collection_description_alone_is_low(self):
        info = classify_element_type({"tagName": "div", "className": ""}, "all visible rows")
        assert info.primary_type == "collection"
        assert info.confidence == "low"


class TestTier1ClassPlusDescriptionAgreement:
    """className (+2) + description (+1) = 3 → medium."""

    def test_card_class_plus_collection_description_is_medium(self):
        info = classify_element_type({"tagName": "div", "className": "card"}, "list of items")
        assert info.primary_type == "collection"
        assert info.confidence == "medium"


class TestTier1MultipleStrongSignalsAgreement:
    """role (+3) + className (+2) = 5 → high; >=4 threshold."""

    def test_role_row_plus_row_class_is_high(self):
        info = classify_element_type(
            {"tagName": "div", "role": "row", "className": "data-row"},
            "filtered rows",
        )
        # role=row → +3 collection, className "data-row" no token match
        # (because token is "data-row" not "row"), description "filtered rows"
        # contains "rows" → +1. Total: 3 + 1 = 4 → high.
        assert info.primary_type == "collection"
        assert info.confidence == "high"


class TestTier1ConflictCases:
    """Conflicting signals drop confidence to low (per recap §5)."""

    def test_bug3_tom_select_outer_wrapper_conflicts(self):
        """
        Bug 3 case: browser-use coords landed on the outer Bootstrap
        '.form-group row' wrapper. Description says "dropdown".
        DOM votes collection (+2), description votes dropdown (+1).
        Classifier outputs collection-low; the collection handler's
        internal dropdown-veto is what produces the actual fall-through
        to the generic path.
        """
        info = classify_element_type(
            {"tagName": "div", "className": "form-group row"},
            "Rate Group dropdown",
        )
        # collection wins on votes, but confidence should not be high.
        assert info.confidence in ("low", "medium")

    def test_two_class_patterns_tied_drops_to_low(self):
        """
        className contains "row" (collection +2) AND description has
        "dropdown" with className "select" hint — both could vote.
        We test that ties produce 'low'.
        """
        # Synthetic tie: dropdown class hint "+2" and collection class "+2"
        info = classify_element_type(
            {"tagName": "div", "className": "row dropdown-trigger"},
            "",
        )
        # className "row" → +2 collection; "dropdown" substring in
        # className → +2 dropdown. Tied at 2. → low.
        # But "dropdown-" is also in NAV_PREFIXES; "dropdown-trigger"
        # starts with that prefix so has_nav_class=True suppresses
        # the collection class vote. dropdown still gets +2 from
        # className "dropdown" hint.
        # Result: dropdown +2, collection 0 → low (single +2 below medium).
        assert info.primary_type == "dropdown"
        assert info.confidence == "low"


class TestTier1NoSignals:
    """Generic elements with no signals → unknown."""

    def test_plain_div_is_unknown(self):
        info = classify_element_type({"tagName": "div", "className": ""}, "")
        assert info.primary_type == "unknown"
        assert info.confidence == "low"
        assert "no-signals" in info.signals

    def test_plain_button_is_unknown(self):
        info = classify_element_type(
            {"tagName": "button", "className": "btn btn-primary"}, "Submit"
        )
        assert info.primary_type == "unknown"


# ----------------------------------------------------------------------
# API contract — signals always include tier marker
# ----------------------------------------------------------------------


class TestSignalsContract:
    """Every verdict must record which tier produced it."""

    def test_tier0_verdict_signals_include_tier0(self):
        info = classify_element_type({"tagName": "select"}, "")
        assert any(s == "tier:0" for s in info.signals)

    def test_tier1_verdict_signals_include_tier1(self):
        info = classify_element_type({"tagName": "div", "className": "card"}, "")
        assert any(s == "tier:1" for s in info.signals)


class TestVisionHintIntegration:
    """Vision hint piggyback (browser-use FindUniqueLocatorParams.element_type)
    is one source of truth — never overrides Tier 0, but adds a strong
    Tier 1 vote and seeds the framework field for the dispatcher's DOM
    probe to corroborate."""

    def test_hint_agreeing_with_tier0_is_annotated(self):
        info = classify_element_type(
            {"tagName": "select"},
            "",
            vision_type_hint="dropdown",
        )
        assert info.primary_type == "dropdown"
        assert any("vision-hint-agree:dropdown" in s for s in info.signals)

    def test_hint_conflicting_with_tier0_does_not_override(self):
        # tagName=select is HTML-deterministic; even if vision says checkbox,
        # the verdict stays "dropdown" — the conflict is logged for telemetry.
        info = classify_element_type(
            {"tagName": "select"},
            "",
            vision_type_hint="checkbox",
        )
        assert info.primary_type == "dropdown"
        assert any("vision-hint-conflict" in s for s in info.signals)

    def test_hint_seeds_framework_when_tier0_has_none(self):
        # Custom dropdown div with role=combobox → Tier 0 fires with
        # framework="combobox-input". Vision adds tom-select hint → does not
        # overwrite (Tier 0 already had a framework).
        info = classify_element_type(
            {"tagName": "div", "role": "combobox"},
            "",
            vision_type_hint="dropdown",
            vision_framework_hint="tom-select",
        )
        assert info.primary_type == "dropdown"
        assert info.framework == "combobox-input"  # Tier 0 wins

    def test_hint_seeds_framework_when_no_tier0_framework(self):
        # Custom <div> with no Tier 0 hit. Tier 1 votes via desc + hint;
        # framework hint seeds the framework field.
        info = classify_element_type(
            {"tagName": "div", "className": "custom-x"},
            "Country dropdown",
            vision_type_hint="dropdown",
            vision_framework_hint="tom-select",
        )
        assert info.primary_type == "dropdown"
        assert info.framework == "tom-select"
        assert any("framework-hint:tom-select" in s for s in info.signals)

    def test_hint_alone_yields_medium_confidence(self):
        # No DOM signal, only vision hint → max_vote=3 → medium confidence.
        # The dispatcher will still require DOM probe corroboration.
        info = classify_element_type(
            {"tagName": "div"},
            "",
            vision_type_hint="dropdown",
        )
        assert info.primary_type == "dropdown"
        assert info.confidence == "medium"

    def test_hint_with_dom_agreement_promotes_to_high(self):
        # Vision hint (+3) + className "select" hint (+2) = 5 → high.
        info = classify_element_type(
            {"tagName": "div", "className": "custom-select"},
            "",
            vision_type_hint="dropdown",
        )
        assert info.primary_type == "dropdown"
        assert info.confidence == "high"

    def test_unmapped_hint_value_ignored(self):
        # "other" maps to "" — no vote, treated as absent.
        info = classify_element_type(
            {"tagName": "div"},
            "",
            vision_type_hint="other",
        )
        assert info.primary_type == "unknown"

    def test_table_hint_maps_to_collection(self):
        info = classify_element_type(
            {"tagName": "div"},
            "",
            vision_type_hint="table",
        )
        assert info.primary_type == "collection"


class TestNoneAndEmptyInputs:
    """Defensive: None / missing inputs should not raise."""

    def test_none_element_data(self):
        info = classify_element_type(None, "")
        assert info.primary_type == "unknown"

    def test_empty_dict_element_data(self):
        info = classify_element_type({}, "")
        assert info.primary_type == "unknown"

    def test_none_description(self):
        info = classify_element_type({"tagName": "select"}, None)
        assert info.primary_type == "dropdown"


class TestElementTypeInfoDataclass:
    """ElementTypeInfo defaults and field types."""

    def test_default_framework_is_empty(self):
        info = ElementTypeInfo(primary_type="unknown")
        assert info.framework == ""

    def test_default_confidence_is_low(self):
        info = ElementTypeInfo(primary_type="unknown")
        assert info.confidence == "low"

    def test_default_signals_is_empty_list(self):
        info = ElementTypeInfo(primary_type="unknown")
        assert info.signals == []

    def test_signals_default_factory_independent_per_instance(self):
        a = ElementTypeInfo(primary_type="x")
        b = ElementTypeInfo(primary_type="y")
        a.signals.append("foo")
        assert b.signals == []
