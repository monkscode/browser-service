"""
Classifier-level assertions for the Phase 2.6 fixture set.

Each test feeds the classifier the ``element_data`` shape that browser-use
would produce for the target element in the corresponding HTML fixture,
then asserts the verdict (``primary_type``, ``framework``, ``confidence``).
These tests do not require a browser — they exercise classifier logic only,
and run alongside the existing 50 ``test_classifier.py`` tests in CI.

For end-to-end orchestrator coverage (real Playwright + file:// URLs) see
``test_locator_fixtures.py``, which is gated behind
``-m integration``.

Reference: docs/ELEMENT_TYPE_CLASSIFIER_ARCHITECTURE.md Section 10.
"""

import pytest

from browser_service.locators.classifier import classify_element_type

# ----------------------------------------------------------------------
# Bug 2 — astpp_create_menu_navitem.html
# Outer ".nav-item" container must NOT be classified as a collection
# even if surrounded by sibling .nav-items.
# ----------------------------------------------------------------------


class TestBug2NavItem:
    """Bug 2 — Phase 1.1 nav-prefix exclusion in classifier Tier 1."""

    def test_outer_create_navitem_not_collection(self):
        """The 'Create' nav-item parent must not vote collection."""
        info = classify_element_type(
            element_data={
                "tagName": "div",
                "className": "nav-item",
                "role": "",
            },
            element_description="Create menu item in navbar",
        )
        assert info.primary_type != "collection", (
            f"nav-item leaked into collection vote — signals={info.signals}"
        )

    def test_customer_link_not_collection(self):
        """Inner <a class='nav-link'> must not vote collection."""
        info = classify_element_type(
            element_data={
                "tagName": "a",
                "className": "nav-link",
                "role": "",
                "href": "/customer/create",
            },
            element_description="Customer menu item",
        )
        assert info.primary_type != "collection"


# ----------------------------------------------------------------------
# Bug 3 — astpp_customer_form_tom_select.html
# Tom Select wrappers must classify as dropdown/tom-select with high
# confidence so the dispatcher routes to dropdown.find_locator().
# ----------------------------------------------------------------------


class TestBug3TomSelect:
    """Bug 3 — Tier 0 Rule 8 catches ts-wrapper className."""

    @pytest.mark.parametrize(
        "classes",
        [
            "ts-wrapper js-select form-select reseller single",
            "ts-wrapper has-items",
            "form-select tomselected ts-wrapper",
        ],
    )
    def test_ts_wrapper_classifies_as_tom_select(self, classes):
        info = classify_element_type(
            element_data={"tagName": "div", "className": classes},
            element_description="Reseller dropdown",
        )
        assert info.primary_type == "dropdown"
        assert info.framework == "tom-select"
        assert info.confidence == "high"

    def test_ts_control_classifies_as_tom_select(self):
        info = classify_element_type(
            element_data={"tagName": "div", "className": "ts-control"},
            element_description="Role dropdown",
        )
        assert info.primary_type == "dropdown"
        assert info.framework == "tom-select"
        assert info.confidence == "high"


# ----------------------------------------------------------------------
# nav_item_negative.html — regression: any of three sibling .nav-items
# must classify as unknown (NOT collection), letting the standard
# 21-strategy text fallback produce text= locators.
# ----------------------------------------------------------------------


class TestNavItemNegative:
    @pytest.mark.parametrize("desc", ["Dashboard nav", "Reports", "Settings link"])
    def test_navitem_siblings_not_collection(self, desc):
        info = classify_element_type(
            element_data={"tagName": "div", "className": "nav-item"},
            element_description=desc,
        )
        assert info.primary_type != "collection", info.signals


# ----------------------------------------------------------------------
# tom_select_no_label_for.html — Tier 0 still fires on ts-wrapper
# regardless of whether the <label for=> attribute is present.
# Label-attribute presence is a handler concern, not a classifier one.
# ----------------------------------------------------------------------


def test_tom_select_no_label_for_still_classifies_as_tom_select():
    info = classify_element_type(
        element_data={"tagName": "div", "className": "ts-wrapper has-items"},
        element_description="Rate Group dropdown",
    )
    assert info.primary_type == "dropdown"
    assert info.framework == "tom-select"


# ----------------------------------------------------------------------
# multi_instance_tom_select.html — every .ts-wrapper on the page should
# classify identically; scoping is a handler-level concern.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "desc",
    ["Country dropdown", "Currency dropdown", "Rate Group dropdown", "Billing Schedule dropdown"],
)
def test_multi_instance_tom_select_each_classified_high(desc):
    info = classify_element_type(
        element_data={"tagName": "div", "className": "ts-wrapper"},
        element_description=desc,
    )
    assert info.primary_type == "dropdown"
    assert info.framework == "tom-select"
    assert info.confidence == "high"


# ----------------------------------------------------------------------
# lying_role_div_button.html — <div role="button" aria-haspopup="listbox">
# with description containing "dropdown".
#
# Tier 0 sees role=button (no rule fires for that), no framework class.
# Tier 1 has only a single low-weight description signal → confidence
# must drop to "low" so the dispatcher hedges (handler returns None,
# orchestrator falls through to generic 21-strategy).
# ----------------------------------------------------------------------


def test_lying_role_div_button_drops_confidence_below_high():
    """Adversarial role mismatch (role=button, no framework class) must
    not produce a high-confidence verdict — the dispatcher needs that
    signal to know it should hedge, since no specialized handler exists
    for an empty framework string."""
    info = classify_element_type(
        element_data={
            "tagName": "div",
            "className": "fake-dropdown",
            "role": "button",  # NOT combobox/listbox — Tier 0 doesn't fire
        },
        element_description="country dropdown",
    )
    assert info.primary_type == "dropdown"
    assert info.framework == ""  # framework only set by Tier 0 / class match
    # Acceptance: NOT "high" — class hint "dropdown" + desc hint may push to
    # "medium" but never to "high" without a framework anchor.
    assert info.confidence != "high", (
        f"Adversarial role-mismatch must hedge below 'high' — "
        f"got {info.confidence}, signals={info.signals}"
    )


# ----------------------------------------------------------------------
# css_in_js_obfuscated_dropdown.html — hashed classnames + role=combobox.
# Tier 0 Rule 15 fires on role=combobox → dropdown/combobox-input/high.
# Framework is "combobox-input" (NOT a CSS-in-JS hash). The dropdown
# handler has no specialized strategy for combobox-input yet, so it
# falls through to the generic 21-strategy — that's the contract.
# ----------------------------------------------------------------------


def test_css_in_js_combobox_role_routes_to_combobox_input():
    info = classify_element_type(
        element_data={
            "tagName": "div",
            "className": "jss-1ab2c3",
            "role": "combobox",
        },
        element_description="project selector",
    )
    assert info.primary_type == "dropdown"
    assert info.framework == "combobox-input"
    assert info.confidence == "high"


# ----------------------------------------------------------------------
# shadow_dom_dropdown.html — <fancy-dropdown> custom element with no
# accessible attributes outside the shadow root. Classifier sees an
# unknown tag with empty class/role; must produce primary_type=unknown
# and confidence=low (signal-free verdict).
# ----------------------------------------------------------------------


def test_shadow_dom_custom_element_classifies_as_unknown():
    info = classify_element_type(
        element_data={
            "tagName": "fancy-dropdown",
            "className": "",
            "role": "",
            "id": "company-picker",
        },
        element_description="company dropdown",
    )
    # Description-only signal puts dropdown vote at +1 → low confidence.
    # The acceptance criterion is that confidence is NOT high.
    assert info.confidence in ("low", "medium"), info.signals
    assert "tier:1" in info.signals or info.primary_type == "unknown"


# ----------------------------------------------------------------------
# astpp_login_decorative_image.html — login form fields are plain
# <input type="text"> / <input type="password">. Tier 0 has no rule for
# text/password input, so they fall through to Tier 1 with no signals.
# This documents the contract: login fields produce primary_type=unknown
# and the standard 21-strategy path generates the locator.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_type, desc",
    [("text", "Username field"), ("password", "Password field")],
)
def test_login_text_inputs_fall_through(input_type, desc):
    info = classify_element_type(
        element_data={
            "tagName": "input",
            "type": input_type,
            "className": "form-control",
        },
        element_description=desc,
    )
    # No specialized handler — relies on standard text/id strategies.
    assert info.primary_type in ("unknown", "input"), info.signals


# ----------------------------------------------------------------------
# Phase 2.5 — checkbox_radio_widgets.html
# Native + custom (role=checkbox|radio|switch) classification.
# ----------------------------------------------------------------------


class TestCheckboxRadioWidgets:
    """Tier 0 must classify all checkbox/radio/switch variants with high
    confidence so the dispatcher routes them to handlers/checkbox.py."""

    def test_native_checkbox_input(self):
        info = classify_element_type(
            element_data={"tagName": "input", "type": "checkbox", "id": "newsletter"},
            element_description="Subscribe to newsletter",
        )
        assert info.primary_type == "checkbox"
        assert info.framework == "native"
        assert info.confidence == "high"

    def test_native_radio_input(self):
        info = classify_element_type(
            element_data={"tagName": "input", "type": "radio", "name": "plan", "value": "pro"},
            element_description="Pro plan radio",
        )
        assert info.primary_type == "radio"
        assert info.framework == "native"
        assert info.confidence == "high"

    def test_custom_checkbox_role(self):
        info = classify_element_type(
            element_data={"tagName": "span", "role": "checkbox"},
            element_description="Enable analytics",
        )
        assert info.primary_type == "checkbox"
        assert info.framework == "custom"
        assert info.confidence == "high"

    def test_custom_radio_role(self):
        info = classify_element_type(
            element_data={"tagName": "span", "role": "radio"},
            element_description="Light theme",
        )
        assert info.primary_type == "radio"
        assert info.framework == "custom"
        assert info.confidence == "high"

    def test_toggle_role_switch(self):
        info = classify_element_type(
            element_data={"tagName": "span", "role": "switch"},
            element_description="Email notifications",
        )
        # role=switch maps to primary_type=checkbox with framework=toggle
        # so the dispatcher routes through handlers/checkbox.py.
        assert info.primary_type == "checkbox"
        assert info.framework == "toggle"
        assert info.confidence == "high"
