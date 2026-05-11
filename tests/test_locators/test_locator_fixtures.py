"""
End-to-end fixture-driven tests for the Phase 2 classifier + dispatcher +
handler chain. Each test launches a real headless Chromium, navigates to a
static HTML fixture in ``locator_fixtures/``, and asserts behavior at one
of two layers:

  1. Handler layer — calls ``dropdown.find_locator()`` / classifier directly
     against a real DOM. Exercises Phase 2.3's Tom Select scoping
     deterministically without going through the orchestrator's text-first
     short-circuits.

  2. Orchestrator layer — calls ``find_unique_locator_at_coordinates(...)``
     end-to-end with realistic element_data. Verifies the full chain
     (classifier → dispatcher → handler → standard fallback) produces a
     usable locator on real DOM.

Marked ``integration`` so they only run when explicitly requested:

    pytest tests/test_locators/test_locator_fixtures.py -m integration -v

Requires ``playwright install chromium``. Default CI runs (``-m "not
integration"``) skip these.

Phase 2.4-rescoped (DOM probe + vision hint piggyback) un-skipped the
CSS-in-JS adversarial fixture: ``role="combobox"`` is a Tier 0 hit and the
probe confirms via the same role signal — no vision-assist required.
Shadow-DOM and Bug 1 cases remain skipped: shadow-DOM aria signals live
inside the shadow root and don't leak to the host element (real bug to
fix when surfaced in production); Bug 1 is an agent-behavior snapshot
that drives the real browser-use loop and is gated on CI cost, not on
the locator pipeline.

Reference: docs/ELEMENT_TYPE_CLASSIFIER_ARCHITECTURE.md Section 10.
"""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.locators import find_unique_locator_at_coordinates
from browser_service.locators.classifier import (
    ElementTypeInfo,
    classify_element_type,
)
from browser_service.locators.handlers import checkbox as checkbox_handler
from browser_service.locators.handlers import dropdown as dropdown_handler


pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent / "locator_fixtures"


def _file_url(name: str) -> str:
    """Convert a fixture filename to a file:// URL Playwright can load."""
    return (FIXTURES_DIR / name).resolve().as_uri()


_ELEMENT_DATA_JS = """
(el) => ({
    tagName: el.tagName.toLowerCase(),
    id: el.id || "",
    className: (typeof el.className === 'string') ? el.className : "",
    role: el.getAttribute('role') || "",
    type: el.getAttribute('type') || "",
    name: el.getAttribute('name') || "",
    href: el.getAttribute('href') || "",
    textContent: (el.textContent || "").trim().slice(0, 80),
})
"""


async def _coords_and_data(page, selector: str):
    """Compute click coords + extract browser-use-style element_data."""
    handle = page.locator(selector).first
    await handle.scroll_into_view_if_needed()
    bbox = await handle.bounding_box()
    if not bbox:
        raise RuntimeError(f"No bounding box for {selector!r}")
    x = bbox["x"] + bbox["width"] / 2
    y = bbox["y"] + bbox["height"] / 2
    data = await handle.evaluate(_ELEMENT_DATA_JS)
    return x, y, data


# ----------------------------------------------------------------------
# Shared browser fixture — one Chromium per test for isolation. Cheap
# enough at this scale; switch to a session-scoped browser if test count
# grows.
# ----------------------------------------------------------------------


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page_obj = await ctx.new_page()
        try:
            yield page_obj
        finally:
            await browser.close()


# ======================================================================
# Bug 2 — Customer link under Create dropdown menu
# ======================================================================


async def test_bug2_customer_link_not_navitem(page):
    """Clicking 'Customer' must produce a text-anchored locator
    (text=, role=link, href= …) — never the parent .nav-item."""
    await page.goto(_file_url("astpp_create_menu_navitem.html"),
                    wait_until="domcontentloaded")
    await page.evaluate(
        "() => document.querySelectorAll('.dropdown-menu')"
        ".forEach(m => { m.style.display = 'block'; })"
    )
    x, y, data = await _coords_and_data(
        page, "a.nav-link[href='/customer/create']"
    )

    result = await find_unique_locator_at_coordinates(
        page=page, x=x, y=y,
        element_id="elem_customer_link",
        element_description="Customer menu item under Create dropdown",
        expected_text="Customer",
        element_data=data,
    )

    assert result["found"] is True, result
    locator = result["best_locator"].lower()
    assert "nav-item" not in locator, (
        f"Bug 2 regression — got nav-item locator: {result['best_locator']}"
    )
    assert "navbar" not in locator
    # Acceptable anchors: visible text, href, link role, or nav-link class.
    assert any(
        token in locator
        for token in ("customer", "/customer/create", "nav-link", "role=link")
    ), f"Locator must anchor on the link, got: {result['best_locator']}"


# ======================================================================
# Bug 3 — Tom Select handler-level scoping (Phase 2.3 directly)
# ======================================================================
#
# These tests bypass the orchestrator and call dropdown.find_locator()
# against a real Playwright page. They verify Phase 2.3 logic in
# isolation, independent of orchestrator-level short-circuits like the
# text-first path.


_HIGH_TS_INFO = ElementTypeInfo(
    primary_type="dropdown",
    framework="tom-select",
    confidence="high",
    signals=["className:ts-wrapper", "tier:0"],
)


async def test_bug3_role_field_strategy_1a_id_anchored(page):
    """Phase 2.3 Strategy 1a: id-anchored input wins for the Role field
    because <select id='permission_id'> has a real (non auto-generated) id."""
    await page.goto(_file_url("astpp_customer_form_tom_select.html"),
                    wait_until="domcontentloaded")
    bbox = await page.locator("#permission_id-ts-control").bounding_box()
    assert bbox is not None
    coords = (bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)

    # element_data shape browser-use would produce when it captures the
    # focusable .ts-control input rather than the outer wrapper.
    elem_data = {
        "tagName": "input",
        "id": "permission_id-ts-control",
        "className": "",
        "role": "combobox",
    }

    result = await dropdown_handler.find_locator(
        page=page,
        element_data=elem_data,
        type_info=_HIGH_TS_INFO,
        element_id="elem_role",
        element_description="Role dropdown",
        expected_text="Customer_permission",
        search_context=page,
        iframe_context=None,
        confirmed_coords=coords,
    )

    assert result is not None, "Strategy 1a should produce a result"
    assert result["element_type"] == "dropdown"
    assert result["dropdown_framework"] == "tom-select"
    assert result["select_id"] == "permission_id"
    assert "permission_id-ts-control" in result["best_locator"], result["best_locator"]


async def test_bug3_timezone_skips_auto_generated_id(page):
    """Phase 2.3 _is_real_select_id guard: ``id='tomselect-6'`` is
    auto-generated and unstable. Strategy 1a/1b must skip it; the
    handler must fall through to label-based strategies and never
    produce a locator anchored on tomselect-6."""
    await page.goto(_file_url("astpp_customer_form_tom_select.html"),
                    wait_until="domcontentloaded")
    bbox = await page.locator("#tomselect-6-ts-control").bounding_box()
    assert bbox is not None
    coords = (bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)

    elem_data = {
        "tagName": "input",
        "id": "tomselect-6-ts-control",
        "className": "",
        "role": "combobox",
    }

    result = await dropdown_handler.find_locator(
        page=page,
        element_data=elem_data,
        type_info=_HIGH_TS_INFO,
        element_id="elem_timezone",
        element_description="Timezone dropdown",
        expected_text="Asia/Kolkata",
        search_context=page,
        iframe_context=None,
        confirmed_coords=coords,
    )

    if result is not None:
        # If the handler did produce a locator, it must NOT anchor on the
        # auto-generated id.
        assert "tomselect-6" not in result["best_locator"], (
            f"Auto-generated id leaked into locator: {result['best_locator']!r}"
        )


@pytest.mark.parametrize(
    "select_id, expected_value",
    [
        ("country_id", "India"),
        ("currency_id", "USD"),
        ("pricelist_id", "Default"),
        ("sweep_id", "Monthly"),
    ],
)
async def test_multi_instance_tom_select_handler_scoping(
    page, select_id, expected_value
):
    """Each Tom Select instance on the multi-instance fixture must scope
    to its own ``{select_id}-ts-control`` — bare ``.ts-control`` would
    match all four. Tests Strategy 1a directly."""
    await page.goto(_file_url("multi_instance_tom_select.html"),
                    wait_until="domcontentloaded")
    bbox = await page.locator(f"#{select_id}-ts-control").bounding_box()
    assert bbox is not None
    coords = (bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)

    elem_data = {
        "tagName": "input",
        "id": f"{select_id}-ts-control",
        "className": "",
        "role": "combobox",
    }

    result = await dropdown_handler.find_locator(
        page=page,
        element_data=elem_data,
        type_info=_HIGH_TS_INFO,
        element_id=f"elem_{select_id}",
        element_description=f"{select_id.replace('_', ' ')} dropdown",
        expected_text=expected_value,
        search_context=page,
        iframe_context=None,
        confirmed_coords=coords,
    )

    assert result is not None
    assert result["dropdown_framework"] == "tom-select"
    assert result["select_id"] == select_id
    assert f"{select_id}-ts-control" in result["best_locator"], (
        f"{select_id} did not scope correctly — got: {result['best_locator']!r}"
    )


async def test_tom_select_no_label_for_handler_succeeds(page):
    """Strategy 1a does not depend on <label for=> — the input id alone
    is enough. This fixture has no `for=` attribute on the label."""
    await page.goto(_file_url("tom_select_no_label_for.html"),
                    wait_until="domcontentloaded")
    bbox = await page.locator("#pricelist_id-ts-control").bounding_box()
    assert bbox is not None
    coords = (bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)

    elem_data = {
        "tagName": "input",
        "id": "pricelist_id-ts-control",
        "className": "",
        "role": "combobox",
    }

    result = await dropdown_handler.find_locator(
        page=page,
        element_data=elem_data,
        type_info=_HIGH_TS_INFO,
        element_id="elem_rate_group",
        element_description="Rate Group dropdown",
        expected_text="Default",
        search_context=page,
        iframe_context=None,
        confirmed_coords=coords,
    )

    assert result is not None
    assert "pricelist_id-ts-control" in result["best_locator"]


# ======================================================================
# Classifier-on-real-DOM — verify class detection sees the same shape
# the live browser-use pipeline does.
# ======================================================================


async def test_classifier_picks_up_ts_wrapper_classname_from_real_dom(page):
    """Sanity check: a real ``.ts-wrapper`` element rendered by Chromium
    yields the same className substring our pure-Python classifier expects
    — i.e., no surprise sanitization or attribute munging."""
    await page.goto(_file_url("astpp_customer_form_tom_select.html"),
                    wait_until="domcontentloaded")
    data = await page.locator("#permission_id + .ts-wrapper").evaluate(
        _ELEMENT_DATA_JS
    )
    info = classify_element_type(data, "Role dropdown")
    assert info.primary_type == "dropdown"
    assert info.framework == "tom-select"
    assert info.confidence == "high"


# ======================================================================
# nav_item_negative — three sibling .nav-items must each get a unique
# locator that is NOT bare ``.nav-item``.
# ======================================================================


@pytest.mark.parametrize("label", ["Dashboard", "Reports", "Settings"])
async def test_navitem_siblings_get_distinct_locators(page, label):
    await page.goto(_file_url("nav_item_negative.html"),
                    wait_until="domcontentloaded")
    handle = page.locator(f"text={label}").first
    bbox = await handle.bounding_box()
    assert bbox is not None
    data = await handle.evaluate(_ELEMENT_DATA_JS)

    result = await find_unique_locator_at_coordinates(
        page=page,
        x=bbox["x"] + bbox["width"] / 2,
        y=bbox["y"] + bbox["height"] / 2,
        element_id=f"elem_{label.lower()}",
        element_description=f"{label} navigation item",
        expected_text=label,
        element_data=data,
    )

    assert result["found"] is True, result
    locator = result["best_locator"].lower()
    # Must not be the bare class — that would match all three siblings.
    assert locator not in ("css=.nav-item", ".nav-item"), result["best_locator"]


# ======================================================================
# Phase 2.5 — checkbox / radio / toggle handler against real DOM
# ======================================================================


_CHECKBOX_HIGH = ElementTypeInfo(
    primary_type="checkbox", framework="native",
    confidence="high", signals=["tagName=input", "type=checkbox", "tier:0"],
)
_RADIO_HIGH = ElementTypeInfo(
    primary_type="radio", framework="native",
    confidence="high", signals=["tagName=input", "type=radio", "tier:0"],
)
_CUSTOM_CHECKBOX_HIGH = ElementTypeInfo(
    primary_type="checkbox", framework="custom",
    confidence="high", signals=["role=checkbox", "tier:0"],
)
_CUSTOM_RADIO_HIGH = ElementTypeInfo(
    primary_type="radio", framework="custom",
    confidence="high", signals=["role=radio", "tier:0"],
)
_TOGGLE_HIGH = ElementTypeInfo(
    primary_type="checkbox", framework="toggle",
    confidence="high", signals=["role=switch", "tier:0"],
)


async def test_native_checkbox_id_anchored(page):
    """`<input type="checkbox" id="newsletter">` must produce id=newsletter
    via Strategy 1 attribute anchor."""
    await page.goto(_file_url("checkbox_radio_widgets.html"),
                    wait_until="domcontentloaded")
    elem_data = {
        "tagName": "input", "type": "checkbox",
        "id": "newsletter", "name": "newsletter",
    }
    result = await checkbox_handler.find_locator(
        page=page, element_data=elem_data, type_info=_CHECKBOX_HIGH,
        element_id="elem", element_description="Subscribe to newsletter",
        expected_text="Subscribe to newsletter",
        search_context=page, iframe_context=None, confirmed_coords=None,
    )
    assert result is not None
    assert result["best_locator"] == "id=newsletter"
    assert result["element_type"] == "checkbox"
    assert result["framework"] == "native"


async def test_native_radio_name_value_anchored(page):
    """Native radios sharing a name must scope on name+value."""
    await page.goto(_file_url("checkbox_radio_widgets.html"),
                    wait_until="domcontentloaded")
    elem_data = {
        "tagName": "input", "type": "radio",
        "name": "plan", "value": "pro",
    }
    # Drop id from element_data to force name+value path; simulates the case
    # where browser-use captured a radio inside a list without an id.
    info = ElementTypeInfo(
        primary_type="radio", framework="native",
        confidence="high",
        signals=["tagName=input", "type=radio", "tier:0"],
    )
    result = await checkbox_handler.find_locator(
        page=page, element_data=elem_data, type_info=info,
        element_id="elem", element_description="Pro plan",
        expected_text="Pro",
        search_context=page, iframe_context=None, confirmed_coords=None,
    )
    assert result is not None
    assert result["best_locator"] == 'input[type="radio"][name="plan"][value="pro"]'
    assert result["element_type"] == "radio"


async def test_native_checkbox_label_text_fallback(page):
    """`Remember me` is a nested `<label><input>...</label>` with no id.
    Strategy 1 misses (no id/name in element_data), Strategy 2 (legacy
    label-based search) succeeds via the nested-input branch."""
    await page.goto(_file_url("checkbox_radio_widgets.html"),
                    wait_until="domcontentloaded")
    elem_data = {"tagName": "input", "type": "checkbox"}
    result = await checkbox_handler.find_locator(
        page=page, element_data=elem_data, type_info=_CHECKBOX_HIGH,
        element_id="elem", element_description="Remember me",
        expected_text="Remember me",
        search_context=page, iframe_context=None, confirmed_coords=None,
    )
    assert result is not None
    # Legacy helper falls back to nested-input search; locator must be
    # name-anchored or id-anchored, never the bare nested CSS chain.
    # Accepted formats: "id=...", "[name=...", "input[type=...][name=..."
    locator = result["best_locator"]
    assert locator.startswith("id=") or '[name="' in locator, locator


async def test_custom_checkbox_role_aria_label(page):
    """Custom-widget `<span role="checkbox" aria-label="...">` must produce
    `[role="checkbox"][aria-label="..."]` (Strategy 3)."""
    await page.goto(_file_url("checkbox_radio_widgets.html"),
                    wait_until="domcontentloaded")
    elem_data = {
        "tagName": "span", "role": "checkbox",
        "id": "custom-analytics",
    }
    # Drop id from element_data to force Strategy 3 — the custom-widget
    # path. (Strategy 1 would otherwise win via id=custom-analytics.)
    elem_data_no_id = {"tagName": "span", "role": "checkbox"}
    result = await checkbox_handler.find_locator(
        page=page, element_data=elem_data_no_id,
        type_info=_CUSTOM_CHECKBOX_HIGH,
        element_id="elem", element_description="Enable analytics",
        expected_text="Enable analytics",
        search_context=page, iframe_context=None, confirmed_coords=None,
    )
    assert result is not None
    assert result["best_locator"] == \
        '[role="checkbox"][aria-label="Enable analytics"]'
    assert result["framework"] == "custom"


async def test_toggle_switch_role_aria_label(page):
    """Custom-widget `<span role="switch">` must produce
    `[role="switch"][aria-label="..."]`."""
    await page.goto(_file_url("checkbox_radio_widgets.html"),
                    wait_until="domcontentloaded")
    elem_data = {"tagName": "span", "role": "switch"}
    result = await checkbox_handler.find_locator(
        page=page, element_data=elem_data, type_info=_TOGGLE_HIGH,
        element_id="elem", element_description="Email notifications",
        expected_text="Email notifications",
        search_context=page, iframe_context=None, confirmed_coords=None,
    )
    assert result is not None
    assert result["best_locator"] == \
        '[role="switch"][aria-label="Email notifications"]'


# ======================================================================
# Adversarial cases — covered by Phase 2.4-rescoped probe + hint design
# ======================================================================


async def test_css_in_js_obfuscated_dropdown_classifies_as_combobox(page):
    """Hashed CSS-in-JS classnames defeat framework-pattern detection,
    but the element exposes ``role="combobox"`` and ``aria-haspopup``
    directly. Tier 0 rule 15 (role=combobox) fires; the probe confirms
    via the same role signal plus aria-haspopup. No vision-assist required.

    This test was originally a Phase 2.4 vision-assist placeholder; the
    Phase 2.4-rescoped two-source-of-truth design covers it for free."""
    await page.goto(_file_url("css_in_js_obfuscated_dropdown.html"),
                    wait_until="domcontentloaded")
    x, y, data = await _coords_and_data(page, "#project-picker")

    # Classifier-only assertion (cheap, no probe).
    type_info = classify_element_type(data, "Project picker dropdown")
    assert type_info.primary_type == "dropdown"
    assert type_info.framework == "combobox-input"
    assert type_info.confidence == "high"
    assert any("role=combobox" in s for s in type_info.signals)
    assert any("tier:0" in s for s in type_info.signals)

    # End-to-end: full pipeline produces a usable locator. The
    # ``combobox-input`` framework has no specialized handler in
    # Phase 2.3 (no Tom Select-style anchor pattern to scope to), so
    # the dropdown handler returns None and the dispatcher falls
    # through to the generic 21-strategy. That's correct behavior —
    # the probe + classifier agreement is what matters here, and the
    # generic path still produces a usable id-anchored locator.
    result = await find_unique_locator_at_coordinates(
        page=page, x=x, y=y, element_id="elem_proj",
        element_description="Project picker dropdown",
        expected_text=None, element_data=data,
    )
    assert result is not None
    assert result.get("best_locator")  # generic fallback found something
    # Probe corroboration is observable in stdout/log even when the
    # generic path runs; metadata stamping is gated on specialized
    # handler success and is intentionally absent here.


async def test_shadow_dom_dropdown_confirmed_via_shadow_root_walk(page):
    """Shadow-DOM-encapsulated dropdowns: the host element
    (``<fancy-dropdown>``) exposes no role/class to the outside DOM,
    but the inner trigger has ``role="combobox"`` plus
    ``aria-haspopup``. The probe's open-shadow-root walk reaches the
    trigger and confirms the dropdown verdict.

    Originally a Phase 2.4 vision-assist placeholder; covered for free
    by the Phase 2.4-rescoped probe + shadow-root walk follow-up."""
    from browser_service.locators.dom_probe import probe_specialized_type

    await page.goto(_file_url("shadow_dom_dropdown.html"),
                    wait_until="domcontentloaded")
    x, y, host_data = await _coords_and_data(page, "fancy-dropdown")

    # Classifier sees only the host: no role, no class, no input type.
    # Tier 0 doesn't fire; Tier 1 yields "unknown" or low-confidence.
    type_info = classify_element_type(host_data, "Company picker dropdown")
    assert type_info.primary_type in ("unknown", "dropdown")  # description-only path

    # The probe (in discovery mode with the dropdown hint) reaches
    # the inner trigger via the open shadow root and confirms.
    probe_result = await probe_specialized_type(
        page=page, suspected_type="dropdown", coords=(x, y),
    )
    assert probe_result["confirmed"] is True
    # Some signal must be tagged with .shadow — proof the walk
    # actually pierced rather than only finding host-level signals.
    assert any(".shadow" in s for s in probe_result["signals"]), \
        f"expected a shadow-walk signal, got: {probe_result['signals']}"
    # role=combobox on the inner trigger should appear in signals.
    assert any("role:combobox" in s for s in probe_result["signals"]), \
        f"expected role:combobox in signals, got: {probe_result['signals']}"


@pytest.mark.skip(
    reason="Bug 1 agent-behavior snapshot: needs a real browser-use loop "
           "with a multimodal LLM. Locator pipeline is not in scope; the "
           "fix lives in clients/inextrix/config.json (covered by "
           "test_inextrix_config.py). Gated on CI cost, not on Phase 2.4."
)
async def test_bug1_decorative_image_does_not_short_circuit_login():
    """Drive a real browser-use agent against this fixture and assert
    it fills the form before declaring login complete."""
