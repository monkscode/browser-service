"""
Dropdown element locator generation.

Dispatches on `type_info.framework` and applies framework-specific scoping
strategies. Currently implements Tom Select in full (Section 6 Strategies
1a–4); other frameworks (native, combobox-input, select2, kendo, react-
select, vue-select, ant-design, material-ui) fall through to the generic
21-strategy path until a real bug forces a specialized implementation.

Tom Select strategies (in priority order):
    1a — Direct id-anchored input:    css=[id="{input_id}"]
    1b — Adjacent-sibling fallback:   css=[id="{select_id}"] + .ts-wrapper .ts-control
    2  — Label-text traversal:        xpath=//label[normalize-space()=$L]/following::div[contains(@class,'ts-wrapper')][1]//div[contains(@class,'ts-control')]
    3  — Form-group ancestor:         xpath=//*[contains(@class,'form-group')][.//label[contains(normalize-space(),$L)]]//div[contains(@class,'ts-control')]
    4  — Coord-based DOM walk:        positional locator scoped to the wrapper containing the click coords
    5  — Vision-assist:               NOT YET IMPLEMENTED — Phase 2.4

Public function:
    find_locator(...) -> Optional[dict]
        Standard handler entry point — see Section 7 of
        docs/ELEMENT_TYPE_CLASSIFIER_ARCHITECTURE.md.

Referenced by:
    - browser_service.locators.smart_locator (dispatcher)

Depends on:
    - re, typing.Optional (stdlib), structlog
    - browser_service.locators.classifier.ElementTypeInfo (type hint only)
"""

import re
import structlog
from typing import TYPE_CHECKING, Optional

from .base import build_locator_result

if TYPE_CHECKING:
    from ..classifier import ElementTypeInfo

logger = structlog.get_logger(__name__)


# ----------------------------------------------------------------------
# Label / description normalization (Section 6 of the architecture doc).
# Both DOM label text and planner descriptions are normalized to the same
# form before comparison: strip required-field markers, strip UI suffixes,
# collapse whitespace.
# ----------------------------------------------------------------------

_LABEL_REQUIRED_MARKER_RE = re.compile(r"\s*\*\s*$")
_LABEL_HTML_REQUIRED_ARTEFACT = "<span> *</span>"
_DESCRIPTION_UI_SUFFIXES: tuple[str, ...] = (
    "dropdown", "field", "input", "select", "picker", "chooser",
)

_TS_CONTROL_SUFFIX = "-ts-control"
_AUTO_GENERATED_SELECT_PREFIX = "tomselect-"


def _css_id_attr(value: str) -> str:
    """Return a CSS attribute selector for an id that is safe for ids containing CSS-special characters."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'[id="{escaped}"]'


def _normalize_label_or_description(text: Optional[str]) -> str:
    """
    Normalize label text from the DOM or description text from the planner
    so they match. Applied transforms (in order):

      1. Strip the literal HTML-encoded artefact `<span> *</span>` if it
         leaked through extraction.
      2. Strip a trailing `*` (with surrounding whitespace) — the
         required-field marker that ASTPP renders as e.g. ``Role *``.
      3. Strip a trailing UI suffix: dropdown / field / input / select /
         picker / chooser. Suffix match is case-insensitive.
      4. Collapse internal whitespace to single spaces and trim.

    Returns an empty string for ``None`` / blank input.
    """
    if not text:
        return ""

    text = text.replace(_LABEL_HTML_REQUIRED_ARTEFACT, "")
    text = _LABEL_REQUIRED_MARKER_RE.sub("", text)
    text = text.strip()

    lower = text.lower()
    for suffix in _DESCRIPTION_UI_SUFFIXES:
        # Word-boundary safe: only strip when preceded by whitespace or
        # when the suffix is the entire string. Avoids "subselect" -> "sub".
        if lower == suffix:
            text = ""
            break
        if lower.endswith(" " + suffix):
            text = text[: len(text) - len(suffix)].rstrip()
            break

    return " ".join(text.split())


def _xpath_string_literal(value: str) -> str:
    """
    Quote a string for safe inclusion as an XPath string literal. Falls
    back to ``concat()`` when the string contains both single and double
    quotes.
    """
    if "'" not in value:
        return f"'{value}'"
    if '"' not in value:
        return f'"{value}"'
    parts = value.split("'")
    return "concat(" + ", \"'\", ".join(f"'{p}'" for p in parts) + ")"


def _is_real_select_id(select_id: Optional[str]) -> bool:
    """
    Tom Select auto-generates ``id="tomselect-N"`` when the underlying
    ``<select>`` has no real id. The N is positional and unstable across
    page renders — a locator built on it cannot be re-derived from a
    description, so we treat such ids as unusable for Strategies 1a/1b
    and fall through to label-based strategies.
    """
    if not select_id:
        return False
    return not select_id.startswith(_AUTO_GENERATED_SELECT_PREFIX)


def _select_id_from_input_id(input_id: Optional[str]) -> Optional[str]:
    """Strip the Tom Select '-ts-control' suffix from a generated input id."""
    if not input_id or not input_id.endswith(_TS_CONTROL_SUFFIX):
        return None
    return input_id[: -len(_TS_CONTROL_SUFFIX)]


# ----------------------------------------------------------------------
# JS probes — run in the page context to discover Tom Select wrapper
# anchors near the click coordinates. Browser-use coords frequently land
# on the outer Bootstrap wrapper rather than the .ts-control input
# itself, so we walk the DOM to find the actual scoping anchor.
# ----------------------------------------------------------------------

# Shared Tom Select DOM walk — finds the nearest .ts-wrapper ancestor.
# Defined once and embedded as an inner function in each probe JS expression
# so the walk logic has a single source of truth.
_TS_FIND_WRAPPER = """\
    function findTsWrapper(startEl) {
        let wrapper = null, cur = startEl;
        while (cur && cur !== document.body) {
            if (cur.classList && cur.classList.contains('ts-wrapper')) { wrapper = cur; break; }
            if (cur.classList && cur.classList.contains('tom-select-custom')) {
                wrapper = cur.querySelector('.ts-wrapper'); if (wrapper) break;
            }
            cur = cur.parentElement;
        }
        return wrapper;
    }
"""

# Shared Tom Select ID extraction — given a .ts-wrapper, returns {selectId, inputId}.
# Used by COORD_PROBE, SCROLL_AND_PROBE, and XPATH_WALK (not STRATEGY4 which only
# needs wrapperIndex).
_TS_EXTRACT_IDS = """\
    function extractIds(wrapper) {
        let select = null;
        if (wrapper.previousElementSibling && wrapper.previousElementSibling.tagName === 'SELECT')
            select = wrapper.previousElementSibling;
        if (!select && wrapper.parentElement)
            select = wrapper.parentElement.querySelector('select');
        const input = wrapper.querySelector('input[role="combobox"]') || wrapper.querySelector('input');
        return { selectId: (select && select.id) || null, inputId: (input && input.id) || null };
    }
"""

# Discovers select_id and input_id for Strategies 1a/1b.
_TS_COORD_PROBE_JS = (
    "(coords) => {\n"
    + _TS_FIND_WRAPPER
    + _TS_EXTRACT_IDS
    + """\

    let el = document.elementFromPoint(coords.x, coords.y);
    if (!el) return null;
    const wrapper = findTsWrapper(el);
    if (!wrapper) return null;
    return extractIds(wrapper);
}
"""
)

# Strategy 4: returns the document-order index of the .ts-wrapper at coords.
# Only needs the walk — does not extract select/input IDs.
_TS_STRATEGY4_JS = (
    "(coords) => {\n"
    + _TS_FIND_WRAPPER
    + """\

    let el = document.elementFromPoint(coords.x, coords.y);
    if (!el) return null;
    const wrapper = findTsWrapper(el);
    if (!wrapper) return null;
    const allWrappers = Array.from(document.querySelectorAll('.ts-wrapper'));
    const idx = allWrappers.indexOf(wrapper);
    if (idx === -1) return null;
    return { wrapperIndex: idx };
}
"""
)

# Source 3: scroll-then-probe — scrolls element into viewport first, then
# runs the same DOM walk as _TS_COORD_PROBE_JS. Used when confirmed_coords
# are offscreen (beyond the viewport) and the initial coord probe returns null.
_TS_SCROLL_AND_PROBE_JS = (
    "async (coords) => {\n"
    + _TS_FIND_WRAPPER
    + _TS_EXTRACT_IDS
    + """\

    window.scrollTo({ top: coords.y - 300, left: 0, behavior: 'instant' });
    await new Promise(r => setTimeout(r, 100));

    const scrollY = window.scrollY;
    const viewY = coords.y - scrollY;
    const viewX = coords.x;

    let el = document.elementFromPoint(viewX, viewY);
    if (!el) return null;
    const wrapper = findTsWrapper(el);
    if (!wrapper) return null;
    return extractIds(wrapper);
}
"""
)

# Label discovery: finds the actual HTML <label> text associated with a
# .ts-wrapper, used for Strategies 2/3 when the agent description doesn't
# match the real label on the page.  (Unused in current implementation;
# see _TS_LABEL_VIA_SELECT_JS and _TS_LABEL_VIA_XPATH_JS below.)
# Removed to avoid SyntaxWarning from unescaped \s in regex.


# Source 4: XPath-based DOM walk — when coords fail, use the element's
# XPath to navigate up to the .ts-wrapper ancestor and find the sibling
# <select> element. Works even when the element is offscreen.
_TS_XPATH_WALK_JS = (
    "(args) => {\n"
    + _TS_FIND_WRAPPER
    + _TS_EXTRACT_IDS
    + """\

    let result;
    try {
        result = document.evaluate(
            args.xpath, document, null,
            XPathResult.FIRST_ORDERED_NODE_TYPE, null
        );
    } catch (e) {
        return null;
    }
    let el = result.singleNodeValue;
    if (!el) return null;
    const wrapper = findTsWrapper(el);
    if (!wrapper) return null;
    return extractIds(wrapper);
}
"""
)

# Label discovery via select_id: finds the <label> whose `for` attribute
# points to the select element, or walks the DOM form-group container
# to find the label text.
_TS_LABEL_VIA_SELECT_JS = r"""
(args) => {
    const selectId = args.selectId;
    if (!selectId) return null;

    // Method 1: label[for=selectId]
    const labelFor = document.querySelector('label[for="' + selectId + '"]');
    if (labelFor) {
        return { label: labelFor.textContent.replace(/\s*\*\s*$/, '').trim() };
    }

    // Method 2: walk up from the select element to find a label in the same container.
    // Cap at 6 levels to avoid picking up unrelated labels from distant ancestors.
    const selectEl = document.getElementById(selectId);
    if (!selectEl) return null;

    let container = selectEl.parentElement;
    let depth = 0;
    while (container && container !== document.body && depth < 6) {
        const label = container.querySelector('label');
        if (label) {
            return { label: label.textContent.replace(/\s*\*\s*$/, '').trim() };
        }
        container = container.parentElement;
        depth++;
    }
    return null;
}
"""

# Label discovery via XPath: walks from the element's XPath to find
# the nearest <label> in the containing form-group.
_TS_LABEL_VIA_XPATH_JS = r"""
(args) => {
    let result;
    try {
        result = document.evaluate(
            args.xpath, document, null,
            XPathResult.FIRST_ORDERED_NODE_TYPE, null
        );
    } catch (e) {
        return null;
    }
    let el = result.singleNodeValue;
    if (!el) return null;

    // Walk up to find a container with a label, skipping ts-wrapper/ts-control
    // (which contain rendered option text, not field labels). Cap at 6 levels.
    let container = el;
    let depth = 0;
    while (container && container !== document.body && depth < 6) {
        const label = container.querySelector('label');
        if (label && !container.classList.contains('ts-wrapper') &&
            !container.classList.contains('ts-control')) {
            return { label: label.textContent.replace(/\s*\*\s*$/, '').trim() };
        }
        container = container.parentElement;
        depth++;
    }
    return null;
}
"""


async def _discover_label_from_dom(
    element_data: dict,
    search_context,
    select_id: Optional[str],
) -> Optional[str]:
    """
    Discover the actual HTML ``<label>`` text associated with a Tom Select
    dropdown from the DOM. This avoids the mismatch between the agent's
    description (a full sentence) and the short label text on the page.

    Sources tried in order:
      1. ``label[for=<select_id>]`` — standard HTML label association
      2. Walk the form-group ancestor from the select element
      3. Walk the form-group ancestor from the element's XPath

    Returns the normalized label text, or None if discovery fails.
    """
    # Source 1 & 2: via select_id
    if select_id and _is_real_select_id(select_id):
        try:
            probe = await search_context.evaluate(
                _TS_LABEL_VIA_SELECT_JS,
                {"selectId": select_id},
            )
            if probe and probe.get("label"):
                label = _normalize_label_or_description(probe["label"])
                if label:
                    return label
        except Exception as e:
            logger.debug("tom_select.label_discovery_select_id_failed", error=str(e))

    # Source 3: via XPath
    if element_data:
        xpath = element_data.get("xpath", "")
        if xpath:
            try:
                probe = await search_context.evaluate(
                    _TS_LABEL_VIA_XPATH_JS,
                    {"xpath": xpath},
                )
                if probe and probe.get("label"):
                    label = _normalize_label_or_description(probe["label"])
                    if label:
                        return label
            except Exception as e:
                logger.debug("tom_select.label_discovery_xpath_failed", error=str(e))

    return None


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


async def find_locator(
    page,
    element_data: dict,
    type_info: "ElementTypeInfo",
    element_id: str,
    element_description: str,
    expected_text: Optional[str],
    search_context,
    iframe_context: Optional[str],
    confirmed_coords: Optional[tuple],
) -> Optional[dict]:
    """
    Dropdown handler entry point.

    Dispatches on ``type_info.framework``:
      - ``"tom-select"`` → Section 6 Strategies 1a → 1b → 2 → 3 → 4
      - all other frameworks → return None (fall through to generic
        21-strategy)

    Returns a result dict with ``element_type='dropdown'`` and
    ``dropdown_framework`` populated when a strategy succeeds; returns
    ``None`` otherwise. Always-fallback contract: never raises.
    """
    framework = type_info.framework

    if framework == "tom-select":
        return await _tom_select(
            element_data=element_data,
            type_info=type_info,
            element_id=element_id,
            element_description=element_description,
            search_context=search_context,
            confirmed_coords=confirmed_coords,
        )

    logger.info("dropdown.handler.no_strategy", framework=framework)
    return None


# ----------------------------------------------------------------------
# Tom Select — Section 6 strategies
# ----------------------------------------------------------------------


async def _tom_select(
    element_data: dict,
    type_info: "ElementTypeInfo",
    element_id: str,
    element_description: str,
    search_context,
    confirmed_coords: Optional[tuple],
) -> Optional[dict]:
    """Run Strategies 1a → 1b → 2 → 3 → 4. Return None if all fail."""

    select_id, input_id = await _resolve_tom_select_ids(
        element_data=element_data,
        search_context=search_context,
        confirmed_coords=confirmed_coords,
    )

    has_real_select_id = _is_real_select_id(select_id)
    has_real_input_id = (
        input_id is not None
        and not input_id.startswith(_AUTO_GENERATED_SELECT_PREFIX)
    )

    # Auto-generated select_ids (tomselect-N) are positionally unstable — the N
    # depends on TomSelect initialisation order in a given browser session and will
    # differ in a fresh Docker session.  Pass None to build_locator_result so the
    # Code Assembler uses the locator-based JS fallback instead of id=tomselect-N.
    result_select_id = select_id if has_real_select_id else None
    if result_select_id is None and select_id is not None:
        logger.info("tom_select.select_id_suppressed", select_id=select_id,
                    reason="auto-generated prefix, unstable across sessions")

    # ---- Strategy 1a: id-anchored input ----
    if has_real_input_id:
        result = await _validate_and_build_result(
            locator=f"css={_css_id_attr(input_id)}",
            strategy_name="Tom Select Strategy 1a (id-anchored input)",
            element_id=element_id,
            element_description=element_description,
            type_info=type_info,
            search_context=search_context,
            select_id=result_select_id,
        )
        if result:
            return result

    # ---- Strategy 1b: adjacent-sibling fallback ----
    if has_real_select_id:
        result = await _validate_and_build_result(
            locator=f"css={_css_id_attr(select_id)} + .ts-wrapper .ts-control",
            strategy_name="Tom Select Strategy 1b (adjacent sibling)",
            element_id=element_id,
            element_description=element_description,
            type_info=type_info,
            search_context=search_context,
            select_id=result_select_id,
        )
        if result:
            return result

    # ----------------------------------------------------------------
    # Strategies 2/3: label-based traversal.
    # Two label sources, tried in priority order:
    #   a) DOM-discovered label — the actual <label> in the HTML.
    #   b) Normalized element_description — fallback when DOM query fails.
    # The agent description is often a sentence ("Rate Group dropdown
    # selector within the Billing Settings section") that never matches
    # the short HTML label ("Rate Group"). Using the DOM label first
    # avoids this mismatch.
    # ----------------------------------------------------------------
    labels_to_try: list[str] = []

    # a) DOM-discovered label via Playwright
    dom_label = await _discover_label_from_dom(
        element_data=element_data,
        search_context=search_context,
        select_id=select_id,
    )
    if dom_label:
        labels_to_try.append(dom_label)
        logger.info("tom_select.label_discovered", label=dom_label)

    # b) Normalized description (fallback)
    desc_label = _normalize_label_or_description(element_description)
    if desc_label and desc_label not in labels_to_try:
        labels_to_try.append(desc_label)

    for label in labels_to_try:
        # ---- Strategy 2: label-text traversal ----
        xpath2 = (
            f"xpath=//label[normalize-space()={_xpath_string_literal(label)}]"
            f"/following::div[contains(@class,'ts-wrapper')][1]"
            f"//div[contains(@class,'ts-control')]"
        )
        result = await _validate_and_build_result(
            locator=xpath2,
            strategy_name="Tom Select Strategy 2 (label-text traversal)",
            element_id=element_id,
            element_description=element_description,
            type_info=type_info,
            search_context=search_context,
            select_id=result_select_id,
        )
        if result:
            return result

        # ---- Strategy 3: form-group ancestor ----
        xpath3 = (
            f"xpath=//*[contains(@class,'form-group')]"
            f"[.//label[contains(normalize-space(),"
            f"{_xpath_string_literal(label)})]]"
            f"//div[contains(@class,'ts-control')]"
        )
        result = await _validate_and_build_result(
            locator=xpath3,
            strategy_name="Tom Select Strategy 3 (form-group ancestor)",
            element_id=element_id,
            element_description=element_description,
            type_info=type_info,
            search_context=search_context,
            select_id=result_select_id,
        )
        if result:
            return result

    # ---- Strategy 4: coordinate-based DOM walk ----
    if confirmed_coords:
        result = await _strategy4_coord_walk(
            element_id=element_id,
            element_description=element_description,
            type_info=type_info,
            search_context=search_context,
            confirmed_coords=confirmed_coords,
        )
        if result:
            return result

    # Strategy 5 (vision) — Phase 2.4.
    logger.info("tom_select.all_strategies_exhausted")
    return None


async def _resolve_tom_select_ids(
    element_data: dict,
    search_context,
    confirmed_coords: Optional[tuple],
) -> tuple[Optional[str], Optional[str]]:
    """
    Discover ``(select_id, input_id)`` for Strategies 1a/1b.

    Sources tried in order:
      1. JS coord probe (most reliable when coords are known and in-viewport)
      2. ``element_data['id']`` if it ends in ``-ts-control``
      3. Scroll-then-probe (when coords exist but are offscreen)
      4. XPath-based DOM walk (when element_data has an XPath)

    Returns ``(None, None)`` when no source produces an id.
    """
    select_id: Optional[str] = None
    input_id: Optional[str] = None

    # ---- Source 1: JS coord probe (in-viewport) ----
    if confirmed_coords:
        try:
            probe = await search_context.evaluate(
                _TS_COORD_PROBE_JS,
                {"x": confirmed_coords[0], "y": confirmed_coords[1]},
            )
        except Exception as e:
            logger.warning("tom_select.coord_probe_failed", error=str(e))
            probe = None

        if probe:
            select_id = probe.get("selectId")
            input_id = probe.get("inputId")
            logger.info("tom_select.coord_probe_resolved", select_id=select_id, input_id=input_id)

    # ---- Source 2: element_data['id'] ending in -ts-control ----
    if not select_id:
        el_id = (element_data.get("id") or "") if element_data else ""
        derived = _select_id_from_input_id(el_id)
        if derived:
            select_id = derived
            input_id = el_id
            logger.info("tom_select.ids_from_element_data", select_id=select_id, input_id=input_id)

    # ---- Source 3: scroll-then-probe (offscreen coords) ----
    # document.elementFromPoint returns null for coords outside the
    # visible viewport. Scroll the target position into view first,
    # then re-run the probe with adjusted viewport-relative coords.
    if not select_id and confirmed_coords:
        try:
            viewport = await search_context.evaluate(
                "() => ({ h: window.innerHeight })"
            )
            vh = viewport.get("h", 0) if viewport else 0
            if confirmed_coords[1] > vh:
                logger.info("tom_select.scroll_to_viewport", coords_y=confirmed_coords[1], viewport_height=vh)
                probe = await search_context.evaluate(
                    _TS_SCROLL_AND_PROBE_JS,
                    {"x": confirmed_coords[0], "y": confirmed_coords[1]},
                )
                if probe:
                    select_id = probe.get("selectId")
                    input_id = probe.get("inputId")
                    logger.info("tom_select.scroll_probe_resolved", select_id=select_id, input_id=input_id)
        except Exception as e:
            logger.warning("tom_select.scroll_probe_failed", error=str(e))

    # ---- Source 4: XPath-based DOM walk ----
    # When coords fail entirely, use the element's XPath (if available)
    # to find the ts-wrapper ancestor and its sibling <select>.
    if not select_id and element_data:
        xpath = element_data.get("xpath", "")
        if xpath:
            try:
                probe = await search_context.evaluate(
                    _TS_XPATH_WALK_JS,
                    {"xpath": xpath},
                )
                if probe:
                    select_id = probe.get("selectId")
                    input_id = probe.get("inputId")
                    logger.info("tom_select.xpath_walk_resolved", select_id=select_id, input_id=input_id)
            except Exception as e:
                logger.warning("tom_select.xpath_walk_failed", error=str(e))

    return select_id, input_id


async def _strategy4_coord_walk(
    element_id: str,
    element_description: str,
    type_info: "ElementTypeInfo",
    search_context,
    confirmed_coords: tuple,
) -> Optional[dict]:
    """
    Coordinate-based DOM walk: locate the ``.ts-wrapper`` containing the
    click coordinates, build a positional XPath that targets the
    ``.ts-control`` inside that specific wrapper.
    """
    try:
        probe = await search_context.evaluate(
            _TS_STRATEGY4_JS,
            {"x": confirmed_coords[0], "y": confirmed_coords[1]},
        )
    except Exception as e:
        logger.warning("tom_select.strategy4_probe_failed", error=str(e))
        return None

    if not probe:
        return None

    idx = probe.get("wrapperIndex")
    if idx is None or idx < 0:
        return None

    # XPath positional indexing is 1-based.
    locator = (
        f"xpath=(//div[contains(@class,'ts-wrapper')])[{idx + 1}]"
        f"//div[contains(@class,'ts-control')]"
    )
    return await _validate_and_build_result(
        locator=locator,
        strategy_name="Tom Select Strategy 4 (coord-based DOM walk)",
        element_id=element_id,
        element_description=element_description,
        type_info=type_info,
        search_context=search_context,
        select_id=None,
    )


# ----------------------------------------------------------------------
# Result construction
# ----------------------------------------------------------------------


async def _validate_and_build_result(
    locator: str,
    strategy_name: str,
    element_id: str,
    element_description: str,
    type_info: "ElementTypeInfo",
    search_context,
    select_id: Optional[str],
) -> Optional[dict]:
    """
    Verify uniqueness (count == 1) of the locator. Build and return the
    standard dropdown result dict on success; return None on count != 1
    or any Playwright error.
    """
    try:
        count = await search_context.locator(locator).count()
    except Exception as e:
        logger.warning("tom_select.strategy_validation_failed", strategy=strategy_name, error=str(e))
        return None

    if count != 1:
        logger.info("tom_select.strategy_not_unique", strategy=strategy_name, count=count, locator=locator)
        return None

    logger.info("tom_select.strategy_succeeded", strategy=strategy_name, locator=locator)

    return build_locator_result(
        element_id=element_id,
        description=element_description,
        best_locator=locator,
        element_type="dropdown",
        strategy_name=strategy_name,
        classifier_confidence=type_info.confidence,
        classifier_signals=type_info.signals,
        count=count,
        dropdown_framework="tom-select",
        select_id=select_id,
    )
