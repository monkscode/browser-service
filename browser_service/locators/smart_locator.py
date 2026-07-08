"""
Smart Locator Finder
====================

Deterministic locator extraction using multiple strategies.
Given coordinates, systematically tries different approaches to find unique locators.
"""

import logging
import math
import re
from typing import Any, Optional

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

logger = logging.getLogger(__name__)

# Configuration Constants
# These values control the behavior of locator finding strategies

# Text validation thresholds
MIN_TEXT_LENGTH = 2  # Minimum text length to use for text-based locators
MAX_TEXT_DISPLAY_LENGTH = 50  # Maximum text length to display in logs (for actual text)
MAX_TEXT_CONTENT_LENGTH = 100  # Maximum text content to extract from elements

# Locator priorities (lower = better)
PRIORITY_ID = 1  # Native ID attribute
PRIORITY_TEST_ID = 2  # data-testid, data-test, data-qa
PRIORITY_NAME = 3  # name attribute
PRIORITY_ARIA_LABEL = 4  # aria-label
PRIORITY_PLACEHOLDER = 5  # placeholder, title
PRIORITY_TEXT = 6  # Visible text content
PRIORITY_ROLE = 7  # ARIA role with name
PRIORITY_CSS_PARENT_ID = 8  # CSS with parent ID context
PRIORITY_CSS_NTH_CHILD = 9  # CSS with nth-child
PRIORITY_CSS_CLASS = 10  # Simple CSS class
PRIORITY_XPATH_PARENT_ID = 11  # XPath with parent ID
PRIORITY_XPATH_PARENT_CLASS = 12  # XPath with parent class and position
PRIORITY_XPATH_TEXT = 13  # XPath with text content
PRIORITY_XPATH_TITLE = 14  # XPath with title
PRIORITY_XPATH_HREF = 15  # XPath with href (for links)
PRIORITY_XPATH_CLASS_POSITION = 16  # XPath with class and position
PRIORITY_XPATH_MULTI_ATTR = 17  # XPath with multiple attributes
PRIORITY_XPATH_FIRST_OF_CLASS = 18  # XPath - first element with class

# Dropdown detection patterns for coordinate-based validation
DROPDOWN_CSS_PATTERNS = [
    'k-multiselect', 'k-dropdown', 'k-combobox',  # Kendo UI
    'select2', 'chosen',  # jQuery plugins
    'MuiSelect', 'MuiAutocomplete',  # Material-UI
    'react-select', 'ng-select',  # React/Angular
    'ts-wrapper', 'ts-control',  # Tom Select
]

DROPDOWN_KEYWORDS = ['dropdown', 'select', 'combobox', 'multiselect', 'picker', 'chooser']

# Max distance (px) between browser-use coordinates and an element's center
# for the element to count as "the one the vision model saw". Shared by
# multi-match disambiguation and the text-singleton identity check (A5).
COORD_DISTANCE_THRESHOLD = 100


def _escape_css_selector(value: str) -> str:
    """
    Escape special characters in CSS selectors.
    
    Handles characters that have special meaning in CSS selectors
    (like :, ., [, ], etc.) by escaping them with backslash.
    
    Args:
        value: The CSS selector value to escape (e.g., class name, id)
        
    Returns:
        Escaped string safe for use in CSS selectors, or empty string if invalid
    """
    if not value or not value.strip():
        return ''
    # CSS escape special characters: : . [ ] ( ) etc
    result = ''
    for char in value:
        if char.isalnum() or char in '-_':
            result += char
        elif char == ' ':
            logger.debug(f"Skipping multi-word class in CSS escape: '{value}'")
            return ''  # Multi-word classes should be handled separately
        else:
            result += f'\\{char}'  # CSS escape
    return result


def is_dropdown_element(
    element_data: dict,
    element_description: Optional[str] = None
) -> bool:
    """
    Multi-layered dropdown detection using all available signals.
    
    Priority order (most reliable first):
    1. Keyword from Test Planner (e.g., "Select Options By")
    2. Element description contains dropdown keywords
    3. ARIA role (combobox, listbox)
    4. Native <select> tag
    5. CSS class patterns (framework-specific)
    
    Args:
        element_data: Dict with element attributes from browser-use DOM
        element_description: Human-readable description of the element
        
    Returns:
        True if element appears to be a dropdown/select component
    """
    # Priority 1: Element description contains dropdown keywords
    if element_description:
        desc = element_description.lower()
        if any(kw in desc for kw in DROPDOWN_KEYWORDS):
            logger.debug(f"🔽 Dropdown detected via description: '{element_description}'")
            return True
    
    # Priority 2: ARIA role (accessibility standard)
    role = element_data.get('role', '').lower() if element_data else ''
    if role in ('combobox', 'listbox'):
        logger.debug(f"🔽 Dropdown detected via ARIA role: '{role}'")
        return True
    
    # Priority 3: Native <select> tag
    tag = element_data.get('tagName', '').lower() if element_data else ''
    if tag == 'select':
        logger.debug("🔽 Dropdown detected via <select> tag")
        return True
    
    # Priority 4: CSS class patterns (framework-specific)
    classes = element_data.get('className', '').lower() if element_data else ''
    for pattern in DROPDOWN_CSS_PATTERNS:
        if pattern.lower() in classes:
            logger.debug(f"🔽 Dropdown detected via CSS class pattern: '{pattern}'")
            return True
    
    return False


async def _validate_by_coordinates(
    page,
    locator: str,
    expected_coords: tuple,
    tolerance: int = 50
) -> tuple[bool, str]:
    """
    Validate locator by checking if element is at expected coordinates.
    
    This is used as an alternative to text-based validation for dropdowns
    where the visible text may not be in the input element itself.
    
    Args:
        page: Playwright page object (can be page or frame)
        locator: The locator string to validate
        expected_coords: Tuple of (x, y) coordinates from browser-use
        tolerance: Maximum pixel distance for coordinate match (default 50px)
        
    Returns:
        Tuple of (is_match: bool, reason: str)
    """
    try:
        element = page.locator(locator)
        count = await element.count()
        
        if count != 1:
            return False, f"not_unique (count={count})"
        
        box = await element.bounding_box()
        if not box:
            return False, "no_bounding_box"
        
        # Calculate center of element
        center_x = box['x'] + box['width'] / 2
        center_y = box['y'] + box['height'] / 2
        
        # Calculate distance from expected coordinates
        dx = abs(center_x - expected_coords[0])
        dy = abs(center_y - expected_coords[1])
        
        if dx < tolerance and dy < tolerance:
            logger.info(f"   ✅ Coordinate validation PASSED: locator at ({center_x:.0f}, {center_y:.0f}), expected {expected_coords}, delta=({dx:.0f}, {dy:.0f})")
            return True, "coordinate_match"
        else:
            logger.warning(f"   ⚠️ Coordinate validation FAILED: locator at ({center_x:.0f}, {center_y:.0f}), expected {expected_coords}, delta=({dx:.0f}, {dy:.0f})")
            return False, f"coord_mismatch: delta=({dx:.0f}, {dy:.0f})"
            
    except Exception as e:
        logger.warning(f"   ⚠️ Coordinate validation error: {e}")
        return False, f"error: {e}"


async def _shorten_xpath(page, full_xpath: str) -> tuple[str, bool]:
    """
    Find shortest unique suffix of xpath.
    
    Progressively shortens the xpath from left to right until finding
    the shortest suffix that still uniquely identifies the element.
    
    Args:
        page: Playwright page or frame_locator context
        full_xpath: Full xpath string (with or without 'xpath=' prefix)
        
    Returns:
        Tuple of (shortened_xpath, was_shortened)
    """
    # Remove leading 'xpath=' if present and normalize
    xpath = full_xpath.replace('xpath=', '')
    if xpath.startswith('/html'):
        xpath = xpath[1:]  # Remove leading /
    
    parts = xpath.split('/')
    
    # Need at least 2 parts to shorten
    if len(parts) < 3:
        return full_xpath, False
    
    # Try progressively shorter suffixes (right to left)
    # Start from 2nd-to-last segment and work backwards
    for start in range(len(parts) - 2, 0, -1):
        suffix = '//' + '/'.join(parts[start:])
        try:
            count = await page.locator(f"xpath={suffix}").count()
            if count == 1:
                logger.info(f"   ✂️ Shortened xpath: {full_xpath} → {suffix}")
                return f"xpath={suffix}", True
        except Exception:
            continue
    
    logger.debug("   ⚠️ Could not shorten xpath (no unique suffix found)")
    return full_xpath if full_xpath.startswith('xpath=') else f"xpath={full_xpath}", False


def _generate_attribute_css(element_data: dict) -> list[dict]:
    """
    Generate CSS locators from element attributes when no direct locators are available.
    
    This is used as an alternative to long xpaths - tries to create stable
    CSS locators using role, type, and class attributes.
    
    Args:
        element_data: Dict with element attributes from browser-use DOM
        
    Returns:
        List of candidate locator dicts with 'locator', 'priority', 'strategy' keys
    """
    candidates = []
    tag = element_data.get('tagName', '').lower()
    role = element_data.get('role', '')
    classes = element_data.get('className', '')
    input_type = element_data.get('type', '')
    
    # Priority A: role attribute (very stable for accessibility)
    if role:
        locator = f"{tag}[role=\"{role}\"]" if tag else f"[role=\"{role}\"]"
        candidates.append({
            'locator': locator,
            'type': 'role-css',
            'priority': 12,  # New priority slot
            'strategy': f'Role-based CSS ({role})',
            'stability': STABLE
        })
        logger.debug(f"   📋 Generated role-based CSS: {locator}")
    
    # Priority B: type attribute for inputs
    if tag == 'input' and input_type:
        locator = f"input[type=\"{input_type}\"]"
        candidates.append({
            'locator': locator,
            'type': 'type-css',
            'priority': 13,
            'strategy': f'Input type CSS ({input_type})',
            'stability': STABLE
        })
        logger.debug(f"   📋 Generated type-based CSS: {locator}")
    
    # Priority C: Semantic class (if class contains meaningful patterns)
    if classes:
        semantic_patterns = ['input', 'select', 'dropdown', 'combo', 'multiselect', 'picker']
        for cls in classes.split():
            if any(p in cls.lower() for p in semantic_patterns):
                escaped_cls = _escape_css_selector(cls)
                if not escaped_cls:
                    continue  # Skip invalid class names
                locator = f".{escaped_cls}"
                candidates.append({
                    'locator': locator,
                    'type': 'class-css',
                    'priority': 14,
                    'strategy': f'Semantic class CSS (.{cls})',
                    'stability': score_stability('class', cls)
                })
                logger.debug(f"   📋 Generated semantic class CSS: {locator}")
                break  # Use only the first semantic class
    
    return candidates


# ========================================
# SHADOW DOM SUPPORT
# ========================================
# These JavaScript snippets handle Shadow DOM traversal for element detection.
# The helper function is defined once and embedded into each snippet to avoid
# duplication while maintaining self-contained JavaScript execution.

# Shared Shadow DOM traversal helper - embedded into each JS snippet
# This function recursively pierces through shadow roots to find the actual
# element at coordinates. Supports Material UI, Salesforce Lightning, etc.
_SHADOW_DOM_HELPER_JS = """
    function getElementFromPointWithShadow(root, x, y) {
        let element = root.elementFromPoint(x, y);
        if (!element) return null;
        
        // Recursively traverse through shadow roots
        while (element && element.shadowRoot) {
            const shadowElement = element.shadowRoot.elementFromPoint(x, y);
            if (shadowElement && shadowElement !== element) {
                element = shadowElement;
            } else {
                break;
            }
        }
        return element;
    }
"""

# Check if element exists at coordinates (returns boolean)
SHADOW_DOM_ELEMENT_FROM_POINT_JS = f"""
(args) => {{
    const {{x, y}} = args;
    {_SHADOW_DOM_HELPER_JS}
    return getElementFromPointWithShadow(document, x, y) ? true : false;
}}
"""

# Get tag name of element at coordinates (returns string or null)
SHADOW_DOM_TAG_NAME_JS = f"""
(args) => {{
    const {{x, y}} = args;
    {_SHADOW_DOM_HELPER_JS}
    const el = getElementFromPointWithShadow(document, x, y);
    return el ? el.tagName.toLowerCase() : null;
}}
"""

# Get full element data at coordinates (returns object or null)
SHADOW_DOM_ELEMENT_DATA_JS = f"""
(args) => {{
    const {{x, y}} = args;
    {_SHADOW_DOM_HELPER_JS}
    const el = getElementFromPointWithShadow(document, x, y);
    if (!el) return null;
    
    // Get text content, preferring innerText (visible text) over textContent
    let textContent = '';
    try {{
        textContent = (el.innerText || el.textContent || '').trim().substring(0, 100);
    }} catch (e) {{
        textContent = '';
    }}
    
    // Get element's bounding rect for coordinates
    const rect = el.getBoundingClientRect();
    
    return {{
        tagName: el.tagName.toLowerCase(),
        id: el.id || '',
        className: el.className || '',
        name: el.getAttribute('name') || '',
        placeholder: el.getAttribute('placeholder') || '',
        ariaLabel: el.getAttribute('aria-label') || '',
        dataTestId: el.getAttribute('data-testid') || el.getAttribute('data-test') || el.getAttribute('data-qa') || '',
        href: el.getAttribute('href') || '',
        type: el.getAttribute('type') || '',
        role: el.getAttribute('role') || '',
        title: el.getAttribute('title') || '',
        textContent: textContent,
        isInShadowDom: el.getRootNode() !== document,
        parentTagName: el.parentElement ? el.parentElement.tagName.toLowerCase() : '',
        parentId: el.parentElement ? (el.parentElement.id || '') : '',
        parentClassName: el.parentElement ? (el.parentElement.className || '') : '',
        coordinates: {{
            x: rect.x + rect.width / 2,
            y: rect.y + rect.height / 2,
            width: rect.width,
            height: rect.height,
            top: rect.top,
            left: rect.left,
            right: rect.right,
            bottom: rect.bottom
        }}
    }};
}}
"""


async def validate_semantic_match(
    node=None,
    expected_text: Optional[str] = None,
    *,
    page=None,
    locator: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Semantic validation that the matched element is the one the user meant.

    Primary path (node is not None): read the canonical "what the LLM saw for this node"
    surface that browser-use 0.12.6 pre-populates on every selector_map entry.
    Probe 16 confirmed this distinguishes target from a wrong-sibling for text-bearing
    buttons, bare inputs, and icon-/aria-only elements — zero CDP round-trips.

    Fallback path (page + locator): count check then evaluate() on the located element.
    Returns (False, ...) immediately when count != 1; otherwise one evaluate() call (~2ms,
    probe 05 pattern). Works with both Playwright Page and FrameLocator callers.
    The haystack includes associated label text (``<label for=>``, wrapping ``<label>``,
    ``aria-labelledby``) so a correctly labeled form control passes without needing a
    placeholder (A2). The node path needs no equivalent: ax_name already carries the
    accessibility-computed name, which resolves labels.

    Anonymous-element behaviour: when expected_text is absent, accept only if the
    haystack is also empty (truly anonymous element — nothing semantic to compare).
    When expected_text IS provided, a match is required regardless of element type;
    nameless icon buttons are accepted only when no expected_text is supplied by the
    caller, so the caller must omit expected_text for icon-only elements rather than
    relying on a blanket carve-out that would accept any interactive element for
    arbitrary text queries.

    Returns (is_match, observed_text).
    """
    haystack = ""
    observed_text = ""
    _haystack_has_content = False  # True when at least one part is non-empty
    _primary_text_len = 0          # Full text length for container ratio check
    _haystack_parts_list: list = []  # Individual fields for per-field word matching

    if node is not None:
        # Primary path — zero CDP round-trips. Read pre-computed attrs.
        ax = getattr(node, "ax_node", None)
        ax_name = (
            getattr(getattr(ax, "name", None), "value", getattr(ax, "name", ""))
            or ""
        )
        meaningful = ""
        try:
            meaningful = node.get_meaningful_text_for_llm() or ""
        except Exception:
            pass
        attrs = node.attributes or {}
        haystack_parts = [
            ax_name,
            meaningful,
            attrs.get("placeholder", ""),
            attrs.get("aria-label", ""),
            attrs.get("value", ""),
            (getattr(node, "node_value", None) or ""),
        ]
        _haystack_has_content = any(str(p).strip() for p in haystack_parts)
        haystack = " | ".join(str(p) for p in haystack_parts).lower()
        observed_text = (meaningful or ax_name).strip()
        _primary_text_len = len(meaningful)
        _haystack_parts_list = [str(p).lower() for p in haystack_parts]

    elif page is not None and locator:
        # Legacy fallback — one evaluate() on the located element.
        # page.locator() works with both Playwright Page and FrameLocator.
        try:
            el = page.locator(locator)
            count = await el.count()
            if count != 1:
                return False, f"[Element count={count}, expected 1]"
            info = await el.evaluate(
                """el => {
                    const tc = (el.textContent || '').trim();
                    const doc = el.ownerDocument;
                    const labelText = el.labels
                        ? Array.from(el.labels)
                            .map(l => (l.textContent || '').trim())
                            .join(' ')
                        : '';
                    const labelledbyText = (el.getAttribute('aria-labelledby') || '')
                        .split(/\\s+/)
                        .filter(Boolean)
                        .map(id => {
                            const n = doc.getElementById(id);
                            return n ? (n.textContent || '').trim() : '';
                        })
                        .join(' ');
                    return {
                        textContent: tc.slice(0, 500),
                        textContentLength: tc.length,
                        innerText: (el.innerText || '').trim().slice(0, 500),
                        placeholder: el.getAttribute('placeholder') || '',
                        ariaLabel: el.getAttribute('aria-label') || '',
                        value: el.value || '',
                        labelText: labelText.slice(0, 500),
                        labelledbyText: labelledbyText.slice(0, 500)
                    };
                }"""
            )
            parts = [
                info.get("textContent", ""),
                info.get("innerText", ""),
                info.get("placeholder", ""),
                info.get("ariaLabel", ""),
                info.get("value", ""),
                info.get("labelText", ""),
                info.get("labelledbyText", ""),
            ]
            _haystack_has_content = any(p.strip() for p in parts)
            haystack = " | ".join(parts).lower()
            # Fallback chain covers every semantic surface so the mismatch
            # log's "got '…'" reflects what the element actually says —
            # empty observed_text now reliably means a surface-less element.
            observed_text = (
                info.get("innerText")
                or info.get("textContent")
                or info.get("ariaLabel")
                or info.get("labelText")
                or info.get("labelledbyText")
                or info.get("placeholder")
                or info.get("value")
                or ""
            ).strip()
            _primary_text_len = info.get("textContentLength", len(info.get("textContent", "")))
            _haystack_parts_list = [p.lower() for p in parts]
        except Exception as e:
            logger.warning(f"   ⚠️ Semantic validation error: {e}")
            return False, f"[Error: {e}]"

    needle = (expected_text or "").strip().lower()
    if not needle:
        # Short-circuit removed. Accept only when haystack is also empty
        # (truly anonymous element — nothing semantic to compare against).
        return (not _haystack_has_content, observed_text)

    # Probe 18 carve-out: SVG-only buttons and CDP fallback nodes have no accessible
    # text surface through no fault of the locator strategy. Rejecting them would silently
    # break delete/edit icon buttons on admin dashboards (demoqa.com/webtables pattern).
    # When the haystack is empty AND the tag is interactive, accept the match.
    # Non-interactive tags (div, span, p) are not carved out — they are more likely to
    # be container false positives and have no interactive affordance to justify this.
    # NOTE: the fast path in find_unique_locator_at_coordinates gates on
    # `children_nodes is not None` to prevent this from firing on CDP fallback nodes in
    # the general locator loop — the carve-out is intentional only for confirmed locators.
    if not _haystack_has_content and node is not None:
        _tag = (
            getattr(node, "tag_name", None)
            or getattr(node, "node_name", None)
            or ""
        ).lower()
        if _tag in {"button", "a", "input", "select", "textarea"}:
            return (True, observed_text)

    # Container rejection: an element whose text surface far exceeds the search term
    # is almost certainly a layout container that incidentally contains the text.
    # _primary_text_len reflects DOM children text (get_all_children_text fallback);
    # elements where get_meaningful_text_for_llm() returns an attribute value (aria-label,
    # value, placeholder) will have a short _primary_text_len and are unaffected.
    _container_threshold = max(len(expected_text) * 40, 500)
    if _primary_text_len > _container_threshold:
        return False, observed_text

    if needle in haystack:
        logger.info(f"   ✅ Semantic match: '{expected_text}' found in element surface")
        return (True, observed_text)

    # Word-level soft match — all significant words must appear in the same field.
    # Handles "login email" matching "Enter your login email" within one field.
    # Per-field prevents cross-field false positives: "delete" (meaningful) + "row"
    # (aria-label) no longer combine to match "delete row".
    words = [w for w in needle.split() if len(w) >= 2]
    if words and any(
        all(w in field for w in words)
        for field in _haystack_parts_list
        if field.strip()
    ):
        logger.info(f"   ✅ Semantic match (word-level): '{expected_text}' matched")
        return (True, observed_text)

    logger.warning(
        f"   ❌ Semantic MISMATCH: expected '{expected_text}', "
        f"got '{observed_text[:MAX_TEXT_CONTENT_LENGTH]}'"
    )
    return (False, observed_text)


# ========================================
# MULTI-ELEMENT COLLECTION DETECTION
# ========================================
# Helpers moved to handlers/collection.py. Re-exported here so in-module
# callers and existing test imports (`from .smart_locator import ...`)
# continue to work unchanged.
from .classifier import classify_element_type
from .handlers import checkbox as _checkbox_handler
from .handlers import collection as _collection_handler
from .handlers import dropdown as _dropdown_handler
from .handlers.checkbox import (
    find_checkbox_or_radio_by_label as _find_checkbox_or_radio_by_label,
    resolve_hidden_input_proxy as _resolve_hidden_input_proxy,
)
from .handlers.dropdown import _xpath_string_literal
from .handlers.collection import (
    _is_collection_element,
    _extract_collection_class,
    _find_collection_locator,
    _find_collection_by_text_traversal,
)


async def _disambiguate_by_coordinates(page, selector: str, x: float, y: float) -> Optional[dict]:
    """
    When multiple elements match a selector, find which one is at or closest to (x, y).
    
    Uses 3-layer approach:
    1. Visible filter - try to reduce to 1 visible element
    2. Bounding box match - find element containing coordinates
    3. Closest distance - fallback if coordinates slightly off
    
    Args:
        page: Playwright page object
        selector: The selector that matched multiple elements
        x, y: Target coordinates
        
    Returns:
        Dict with 'locator' and 'disambiguated': True if found, None otherwise
    """
    try:
        locator = page.locator(selector)
        count = await locator.count()
        
        if count <= 1:
            return None  # Nothing to disambiguate
        
        logger.info(f"   🔍 DISAMBIGUATE: '{selector}' has {count} matches, using coordinates ({x}, {y})")
        
        # Track which selector to use for nth indexing
        base_selector_for_nth = selector
        
        # ========================================
        # Layer 1: Visible Filter
        # ========================================
        try:
            visible_selector = f"{selector} >> visible=true"
            visible_locator = page.locator(visible_selector)
            visible_count = await visible_locator.count()
            
            if visible_count == 1:
                logger.info(f"   ✅ DISAMBIGUATED (visible filter): Only 1 visible element")
                return {'locator': visible_selector, 'disambiguated': True, 'strategy': 'visible_filter'}
            elif visible_count < count:
                # Reduced count, use visible locator for next checks
                # IMPORTANT: Update base_selector_for_nth to use visible filter
                locator = visible_locator
                count = visible_count
                base_selector_for_nth = visible_selector  # FIX: Use visible selector for nth
                logger.info(f"   📉 Visible filter reduced to {visible_count} elements")
        except Exception as e:
            logger.info(f"   Visible filter failed: {e}")
        
        # ========================================
        # Layer 2: Bounding Box Match (exact hit)
        # ========================================
        best_exact_idx = -1
        for i in range(count):
            try:
                element = locator.nth(i)
                box = await element.bounding_box()
                
                if box:
                    # Check if (x, y) is inside this element's bounding box
                    if (box['x'] <= x <= box['x'] + box['width'] and
                        box['y'] <= y <= box['y'] + box['height']):
                        best_exact_idx = i
                        logger.info(f"   ✅ DISAMBIGUATED (bounding box): Coordinates inside element {i}")
                        break
            except Exception as e:
                logger.info(f"   Bounding box check failed for element {i}: {e}")
        
        if best_exact_idx >= 0:
            indexed_selector = f"{base_selector_for_nth} >> nth={best_exact_idx}"
            return {'locator': indexed_selector, 'disambiguated': True, 'strategy': 'bounding_box'}
        
        # ========================================
        # Layer 3: Closest Distance (fallback)
        # ========================================
        min_distance = float('inf')
        closest_idx = -1

        for i in range(count):
            try:
                element = locator.nth(i)
                box = await element.bounding_box()
                
                if box:
                    # Calculate distance to box center
                    center_x = box['x'] + box['width'] / 2
                    center_y = box['y'] + box['height'] / 2
                    distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_idx = i
            except Exception as e:
                logger.info(f"   Distance check failed for element {i}: {e}")
        
        if closest_idx >= 0 and min_distance < COORD_DISTANCE_THRESHOLD:
            indexed_selector = f"{base_selector_for_nth} >> nth={closest_idx}"
            logger.info(f"   ✅ DISAMBIGUATED (closest distance): Element {closest_idx} is {min_distance:.1f}px away")
            return {'locator': indexed_selector, 'disambiguated': True, 'strategy': 'closest_distance', 'distance': min_distance}

        logger.info(f"   ⚠️ DISAMBIGUATION FAILED: No element within {COORD_DISTANCE_THRESHOLD}px (closest: {min_distance:.1f}px)")
        return None
        
    except Exception as e:
        logger.info(f"   Disambiguation error: {e}")
        return None


async def _upgrade_to_visible_only(search_context, locator: str) -> Optional[str]:
    """
    Visibility-aware uniqueness rescue (G2 / Task A).

    Sites routinely keep hidden template DOM in the page permanently —
    closed Bootstrap modals with duplicated ids, mobile/desktop dual
    navs, inactive tab panels. A candidate that is unique among VISIBLE
    elements then fails the raw count()==1 check against its hidden
    twins and the cascade decays toward parent-CSS/positional locators
    (ASTPP audit: up to 19 duplicated ids per page).

    When the candidate matches >1 elements but exactly one is visible,
    return the composite ``{locator} >> visible=true`` instead of
    discarding it. Generic by construction: the ``>>`` chain works for
    every selector engine (css/#id, [attr=...], text=, xpath=), so all
    candidate families take the same path on any site. The visibility
    filter re-evaluates at RF runtime and selects whichever copy is
    visible then — the QA intent ("the Save you can see"). If several
    copies are visible at runtime the test fails loudly on strict-mode
    ambiguity rather than silently clicking a wrong element.

    Stability is a property of the base candidate: ``visible=true``
    encodes no DOM order, so callers keep the base stability score
    (classify_locator agrees — no positional pattern matches).

    Returns the composite locator string, or None when the upgrade does
    not apply (0 or 2+ visible matches, or the probe errored).
    """
    composite = f"{locator} >> visible=true"
    try:
        visible_count = await search_context.locator(composite).count()
    except Exception as e:
        logger.debug(f"   Visible-only upgrade check failed for {locator!r}: {e}")
        return None

    if visible_count == 1:
        logger.info(
            f"   👁️ VISIBILITY UPGRADE: '{locator}' matches multiple elements "
            f"but exactly one is visible → '{composite}'"
        )
        return composite
    return None


async def _upgrade_to_row_anchor(
    search_context,
    locator: str,
    row_anchor_text: Optional[str],
    x: Optional[float] = None,
    y: Optional[float] = None,
) -> Optional[dict]:
    """
    Row-anchored uniqueness rescue (G1/G8 / Task B).

    Data grids repeat identical action controls on every row (ASTPP:
    ``a[title="Edit"]`` x12). The only thing that distinguishes "Edit
    customer 64625" from the other Edit links is the ROW'S DATA, and
    only the QA step knows which datum identifies the row — so the
    anchor comes from the vision agent (``row_anchor_text``), never
    from guessing a cell value that may be data-bound (balance,
    timestamps).

    When the candidate matches >1 elements, try the row-scoped chain
    ``tr:has-text("{anchor}") >> {candidate}`` (then ``li:`` for
    repeated card/list layouts). Emit it only when exactly one element
    matches — rescued by the visible-only filter when a hidden template
    row duplicates the anchor — and, when coordinates are available,
    only when that element is the one the vision model actually saw.

    XPath-shaped candidates are skipped: whether a chained
    ``xpath=//...`` resolves relative to the row or from the document
    root depends on the selector-engine version (verified relative in
    our pinned Python Playwright, but the generated test runs under RF
    Browser's own bundled engine). A locator whose meaning can differ
    between discovery and runtime is exactly what this feature exists
    to eliminate, and xpath candidates are last-resort priority anyway.

    Anchored-on-QA-data is as stable as the QA case itself: callers
    keep the base candidate's stability score (classify_locator agrees
    — ``:has-text()`` carries no positional pattern). If the anchor
    data changes rows at RF runtime, the locator follows the data; if
    a second row ever matches, the test fails loudly on strict-mode
    ambiguity instead of silently acting on a wrong row.

    Returns:
        ``{'locator': composite}`` on success;
        ``{'ambiguous': True}`` when the anchor matches several rows
        (Option 1: caller falls through to existing behavior and flags
        the payload ``row_anchor_ambiguous`` — demote, never delete);
        ``None`` when the upgrade does not apply.
    """
    anchor = (row_anchor_text or '').strip()
    if not anchor:
        return None
    if locator.startswith(('xpath=', '//', '(//')):
        logger.debug(
            f"   Row-anchor upgrade skipped for xpath candidate {locator!r} "
            f"(chained-xpath scoping is engine-version-dependent)"
        )
        return None

    escaped = anchor.replace('\\', '\\\\').replace('"', '\\"')

    for container in ('tr', 'li'):
        composite = f'{container}:has-text("{escaped}") >> {locator}'
        try:
            count = await search_context.locator(composite).count()
        except Exception as e:
            logger.debug(f"   Row-anchor probe failed for {composite!r}: {e}")
            return None

        if count == 0:
            continue

        if count > 1:
            # A hidden template row (modal grid copy) may duplicate the
            # anchor — unique-among-visible still wins (same rescue as G2).
            visible = await _upgrade_to_visible_only(search_context, composite)
            if visible:
                composite = visible
            else:
                logger.info(
                    f"   ⚠️ ROW ANCHOR AMBIGUOUS: '{anchor}' matches "
                    f"{count} elements via {container}-scoped chain — "
                    f"falling through (flagged)"
                )
                return {'ambiguous': True}

        # Exactly one match — when vision coordinates are available,
        # require them to land on it: a unique match in a DIFFERENT row
        # (anchor text elsewhere) must not be emitted.
        if x is not None and y is not None:
            is_match, reason = await _singleton_matches_coordinates(
                search_context, composite, x, y
            )
            if not is_match:
                logger.info(
                    f"   ⚠️ Row-anchored match is not the element vision "
                    f"saw ({reason}) — skipping row anchor"
                )
                return None

        logger.info(
            f"   📌 ROW ANCHOR UPGRADE: '{locator}' anchored to the row "
            f"containing '{anchor}' → '{composite}'"
        )
        return {'locator': composite}

    return None


async def _singleton_matches_coordinates(page, selector: str, x: float, y: float) -> tuple[bool, str]:
    """
    Identity check for a count==1 text match: is it the element the vision
    model saw at (x, y)? (A5)

    Same contract as multi-match disambiguation: the coordinates must fall
    inside the element's bounding box, or its center must be within
    COORD_DISTANCE_THRESHOLD. A hidden element (no bounding box) can never
    be what vision saw. Fails open on errors — the text already matched,
    and a transient check failure is not evidence of a wrong element.

    Returns:
        Tuple of (is_match: bool, reason: str)
    """
    try:
        box = await page.locator(selector).bounding_box()
        if not box:
            return False, "hidden (no bounding box)"

        if (box['x'] <= x <= box['x'] + box['width'] and
                box['y'] <= y <= box['y'] + box['height']):
            return True, "coords inside bounding box"

        center_x = box['x'] + box['width'] / 2
        center_y = box['y'] + box['height'] / 2
        distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        if distance < COORD_DISTANCE_THRESHOLD:
            return True, f"center {distance:.0f}px from coords"
        return False, f"center {distance:.0f}px from coords (limit {COORD_DISTANCE_THRESHOLD}px)"
    except Exception as e:
        logger.warning(f"   ⚠️ Singleton identity check errored — accepting unchecked: {e}")
        return True, f"identity check skipped ({e})"


async def _checkbox_evidence_corroborated(
    probe_page,
    x: Optional[float],
    y: Optional[float],
    iframe_context: Optional[str],
    prefer_radio: bool,
) -> bool:
    """
    Second source of truth for the text-first checkbox/radio detour:
    does the live DOM actually have checkbox/radio structure at the
    click coordinates?

    Mirrors the STEP-0 dispatcher contract (classifier + probe must
    agree) for the path where no element_data exists: the description /
    vision hint is source one, this probe is source two.

    Returns True when the probe confirms. Inside iframes the probe's
    ``page.evaluate`` runs on the top document and can't see the frame's
    DOM — there the claim (description keywords or vision hint) gates
    alone, preserving iframe checkbox support.
    """
    if iframe_context:
        return True
    if probe_page is None or x is None or y is None:
        return False

    from .dom_probe import probe_specialized_type
    order = ("radio", "checkbox") if prefer_radio else ("checkbox", "radio")
    for suspected in order:
        result = await probe_specialized_type(
            probe_page, suspected, coords=(x, y)
        )
        if result["confirmed"]:
            return True
    return False


async def _find_element_by_expected_text(
    page,
    expected_text: str,
    element_description: str,
    x: float = None,
    y: float = None,
    vision_type_hint: Optional[str] = None,
    probe_page=None,
    iframe_context: Optional[str] = None,
    row_anchor_text: Optional[str] = None,
) -> Optional[dict]:
    """
    Try to find element directly by the expected visible text.
    This is the TEXT-FIRST approach - more reliable than coordinates.
    
    CHECKBOX/RADIO DETOUR: when the step gives real checkbox/radio
    evidence (description keywords or vision hint) AND the DOM probe
    confirms matching structure at the click coordinates, return the
    actual input element instead of the text label — clicking bare label
    text doesn't toggle inputs without a proper <label> association.
    The detour never triggers on text shape alone (A1): a short label
    like "Page 2" must not start a checkbox hunt.

    Args:
        page: Locator search context (Playwright page or frame_locator)
        expected_text: The actual text AI sees on the element
        element_description: Human-readable description (for context)
        x, y: Click coordinates from browser-use (probe corroboration)
        vision_type_hint: LLM's visual classification for the element
        probe_page: Page-level object for the DOM probe (page.evaluate
            doesn't exist on frame_locator search contexts)
        iframe_context: Iframe locator when the element is inside a
            frame — the probe can't see into frames, so description or
            vision evidence alone gates the detour there

    Returns:
        Dict with 'locator' and optionally 'element_type' if found, None otherwise.
        For backward compatibility, returns string locator for non-checkbox elements.
    """
    if not expected_text or len(expected_text.strip()) < MIN_TEXT_LENGTH:
        return None
    
    text = expected_text.strip()
    desc_lower = element_description.lower() if element_description else ""
    
    logger.info(f"🔍 TEXT-FIRST: Searching for element with text '{text}'")
    
    # ========================================
    # SPECIAL HANDLING: Checkbox/Radio Elements
    # ========================================
    # Two-source gate (same contract as the STEP-0 dispatcher): the
    # detour runs only when the step CLAIMS a checkbox/radio (description
    # keywords or vision hint) AND the live DOM probe finds matching
    # structure at the click coordinates. Text shape (length, leading
    # word) is not evidence — that heuristic sent 23% of text searches
    # into checkbox hunting with a 0% hit rate and could return a
    # checkbox for a pagination step (A1/C7).

    # Early exit for obvious non-form elements
    skip_checkbox_check = False
    if element_description:
        # Keywords that clearly indicate non-form elements
        non_form_keywords = ['button', 'link', 'heading', 'title', 'paragraph', 'span', 'div text', 'label text', 'banner', 'menu item']
        if any(keyword in desc_lower for keyword in non_form_keywords):
            skip_checkbox_check = True
            logger.info(f"   ⏩ Skipping checkbox detection - element is clearly not a form input")

    if not skip_checkbox_check:
        # 'select the' intentionally absent: it's dropdown phrasing and
        # dragged select steps into checkbox hunting.
        is_checkbox_context = any(keyword in desc_lower for keyword in [
            'checkbox', 'check box', 'radio', 'toggle', 'check the',
            'tick', 'untick', 'check mark', 'input element for'
        ])
        hint_normalized = (vision_type_hint or '').lower().strip()
        hint_is_checkbox = hint_normalized in ('checkbox', 'radio')

        if is_checkbox_context or hint_is_checkbox:
            prefer_radio = 'radio' in desc_lower or hint_normalized == 'radio'
            corroborated = await _checkbox_evidence_corroborated(
                probe_page=probe_page, x=x, y=y,
                iframe_context=iframe_context, prefer_radio=prefer_radio,
            )
            if corroborated:
                logger.info(f"   🎯 Checkbox/Radio context detected - checking for input element")
                checkbox_result = await _find_checkbox_or_radio_by_label(page, text)

                if checkbox_result:
                    # G3: a styled switch's real input can be display:none —
                    # discovery-time JS clicks succeed on it, but the
                    # generated RF Click would time out. Redirect to the
                    # visible clickable and keep the input for state reads.
                    proxy_info = await _resolve_hidden_input_proxy(
                        page, checkbox_result['locator']
                    )
                    if proxy_info:
                        checkbox_result = {
                            **checkbox_result,
                            'hidden_input': True,
                            'input_locator': checkbox_result['locator'],
                        }
                        if proxy_info.get('locator'):
                            checkbox_result['locator'] = proxy_info['locator']
                            checkbox_result['proxy_kind'] = proxy_info['proxy_kind']
                    # Return the checkbox/radio input locator instead of text
                    logger.info(f"   ✅ Returning checkbox/radio locator: {checkbox_result['locator']}")
                    return checkbox_result
                else:
                    logger.info(f"   ⚠️ No checkbox/radio found, falling back to text-based search")
            else:
                logger.info(
                    f"   ⏩ Checkbox/radio claimed but DOM probe found no "
                    f"matching structure at ({x}, {y}) - skipping detour"
                )
    
    # ========================================
    # Standard Text-Based Search
    # ========================================
    # Build list of selectors to try based on expected_text
    selectors_to_try = []
    
    # Exact text match (highest priority)
    selectors_to_try.append(f'text="{text}"')
    
    # Role-based with exact name (very reliable for buttons/links)
    if "button" in desc_lower or any(word in text.lower() for word in ['submit', 'add', 'delete', 'save', 'cancel', 'ok', 'yes', 'no']):
        selectors_to_try.extend([
            f'role=button[name="{text}"]',
            f'button:has-text("{text}")',
        ])
    
    if "link" in desc_lower:
        selectors_to_try.extend([
            f'role=link[name="{text}"]',
            f'a:has-text("{text}")',
        ])
    
    # Generic text-based selectors
    # NOTE: Removed *:has-text() - it's too broad (matches every ancestor container)
    selectors_to_try.extend([
        f'[aria-label="{text}"]',
        f'[title="{text}"]',
        f'[placeholder="{text}"]',
    ])
    
    # Try partial matches if text is long
    if len(text) > 20:
        short_text = text[:20]
        selectors_to_try.extend([
            f'text="{short_text}"',
            # NOTE: Removed *:has-text() for partial text - same issue as above
        ])
    
    # Try each selector
    for selector in selectors_to_try:
        try:
            count = await page.locator(selector).count()
            if count == 1:
                # A singleton is only "lucky" — verify it is the element the
                # vision model saw before trusting it (A5). A hidden duplicate
                # of the text must not shadow the real element; the next
                # selector (e.g. role=button[name=…]) may match it correctly.
                if x is not None and y is not None:
                    is_match, reason = await _singleton_matches_coordinates(page, selector, x, y)
                    if not is_match:
                        logger.info(
                            f"   ⚠️ '{selector}' is unique but {reason} — "
                            f"not the element at ({x}, {y}), trying next selector"
                        )
                        continue
                logger.info(f"   ✅ TEXT-FIRST SUCCESS: Found unique element with '{selector}'")
                # Return as dict for consistency, but no special element_type
                return {'locator': selector}
            elif count > 1:
                # G1: row-anchored rescue comes BEFORE nth= disambiguation —
                # anchoring to the QA-named row data survives reorder; nth=
                # silently acts on a different row when sort/data changes.
                row_ambiguous = False
                if row_anchor_text:
                    row = await _upgrade_to_row_anchor(
                        page, selector, row_anchor_text, x=x, y=y
                    )
                    if row and row.get('locator'):
                        logger.info(f"   ✅ TEXT-FIRST SUCCESS (row-anchored): {row['locator']}")
                        return {'locator': row['locator'], 'row_anchored': True}
                    row_ambiguous = bool(row and row.get('ambiguous'))
                # Try to disambiguate using coordinates if available
                if x is not None and y is not None:
                    result = await _disambiguate_by_coordinates(page, selector, x, y)
                    if result:
                        if row_ambiguous:
                            # Option 1 (owner, 2026-07-08): ambiguous anchor
                            # falls through to today's behavior, flagged so
                            # nlrf can warn — demote, never delete.
                            result['row_anchor_ambiguous'] = True
                        logger.info(f"   ✅ TEXT-FIRST SUCCESS (disambiguated): {result['locator']}")
                        return result
                else:
                    # G2: no coordinates to disambiguate with, but hidden
                    # duplicates must not kill a visible-unique text match.
                    upgraded = await _upgrade_to_visible_only(page, selector)
                    if upgraded:
                        logger.info(f"   ✅ TEXT-FIRST SUCCESS (visible-only): {upgraded}")
                        return {'locator': upgraded}
                    logger.info(f"   ⚠️ Multiple matches ({count}) for: {selector} (no coords for disambiguation)")
            # count == 0: no matches, try next
        except Exception as e:
            logger.info(f"   ⚠️ Selector failed: {selector} - {e}")
            pass
    
    logger.info(f"   ⚠️ TEXT-FIRST: No unique element found for text '{text}'")
    return None


async def _find_element_by_description(
    page, description: str, row_anchor_text: Optional[str] = None
) -> Optional[str]:
    """
    Fallback: Try to find element by its description when coordinates fail.
    Returns the unique locator string if found, None otherwise.
    
    This is used when document.elementFromPoint() returns BODY/HTML,
    which happens when coordinates land in empty space (common with centered layouts).
    
    Strategy: Use Playwright's semantic locators based on the element description.
    This is more reliable than coordinate-based approach since it matches what
    the AI "sees" (text, role, label) rather than pixel positions.
    """
    if not description:
        return None
    
    # Extract key words from description (e.g., "Add Element button" -> ["Add", "Element"])
    # Also handle common variations
    desc_lower = description.lower()
    keywords = description.replace("button", "").replace("link", "").replace("input", "").replace("field", "").strip().split()
    search_text = " ".join(keywords[:3])  # Use first 3 words max
    
    # Also try the full description as-is (without role words)
    full_text = " ".join(keywords)
    
    try:
        # Priority-ordered selectors based on Playwright best practices
        # Using semantic locators that match what the AI "sees"
        selectors_to_try = []
        
        # If description mentions "button", prioritize button locators
        if "button" in desc_lower:
            selectors_to_try.extend([
                f'role=button[name="{search_text}"]',
                f'role=button[name="{full_text}"]',
                f'button:has-text("{search_text}")',
                f'button >> text="{search_text}"',
            ])
        
        # If description mentions "link", prioritize link locators
        if "link" in desc_lower:
            selectors_to_try.extend([
                f'role=link[name="{search_text}"]',
                f'role=link[name="{full_text}"]',
                f'a:has-text("{search_text}")',
            ])
        
        # If description mentions input/field, prioritize input locators
        if "input" in desc_lower or "field" in desc_lower:
            selectors_to_try.extend([
                f'role=textbox[name="{search_text}"]',
                f'input[placeholder*="{search_text}"]',
                f'input[name*="{search_text}"]',
            ])
        
        # Generic selectors that work for any element type
        selectors_to_try.extend([
            f'text="{search_text}"',
            f'text="{full_text}"',
            f'[aria-label*="{search_text}"]',
            f'[title*="{search_text}"]',
            f'[role="button"]:has-text("{search_text}")',
            f'button:has-text("{search_text}")',
            f'a:has-text("{search_text}")',
        ])
        
        # Try each selector
        for selector in selectors_to_try:
            try:
                count = await page.locator(selector).count()
                if count == 1:
                    logger.info(f"   ✅ Found unique element with semantic locator: {selector}")
                    return selector
                elif count > 1:
                    # G1: row-anchored rescue for per-row action controls.
                    # This path has no coordinates, so ambiguity cannot be
                    # flagged here — it falls through unchanged.
                    if row_anchor_text:
                        row = await _upgrade_to_row_anchor(
                            page, selector, row_anchor_text
                        )
                        if row and row.get('locator'):
                            return row['locator']
                    # G2: hidden duplicates — unique-among-visible still wins.
                    upgraded = await _upgrade_to_visible_only(page, selector)
                    if upgraded:
                        return upgraded
                    logger.info(f"   ⚠️ Multiple matches ({count}) for: {selector}")
                # count == 0: no matches, try next
            except Exception as e:
                logger.info(f"   ⚠️ Selector failed: {selector} - {e}")
                pass

        logger.warning(f"   ❌ No unique element found for description: {description}")
        return None
    except Exception as e:
        logger.info(f"   Error in fallback search: {e}")
        return None


# =============================================================================
# ACCESSIBILITY API FALLBACK STRATEGY (STEP 2.5)
# =============================================================================
# These functions use Playwright's accessibility features to generate robust
# role-based locators. This is a fallback that queries the LIVE DOM (not cached
# indices), solving stale index issues and working reliably with dynamic content.
# =============================================================================


async def _find_element_by_playwright_role(
    search_context,
    expected_text: str,
    element_description: str,
    iframe_context: Optional[str] = None
) -> Optional[dict]:
    """
    Find element using Playwright's native getBy* APIs (100% MCP parity).
    
    This is COORDINATE-INDEPENDENT and follows Microsoft's recommended approach.
    Uses the accessibility tree directly without needing pixel coordinates.
    
    Priority order (Microsoft/Playwright MCP recommendation):
    1. getByRole with name - Most stable, matches ARIA roles (exact match)
    2. getByLabel - For form inputs with associated labels
    3. getByPlaceholder - For text inputs with placeholder
    4. getByRole (partial) - Role match with partial name
    5. getByAltText - For images with alt attribute
    6. getByTitle - For elements with title attribute
    
    Args:
        search_context: Page or frame_locator for the target context
        expected_text: The visible text/label to search for
        element_description: Human-readable description for context
        iframe_context: Optional iframe locator for composite locators
        
    Returns:
        Dict with locator, count, and metadata if found, None otherwise
    """
    if not expected_text or len(expected_text.strip()) < MIN_TEXT_LENGTH:
        return None
    
    text = expected_text.strip()
    desc_lower = element_description.lower() if element_description else ""
    
    logger.info(f"   🎯 PLAYWRIGHT ROLE: Trying native Playwright accessibility APIs")
    logger.info(f"      Text: '{text[:40]}...' | Description: '{desc_lower[:40]}...'")
    
    def apply_iframe_prefix(locator: str) -> str:
        if iframe_context and not locator.startswith(iframe_context):
            return f"{iframe_context} >>> {locator}"
        return locator
    
    # Map description keywords to likely roles
    role_hints = []
    if any(kw in desc_lower for kw in ['button', 'submit', 'click', 'press']):
        role_hints.append('button')
    if any(kw in desc_lower for kw in ['link', 'navigate', 'go to']):
        role_hints.append('link')
    if any(kw in desc_lower for kw in ['input', 'text', 'field', 'enter', 'type']):
        role_hints.extend(['textbox', 'combobox', 'searchbox'])
    if any(kw in desc_lower for kw in ['checkbox', 'check', 'tick']):
        role_hints.append('checkbox')
    if any(kw in desc_lower for kw in ['radio', 'option']):
        role_hints.append('radio')
    if any(kw in desc_lower for kw in ['tab']):
        role_hints.append('tab')
    if any(kw in desc_lower for kw in ['menu', 'dropdown']):
        role_hints.extend(['menuitem', 'option'])
    if any(kw in desc_lower for kw in ['heading', 'title']):
        role_hints.append('heading')
    
    # Common ARIA roles to try for interactive elements
    common_roles = ['button', 'link', 'textbox', 'combobox', 'checkbox', 
                    'radio', 'tab', 'menuitem', 'option', 'searchbox']
    
    # Prioritize hinted roles, then try common roles
    roles_to_try = role_hints + [r for r in common_roles if r not in role_hints]
    
    # Track what we tried for debugging
    attempts = []
    
    try:
        # Strategy 1: getByRole with exact name match
        for role in roles_to_try:
            try:
                locator_obj = search_context.get_by_role(role, name=text, exact=True)
                count = await locator_obj.count()
                attempts.append(f"role={role}[name=\"{text}\"] -> {count}")
                
                if count == 1:
                    # Convert to string locator for Robot Framework compatibility
                    locator_str = f'role={role}[name="{text}"]'
                    locator_str = apply_iframe_prefix(locator_str)
                    
                    logger.info(f"   ✅ PLAYWRIGHT ROLE SUCCESS: {locator_str}")
                    return {
                        'locator': locator_str,
                        'count': 1,
                        'unique': True,
                        'role': role,
                        'accessible_name': text,
                        'element_type': role,
                        'strategy': f'playwright_get_by_role_{role}'
                    }
            except Exception:
                pass  # Role not applicable, continue
        
        # Strategy 2: getByLabel - for form inputs
        try:
            locator_obj = search_context.get_by_label(text, exact=True)
            count = await locator_obj.count()
            attempts.append(f"label=\"{text}\" -> {count}")
            
            if count == 1:
                locator_str = f'[aria-label="{text}"]'  # Closest RF equivalent
                locator_str = apply_iframe_prefix(locator_str)
                
                logger.info(f"   ✅ PLAYWRIGHT LABEL SUCCESS: {locator_str}")
                return {
                    'locator': locator_str,
                    'count': 1,
                    'unique': True,
                    'role': 'textbox',  # Most common for labeled inputs
                    'accessible_name': text,
                    'element_type': 'labeled-input',
                    'strategy': 'playwright_get_by_label'
                }
        except Exception:
            pass
        
        # Strategy 3: getByPlaceholder - for text inputs
        try:
            locator_obj = search_context.get_by_placeholder(text, exact=True)
            count = await locator_obj.count()
            attempts.append(f"placeholder=\"{text}\" -> {count}")
            
            if count == 1:
                locator_str = f'[placeholder="{text}"]'
                locator_str = apply_iframe_prefix(locator_str)
                
                logger.info(f"   ✅ PLAYWRIGHT PLACEHOLDER SUCCESS: {locator_str}")
                return {
                    'locator': locator_str,
                    'count': 1,
                    'unique': True,
                    'role': 'textbox',
                    'accessible_name': text,
                    'element_type': 'placeholder-input',
                    'strategy': 'playwright_get_by_placeholder'
                }
        except Exception:
            pass
        
        # Strategy 4: getByRole with partial name match (less strict)
        for role in roles_to_try[:5]:  # Only try top 5 roles for partial match
            try:
                locator_obj = search_context.get_by_role(role, name=text, exact=False)
                count = await locator_obj.count()
                attempts.append(f"role={role}[name~=\"{text}\"] -> {count}")
                
                if count == 1:
                    locator_str = f'role={role}[name="{text}"]'
                    locator_str = apply_iframe_prefix(locator_str)
                    
                    logger.info(f"   ✅ PLAYWRIGHT ROLE (partial) SUCCESS: {locator_str}")
                    return {
                        'locator': locator_str,
                        'count': 1,
                        'unique': True,
                        'role': role,
                        'accessible_name': text,
                        'element_type': role,
                        'strategy': f'playwright_get_by_role_{role}_partial'
                    }
            except Exception:
                pass
        
        # Strategy 5: getByAltText - for images (MCP parity)
        try:
            locator_obj = search_context.get_by_alt_text(text, exact=True)
            count = await locator_obj.count()
            attempts.append(f"alt=\"{text}\" -> {count}")
            
            if count == 1:
                locator_str = f'[alt="{text}"]'
                locator_str = apply_iframe_prefix(locator_str)
                
                logger.info(f"   ✅ PLAYWRIGHT ALT TEXT SUCCESS: {locator_str}")
                return {
                    'locator': locator_str,
                    'count': 1,
                    'unique': True,
                    'role': 'img',
                    'accessible_name': text,
                    'element_type': 'image',
                    'strategy': 'playwright_get_by_alt_text'
                }
        except Exception:
            pass
        
        # Strategy 6: getByTitle - for elements with title attribute (MCP parity)
        try:
            locator_obj = search_context.get_by_title(text, exact=True)
            count = await locator_obj.count()
            attempts.append(f"title=\"{text}\" -> {count}")
            
            if count == 1:
                locator_str = f'[title="{text}"]'
                locator_str = apply_iframe_prefix(locator_str)
                
                logger.info(f"   ✅ PLAYWRIGHT TITLE SUCCESS: {locator_str}")
                return {
                    'locator': locator_str,
                    'count': 1,
                    'unique': True,
                    'role': 'generic',
                    'accessible_name': text,
                    'element_type': 'titled-element',
                    'strategy': 'playwright_get_by_title'
                }
        except Exception:
            pass
        
        # Log attempts for debugging
        logger.info(f"   ⚠️ PLAYWRIGHT ROLE: No unique match found")
        if attempts:
            logger.info(f"      Tried: {', '.join(attempts[:5])}...")
        
        return None
        
    except Exception as e:
        logger.warning(f"   ⚠️ PLAYWRIGHT ROLE error: {e}")
        return None



async def _find_element_via_accessibility_tree(
    page,
    expected_text: Optional[str] = None,
    element_description: Optional[str] = None,
    iframe_context: Optional[str] = None
) -> Optional[dict]:
    """
    STEP 2.5c: Full Accessibility Tree Search (MCP Parity)
    
    Uses Playwright's NATIVE accessibility APIs to search the entire page.
    No custom JavaScript - automatically benefits from Playwright updates.
    
    Capabilities:
    - Full Accessibility Tree: Uses Playwright's built-in getBy* methods
    - Tree Search: Searches all elements by role/text/label
    - Fallback Matching: Exact → Partial → Contains (via regex)
    - No Coordinate Dependency: Pure text/role based search
    
    Args:
        page: Playwright page object
        expected_text: Text to search for in element names
        element_description: Description to derive role hints
        iframe_context: Optional iframe locator prefix
        
    Returns:
        Dict with locator, count, and metadata if found, None otherwise
    """
    import re
    
    if not expected_text:
        return None
    
    logger.info(f"   🌳 STEP 2.5c: Searching via Playwright native APIs for '{expected_text}'")
    
    def apply_iframe_prefix(locator: str) -> str:
        if iframe_context and not locator.startswith(iframe_context):
            return f"{iframe_context} >>> {locator}"
        return locator
    
    # Derive role hints from description
    desc_lower = element_description.lower() if element_description else ""
    role_hints = []
    if any(kw in desc_lower for kw in ['button', 'submit', 'click']):
        role_hints.append('button')
    if any(kw in desc_lower for kw in ['link', 'navigate', 'go to', 'menu']):
        role_hints.append('link')
    if any(kw in desc_lower for kw in ['input', 'text', 'field', 'enter', 'type', 'search']):
        role_hints.extend(['textbox', 'combobox', 'searchbox'])
    if any(kw in desc_lower for kw in ['checkbox', 'check']):
        role_hints.append('checkbox')
    if any(kw in desc_lower for kw in ['dropdown', 'select']):
        role_hints.extend(['combobox', 'listbox'])
    
    # If no role hints from description, try all common roles
    if not role_hints:
        role_hints = ['link', 'button', 'textbox', 'combobox', 'checkbox', 'menuitem', 'tab']
    
    # Create regex pattern for flexible matching
    escaped_text = re.escape(expected_text)
    text_pattern = re.compile(escaped_text, re.IGNORECASE)
    
    try:
        # Strategy 1: Try role-based search with Playwright's get_by_role
        # This uses Playwright's built-in accessibility tree traversal
        for role in role_hints:
            try:
                # Exact match
                locator_obj = page.get_by_role(role, name=expected_text, exact=True)
                count = await locator_obj.count()
                
                if count == 1:
                    safe_name = expected_text.replace('"', '\\"')
                    locator_str = f'role={role}[name="{safe_name}"]'
                    locator_str = apply_iframe_prefix(locator_str)
                    
                    logger.info(f"   ✅ NATIVE API SUCCESS (exact): {locator_str}")
                    return {
                        'locator': locator_str,
                        'count': 1,
                        'unique': True,
                        'role': role,
                        'accessible_name': expected_text,
                        'element_type': role,
                        'strategy': 'playwright_native_exact'
                    }
                
                # Partial match (case-insensitive, substring)
                locator_obj = page.get_by_role(role, name=expected_text, exact=False)
                count = await locator_obj.count()
                
                if count == 1:
                    safe_name = expected_text.replace('"', '\\"')
                    locator_str = f'role={role}[name="{safe_name}"]'
                    locator_str = apply_iframe_prefix(locator_str)
                    
                    logger.info(f"   ✅ NATIVE API SUCCESS (partial): {locator_str}")
                    return {
                        'locator': locator_str,
                        'count': 1,
                        'unique': True,
                        'role': role,
                        'accessible_name': expected_text,
                        'element_type': role,
                        'strategy': 'playwright_native_partial'
                    }
                    
                # Regex match for more flexible matching
                locator_obj = page.get_by_role(role, name=text_pattern)
                count = await locator_obj.count()
                
                if count == 1:
                    safe_name = expected_text.replace('"', '\\"')
                    locator_str = f'role={role}[name="{safe_name}"]'
                    locator_str = apply_iframe_prefix(locator_str)
                    
                    logger.info(f"   ✅ NATIVE API SUCCESS (regex): {locator_str}")
                    return {
                        'locator': locator_str,
                        'count': 1,
                        'unique': True,
                        'role': role,
                        'accessible_name': expected_text,
                        'element_type': role,
                        'strategy': 'playwright_native_regex'
                    }
                    
            except Exception as e:
                logger.debug(f"   Role {role} search failed: {e}")
                continue
        
        # Strategy 2: Try get_by_text (searches visible text content)
        try:
            locator_obj = page.get_by_text(expected_text, exact=True)
            count = await locator_obj.count()
            
            if count == 1:
                # Get the text locator string
                locator_str = f'text="{expected_text}"'
                locator_str = apply_iframe_prefix(locator_str)
                
                logger.info(f"   ✅ NATIVE TEXT SUCCESS: {locator_str}")
                return {
                    'locator': locator_str,
                    'count': 1,
                    'unique': True,
                    'accessible_name': expected_text,
                    'element_type': 'text',
                    'strategy': 'playwright_get_by_text'
                }
        except Exception:
            pass
        
        # Strategy 3: Try get_by_text with partial match
        try:
            locator_obj = page.get_by_text(expected_text, exact=False)
            count = await locator_obj.count()
            
            if count == 1:
                locator_str = f'text="{expected_text}"'
                locator_str = apply_iframe_prefix(locator_str)
                
                logger.info(f"   ✅ NATIVE TEXT (partial) SUCCESS: {locator_str}")
                return {
                    'locator': locator_str,
                    'count': 1,
                    'unique': True,
                    'accessible_name': expected_text,
                    'element_type': 'text',
                    'strategy': 'playwright_get_by_text_partial'
                }
        except Exception:
            pass
        
        # Strategy 4: Try get_by_label (for form elements)
        try:
            locator_obj = page.get_by_label(expected_text, exact=False)
            count = await locator_obj.count()
            
            if count == 1:
                locator_str = f'label="{expected_text}"'
                locator_str = apply_iframe_prefix(locator_str)
                
                logger.info(f"   ✅ NATIVE LABEL SUCCESS: {locator_str}")
                return {
                    'locator': locator_str,
                    'count': 1,
                    'unique': True,
                    'accessible_name': expected_text,
                    'element_type': 'label',
                    'strategy': 'playwright_get_by_label'
                }
        except Exception:
            pass
        
        logger.info(f"   ⚠️ Native API search found no unique match for '{expected_text}'")
        return None
        
    except Exception as e:
        logger.warning(f"   ⚠️ Native API search error: {e}")
        return None



async def _get_element_accessibility_info(
    search_context,
    x: float,
    y: float
) -> Optional[dict]:
    """
    Query the live DOM to get accessibility information for element at coordinates.
    
    Unlike cached element indices, this always reflects the CURRENT page state,
    solving stale index issues after search/filter/AJAX operations.
    
    Args:
        search_context: Page or frame_locator for the target context
        x, y: Coordinates of the target element
        
    Returns:
        Dict with role, accessible name, and other info, or None if failed
    """
    try:
        result = await search_context.evaluate("""
            ({x, y}) => {
                const element = document.elementFromPoint(x, y);
                if (!element) return null;
                
                // Get explicit role or derive from HTML semantics
                const explicitRole = element.getAttribute('role');
                const tagName = element.tagName.toLowerCase();
                
                // Map common HTML elements to implicit ARIA roles
                let implicitRole = null;
                if (tagName === 'button') implicitRole = 'button';
                else if (tagName === 'a' && element.hasAttribute('href')) implicitRole = 'link';
                else if (tagName === 'input') {
                    const type = element.getAttribute('type') || 'text';
                    if (type === 'checkbox') implicitRole = 'checkbox';
                    else if (type === 'radio') implicitRole = 'radio';
                    else if (type === 'submit' || type === 'button') implicitRole = 'button';
                    else if (type === 'search') implicitRole = 'searchbox';
                    else implicitRole = 'textbox';
                }
                else if (tagName === 'textarea') implicitRole = 'textbox';
                else if (tagName === 'select') implicitRole = 'combobox';
                else if (tagName === 'table') implicitRole = 'table';
                else if (tagName === 'tr') implicitRole = 'row';
                else if (tagName === 'td') implicitRole = 'cell';
                else if (tagName === 'th') implicitRole = 'columnheader';
                else if (tagName === 'ul' || tagName === 'ol') implicitRole = 'list';
                else if (tagName === 'li') implicitRole = 'listitem';
                else if (tagName === 'nav') implicitRole = 'navigation';
                else if (tagName === 'dialog') implicitRole = 'dialog';
                else if (['h1','h2','h3','h4','h5','h6'].includes(tagName)) implicitRole = 'heading';
                else if (tagName === 'img') implicitRole = 'img';
                
                const role = explicitRole || implicitRole;
                
                // Get accessible name following W3C AccName algorithm
                let accessibleName = null;
                
                // 1. aria-labelledby
                const labelledBy = element.getAttribute('aria-labelledby');
                if (labelledBy) {
                    const labels = labelledBy.split(' ')
                        .map(id => document.getElementById(id)?.textContent?.trim())
                        .filter(Boolean);
                    if (labels.length) accessibleName = labels.join(' ');
                }
                
                // 2. aria-label
                if (!accessibleName) {
                    accessibleName = element.getAttribute('aria-label');
                }
                
                // 3. Associated label for form elements
                if (!accessibleName && element.id) {
                    const label = document.querySelector(`label[for="${element.id}"]`);
                    if (label) accessibleName = label.textContent?.trim();
                }
                
                // 4. title attribute
                if (!accessibleName) {
                    accessibleName = element.getAttribute('title');
                }
                
                // 5. For buttons/links, use text content
                if (!accessibleName) {
                    const interactiveRoles = ['button', 'link', 'tab', 'menuitem', 'option'];
                    if (interactiveRoles.includes(role) || ['BUTTON', 'A'].includes(element.tagName)) {
                        const text = element.textContent?.trim();
                        if (text && text.length < 100) accessibleName = text;
                    }
                }
                
                // 6. alt for images
                if (!accessibleName && tagName === 'img') {
                    accessibleName = element.getAttribute('alt');
                }
                
                // 7. placeholder for inputs
                if (!accessibleName && ['input', 'textarea'].includes(tagName)) {
                    accessibleName = element.getAttribute('placeholder');
                }
                
                // Check if element is part of a collection
                const collectionRoles = ['row', 'listitem', 'option', 'treeitem', 'menuitem', 'cell', 'gridcell'];
                const isCollection = collectionRoles.includes(role);
                
                return {
                    role: role,
                    accessibleName: accessibleName,
                    tagName: tagName,
                    isCollection: isCollection,
                    id: element.id || null,
                    className: element.className || null
                };
            }
        """, {'x': x, 'y': y})
        
        if result:
            logger.info(f"   🔍 Accessibility info: role={result.get('role')}, name='{str(result.get('accessibleName', ''))[:30]}...'")
        return result
        
    except Exception as e:
        logger.warning(f"   ⚠️ Failed to get accessibility info: {e}")
        return None


async def _find_table_via_accessibility(
    search_context,
    x: float,
    y: float,
    expected_text: Optional[str] = None
) -> Optional[dict]:
    """
    Specialized table/grid detection using accessibility API.
    
    Handles tables that standard CSS selectors miss:
    - React-Table, AG Grid, MUI Table (via role attributes)
    - Custom data grids with proper ARIA markup
    - Dynamic tables after filtering/searching
    """
    try:
        # JavaScript to detect table/grid and extract ALL visible data row texts
        table_info = await search_context.evaluate("""
            ({x, y, expectedText}) => {
                let element = document.elementFromPoint(x, y);
                if (!element) return { found: false };
                
                let current = element;
                let rowElement = null;
                
                while (current && current !== document.body) {
                    const role = current.getAttribute('role');
                    const tag = current.tagName.toLowerCase();
                    
                    if (role === 'row' || tag === 'tr') {
                        rowElement = current;
                    }
                    
                    if (role === 'grid' || role === 'table' || tag === 'table') {
                        const rows = current.querySelectorAll('[role="row"], tr');
                        
                        // Filter out header rows (containing th or with columnheader role)
                        const dataRows = Array.from(rows).filter(r => {
                            const isHeader = r.querySelector('th') || 
                                           r.getAttribute('role') === 'columnheader' ||
                                           r.closest('thead');
                            return !isHeader;
                        });
                        
                        // Extract text content from ALL data rows (skip empty but keep all non-empty)
                        const rowTexts = dataRows
                            .map(r => r.textContent.trim())
                            .filter(text => text.length > 0);  // Skip blank rows
                        
                        // Find the best CSS selector for targeting data rows
                        // Priority: 1) role=row not in header, 2) tbody tr, 3) common table patterns
                        let rowSelector = null;
                        let selectorType = null;
                        
                        // Try React-Table pattern first (most specific)
                        const rtRows = current.querySelectorAll('.rt-tbody .rt-tr-group');
                        if (rtRows.length > 0) {
                            rowSelector = '.rt-tbody .rt-tr-group';
                            selectorType = 'react-table';
                        }
                        // Try standard table tbody rows
                        else if (tag === 'table') {
                            const tbodyRows = current.querySelectorAll('tbody tr');
                            if (tbodyRows.length > 0) {
                                rowSelector = 'tbody tr';
                                selectorType = 'html-table';
                            }
                        }
                        // Try ARIA grid with rowgroup (excludes header)
                        else if (role === 'grid') {
                            const rowgroupRows = current.querySelectorAll('[role="rowgroup"]:not(:first-child) [role="row"]');
                            if (rowgroupRows.length > 0) {
                                rowSelector = '[role="rowgroup"]:not(:first-child) [role="row"]';
                                selectorType = 'aria-grid';
                            } else {
                                // Fallback: all rows except those with th
                                rowSelector = '[role="row"]:not(:has(th)):not([role="columnheader"])';
                                selectorType = 'aria-row-generic';
                            }
                        }
                        // Generic fallback using role=row
                        if (!rowSelector) {
                            rowSelector = '[role="row"]';
                            selectorType = 'role-row-generic';
                        }
                        
                        return {
                            found: true,
                            role: role || (tag === 'table' ? 'table' : null),
                            ariaLabel: current.getAttribute('aria-label'),
                            totalRows: rows.length,
                            dataRowCount: dataRows.length,
                            visibleRowCount: rowTexts.length,
                            hadRowElement: !!rowElement,
                            tag: tag,
                            rowTexts: rowTexts,
                            rowSelector: rowSelector,
                            selectorType: selectorType
                        };
                    }
                    current = current.parentElement;
                }
                // No table/grid found - return debug info about what we found
                return { 
                    found: false, 
                    debugInfo: element ? `tag=${element.tagName.toLowerCase()}, role=${element.getAttribute('role')}, class=${element.className?.substring(0,50)}` : 'no element at coords'
                };
            }
        """, {'x': x, 'y': y, 'expectedText': expected_text})
        
        if not table_info.get('found'):
            # Debug: Log what we found at coordinates
            debug_info = table_info.get('debugInfo', 'No debug info')
            logger.info(f"   ⚠️ No table/grid found at coordinates ({x}, {y})")
            logger.info(f"   🔍 Debug: {debug_info}")
            return None
        
        visible_rows = table_info.get('visibleRowCount', 0)
        data_rows = table_info.get('dataRowCount', 0)
        row_texts = table_info.get('rowTexts', [])
        row_selector = table_info.get('rowSelector', 'role=row')
        selector_type = table_info.get('selectorType', 'generic')
        
        logger.info(f"   📊 Found {table_info.get('role') or 'table'} with {data_rows} data rows ({visible_rows} non-empty)")
        logger.info(f"   📍 Row selector: {row_selector} (type: {selector_type})")
        
        role = table_info.get('role') or 'table'
        aria_label = table_info.get('ariaLabel')
        
        # Verify the row selector works
        actual_count = await search_context.locator(row_selector).count()
        
        if actual_count == 0:
            # Fallback to role=row if specific selector fails
            logger.info(f"   ⚠️ Selector '{row_selector}' found 0 matches, falling back to role=row")
            row_selector = 'role=row'
            actual_count = await search_context.locator(row_selector).count()
        
        if actual_count == 0:
            logger.info(f"   ⚠️ No rows found with any selector")
            return None
        
        logger.info(f"   ✅ Verified: {actual_count} rows found with '{row_selector}'")
        
        # Log first few row texts for debugging (limit to 3)
        if row_texts:
            for i, text in enumerate(row_texts[:3]):
                preview = text[:60] + '...' if len(text) > 60 else text
                logger.info(f"   📝 Row {i+1}: {preview}")
            if len(row_texts) > 3:
                logger.info(f"   📝 ... and {len(row_texts) - 3} more rows")
        
        # Return result with ALL row data for FOR loop validation in Robot Framework
        return {
            'locator': row_selector,  # Locator for ALL data rows (not pre-filtered)
            'count': actual_count,
            'unique': False,  # Always False for table rows (collection)
            'role': 'row',
            'element_type': 'table-rows',  # Triggers FOR loop generation in CrewAI
            'strategy': f'accessibility_table_{selector_type}',
            # Additional metadata for validation
            'row_texts': row_texts,  # Actual text content of each row
            'visible_row_count': visible_rows,
            'validation_text': expected_text,  # Text to validate each row contains
            'table_role': role,
            'table_label': aria_label
        }
        
    except Exception as e:
        logger.warning(f"   ⚠️ Table accessibility detection failed: {e}")
        return None


async def _find_element_via_accessibility(
    page,
    x: float,
    y: float,
    element_description: str,
    expected_text: Optional[str] = None,
    search_context=None,
    iframe_context: Optional[str] = None
) -> Optional[dict]:
    """
    STEP 2.5: Accessibility API Fallback Strategy
    
    Uses Playwright's live DOM query to generate robust role-based locators.
    Queries CURRENT page state (not stale indices), works after search/filter.
    
    Browser Library Compatible Output:
    - role=button[name="Submit"]
    - role=grid
    - role=row
    - role=textbox[name="Search"]
    """
    logger.info("=" * 60)
    logger.info("STEP 2.5: Trying ACCESSIBILITY API fallback")
    logger.info("=" * 60)
    
    ctx = search_context if search_context is not None else page
    desc_lower = element_description.lower() if element_description else ""
    
    def apply_iframe_prefix(locator: str) -> str:
        if iframe_context and not locator.startswith(iframe_context):
            return f"{iframe_context} >>> {locator}"
        return locator
    
    # === STRATEGY 2.5a: COORDINATE-INDEPENDENT (Playwright Native APIs) ===
    # Try this FIRST - uses getByRole, getByLabel, etc. without coordinates
    # This is the Microsoft-recommended approach for accessibility
    if expected_text:
        pw_role_result = await _find_element_by_playwright_role(
            ctx, expected_text, element_description, iframe_context
        )
        if pw_role_result:
            logger.info(f"   ✅ STEP 2.5a SUCCESS (coordinate-independent): {pw_role_result['locator']}")
            return pw_role_result
        logger.info(f"   ⚠️ STEP 2.5a: Coordinate-independent approach failed, trying coordinate-based...")
    
    # === STRATEGY 2.5b: COORDINATE-DEPENDENT (original approach) ===
    # Falls back to querying element at coordinates when no expected_text or 2.5a fails
    
    # Check for table/grid request
    table_keywords = ['table', 'grid', 'row', 'rows', 'cell', 'data', 'result', 'record', 'entry']
    is_table_request = any(kw in desc_lower for kw in table_keywords)
    
    if is_table_request:
        logger.info(f"   🔍 Description suggests table/grid - trying table detection")
        table_result = await _find_table_via_accessibility(ctx, x, y, expected_text)
        
        if table_result:
            table_result['locator'] = apply_iframe_prefix(table_result['locator'])
            if table_result.get('filtered_locator'):
                table_result['filtered_locator'] = apply_iframe_prefix(table_result['filtered_locator'])
            logger.info(f"   ✅ ACCESSIBILITY TABLE: {table_result['locator']}")
            return table_result

    
    acc_info = await _get_element_accessibility_info(ctx, x, y)
    
    if not acc_info or not acc_info.get('role'):
        logger.info(f"   ⚠️ No accessibility role found at coordinates ({x}, {y})")
        return None
    
    role = acc_info['role']
    accessible_name = acc_info.get('accessibleName')
    is_collection = acc_info.get('isCollection', False)
    
    if accessible_name:
        safe_name = accessible_name.replace('"', '\\"')
        locator = f'role={role}[name="{safe_name}"]'
    else:
        locator = f'role={role}'
    
    try:
        count = await ctx.locator(locator).count()
        
        if count == 0 and accessible_name:
            logger.info(f"   ⚠️ Role locator found 0 matches: {locator}")
            
            # Strategy 2.5b-1: Try Playwright's native get_by_role with EXACT match
            try:
                native_locator = ctx.get_by_role(role, name=accessible_name, exact=True)
                native_count = await native_locator.count()
                if native_count == 1:
                    locator = f'role={role}[name="{safe_name}"]'
                    count = native_count
                    logger.info(f"   ✅ Playwright native exact match found: {locator}")
            except Exception:
                pass
            
            # Strategy 2.5b-2: Try Playwright's native get_by_role with PARTIAL match
            if count == 0:
                try:
                    native_locator = ctx.get_by_role(role, name=accessible_name, exact=False)
                    native_count = await native_locator.count()
                    if native_count == 1:
                        locator = f'role={role}[name="{safe_name}"]'
                        count = native_count
                        logger.info(f"   ✅ Playwright native partial match found: {locator}")
                except Exception:
                    pass
            
            # Strategy 2.5b-3: Try CSS contains text selector
            if count == 0:
                try:
                    text_locator = f'{role}:has-text("{safe_name}")'
                    text_count = await ctx.locator(text_locator).count()
                    if text_count == 1:
                        locator = f'role={role}:has-text("{safe_name}")'
                        count = text_count
                        logger.info(f"   ✅ Text contains match found: {locator}")
                except Exception:
                    pass
            
            # Strategy 2.5b-4: Try role with normalized whitespace
            if count == 0:
                try:
                    normalized_name = ' '.join(accessible_name.split())
                    if normalized_name != accessible_name:
                        norm_locator = f'role={role}[name="{normalized_name}"]'
                        norm_count = await ctx.locator(norm_locator).count()
                        if norm_count == 1:
                            locator = norm_locator
                            count = norm_count
                            logger.info(f"   ✅ Normalized whitespace match found: {locator}")
                except Exception:
                    pass
            
            # If all strategies failed, try STEP 2.5c (Full Accessibility Tree Search)
            if count == 0:
                logger.info(f"   ⚠️ All 2.5b matching strategies failed for '{accessible_name}' - trying 2.5c tree search...")
                tree_result = await _find_element_via_accessibility_tree(
                    page, expected_text, element_description, iframe_context
                )
                if tree_result:
                    return tree_result
                return None
        
        if count == 0:
            return None
        
        # Only accept unique locators (count=1) for non-collection elements
        if count > 1 and not is_collection:
            logger.info(f"   ❌ Accessibility locator not unique: {locator} ({count} matches) - trying 2.5c tree search...")
            tree_result = await _find_element_via_accessibility_tree(
                page, expected_text, element_description, iframe_context
            )
            if tree_result:
                return tree_result
            return None
        
        locator = apply_iframe_prefix(locator)
        
        logger.info(f"   ✅ ACCESSIBILITY SUCCESS: {locator} ({count} matches)")
        
        return {
            'locator': locator,
            'count': count,
            'unique': count == 1 and not is_collection,
            'role': role,
            'accessible_name': accessible_name,
            'element_type': role,
            'strategy': 'accessibility_role'
        }
        
    except Exception as e:
        logger.warning(f"   ⚠️ Error validating accessibility locator: {e}")
        return None


async def _find_table_rows_by_description(
    page,
    description: str,
    expected_text: Optional[str] = None
) -> Optional[dict]:
    """
    Find table rows when the description indicates we're looking for table rows.
    
    This handles scenarios like:
    - "all visible data rows"
    - "table rows after filtering"
    - "filtered results in table"
    
    Common table row patterns for different frameworks:
    - React-Table: .rt-tbody .rt-tr-group
    - Standard HTML: table tbody tr
    - ARIA grids: [role="grid"] [role="row"]
    
    Args:
        page: Playwright page object
        description: Element description from CrewAI
        expected_text: Optional text that should appear in the rows
        
    Returns:
        Dict with 'locator', 'count', and 'element_type' if found, None otherwise
    """
    if not description:
        return None
    
    desc_lower = description.lower()
    
    # Keywords that indicate we're looking for table rows (not individual cells)
    table_row_keywords = [
        # Explicit row keywords
        'table row', 'data row', 'table body', 'visible row', 'filtered row',
        'all rows', 'row result', 'matching row', 'search result', 'result row',
        'rows in table', 'rows within', 'data rows',
        # Table-related keywords (when user wants to verify table data)
        'data table', 'main table', 'content table', 'result table',
        'table on', 'table after', 'filtered table', 'search table',
        # Content area patterns (table displaying results)
        'table displaying', 'displaying results', 'content area of the table',
        'table content', 'table results', 'results in table'
    ]
    
    # Check if description mentions table rows
    is_table_row_request = any(keyword in desc_lower for keyword in table_row_keywords)
    
    if not is_table_row_request:
        return None
    
    logger.info(f"🔍 TABLE-ROW-FINDER: Description mentions table rows")
    
    # Common table row locators for different frameworks (ordered by specificity)
    table_row_locators = [
        # React-Table (demoqa, etc.)
        ('.rt-tbody .rt-tr-group', 'react-table-rows'),
        ('.rt-tbody > .rt-tr-group', 'react-table-rows-direct'),
        # Standard HTML tables
        ('table tbody tr', 'html-table-rows'),
        ('table > tbody > tr', 'html-table-rows-direct'),
        # ARIA grids
        ('[role="grid"] [role="row"]:not([role="columnheader"])', 'aria-grid-rows'),
        ('[role="rowgroup"] [role="row"]', 'aria-rowgroup-rows'),
        # Common data table classes
        ('.table-body tr', 'table-body-rows'),
        ('.data-table tbody tr', 'data-table-rows'),
        # AG Grid
        ('.ag-body-viewport .ag-row', 'ag-grid-rows'),
        # Material UI Table
        ('.MuiTableBody-root .MuiTableRow-root', 'mui-table-rows'),
    ]
    
    for locator, locator_type in table_row_locators:
        try:
            count = await page.locator(locator).count()
            
            if count >= 1:
                logger.info(f"   📋 Found {count} rows with: {locator}")
                
                # If expected_text provided, this is a TABLE VERIFICATION scenario
                if expected_text:
                    # Get first word of expected text for partial matching
                    first_word = expected_text.split()[0] if expected_text.split() else expected_text
                    
                    # Build a filtered locator that matches only rows with the text
                    filtered_locator = f'{locator}:has-text("{first_word}")'
                    
                    # Check if any row contains the expected text
                    matching_rows = page.locator(filtered_locator)
                    matching_count = await matching_rows.count()
                    
                    if matching_count >= 1:
                        logger.info(f"   ✅ {matching_count} rows contain '{first_word}'")
                        logger.info(f"   🔍 This is a TABLE-VERIFICATION scenario")
                        
                        # Return enriched metadata for table verification
                        return {
                            'locator': locator,  # Base row locator (matches all rows)
                            'filtered_locator': filtered_locator,  # Locator for rows with text
                            'count': count,  # Total row count
                            'matching_count': matching_count,  # Rows matching filter
                            'filter_text': first_word,  # The text to verify
                            'element_type': 'table-verification',  # Special type for verification
                            'locator_type': locator_type
                        }
                    else:
                        logger.info(f"   ⚠️ Rows found but none contain '{first_word}'")
                        continue
                else:
                    # No expected_text, return basic table-rows type
                    return {
                        'locator': locator,
                        'count': count,
                        'element_type': 'table-rows',
                        'locator_type': locator_type
                    }
                    
        except Exception as e:
            logger.info(f"   ⚠️ Locator failed: {locator} - {e}")
            continue
    
    logger.info(f"   ⚠️ TABLE-ROW-FINDER: No table rows found on page")
    return None


async def _refine_cell_to_clickable_element(
    page,
    cell_locator: str,
    expected_text: str
) -> Optional[str]:
    """
    Refine a table cell locator to find a specific clickable element inside.
    
    When a td contains multiple elements (e.g., "edit" and "delete" links),
    this function attempts to find the exact element matching expected_text.
    
    Refinement Priority (for QA automation best practices):
    1. Links (<a>) - Most common for table actions
    2. Buttons (<button>) - Standard clickable elements
    3. ARIA buttons ([role="button"]) - Custom button implementations
    4. Elements with aria-label (icon buttons)
    5. Elements with title attribute (tooltip elements)
    6. Any element with matching text (last resort)
    
    Args:
        page: Playwright page object
        cell_locator: The td cell locator
        expected_text: The text to find inside the cell
        
    Returns:
        Refined locator string if found, None otherwise
    """
    if not expected_text or not expected_text.strip():
        return None
    
    text = expected_text.strip()
    
    # Refinement strategies in priority order
    # Using >> for Playwright's chained locator syntax
    refinement_strategies = [
        # 1. Links - most common for table actions like "edit", "delete", "view"
        (f'{cell_locator} >> a:has-text("{text}")', 'link'),
        (f'{cell_locator} >> a:text("{text}")', 'link-exact'),
        
        # 2. Buttons - standard clickable elements
        (f'{cell_locator} >> button:has-text("{text}")', 'button'),
        (f'{cell_locator} >> button:text("{text}")', 'button-exact'),
        
        # 3. ARIA buttons - custom button implementations
        (f'{cell_locator} >> [role="button"]:has-text("{text}")', 'aria-button'),
        
        # 4. Icon buttons with aria-label
        (f'{cell_locator} >> [aria-label="{text}" i]', 'aria-label'),
        (f'{cell_locator} >> [aria-label*="{text}" i]', 'aria-label-partial'),
        
        # 5. Elements with title attribute (tooltips)
        (f'{cell_locator} >> [title="{text}" i]', 'title'),
        (f'{cell_locator} >> [title*="{text}" i]', 'title-partial'),
        
        # 6. Input elements with matching value
        (f'{cell_locator} >> input[value="{text}" i]', 'input-value'),
        
        # 7. Any clickable element with text (span, div with onclick, etc.)
        (f'{cell_locator} >> :text("{text}")', 'any-text'),
    ]
    
    logger.info(f"   🔍 Refining cell locator to find clickable element with text '{text}'")
    
    for refined_locator, strategy_name in refinement_strategies:
        try:
            count = await page.locator(refined_locator).count()
            
            if count == 1:
                logger.info(f"   ✅ Refined to {strategy_name}: {refined_locator}")
                return refined_locator
            elif count > 1:
                logger.info(f"   ⚠️ Multiple matches ({count}) for {strategy_name}")
            # count == 0: no matches, try next strategy
            
        except Exception as e:
            logger.info(f"   ⚠️ Refinement failed for {strategy_name}: {e}")
            continue
    
    logger.info(f"   ⚠️ Could not refine cell to specific element, using cell locator")
    return None


async def _find_table_cell_by_structured_info(
    page, 
    table_cell_info: Optional[dict] = None,
    description: str = "",
    expected_text: Optional[str] = None
) -> Optional[dict]:
    """
    Find a table cell element using structured table_cell_info from BrowserUse agent.
    
    This function uses STRUCTURED INPUT from BrowserUse (preferred) rather than parsing
    natural language descriptions with regex (brittle).
    
    ENHANCED: When expected_text is provided and matches content inside the cell,
    attempts to refine the locator to target the specific clickable element (link, button)
    rather than the entire cell. This is critical for cells with multiple actions.
    
    Structured Format (from BrowserUse agent):
    {
        "table_heading": "Example 1",   # Text near/above the table (primary identifier)
        "table_index": 1,               # Fallback: nth table on page (1-indexed)
        "row": 1,                        # Row number (1-indexed)
        "column": 2,                     # Column number (1-indexed)
    }
    
    Args:
        page: Playwright page object
        table_cell_info: Structured dict with table/row/column info (from BrowserUse)
        description: Human-readable description (for logging only)
        expected_text: Optional expected text content for validation AND refinement
        
    Returns:
        Dict with 'locator' and 'element_type' keys if found, None otherwise
    """
    if not table_cell_info:
        logger.info(f"   ⚠️ No structured table_cell_info provided for: {description}")
        return None
    
    # Extract structured info
    table_heading = table_cell_info.get('table_heading')
    table_index = table_cell_info.get('table_index', 1)
    row = table_cell_info.get('row')
    column = table_cell_info.get('column')
    
    # Validate required fields
    if row is None or column is None:
        logger.warning(f"   ⚠️ Missing row ({row}) or column ({column}) in table_cell_info")
        return None
    
    logger.info(f"🔍 TABLE-CELL-FINDER: Using structured info")
    logger.info(f"   📋 Table heading: {table_heading or 'N/A'}, Index: {table_index}")
    logger.info(f"   📋 Row: {row}, Column: {column}")
    if expected_text:
        logger.info(f"   📋 Expected text: '{expected_text}'")
    
    # ========================================
    # Build Locator Strategies for the Cell
    # ========================================
    locators_to_try = []
    
    # Strategy 1: If table_heading provided, find table near that heading
    if table_heading:
        # XPath to find table following a heading with specific text
        locators_to_try.extend([
            # Table following h3 with text
            f'xpath=//h3[contains(text(), "{table_heading}")]/following-sibling::table[1]//tbody/tr[{row}]/td[{column}]',
            # Table following any heading with text
            f'xpath=//*[self::h1 or self::h2 or self::h3 or self::h4][contains(text(), "{table_heading}")]/following-sibling::table[1]//tbody/tr[{row}]/td[{column}]',
            # Table with caption containing text
            f'xpath=//table[.//caption[contains(text(), "{table_heading}")]]//tbody/tr[{row}]/td[{column}]',
        ])
    
    # Strategy 2: Use table_index (nth table on page)
    table_num = table_index if table_index else 1
    locators_to_try.extend([
        # CSS selector with nth-of-type (works with tables having tbody)
        f'table:nth-of-type({table_num}) tbody tr:nth-child({row}) td:nth-child({column})',
        # XPath selector (very reliable for tables)
        f'xpath=(//table)[{table_num}]//tbody/tr[{row}]/td[{column}]',
        # CSS without tbody (some tables don't use tbody)
        f'table:nth-of-type({table_num}) tr:nth-child({row}) td:nth-child({column})',
        # Direct XPath without tbody
        f'xpath=(//table)[{table_num}]//tr[{row}]/td[{column}]',
    ])
    
    # Strategy 3: Using role=table with nth-of-type
    locators_to_try.append(
        f'[role="table"]:nth-of-type({table_num}) [role="row"]:nth-child({row}) [role="cell"]:nth-child({column})'
    )
    
    # Try each locator to find the cell
    for cell_locator in locators_to_try:
        try:
            count = await page.locator(cell_locator).count()
            
            if count == 1:
                # Cell found! Now determine what to return
                
                if expected_text:
                    # Validate that expected_text is somewhere in this cell
                    is_match, actual_text = await validate_semantic_match(None, expected_text, page=page, locator=cell_locator)
                    
                    if not is_match:
                        logger.info(f"   ⚠️ Locator found but text mismatch: {cell_locator}")
                        continue  # Try next locator
                    
                    logger.info(f"   ✅ TABLE-CELL found with text match: {cell_locator}")
                    
                    # ========================================
                    # REFINEMENT: Try to find specific clickable element inside
                    # ========================================
                    # This handles cases like <td><a>edit</a> <a>delete</a></td>
                    # where we want to target the specific "edit" link, not the whole cell
                    
                    refined_locator = await _refine_cell_to_clickable_element(
                        page, cell_locator, expected_text
                    )
                    
                    if refined_locator:
                        # Successfully refined to a specific element inside the cell
                        return {
                            'locator': refined_locator, 
                            'element_type': 'table-cell-element',
                            'cell_locator': cell_locator  # Keep original cell for reference
                        }
                    else:
                        # Refinement failed, return the cell locator
                        # This is correct for cells where the text IS the content (e.g., <td>$45.00</td>)
                        logger.info(f"   📝 Using cell locator (no refinable inner element)")
                        return {'locator': cell_locator, 'element_type': 'table-cell'}
                else:
                    # No expected_text, just return the cell locator
                    logger.info(f"   ✅ TABLE-CELL locator found: {cell_locator}")
                    return {'locator': cell_locator, 'element_type': 'table-cell'}
            
            elif count > 1:
                logger.info(f"   ⚠️ Multiple matches ({count}) for: {cell_locator}")
            # count == 0: no matches, try next
            
        except Exception as e:
            logger.info(f"   ⚠️ Locator failed: {cell_locator} - {e}")
            continue
    
    logger.info(f"   ⚠️ TABLE-CELL-FINDER: No unique locator found for Row {row}, Col {column}")
    return None


def _attach_classifier_metadata(
    result: dict,
    type_info,
    probe_result: Optional[dict],
    vision_type_hint: Optional[str],
) -> None:
    """
    Stamp the classifier verdict + DOM probe verdict + vision hint onto
    a successful handler result. Lets the Code Assembler route on
    structured signals instead of guessing from element_type alone, and
    gives debug tooling a clean trail when classification was wrong.
    """
    if "element_type" not in result:
        result["element_type"] = type_info.primary_type
    result.setdefault("dropdown_framework", type_info.framework or "")
    result["classifier_confidence"] = type_info.confidence
    result["classifier_signals"] = list(type_info.signals)
    if probe_result is not None:
        result["dom_probe_confirmed"] = probe_result["confirmed"]
        result["dom_probe_framework"] = probe_result["framework"]
        result["dom_probe_signals"] = probe_result["signals"]
    if vision_type_hint:
        result["vision_type_hint"] = vision_type_hint


def _build_element_data_candidates(element_data: dict) -> list[dict]:
    """
    Build prioritized locator candidates from browser-use DOM attributes.

    Pure candidate generation for STEP 0 (element-data locators): direct
    attributes first (id, test id, name, aria-label, placeholder), then
    parent-context CSS for elements without id/name. Uniqueness validation
    happens in the caller.
    """
    locator_candidates = []

    # Priority 1: ID (most stable — unless the VALUE is session-generated;
    # stability demotes those below stable candidates in the caller's sort)
    if element_data.get('id'):
        element_id_val = element_data['id']
        id_stability = score_stability('id', element_id_val)
        # Handle numeric IDs with attribute selector
        if element_id_val.isdigit():
            locator_candidates.append({
                'locator': f'[id="{element_id_val}"]',
                'type': 'id-attr',
                'priority': PRIORITY_ID,
                'strategy': 'ID attribute selector (numeric ID)',
                'stability': id_stability
            })
        else:
            locator_candidates.append({
                'locator': f'#{element_id_val}',  # Use CSS ID selector - Playwright native format
                'type': 'id',
                'priority': PRIORITY_ID,
                'strategy': 'ID selector from element_data',
                'stability': id_stability
            })
    
    # Priority 2: test attribute (very stable for testing)
    # dataTestId may have been sourced from data-test rather than data-testid
    # (see _extract_dom_node_attributes) — emit the attribute that actually
    # exists on the element or the selector matches 0 elements.
    if element_data.get('dataTestId'):
        test_attr = element_data.get('dataTestAttr') or 'data-testid'
        locator_candidates.append({
            'locator': f'[{test_attr}="{element_data["dataTestId"]}"]',
            'type': test_attr,
            'priority': PRIORITY_TEST_ID,
            'strategy': f'{test_attr} from element_data',
            'stability': score_stability(test_attr, element_data['dataTestId'])
        })

    # Priority 3: name attribute
    if element_data.get('name'):
        locator_candidates.append({
            'locator': f'[name="{element_data["name"]}"]',
            'type': 'name',
            'priority': PRIORITY_NAME,
            'strategy': 'Name attribute from element_data',
            'stability': score_stability('name', element_data['name'])
        })
    
    # Priority 4: aria-label (with role if available)
    if element_data.get('ariaLabel'):
        aria_label = element_data['ariaLabel']
        # aria-labels carry visible content: "Cart (3 items)" dies at RF
        # runtime when the count changes (B3).
        aria_stability = VOLATILE if is_dynamic_text(aria_label) else STABLE
        role = element_data.get('role')
        if role:
            locator_candidates.append({
                'locator': f'[role="{role}"][aria-label="{aria_label}"]',
                'type': 'aria-role',
                'priority': PRIORITY_ARIA_LABEL,
                'strategy': 'ARIA label + role from element_data',
                'stability': aria_stability
            })
        else:
            locator_candidates.append({
                'locator': f'[aria-label="{aria_label}"]',
                'type': 'aria-label',
                'priority': PRIORITY_ARIA_LABEL,
                'strategy': 'ARIA label from element_data',
                'stability': aria_stability
            })
    
    # Priority 5: placeholder (for inputs)
    if element_data.get('placeholder'):
        locator_candidates.append({
            'locator': f'[placeholder="{element_data["placeholder"]}"]',
            'type': 'placeholder',
            'priority': PRIORITY_PLACEHOLDER,
            'strategy': 'Placeholder attribute from element_data',
            'stability': STABLE
        })
    
    # Priority 5.5: Parent-context CSS locators (for elements without id/name)
    # When element lacks direct id/name but has parent with id/class, generate
    # stable CSS selectors like "#parentId input" or ".parentClass input"
    # This is MORE STABLE than xpath because it uses semantic anchors
    if not element_data.get('id') and not element_data.get('name'):
        tag_name = element_data.get('tagName', '')
        parent_id = element_data.get('parentId', '')
        parent_class = element_data.get('parentClass', '')
        input_type = element_data.get('type', '')
        
        # Build CSS selector using parent context
        escaped_parent_id = _escape_css_selector(parent_id)
        if escaped_parent_id and tag_name:
            # Use parent id + tag name (e.g., "#formContainer input")
            css_locator = f'#{escaped_parent_id} {tag_name}'
            if input_type:
                # Be more specific for inputs (e.g., "#formContainer input[type='text']")
                css_locator = f'#{escaped_parent_id} {tag_name}[type="{input_type}"]'
            locator_candidates.append({
                'locator': css_locator,
                'type': 'parent-id-css',
                'priority': PRIORITY_CSS_PARENT_ID,
                'strategy': f'Parent ID context + tag (#{parent_id} {tag_name})',
                'stability': score_stability('id', parent_id)
            })
            logger.info(f"   📋 Added parent-context CSS: {css_locator}")
        
        elif parent_class and tag_name:
            # Use first significant class from parent (escape special chars)
            first_class = parent_class.split()[0] if ' ' in parent_class else parent_class
            escaped_class = _escape_css_selector(first_class)
            if escaped_class:
                css_locator = f'.{escaped_class} {tag_name}'
                if input_type:
                    css_locator = f'.{escaped_class} {tag_name}[type="{input_type}"]'
                locator_candidates.append({
                    'locator': css_locator,
                    'type': 'parent-class-css',
                    'priority': PRIORITY_CSS_CLASS,
                    'strategy': f'Parent class context + tag (.{first_class} {tag_name})',
                    'stability': score_stability('class', first_class)
                })
                logger.info(f"   📋 Added parent-context CSS: {css_locator}")

    return locator_candidates


async def _generate_locators_from_element_data(
    search_context,  # Can be page or frame_locator when in iframe context
    element_data: dict[str, Any],
    element_id: str,
    element_description: str,
    expected_text: Optional[str] = None,
    iframe_context: Optional[str] = None,  # Pass iframe context to allow xpath for iframe elements
    confirmed_coords: Optional[tuple] = None,  # (x, y) from browser-use for coordinate validation
    vision_type_hint: Optional[str] = None,  # LLM's visual classification (1 of 2 sources of truth)
    vision_framework_hint: Optional[str] = None,  # LLM's framework guess
    page=None,  # Page-level reference for DOM probe (vs. search_context which can be frame_locator)
    row_anchor_text: Optional[str] = None,  # Row-identifying datum from the QA step (G1/Task B)
) -> Optional[dict]:
    """
    Generate and validate locators from element_data extracted from browser-use DOM.
    
    This is the FASTEST approach - we already have element attributes from browser-use DOM,
    so we can immediately try to generate locators without coordinate-based JavaScript.
    
    Priority order (most stable first):
    1. id attribute → id=xxx
    2. data-testid attribute → [data-testid="xxx"]
    3. name attribute → [name="xxx"]
    4. aria-label + role → [role="xxx"][aria-label="yyy"]
    5. placeholder → [placeholder="xxx"]
    6. xpath (from browser-use) → direct xpath
    
    For TABLE elements (td/th with xpath):
    - The xpath already contains precise table cell location
    - e.g., /html/body/.../table/tbody/tr[1]/td[2]
    
    Args:
        search_context: Playwright page or frame_locator object (for iframe elements)
        element_data: Dict with element attributes from browser-use DOM:
                     {tagName, id, name, className, ariaLabel, placeholder, 
                      title, role, dataTestId, xpath, textContent}
        element_id: Element identifier
        element_description: Human-readable description
        expected_text: Optional expected text for semantic validation
        
    Returns:
        Complete result dict if locator found, None otherwise
    """
    if not element_data:
        return None
    
    logger.info(f"🔍 STEP 0: Trying ELEMENT-DATA locators from browser-use DOM")
    logger.info(f"   Tag: <{element_data.get('tagName', '?')}>")
    
    # Track what we have for debugging
    attrs_found = []
    if element_data.get('id'):
        attrs_found.append(f"id='{element_data['id']}'")
    if element_data.get('dataTestId'):
        attrs_found.append(f"data-testid='{element_data['dataTestId']}'")
    if element_data.get('name'):
        attrs_found.append(f"name='{element_data['name']}'")
    if element_data.get('ariaLabel'):
        attrs_found.append(f"aria-label='{element_data['ariaLabel']}'")
    if element_data.get('xpath'):
        attrs_found.append(f"xpath available")
    
    if attrs_found:
        logger.info(f"   Available attributes: {', '.join(attrs_found)}")
    else:
        logger.info(f"   ⚠️ No usable attributes — falling back to classifier/probe path")
    
    # ========================================
    # CLASSIFIER + DOM PROBE + PER-TYPE DISPATCHER
    # ========================================
    # Two-source-of-truth gate: a specialized handler only runs when BOTH
    # the classifier (DOM signals + vision hint) AND a live Playwright
    # DOM probe agree. Vision-only or DOM-only verdicts are insufficient;
    # the probe is the empirical second source.
    #
    # Modes:
    #   - confirmation: classifier verdict is a specialized type → probe
    #     must find structural signals before the handler runs
    #   - discovery: classifier verdict is "unknown" but the vision hint
    #     suggests a specialized type → probe runs anyway; if structure
    #     is found, we promote the verdict and route
    #
    # Probe rejection → fall through to the generic 21-strategy below.
    # See docs/ELEMENT_TYPE_CLASSIFIER_ARCHITECTURE.md §5 + the
    # two-source-of-truth design discussion.
    type_info = classify_element_type(
        element_data,
        element_description,
        vision_type_hint=vision_type_hint,
        vision_framework_hint=vision_framework_hint,
    )
    logger.info(
        f"   🏷️ Classified: {type_info.primary_type}/{type_info.framework} "
        f"(confidence={type_info.confidence}, signals={type_info.signals})"
    )

    # Determine which type to probe and whether we're in confirmation or
    # discovery mode. Discovery mode kicks in when the classifier said
    # "unknown" but the vision hint mapped to a specialized type.
    suspected_type = ""
    probe_mode = ""
    if type_info.primary_type in ("dropdown", "collection", "checkbox", "radio"):
        suspected_type = type_info.primary_type
        probe_mode = "confirmation"
    elif type_info.primary_type == "unknown" and vision_type_hint:
        from .classifier import map_vision_hint
        mapped = map_vision_hint(vision_type_hint)
        if mapped in ("dropdown", "collection", "checkbox", "radio"):
            suspected_type = mapped
            probe_mode = "discovery"

    probe_result = None
    if suspected_type and page is not None:
        from .dom_probe import probe_specialized_type
        probe_result = await probe_specialized_type(
            page=page,
            suspected_type=suspected_type,
            coords=confirmed_coords,
            candidate_xpath=(
                None if iframe_context else element_data.get("xpath") or None
            ),
        )
        logger.info(
            f"   🔍 DOM probe ({probe_mode}, {suspected_type}): "
            f"confirmed={probe_result['confirmed']}, "
            f"framework={probe_result['framework']!r}, "
            f"signals={probe_result['signals'][:5]}"
        )

        if probe_result["confirmed"]:
            # Probe agrees — commit. If probe detected a framework that the
            # classifier didn't (or contradicts the hinted one), prefer
            # the probe's verdict because it came from the live DOM.
            if probe_result["framework"]:
                if (type_info.framework
                        and type_info.framework != probe_result["framework"]):
                    type_info.signals.append(
                        f"probe-overrides-framework:"
                        f"hint={type_info.framework},"
                        f"probe={probe_result['framework']}"
                    )
                type_info.framework = probe_result["framework"]
            type_info.signals.append(f"probe:confirmed({probe_mode})")
            type_info.signals.extend(
                f"probe:{s}" for s in probe_result["signals"][:10]
            )
            # Discovery mode promotion: probe found structure where the
            # classifier saw none. Adopt the suspected type.
            if probe_mode == "discovery":
                type_info.primary_type = suspected_type
                # Discovery is empirical evidence, but only one DOM source
                # confirmed — keep confidence at medium so downstream is
                # aware this isn't a multi-source agreement.
                type_info.confidence = "medium"
        else:
            # Probe rejects — never run the specialized handler. Fall
            # through to the generic 21-strategy. Telemetry: the conflict
            # signal makes silent classification mistakes audible.
            type_info.signals.append(f"probe:rejected({probe_mode})")
            if probe_mode == "confirmation":
                type_info.signals.append(
                    f"probe-conflict:classifier_said={type_info.primary_type},"
                    f"probe_found_no_structure"
                )
                # Demote to unknown so the generic path runs cleanly.
                type_info.primary_type = "unknown"
                type_info.framework = ""
                type_info.confidence = "low"

    # Now dispatch — only specialized types that survived probe corroboration
    # reach their handler.
    if type_info.primary_type == "dropdown":
        dropdown_result = await _dropdown_handler.find_locator(
            page=search_context,
            element_data=element_data,
            type_info=type_info,
            element_id=element_id,
            element_description=element_description,
            expected_text=expected_text,
            search_context=search_context,
            iframe_context=iframe_context,
            confirmed_coords=confirmed_coords,
        )
        if dropdown_result is not None:
            _attach_classifier_metadata(
                dropdown_result, type_info, probe_result, vision_type_hint
            )
            return dropdown_result
        # Else: fall through to the generic 21-strategy below.

    elif type_info.primary_type == "collection":
        collection_result = await _collection_handler.find_locator(
            page=search_context,
            element_data=element_data,
            type_info=type_info,
            element_id=element_id,
            element_description=element_description,
            expected_text=expected_text,
            search_context=search_context,
            iframe_context=iframe_context,
            confirmed_coords=confirmed_coords,
        )
        if collection_result is not None:
            _attach_classifier_metadata(
                collection_result, type_info, probe_result, vision_type_hint
            )
            return collection_result
        # Else: fall through to the generic 21-strategy below.

    elif type_info.primary_type in ("checkbox", "radio"):
        # Phase 2.5 — native (<input type="checkbox|radio">),
        # custom (role="checkbox|radio"), and toggle (role="switch")
        # all route through handlers/checkbox.find_locator(). Returns
        # None on miss for the always-fallback contract.
        checkbox_result = await _checkbox_handler.find_locator(
            page=search_context,
            element_data=element_data,
            type_info=type_info,
            element_id=element_id,
            element_description=element_description,
            expected_text=expected_text,
            search_context=search_context,
            iframe_context=iframe_context,
            confirmed_coords=confirmed_coords,
        )
        if checkbox_result is not None:
            _attach_classifier_metadata(
                checkbox_result, type_info, probe_result, vision_type_hint
            )
            return checkbox_result
        # Else: fall through to the generic 21-strategy below.

    # ========================================
    # SINGLE ELEMENT LOCATOR GENERATION
    # ========================================
    # Generate candidate locators in priority order
    locator_candidates = _build_element_data_candidates(element_data)
    
    # ========================================
    # SMART LOCATOR FALLBACK (when no id/name/aria-label available)
    # ========================================
    # Priority order:
    # 1. Attribute-based CSS (role, type, semantic class) - stable
    # 2. Shortened xpath (unique suffix) - more stable than full xpath
    # 3. Full xpath - last resort, fragile
    
    has_semantic_locators = len(locator_candidates) > 0  # id, name, aria-label, etc.
    
    # STEP A: Try attribute-based CSS from element data (role, type, class)
    # These are more stable than xpath and work even when element has no id
    attr_css_candidates = _generate_attribute_css(element_data)
    if attr_css_candidates:
        logger.info(f"   📋 Generated {len(attr_css_candidates)} attribute-based CSS candidates")
        locator_candidates.extend(attr_css_candidates)
    
    # STEP B: Handle xpath - shorten if possible, use full as last resort
    if element_data.get('xpath'):
        in_iframe = iframe_context is not None
        
        # Only use xpath if:
        # 1. No expected_text provided (can't use TEXT-FIRST), OR
        # 2. We have better locators to try first (id, name, etc.), OR
        # 3. We're inside an iframe
        should_use_xpath = not expected_text or has_semantic_locators or in_iframe
        
        if should_use_xpath:
            full_xpath = element_data['xpath']
            
            # Add shortened xpath with higher priority (more stable)
            shortened_xpath, was_shortened = await _shorten_xpath(search_context, full_xpath)
            if was_shortened:
                locator_candidates.append({
                    'locator': shortened_xpath,
                    'type': 'shortened-xpath',
                    'priority': 15,  # Before full xpath
                    'strategy': 'Shortened XPath (unique suffix)',
                    'stability': classify_locator(shortened_xpath)
                })

            # Add full xpath as last resort with lowest priority
            strategy_note = 'Full XPath' + (' (iframe element)' if in_iframe else ' (last resort)')
            full_xpath_locator = f'xpath={full_xpath}'
            locator_candidates.append({
                'locator': full_xpath_locator,
                'type': 'full-xpath',
                'priority': 19,  # Demoted to last resort
                'strategy': strategy_note,
                'stability': classify_locator(full_xpath_locator)
            })
            
            if in_iframe:
                logger.info(f"   📋 Using xpath for iframe element (attribute CSS or shortened preferred)")
        else:
            # Skip xpath - let TEXT-FIRST (STEP 1) handle with disambiguation
            logger.info(f"   ⏭️ Skipping xpath - expected_text available, will use TEXT-FIRST strategy")
    
    # Sort candidates by stability tier first, then priority (E1): a
    # session-generated id (ext-gen1042) sinks below a stable name= but
    # stays in the list as a last resort — demoted, never deleted, so a
    # volatile-only element is still returned (marked) and found=false
    # cannot rise (locked decision #3).
    locator_candidates.sort(
        key=lambda c: (
            stability_rank(c.get('stability', STABLE)),
            c.get('priority', 100),
        )
    )

    # Try each candidate locator in priority order
    row_anchor_ambiguous_seen = False
    for candidate in locator_candidates:
        locator = candidate['locator']
        try:
            count = await search_context.locator(locator).count()

            # G1: per-row action controls (Edit/Delete icons repeated on
            # every grid row) are only distinguishable by the row's data —
            # anchor to the QA-named row before any other rescue.
            row_anchored = False
            if count > 1 and row_anchor_text:
                _rc = confirmed_coords or (None, None)
                row = await _upgrade_to_row_anchor(
                    search_context, locator, row_anchor_text,
                    x=_rc[0], y=_rc[1],
                )
                if row and row.get('locator'):
                    locator = row['locator']
                    count = 1
                    row_anchored = True
                elif row and row.get('ambiguous'):
                    row_anchor_ambiguous_seen = True

            # G2: hidden duplicates (closed modals, dual navs) must not kill
            # the candidate when exactly one match is visible — upgrade to a
            # visible-only composite and validate that instead.
            visibility_filtered = False
            if count > 1:
                upgraded = await _upgrade_to_visible_only(search_context, locator)
                if upgraded:
                    locator = upgraded
                    count = 1
                    visibility_filtered = True

            if count == 1:
                # SEMANTIC VALIDATION: Verify we found the RIGHT element
                semantic_match = True
                actual_text = ""
                validation_method = "text"
                
                if expected_text:
                    semantic_match, actual_text = await validate_semantic_match(None, expected_text, page=search_context, locator=locator)
                    
                    if not semantic_match:
                        # Form controls (<input>, <select>, <textarea>): the semantic
                        # check above already covers placeholder, aria-label, value,
                        # and associated label text, so a labeled or placeholder-bearing
                        # field passes on its own merits. When it still fails, a unique
                        # structural locator (id, name, data-testid) is accepted anyway —
                        # some fields are genuinely surface-less and structural uniqueness
                        # is the only identification available (A2). The acceptance is
                        # UNVERIFIED and reported as such: semantic_match stays False with
                        # validation_method="form_control_structural". Narrowing the
                        # acceptance to empty-surface fields only is deferred (owner,
                        # 2026-07-04) until these logs show a real wrong-field acceptance
                        # — watch for the warning below with a non-empty surface.
                        is_form_control = element_data.get('tagName', '').lower() in ('input', 'select', 'textarea')

                        # For DROPDOWNS: Use coordinate-based validation instead of text
                        # Dropdowns often have placeholder text in sibling elements, not the input itself
                        is_dropdown = is_dropdown_element(element_data, element_description)

                        if is_form_control:
                            validation_method = "form_control_structural"
                            if actual_text:
                                logger.warning(
                                    f"   ⚠️ {candidate['type']}: form control surface ('{actual_text[:MAX_TEXT_CONTENT_LENGTH]}') "
                                    f"does not match expected '{expected_text}' — accepting unique structural locator UNVERIFIED (A2 carve-out)"
                                )
                            else:
                                logger.info(
                                    f"   ✅ {candidate['type']}: form control with empty semantic surface — accepting on structural uniqueness (unverified)"
                                )
                        elif is_dropdown and confirmed_coords:
                            logger.info(f"   🔽 Dropdown detected - trying coordinate validation instead of text")
                            coord_match, coord_reason = await _validate_by_coordinates(
                                search_context, locator, confirmed_coords
                            )
                            if coord_match:
                                # Accept locator based on coordinates - trust browser-use vision
                                semantic_match = True
                                validation_method = "coordinates"
                                logger.info(f"   ✅ Dropdown validated via coordinates at {confirmed_coords}")
                            else:
                                # Semantic text AND coordinates both flagged the wrong
                                # element — two independent signals are never overridden
                                # (A3). Reject and let later candidates / pipeline steps
                                # find it or fail loudly.
                                logger.info(
                                    f"   ⚠️ {candidate['type']}: coordinate validation also failed ({coord_reason}) "
                                    f"— rejecting despite uniqueness (trying next)"
                                )
                                continue
                        else:
                            # Not a form control or dropdown - standard text mismatch handling
                            logger.info(f"   ⚠️ {candidate['type']}: unique but text mismatch (trying next)")
                            logger.info(f"      Expected: '{expected_text}', Actual: '{actual_text}'")
                            continue  # Try next locator
                
                logger.info(f"   ✅ ELEMENT-DATA locator found: {locator}")
                logger.info(f"      Strategy: {candidate['strategy']}")
                
                # Check if this is a table element
                tag_name = element_data.get('tagName', '').lower()
                element_type = None
                if tag_name in ['td', 'th']:
                    element_type = 'table-cell'
                    logger.info(f"      Element type: table-cell (from <{tag_name}>)")
                elif tag_name == 'tr':
                    element_type = 'table-row'
                    logger.info(f"      Element type: table-row")
                
                candidate_stability = candidate.get('stability', STABLE)
                if candidate_stability == VOLATILE:
                    logger.warning(
                        f"   ⚠️ Emitting VOLATILE locator {locator} — no stable "
                        f"candidate validated; expect breakage in a fresh session"
                    )

                return {
                    'element_id': element_id,
                    'description': element_description,
                    'found': True,
                    'best_locator': locator,
                    'element_type': element_type,
                    'stability': candidate_stability,
                    **({'visibility_filtered': True} if visibility_filtered else {}),
                    **({'row_anchored': True} if row_anchored else {}),
                    **({'row_anchor_ambiguous': True} if row_anchor_ambiguous_seen else {}),
                    'all_locators': [{
                        'type': candidate['type'],
                        'locator': locator,
                        'priority': candidate['priority'],
                        'strategy': candidate['strategy'],
                        'count': count,
                        'unique': True,
                        'valid': True,
                        'validated': True,
                        'semantic_match': semantic_match,
                        'validation_method': validation_method,
                        'stability': candidate_stability,
                        **({'visibility_filtered': True} if visibility_filtered else {}),
                        **({'row_anchored': True} if row_anchored else {})
                    }],
                    'element_info': {
                        'tagName': element_data.get('tagName', ''),
                        'id': element_data.get('id', ''),
                        'textContent': element_data.get('textContent', ''),
                        'actual_text': actual_text,
                        'source': 'element_data'
                    },
                    'coordinates': element_data.get('coordinates', {}),
                    'validation_summary': {
                        'total_generated': len(locator_candidates),
                        'valid': 1,
                        'unique': 1,
                        'validated': 1,
                        'best_type': candidate['type'],
                        'best_strategy': candidate['strategy'],
                        'validation_method': validation_method
                    },
                    'validated': True,
                    'count': count,
                    'unique': True,
                    'valid': True,
                    'semantic_match': semantic_match,
                    'validation_method': 'playwright'
                }
            
            elif count > 1:
                logger.info(f"   ⚠️ {candidate['type']}: not unique (count={count})")
            else:  # count == 0
                logger.info(f"   ⚠️ {candidate['type']}: not found (count=0)")
                
        except Exception as e:
            logger.info(f"   ⚠️ {candidate['type']}: validation failed - {e}")
            continue
    
    logger.info(f"   ⚠️ ELEMENT-DATA: No unique locator found, falling back to other strategies")
    return None


def _build_coordinate_strategies(element_data: dict) -> list[dict]:
    """
    Build the coordinate-fallback locator strategies from extracted element data.

    Pure candidate generation for the 21-strategy cascade (STEP 3): given the
    element attributes read from the DOM at the confirmed coordinates, emit
    every applicable strategy in priority order (lower = better). Uniqueness
    validation happens in _validate_strategy_candidates.
    """
    locator_strategies = []

    # Strategy 1: ID (Priority 1 - Best)
    if element_data['id']:
        locator_strategies.append({
            'type': 'id',
            'locator': f"id={element_data['id']}",
            'priority': PRIORITY_ID,
            'strategy': 'Native ID attribute'
        })

    # Strategy 2: data-testid (Priority 2)
    if element_data['dataTestId']:
        locator_strategies.append({
            'type': 'data-testid',
            'locator': f"data-testid={element_data['dataTestId']}",
            'priority': PRIORITY_TEST_ID,
            'strategy': 'Test ID attribute'
        })

    # Strategy 3: data-test (Priority 2)
    if element_data['dataTest']:
        locator_strategies.append({
            'type': 'data-test',
            'locator': f"data-test={element_data['dataTest']}",
            'priority': PRIORITY_TEST_ID,
            'strategy': 'Test attribute'
        })

    # Strategy 4: data-qa (Priority 2)
    # Playwright has no data-qa selector engine (data-testid/data-test are
    # built-in, data-qa is not) — only the CSS attribute form resolves.
    if element_data['dataQa']:
        data_qa_escaped = element_data['dataQa'].replace('"', '\\"')
        locator_strategies.append({
            'type': 'data-qa',
            'locator': f'[data-qa="{data_qa_escaped}"]',
            'priority': PRIORITY_TEST_ID,
            'strategy': 'QA attribute'
        })

    # Strategy 5: name (Priority 3)
    # Browser Library (Playwright) has no name= engine — only the CSS
    # attribute form resolves.
    if element_data['name']:
        name_escaped = element_data['name'].replace('"', '\\"')
        locator_strategies.append({
            'type': 'name',
            'locator': f'[name="{name_escaped}"]',
            'priority': PRIORITY_NAME,
            'strategy': 'Name attribute'
        })

    # Strategy 6: aria-label (Priority 4)
    if element_data['ariaLabel']:
        aria_label_escaped = element_data['ariaLabel'].replace('"', '\\"')
        locator_strategies.append({
            'type': 'aria-label',
            'locator': f'[aria-label="{aria_label_escaped}"]',
            'priority': PRIORITY_ARIA_LABEL,
            'strategy': 'ARIA label'
        })

    # Strategy 7: placeholder (Priority 5)
    if element_data['placeholder']:
        placeholder_escaped = element_data['placeholder'].replace('"', '\\"')
        locator_strategies.append({
            'type': 'placeholder',
            'locator': f'[placeholder="{placeholder_escaped}"]',
            'priority': PRIORITY_PLACEHOLDER,
            'strategy': 'Placeholder attribute'
        })

    # Strategy 8: title (Priority 5)
    if element_data['title']:
        title_escaped = element_data['title'].replace('"', '\\"')
        locator_strategies.append({
            'type': 'title',
            'locator': f'[title="{title_escaped}"]',
            'priority': PRIORITY_PLACEHOLDER,
            'strategy': 'Title attribute'
        })

    # Strategy 9: Text content (Priority 6)
    if element_data['innerText'] and len(element_data['innerText']) > MIN_TEXT_LENGTH:
        # Escape quotes in text
        text = element_data['innerText'].replace('"', '\\"')
        locator_strategies.append({
            'type': 'text',
            'locator': f'text="{text}"',
            'priority': PRIORITY_TEXT,
            'strategy': 'Visible text content'
        })

    # Strategy 10: Role + Name (Priority 7)
    if element_data['role'] and element_data['innerText']:
        text = element_data['innerText'].replace('"', '\\"')
        locator_strategies.append({
            'type': 'role',
            'locator': f'role={element_data["role"]}[name="{text}"]',
            'priority': PRIORITY_ROLE,
            'strategy': 'ARIA role with name'
        })

    # Strategy 11: CSS with parent ID context (Priority 8)
    # Raw id/class values can carry CSS meta characters (Tailwind w-1/2,
    # md:flex) — escape both, skip when unescapable (same as STEP 0).
    if element_data['parentId'] and element_data['className']:
        first_class = element_data['className'].split(
        )[0] if element_data['className'] else ''
        escaped_parent_id = _escape_css_selector(element_data['parentId'])
        escaped_class = _escape_css_selector(first_class)
        if escaped_parent_id and escaped_class:
            locator_strategies.append({
                'type': 'css-parent-id',
                'locator': f"#{escaped_parent_id} {element_data['tagName']}.{escaped_class}",
                'priority': PRIORITY_CSS_PARENT_ID,
                'strategy': 'CSS with parent ID context'
            })

    # Strategy 12: CSS with nth-child (Priority 9)
    if element_data['siblingIndex'] and element_data['parentClass']:
        first_parent_class = element_data['parentClass'].split(
        )[0] if element_data['parentClass'] else ''
        escaped_parent_class = _escape_css_selector(first_parent_class)
        if escaped_parent_class:
            locator_strategies.append({
                'type': 'css-nth-child',
                'locator': f".{escaped_parent_class} > {element_data['tagName']}:nth-child({element_data['siblingIndex']})",
                'priority': PRIORITY_CSS_NTH_CHILD,
                'strategy': 'CSS with nth-child'
            })

    # Strategy 13: Simple CSS class (Priority 10)
    if element_data['className']:
        first_class = element_data['className'].split(
        )[0] if element_data['className'] else ''
        escaped_class = _escape_css_selector(first_class)
        if escaped_class:
            locator_strategies.append({
                'type': 'css-class',
                'locator': f"{element_data['tagName']}.{escaped_class}",
                'priority': PRIORITY_CSS_CLASS,
                'strategy': 'Simple CSS class'
            })

    # Strategy 14: XPath with parent ID (Priority 11)
    if element_data['parentId']:
        locator_strategies.append({
            'type': 'xpath-parent-id',
            'locator': f"xpath=//*[@id='{element_data['parentId']}']//{element_data['tagName']}",
            'priority': PRIORITY_XPATH_PARENT_ID,
            'strategy': 'XPath with parent ID'
        })

    # Strategy 15: XPath with parent class and position (Priority 12)
    if element_data['parentClass'] and element_data['siblingIndex']:
        first_parent_class = element_data['parentClass'].split(
        )[0] if element_data['parentClass'] else ''
        if first_parent_class:
            locator_strategies.append({
                'type': 'xpath-parent-class-position',
                'locator': f"xpath=//*[contains(@class, '{first_parent_class}')]//{element_data['tagName']}[{element_data['siblingIndex']}]",
                'priority': PRIORITY_XPATH_PARENT_CLASS,
                'strategy': 'XPath with parent class and position'
            })

    # Strategy 16: XPath with text (Priority 13)
    # XPath 1.0 has no backslash escaping — _xpath_string_literal builds a
    # valid literal (concat() when both quote types are present).
    if element_data['innerText'] and len(element_data['innerText']) > MIN_TEXT_LENGTH:
        text_literal = _xpath_string_literal(
            element_data['innerText'][:MAX_TEXT_DISPLAY_LENGTH]
        )
        locator_strategies.append({
            'type': 'xpath-text',
            'locator': f"xpath=//{element_data['tagName']}[contains(text(), {text_literal})]",
            'priority': PRIORITY_XPATH_TEXT,
            'strategy': 'XPath with text content'
        })

    # Strategy 17: XPath with title attribute (Priority 14)
    if element_data['title']:
        title_literal = _xpath_string_literal(element_data['title'])
        locator_strategies.append({
            'type': 'xpath-title',
            'locator': f"xpath=//{element_data['tagName']}[@title={title_literal}]",
            'priority': PRIORITY_XPATH_TITLE,
            'strategy': 'XPath with title attribute'
        })

    # Strategy 18: XPath with href (for links) (Priority 15)
    if element_data['href'] and element_data['tagName'] == 'a':
        # Use partial href match
        href_part = element_data['href'].split('?')[0].split('#')[0]
        # Safe slicing to prevent IndexError when href_part is empty or too short
        if href_part and len(href_part) > 0:
            href_slice = href_part[-MAX_TEXT_DISPLAY_LENGTH:] if len(href_part) >= MAX_TEXT_DISPLAY_LENGTH else href_part
            locator_strategies.append({
                'type': 'xpath-href',
                'locator': f"xpath=//a[contains(@href, '{href_slice}')]",
                'priority': PRIORITY_XPATH_HREF,
                'strategy': 'XPath with href'
            })

    # Strategy 19: XPath with class and position (Priority 16)
    if element_data['className'] and element_data['siblingIndex']:
        first_class = element_data['className'].split(
        )[0] if element_data['className'] else ''
        if first_class:
            locator_strategies.append({
                'type': 'xpath-class-position',
                'locator': f"xpath=(//{element_data['tagName']}[contains(@class, '{first_class}')])[{element_data['siblingIndex']}]",
                'priority': PRIORITY_XPATH_CLASS_POSITION,
                'strategy': 'XPath with class and position'
            })

    # Strategy 20: XPath with multiple attributes (Priority 17)
    if element_data['className'] and element_data['innerText']:
        first_class = element_data['className'].split(
        )[0] if element_data['className'] else ''
        text = element_data['innerText'][:30]
        if first_class and text:
            text_literal = _xpath_string_literal(text)
            locator_strategies.append({
                'type': 'xpath-multi-attr',
                'locator': f"xpath=//{element_data['tagName']}[contains(@class, '{first_class}') and contains(text(), {text_literal})]",
                'priority': PRIORITY_XPATH_MULTI_ATTR,
                'strategy': 'XPath with class and text'
            })

    # Strategy 21: XPath - first of type with class (Priority 18)
    if element_data['className']:
        first_class = element_data['className'].split(
        )[0] if element_data['className'] else ''
        if first_class:
            locator_strategies.append({
                'type': 'xpath-first-of-class',
                'locator': f"xpath=(//{element_data['tagName']}[contains(@class, '{first_class}')])[1]",
                'priority': PRIORITY_XPATH_FIRST_OF_CLASS,
                'strategy': 'XPath - first element with class'
            })

    # Stability annotation (E1): every strategy carries its tier so the
    # caller's ordering, the early-exit, and the PHASE-2 re-ranker all
    # read one verdict.
    for strategy in locator_strategies:
        strategy['stability'] = _score_strategy_stability(strategy, element_data)

    return locator_strategies


def _worst_stability(*tiers: str) -> str:
    """Return the most fragile of the given tiers (highest rank)."""
    return max(tiers, key=stability_rank)


def _score_strategy_stability(strategy: dict, element_data: dict) -> str:
    """
    Classify one STEP-3 strategy by the raw material it embeds.

    Position dominates (nth-child, numeric XPath predicates, ordinal group
    indexes encode today's DOM order — B2); attribute-backed strategies
    score their source VALUE (B1); content-backed strategies check for
    data-bound text like "Cart (3 items)" (B3).
    """
    if is_positional_locator(strategy['locator']):
        return POSITIONAL

    stype = strategy['type']
    inner_text = element_data.get('innerText', '') or ''
    first_class = (element_data.get('className', '') or '').split()[0] \
        if (element_data.get('className', '') or '').strip() else ''

    if stype == 'id':
        return score_stability('id', element_data.get('id', ''))
    if stype == 'data-testid':
        return score_stability('data-testid', element_data.get('dataTestId', ''))
    if stype == 'data-test':
        return score_stability('data-test', element_data.get('dataTest', ''))
    if stype == 'data-qa':
        return score_stability('data-qa', element_data.get('dataQa', ''))
    if stype == 'name':
        return score_stability('name', element_data.get('name', ''))
    if stype == 'aria-label':
        return VOLATILE if is_dynamic_text(element_data.get('ariaLabel', '')) else STABLE
    if stype in ('title', 'xpath-title'):
        return VOLATILE if is_dynamic_text(element_data.get('title', '')) else STABLE
    if stype in ('text', 'xpath-text'):
        return VOLATILE if is_dynamic_text(inner_text) else STABLE
    if stype == 'role':
        return VOLATILE if is_dynamic_text(inner_text) else STABLE
    if stype == 'placeholder':
        return STABLE
    if stype == 'css-parent-id':
        return _worst_stability(
            score_stability('id', element_data.get('parentId', '')),
            score_stability('class', first_class),
        )
    if stype == 'xpath-parent-id':
        return score_stability('id', element_data.get('parentId', ''))
    if stype == 'css-class':
        return score_stability('class', first_class)
    if stype == 'xpath-multi-attr':
        return _worst_stability(
            score_stability('class', first_class),
            VOLATILE if is_dynamic_text(inner_text[:30]) else STABLE,
        )
    if stype == 'xpath-href':
        return STABLE  # query/fragment already stripped by the builder

    return classify_locator(strategy['locator'])


async def _validate_strategy_candidates(
    search_context,
    sorted_strategies: list,
    expected_text: Optional[str] = None,
    row_anchor_text: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
) -> list[dict]:
    """
    Validate strategy candidates for uniqueness against the live DOM.

    Runs count() per candidate in priority order, recording the outcome on
    each. Early-exits once a high-priority candidate (priority <= PRIORITY_NAME:
    ID, test attribute, name) validates as unique - remaining candidates stay
    unvalidated. When expected_text is given, the early-exit additionally
    requires a semantic match: a unique-but-wrong-text id must not stop the
    loop, or Step 5 (authoritative selection) never sees the lower-priority
    strategies that point at the right element.
    """
    validated_locators = []

    for idx, strategy in enumerate(sorted_strategies, 1):
        try:
            # Log strategy attempt (DEBUG level - verbose details)
            logger.info(f"🔍 Strategy {idx}/{len(sorted_strategies)}: {strategy['type']} (priority={strategy['priority']})")
            logger.info(f"   Locator: {strategy['locator']}")
            logger.info(f"   Strategy: {strategy['strategy']}")
            
            # Validate with Playwright
            # NOTE: Use search_context (either page or frame_locator) for validation
            # This ensures iframe locators are validated inside the iframe
            count = await search_context.locator(strategy['locator']).count()

            # G1: row-anchored rescue for per-row action controls — the
            # QA-named row datum scopes the repeated candidate to its row;
            # stability stays that of the base (data-derived anchor).
            if count > 1 and row_anchor_text:
                row = await _upgrade_to_row_anchor(
                    search_context, strategy['locator'], row_anchor_text,
                    x=x, y=y,
                )
                if row and row.get('locator'):
                    strategy = {
                        **strategy,
                        'locator': row['locator'],
                        'row_anchored': True,
                    }
                    count = 1
                elif row and row.get('ambiguous'):
                    # Option 1: fall through flagged — demote, never delete.
                    strategy = {**strategy, 'row_anchor_ambiguous': True}

            # G2: rescue candidates whose only duplicates are hidden DOM
            # (closed modals, dual navs) — unique-among-visible upgrades to
            # a visible-only composite; stability stays that of the base.
            if count > 1:
                upgraded = await _upgrade_to_visible_only(
                    search_context, strategy['locator']
                )
                if upgraded:
                    strategy = {
                        **strategy,
                        'locator': upgraded,
                        'visibility_filtered': True,
                    }
                    count = 1

            # Determine validation status
            is_unique = (count == 1)
            is_valid = (count == 1)  # Only unique locators are valid
            
            validated_locators.append({
                **strategy,
                'count': count,
                'unique': is_unique,
                'valid': is_valid,
                'validated': True,
                'validation_method': 'playwright'
            })

            # Log validation result with detailed status
            if is_unique:
                logger.info(f"   ✅ VALID & UNIQUE: count={count}, unique={is_unique}, valid={is_valid}")
                
                # OPTIMIZATION: Early exit for high-priority unique locators
                # If we found a high-priority unique locator (ID, test-id, name), stop searching
                # Priority 1-3 are considered "high-priority" (ID, test attributes, name)
                # — but only when the VALUE is stable (E1): a volatile id
                # (ext-gen1042) must not stop the cascade before stable
                # lower-priority strategies get validated.
                if (
                    strategy['priority'] <= PRIORITY_NAME  # PRIORITY_NAME = 3
                    and strategy.get('stability', STABLE) == STABLE
                ):
                    semantic_ok = True
                    if expected_text:
                        is_match, observed_text = await validate_semantic_match(
                            None, expected_text,
                            page=search_context, locator=strategy['locator']
                        )
                        if not is_match:
                            semantic_ok = False
                            logger.info(
                                f"   🛑 SEMANTIC VETO on early exit: expected '{expected_text}', "
                                f"got '{observed_text[:MAX_TEXT_CONTENT_LENGTH]}' - continuing validation"
                            )
                    if semantic_ok:
                        logger.info(f"   ⚡ EARLY EXIT: High-priority unique locator found (priority={strategy['priority']})")
                        logger.info(f"   Skipping validation of {len(sorted_strategies) - idx} remaining strategies")
                        break  # Exit the loop early
                    
            elif count > 1:
                logger.info(f"   ❌ NOT UNIQUE: count={count}, unique={is_unique}, valid={is_valid}")
            elif count == 0:
                logger.info(f"   ❌ NOT FOUND: count={count}, unique={is_unique}, valid={is_valid}")
            else:
                logger.info(f"   ⚠️ UNEXPECTED: count={count}, unique={is_unique}, valid={is_valid}")

        except Exception as e:
            logger.warning(f"   ❌ VALIDATION ERROR: {type(e).__name__}: {e}")
            logger.warning(f"   Locator: {strategy['locator']}")
            validated_locators.append({
                **strategy,
                'count': 0,  # Set to 0 instead of None for consistency
                'unique': False,
                'valid': False,
                'validated': False,
                'validation_error': str(e),
                'validation_method': 'playwright'
            })

    return validated_locators


async def find_unique_locator_at_coordinates(
    page,
    x: float,
    y: float,
    element_id: str,
    element_description: str,
    expected_text: Optional[str] = None,
    element_data: Optional[dict] = None,  # Element attributes from browser-use DOM (id, class, text, etc.)
    search_context=None,  # Either page or frame_locator for iframe context
    iframe_context: Optional[str] = None,  # Iframe locator (e.g., 'iframe[id="main"]') for composite locators
    is_collection: Optional[bool] = None,  # Collection flag for multi-element detection
    browser_session=None,  # BrowserSession for resolved_node lookup (DELTA 1)
    vision_type_hint: Optional[str] = None,  # LLM's visual classification (1 of 2 sources of truth)
    vision_framework_hint: Optional[str] = None,  # LLM's framework guess
    row_anchor_text: Optional[str] = None,  # Row-identifying datum from the QA step (G1/Task B)
) -> dict:
    """
    Find a unique locator for an element using a semantic-first approach.

    Strategy Priority (Semantic-First):
    0. ELEMENT DATA: If element_data is provided (from browser-use DOM), generate locators from those attributes
    1. TEXT-FIRST: Semantic locators from expected_text - Most reliable, uses actual visible text
    2. SEMANTIC: Locators from description - Fallback when expected_text not available
    3. COORDINATE: Coordinate-based extraction + 21 strategies - Last resort when semantic fails

    The semantic-first approach is more reliable because:
    - Doesn't depend on viewport size or layout (centered layouts won't break it)
    - Matches what the AI "sees" (text, role, label)
    - Produces more stable locators (text=, role=, aria-label)

    SEMANTIC VALIDATION:
    - If expected_text is provided, we validate that the found element's actual text
      matches the expected text (case-insensitive, substring match)
    - This prevents "unique but wrong element" bugs where coordinates land on wrong element
    
    IFRAME SUPPORT:
    - If iframe_context is provided (e.g., 'iframe[id="main"]'), the element is inside an iframe
    - search_context will be frame_locator instead of page for correct DOM searches
    - Returned locator will be composite format: "iframe_context >>> locator"

    Args:
        page: Playwright page object (for coordinate-based fallback JavaScript)
        x: X coordinate of element center (used as fallback)
        y: Y coordinate of element center (used as fallback)
        element_id: Element identifier (elem_1, elem_2, etc.)
        element_description: Human-readable description (primary source for semantic locators)
        expected_text: The actual visible text AI sees on the element (e.g., "Submit", "Nike Air Max 270").
                      Used for semantic validation AND for text-first locator search.
        element_data: Optional dict with element attributes from browser-use DOM:
                     {"tagName": "a", "id": "", "textContent": "Services", "href": "/services", ...}
        search_context: The context to use for locator searches (page or frame_locator).
                       If None, defaults to page. Use frame_locator for iframe elements.
        iframe_context: Optional iframe locator (e.g., 'iframe[id="main"]') for composite locator generation.

    Returns:
        Dict with best_locator, all_locators, validation_summary, validation_method, semantic_match
    """

    logger.info(f"🎯 Finding unique locator for {element_id}")
    logger.info(f"   Description: '{element_description}'")
    if expected_text:
        logger.info(f"   Expected text: '{expected_text}'")
    if element_data:
        logger.info(f"   Element data from index: <{element_data.get('tagName', '?')}> id='{element_data.get('id', '')}' text='{element_data.get('textContent', '')[:30]}...'")
    if iframe_context:
        logger.info(f"   🖼️ Iframe context: {iframe_context}")
    logger.info(f"   Coordinates: ({x}, {y}) [fallback]")

    # DELTA 1: Resolve the DOM node at (x, y) once via get_dom_element_at_coordinates.
    # Cache-hit path returns a fully-populated selector_map node (children_nodes is a list).
    # CDP-fallback path (element not in cache) returns a minimal node with children_nodes=None
    # and ax_node=None — Step 5 routes such nodes to the slow per-locator evaluate() path.
    resolved_node = None
    if browser_session is not None and expected_text:
        try:
            resolved_node = await browser_session.get_dom_element_at_coordinates(int(x), int(y))
        except Exception as e:
            logger.warning(
                f"get_dom_element_at_coordinates failed at ({x}, {y}) "
                f"expected={expected_text!r}: {e}"
            )
            resolved_node = None

    # ========================================
    # SEARCH CONTEXT: Use iframe context if provided
    # ========================================
    # search_context is either page (for main page elements) or
    # frame_locator (for elements inside iframes)
    if search_context is None:
        search_context = page
    
    # Helper function to create composite locator for iframe elements
    def _make_composite_locator(locator: str) -> str:
        """Prefix locator with iframe context if element is inside iframe."""
        if iframe_context and not locator.startswith(iframe_context):
            return f"{iframe_context} >>> {locator}"
        return locator
    
    # Helper function to apply iframe prefix to entire result
    def _apply_iframe_prefix_to_result(result: dict) -> dict:
        """Apply iframe prefix to best_locator and all entries in all_locators."""
        if not iframe_context:
            return result
        
        # Apply to best_locator
        if result.get('best_locator'):
            result['best_locator'] = _make_composite_locator(result['best_locator'])
        
        # Apply to ALL locators in all_locators array
        for loc in result.get('all_locators', []):
            if loc.get('locator') and not loc['locator'].startswith(iframe_context):
                loc['locator'] = _make_composite_locator(loc['locator'])

        # An ordinal iframe hop (iframe >> nth=N >>> ...) encodes DOM order:
        # the whole composite is positional even when the inner locator is
        # stable (B2).
        best = result.get('best_locator', '')
        if best and is_positional_locator(best):
            result['stability'] = POSITIONAL
            for loc in result.get('all_locators', []):
                loc['stability'] = POSITIONAL

        result['iframe_context'] = iframe_context
        return result
    
    # ========================================
    # PRE-CHECK: Reset element_data if it's the iframe container
    # ========================================
    # When browser-use provides element_data for an iframe but the user wants
    # an element INSIDE the iframe, we must reset element_data so STEP 1/2/3
    # will search inside the iframe for the actual target element.
    if element_data and iframe_context:
        element_tag = element_data.get('tagName', '').lower()
        if element_tag == 'iframe':
            logger.info(f"⚠️ element_data is the iframe container (tagName={element_tag})")
            logger.info(f"   Resetting element_data - will search inside {iframe_context} for actual element")
            element_data = None  # Force STEP 1/2/3 to find element inside iframe
    
    # ========================================
    # APPROACH METRICS: Build base dict for pattern analysis
    # ========================================
    # Captures element characteristics to enable future pattern analysis
    # (e.g., "buttons work better with text_first approach")
    # NOTE: This must be AFTER the iframe reset above to capture the actual
    # target element's characteristics, not the iframe container's.
    _approach_metrics_base = {
        'element_tag': element_data.get('tagName', '').lower() if element_data else '',
        'has_id': bool(element_data.get('id')) if element_data else False,
        'has_text_content': bool(element_data.get('textContent', '').strip()) if element_data else False,
        'element_data_available': bool(element_data),
        'is_collection': is_collection is True,
        'is_in_iframe': bool(iframe_context),
    }
    
    # ========================================
    # STEP 0: ELEMENT-DATA approach (highest priority - FASTEST)
    # ========================================
    # When element_index is provided, we already have element attributes from browser-use DOM.
    # This is the FASTEST approach - no coordinate-based JavaScript needed.
    if element_data:
        result = await _generate_locators_from_element_data(
            search_context, element_data, element_id, element_description, expected_text,
            iframe_context=iframe_context,  # Pass iframe context for proper xpath/CSS handling
            confirmed_coords=(x, y) if x is not None and y is not None else None,  # Use explicit None checks (0 is valid coord)
            vision_type_hint=vision_type_hint,  # 1 of 2 sources of truth (probe is the other)
            vision_framework_hint=vision_framework_hint,
            page=page,  # Page-level for DOM probe (search_context may be frame_locator)
            row_anchor_text=row_anchor_text,  # Row-scoped rescue for per-row actions (G1)
        )
        if result:
            # Add approach metrics for pattern analysis
            result['approach_metrics'] = {
                **_approach_metrics_base,
                'locator_approach': 'element_data',
                'fallback_depth': 1,
                'success': True,
            }
            # Add iframe prefix to best_locator AND all_locators
            return _apply_iframe_prefix_to_result(result)
    # ========================================
    # STEP 0.5: Collection detection (hybrid: is_collection flag + keyword fallback)
    # ========================================
    # Priority 1: Explicit is_collection=True from custom action (most reliable)
    # Priority 2: Fallback keyword detection in description (backup)
    #
    # DESIGN: If CrewAI determined this is a collection, trust that decision
    # and return a multi-element locator even if only 1 element is currently visible.
    
    explicit_collection = is_collection is True
    keyword_collection = _is_collection_element({}, element_description) if element_description else False
    
    is_collection_request = explicit_collection or keyword_collection
    
    if is_collection_request and expected_text:
        logger.info(f"🔍 Step 0.5: Collection detected - trying multi-element approach")
        if explicit_collection:
            logger.info(f"   Detection method: is_collection=True (from custom action)")
        else:
            logger.info(f"   Detection method: collection keywords in description (fallback)")
        
        # Try text-traversal to find collection (works even without element_data)
        collection_result = await _find_collection_by_text_traversal(search_context, expected_text)
        
        if collection_result:
            locator = collection_result.get('locator')
            count = collection_result.get('count', 0)
            
            # DESIGN DECISION: If collection is explicit, return collection locator regardless of count
            # This handles filtered results (search returns 1 row) correctly
            should_return_collection = explicit_collection or count > 1
            
            if should_return_collection:
                logger.info(f"   ✅ COLLECTION locator found: {locator} (count={count})")
                
                # Apply iframe prefix if needed
                if iframe_context:
                    locator = _make_composite_locator(locator)
                
                collection_stability = classify_locator(locator)
                return {
                    'element_id': element_id,
                    'description': element_description,
                    'found': True,
                    'best_locator': locator,
                    'stability': collection_stability,
                    'is_collection': True,
                    'element_type': 'collection',
                    'all_locators': [{
                        'type': 'collection',
                        'locator': locator,
                        'priority': 0,
                        'strategy': 'Collection via text-traversal',
                        'count': count,
                        'unique': False,  # Collections are never unique (even if count==1)
                        'valid': True,
                        'validated': True,
                        'semantic_match': True,
                        'validation_method': 'playwright',
                        'stability': collection_stability
                    }],
                    'element_info': {
                        'expected_text': expected_text,
                        'row_class': collection_result.get('row_class', ''),
                        'explicit_collection': explicit_collection
                    },
                    'coordinates': {'x': x, 'y': y, 'note': 'Collection found - coordinates used as hint only'},
                    'validation_summary': {
                        'total_generated': 1,
                        'valid': 1,
                        'unique': 0,  # Collections are never unique
                        'validated': 1,
                        'best_type': 'collection',
                        'best_strategy': 'Text-traversal collection finder',
                        'validation_method': 'playwright'
                    },
                    'validated': True,
                    'count': count,
                    'unique': False,  # Collections are never unique (even if count==1)
                    'valid': True,
                    'semantic_match': True,
                    'validation_method': 'playwright',
                    # Approach metrics for pattern analysis
                    'approach_metrics': {
                        **_approach_metrics_base,
                        'locator_approach': 'collection',
                        'fallback_depth': 3,
                        'success': True,
                    }
                }
            else:
                logger.info(f"   ⚠️ Collection found but count={count}, no explicit flag, falling through to single-element")
        else:
            logger.info(f"   ⚠️ Collection text-traversal failed, falling through to single-element approaches")
    
    # ========================================
    # STEP 1: Try TEXT-FIRST approach (using expected_text)
    # ========================================
    # This is the MOST RELIABLE approach - uses the actual text AI sees
    # (only runs if not a table-row scenario, or if table-row detection failed)
    if expected_text and expected_text.strip():
        logger.info(f"🔍 Step 1: Trying TEXT-FIRST locators from expected_text: '{expected_text}'")
        
        text_result = await _find_element_by_expected_text(
            search_context, expected_text, element_description, x, y,
            vision_type_hint=vision_type_hint,
            probe_page=page,  # Page-level for the DOM probe (search_context may be a frame_locator)
            iframe_context=iframe_context,
            row_anchor_text=row_anchor_text,  # Row-scoped rescue for per-row actions (G1)
        )
        
        if text_result:
            # text_result is now a dict with 'locator' and optionally 'element_type'
            text_locator = text_result.get('locator')
            element_type = text_result.get('element_type')  # 'checkbox', 'radio', or None
            
            # Add iframe prefix if needed
            if iframe_context:
                text_locator = _make_composite_locator(text_locator)
            
            logger.info(f"✅ TEXT-FIRST locator found: {text_locator}" + (f" (element_type={element_type})" if element_type else ""))
            
            # Determine strategy name based on whether it's a checkbox/radio
            if element_type:
                strategy_name = f'Checkbox/Radio INPUT locator (type={element_type})'
                locator_type = f'{element_type}-input'
            else:
                strategy_name = 'Text-first locator from expected_text'
                locator_type = 'text-first'
            
            # Text-first locators can carry nth= disambiguation (positional)
            # or data-bound text like "Cart (3 items)" (volatile) — classify
            # the finished locator so the payload is honest about it.
            text_first_stability = classify_locator(text_locator)

            # Carry the hidden-input redirect contract (G3) into element_info
            # so the Assembler can click the proxy but read state from the
            # real input.
            _text_element_info = {'expected_text': expected_text}
            if element_type:
                _text_element_info['element_type'] = element_type
            for _g3_key in ('hidden_input', 'input_locator', 'proxy_kind'):
                if _g3_key in text_result:
                    _text_element_info[_g3_key] = text_result[_g3_key]
            return {
                'element_id': element_id,
                'description': element_description,
                'found': True,
                'best_locator': text_locator,
                'stability': text_first_stability,
                # G1 row-anchor contract: anchored composite, or ambiguous
                # anchor that fell through to nth= (nlrf warns on the flag).
                **({'row_anchored': True} if text_result.get('row_anchored') else {}),
                **({'row_anchor_ambiguous': True} if text_result.get('row_anchor_ambiguous') else {}),
                'element_type': element_type,  # NEW: Pass element_type to caller
                'all_locators': [{
                    'type': locator_type,
                    'locator': text_locator,  # Already has iframe prefix from line 2400
                    'priority': 0,
                    'strategy': strategy_name,
                    'count': 1,
                    'unique': True,
                    'valid': True,
                    'validated': True,
                    'semantic_match': True,  # By definition, text-first is semantically correct
                    'validation_method': 'playwright',
                    'stability': text_first_stability
                }],
                'element_info': _text_element_info,
                'coordinates': {'x': x, 'y': y, 'note': 'Not used - text-first approach succeeded'},
                'validation_summary': {
                    'total_generated': 1,
                    'valid': 1,
                    'unique': 1,
                    'validated': 1,
                    'best_type': locator_type,
                    'best_strategy': strategy_name,
                    'validation_method': 'playwright'
                },
                # Top-level validation fields (required by workflow validation)
                'validated': True,
                'count': 1,
                'unique': True,
                'valid': True,
                'semantic_match': True,
                'validation_method': 'playwright',
                # Approach metrics for pattern analysis
                'approach_metrics': {
                    **_approach_metrics_base,
                    'locator_approach': 'text_first',
                    'fallback_depth': 4,
                    'success': True,
                }
            }
        else:
            logger.info(f"⚠️ TEXT-FIRST approach failed - trying table cell locators")
    else:
        logger.info(f"⚠️ No expected_text provided - skipping TEXT-FIRST approach")
    
    # ========================================
    # STEP 2: Try SEMANTIC LOCATORS from description (fallback)
    # ========================================
    # This is a fallback when expected_text is not available or didn't work
    if element_description and element_description.strip():
        logger.info(f"🔍 Step 2: Trying SEMANTIC locators from description: '{element_description}'")
        
        semantic_locator = await _find_element_by_description(
            search_context, element_description, row_anchor_text=row_anchor_text
        )
        
        if semantic_locator:
            # Add iframe prefix if needed
            if iframe_context:
                semantic_locator = _make_composite_locator(semantic_locator)
            
            # If expected_text provided, validate that we found the right element
            semantic_match = True
            actual_text = ""
            validation_method = "playwright"
            accept = True
            if expected_text:
                semantic_match, actual_text = await validate_semantic_match(None, expected_text, page=search_context, locator=semantic_locator)
                accept = semantic_match
                if not semantic_match:
                    logger.warning(f"⚠️ Description-based locator found BUT text doesn't match!")
                    logger.warning(f"   Expected: '{expected_text}'")
                    logger.warning(f"   Actual: '{actual_text}'")
                    logger.info("   Continuing to coordinate-based approach...")
                    # Don't return - continue to try coordinates
                else:
                    logger.info(f"✅ Semantic locator is correct (text matches)")
            else:
                # The locator was built from description keywords alone
                # (substring matchers) and there is no expected_text to check
                # it against. Acceptance unchanged, but reported UNVERIFIED
                # instead of validated (A6).
                semantic_match = False
                validation_method = "description_derived"
                logger.info(
                    "   ⚠️ No expected_text — accepting description-derived locator UNVERIFIED (A6)"
                )

            if accept:
                logger.info(f"✅ Semantic locator found: {semantic_locator}")
                semantic_stability = classify_locator(semantic_locator)
                return {
                    'element_id': element_id,
                    'description': element_description,
                    'found': True,
                    'best_locator': semantic_locator,
                    'stability': semantic_stability,
                    'all_locators': [{
                        'type': 'semantic',
                        'locator': semantic_locator,
                        'priority': 0,
                        'strategy': 'Semantic locator from description',
                        'count': 1,
                        'unique': True,
                        'valid': True,
                        'validated': True,
                        'semantic_match': semantic_match,
                        'validation_method': validation_method,
                        'stability': semantic_stability
                    }],
                    'element_info': {'description': element_description, 'actual_text': actual_text} if actual_text else {'description': element_description},
                    'coordinates': {'x': x, 'y': y, 'note': 'Not used - semantic approach succeeded'},
                    'validation_summary': {
                        'total_generated': 1,
                        'valid': 1,
                        'unique': 1,
                        'validated': 1,
                        'best_type': 'semantic',
                        'best_strategy': 'Semantic locator from description',
                        'validation_method': validation_method
                    },
                    # Top-level validation fields (required by workflow validation)
                    'validated': True,
                    'count': 1,
                    'unique': True,
                    'valid': True,
                    'semantic_match': semantic_match,
                    'validation_method': 'playwright',
                    # Approach metrics for pattern analysis
                    'approach_metrics': {
                        **_approach_metrics_base,
                        'locator_approach': 'semantic',
                        'fallback_depth': 5,
                        'success': True,
                    }
                }
        else:
            logger.info(f"⚠️ Semantic approach failed - falling back to accessibility API")
    else:
        logger.info(f"⚠️ No description provided - skipping semantic approach")
    
    # ========================================
    # STEP 2.5: ACCESSIBILITY API FALLBACK
    # ========================================
    # Uses Playwright's live DOM query to generate robust role-based locators.
    # This is fundamentally different from cached element indices:
    # - Queries CURRENT page state (not stale indices)
    # - Works after search/filter/AJAX without refresh
    # - Handles tables, grids, dialogs, menus, etc.
    logger.info(f"🔍 Step 2.5: Trying ACCESSIBILITY API fallback")
    
    accessibility_result = await _find_element_via_accessibility(
        page=page,
        x=x,
        y=y,
        element_description=element_description,
        expected_text=expected_text,
        search_context=search_context,
        iframe_context=iframe_context
    )
    
    if accessibility_result and accessibility_result.get('locator'):
        locator = accessibility_result['locator']
        logger.info(f"✅ ACCESSIBILITY FALLBACK SUCCESS: {locator}")
        
        # Validate against expected_text if provided
        semantic_match = True
        if expected_text and accessibility_result.get('accessible_name'):
            expected_lower = expected_text.lower()
            accessible_lower = accessibility_result['accessible_name'].lower()
            semantic_match = expected_lower in accessible_lower or accessible_lower in expected_lower
            if not semantic_match:
                logger.info(f"   ⚠️ Semantic mismatch: expected '{expected_text}' but found '{accessibility_result['accessible_name']}'")
        
        accessibility_stability = classify_locator(locator)
        return _apply_iframe_prefix_to_result({
            # CRITICAL: workflow.py extraction requires these fields
            'element_id': element_id,
            'description': element_description,
            'found': True,  # CRITICAL: Required by registration.py to recognize as success
            'best_locator': locator,
            'stability': accessibility_stability,
            'all_locators': [{'locator': locator, 'method': 'accessibility_role', 'priority': 1,
                              'stability': accessibility_stability}],
            'preferred_method': 'accessibility_role',
            'validated': True,
            'count': accessibility_result.get('count', 1),
            'unique': accessibility_result.get('unique', True),
            'valid': True,
            'semantic_match': semantic_match,
            'validation_method': 'playwright',
            # Pass through element_type for table-rows FOR loop handling
            'element_type': accessibility_result.get('element_type'),
            'row_texts': accessibility_result.get('row_texts'),
            'validation_text': accessibility_result.get('validation_text'),
            # Element info for debugging and metrics
            'element_info': {
                'role': accessibility_result.get('role', 'unknown'),
                'aria_label': accessibility_result.get('table_label'),
                'source': 'accessibility_fallback',
                'selector_type': accessibility_result.get('selector_type', 'unknown')
            },
            # Add validation_summary for actions.py logging
            'validation_summary': {
                'total_generated': 1,
                'valid': 1,
                'unique': 1 if accessibility_result.get('unique', True) else 0,
                'validated': 1,
                'not_found': 0,
                'not_unique': 0 if accessibility_result.get('unique', True) else 1,
                'errors': 0,
                'best_type': accessibility_result.get('role', 'accessibility'),
                'best_strategy': accessibility_result.get('strategy', 'accessibility_role'),
                'validation_method': 'playwright'
            },
            'approach_metrics': {
                **_approach_metrics_base,
                'locator_approach': 'accessibility',
                'strategy_used': accessibility_result.get('strategy', 'accessibility_role'),
                'role': accessibility_result.get('role'),
                'fallback_depth': 6,  # Accessibility fallback
                'success': True,
            }
        })

    else:
        logger.info(f"⚠️ Accessibility fallback failed - falling back to coordinate-based approach")
    
    # ========================================
    # STEP 3: FALLBACK - Coordinate-based approach (21 strategies)
    # ========================================
    logger.info(f"🔍 Step 3: Using COORDINATE-based approach at ({x}, {y})")
    
    # For iframe elements, we need to run JavaScript INSIDE the iframe
    # using relative coordinates (subtracting iframe position)
    eval_context = page  # Default: run JS on main page
    eval_x, eval_y = x, y  # Default: use original coordinates
    
    if iframe_context:
        logger.info(f"🖼️ Element is inside iframe {iframe_context} - switching to iframe context")
        try:
            # Get the iframe's Frame object for JavaScript execution
            iframe_locator = page.locator(iframe_context)
            iframe_box = await iframe_locator.bounding_box()
            
            if iframe_box:
                # Calculate relative coordinates inside iframe
                eval_x = x - iframe_box['x']
                eval_y = y - iframe_box['y']
                logger.info(f"   📐 Relative coordinates: ({eval_x:.1f}, {eval_y:.1f})")
                
                # Get the iframe's content frame for JS execution
                # NOTE: Must use element_handle() first, then content_frame()
                iframe_element = await iframe_locator.element_handle()
                if iframe_element:
                    eval_context = await iframe_element.content_frame()
                    if eval_context:
                        logger.info(f"   ✅ Got iframe content frame for JavaScript execution")
                    else:
                        logger.warning(f"   ⚠️ Could not get iframe content frame - using main page")
                        eval_context = page
                        eval_x, eval_y = x, y
                else:
                    logger.warning(f"   ⚠️ Could not get iframe element handle - using main page")
                    eval_context = page
                    eval_x, eval_y = x, y
            else:
                logger.warning(f"   ⚠️ Could not get iframe bounding box - using main page")
        except Exception as e:
            logger.warning(f"   ⚠️ Error getting iframe context: {e}")
            eval_context = page
            eval_x, eval_y = x, y
    
    # Get the element at coordinates
    try:
        # Use Playwright to get element at coordinates (Shadow DOM aware)
        # For iframe elements, this runs inside the iframe with relative coords
        element_exists = await eval_context.evaluate(
            SHADOW_DOM_ELEMENT_FROM_POINT_JS,
            {"x": eval_x, "y": eval_y}
        )

        if not element_exists:
            logger.error(f"❌ No element found at coordinates ({x}, {y})")
            return {
                "element_id": element_id,
                "description": element_description,
                "found": False,
                "error": f"No element at coordinates ({x}, {y}) and semantic approach also failed",
                # Track failure metrics for pattern analysis
                'approach_metrics': {
                    **_approach_metrics_base,
                    'locator_approach': 'coordinate_fallback',
                    'fallback_depth': 7,
                    'success': False,
                }
            }
        
        # Check if we got BODY or HTML (coordinates landed in empty space) - Shadow DOM aware
        tag_check = await eval_context.evaluate(
            SHADOW_DOM_TAG_NAME_JS,
            {"x": eval_x, "y": eval_y}
        )
        
        if tag_check in ['body', 'html']:
            # Both semantic AND coordinate approaches failed
            logger.error(f"❌ Coordinates ({x}, {y}) landed on {tag_check.upper()} (empty space)")
            logger.error(f"   Both semantic and coordinate approaches failed for: {element_description}")
            return {
                'element_id': element_id,
                'description': element_description,
                'found': False,
                'error': f"Semantic approach failed and coordinates ({x}, {y}) landed on {tag_check.upper()} (empty space)",
                'coordinates': {'x': x, 'y': y},
                'validation_summary': {
                    'total_generated': 0,
                    'valid': 0,
                    'unique': 0,
                    'validated': 0,
                    'best_type': None,
                    'best_strategy': None,
                    'validation_method': 'playwright'
                },
                # Track failure metrics for pattern analysis
                'approach_metrics': {
                    **_approach_metrics_base,
                    'locator_approach': 'coordinate_fallback',
                    'fallback_depth': 7,
                    'success': False,
                }
            }

    except Exception as e:
        logger.error(f"❌ Error getting element at coordinates: {e}")
        return {
            "element_id": element_id,
            "found": False,
            "error": str(e),
            # Track failure metrics for pattern analysis
            'approach_metrics': {
                **_approach_metrics_base,
                'locator_approach': 'coordinate_fallback',
                'fallback_depth': 7,
                'success': False,
            }
        }

    # Step 2: Extract all possible attributes from the element (Shadow DOM aware)
    # For iframe elements, this runs inside the iframe with relative coords
    try:
        element_data = await eval_context.evaluate(
            """(coords) => {
                // Shadow DOM aware element detection
                function getElementFromPointWithShadow(root, x, y) {
                    let element = root.elementFromPoint(x, y);
                    if (!element) return null;
                    while (element && element.shadowRoot) {
                        const shadowElement = element.shadowRoot.elementFromPoint(x, y);
                        if (shadowElement && shadowElement !== element) {
                            element = shadowElement;
                        } else {
                            break;
                        }
                    }
                    return element;
                }
                
                const element = getElementFromPointWithShadow(document, coords.x, coords.y);
                if (!element) return null;
                
                const rect = element.getBoundingClientRect();
                
                // Get all attributes
                const attrs = {};
                for (let attr of element.attributes) {
                    attrs[attr.name] = attr.value;
                }
                
                // Get computed role
                let computedRole = element.getAttribute('role');
                if (!computedRole) {
                    // Try to infer role from tag
                    const tagRoleMap = {
                        'button': 'button',
                        'a': 'link',
                        'input': element.type || 'textbox',
                        'textarea': 'textbox',
                        'select': 'combobox',
                        'img': 'img',
                        'h1': 'heading', 'h2': 'heading', 'h3': 'heading',
                        'nav': 'navigation',
                        'main': 'main',
                        'header': 'banner',
                        'footer': 'contentinfo'
                    };
                    computedRole = tagRoleMap[element.tagName.toLowerCase()];
                }
                
                return {
                    tagName: element.tagName.toLowerCase(),
                    id: element.id || '',
                    name: element.name || '',
                    className: element.className || '',
                    textContent: element.textContent?.trim().substring(0, 100) || '',
                    innerText: element.innerText?.trim().substring(0, 100) || '',
                    value: element.value || '',
                    placeholder: element.placeholder || '',
                    title: element.title || '',
                    alt: element.alt || '',
                    href: element.href || '',
                    src: element.src || '',
                    type: element.type || '',
                    ariaLabel: element.getAttribute('aria-label') || '',
                    ariaDescribedby: element.getAttribute('aria-describedby') || '',
                    dataTestId: element.getAttribute('data-testid') || '',
                    dataTest: element.getAttribute('data-test') || '',
                    dataQa: element.getAttribute('data-qa') || '',
                    role: computedRole || '',
                    attributes: attrs,
                    coordinates: {
                        x: rect.left + rect.width / 2,
                        y: rect.top + rect.height / 2
                    },
                    // Get parent context
                    parentId: element.parentElement?.id || '',
                    parentClass: element.parentElement?.className || '',
                    // Get position among siblings
                    siblingIndex: Array.from(element.parentElement?.children || []).indexOf(element) + 1,
                    totalSiblings: element.parentElement?.children.length || 0
                };
            }""",
            {"x": eval_x, "y": eval_y}
        )

        if not element_data:
            logger.error(f"❌ Could not extract element data")
            return {
                "element_id": element_id,
                "description": element_description,
                "found": False,
                "error": "Could not extract element data"
            }

        logger.info(
            f"📋 Element data: tag={element_data['tagName']}, id={element_data['id']}, text=\"{element_data['textContent'][:30]}...\"")

    except Exception as e:
        logger.error(f"❌ Error extracting element data: {e}")
        return {
            "element_id": element_id,
            "description": element_description,
            "found": False,
            "error": str(e)
        }

    # Step 3: Try multiple locator strategies in priority order
    locator_strategies = _build_coordinate_strategies(element_data)

    logger.info(
        f"🔍 Generated {len(locator_strategies)} locator strategies to test")

    # Step 4: Validate each strategy
    # Sort strategies by priority for optimal early exit
    # Lower priority number = better locator (1=ID is best, 18=XPath-first-of-class is worst)
    # Stability tier first, then priority (E1): volatile ids and positional
    # strategies sink below stable candidates but stay in the cascade as
    # last resorts — demoted, never deleted.
    sorted_strategies = sorted(
        locator_strategies,
        key=lambda x: (stability_rank(x.get('stability', STABLE)), x['priority']),
    )
    validated_locators = await _validate_strategy_candidates(
        search_context, sorted_strategies, expected_text=expected_text,
        row_anchor_text=row_anchor_text, x=x, y=y,
    )

    # Step 5: Select best locator (unique, lowest priority number)
    # WITH SEMANTIC VALIDATION if expected_text is provided
    unique_locators = [loc for loc in validated_locators if loc.get(
        'valid') and loc.get('unique')]

    best_locator_obj = None  # Initialize to None
    semantic_match = True  # Assume match unless expected_text is provided
    actual_text = ""
    
    if unique_locators:
        # Sort by stability tier, then priority (E1): a volatile id only
        # wins when no stable candidate validated.
        sorted_locators = sorted(
            unique_locators,
            key=lambda x: (stability_rank(x.get('stability', STABLE)), x['priority']),
        )
        
        # If expected_text is provided, find a locator that ALSO matches semantically
        if expected_text:
            logger.info(f"🔍 Checking semantic match for {len(sorted_locators)} unique locators...")

            # _fast_path_resolved: True once a definitive decision is reached in the
            # fast path (either "matched + identity confirmed" or "semantic mismatch").
            # When it stays False, the slow per-locator path runs below.
            _fast_path_resolved = False

            if resolved_node is not None and resolved_node.children_nodes is not None:
                # Fast path: cache-hit node has full AX/text data — one semantic check
                # covers all candidates without CDP round-trips.
                # CDP fallback nodes (children_nodes=None) have incomplete text data
                # and are routed to the slow path instead.
                is_match, actual_text = await validate_semantic_match(resolved_node, expected_text)
                semantic_match = is_match
                if is_match:
                    # The element at (x, y) matches expected_text. Now verify that the
                    # top-priority locator resolves to that same element. A locator
                    # generated from text or attributes can uniquely match a *different*
                    # element on the page while the resolved_node proves only the element
                    # at (x, y) is correct.
                    _best = sorted_locators[0]
                    _identity_ok = False
                    try:
                        bbox = await search_context.locator(_best['locator']).bounding_box()
                        if bbox is not None:
                            _TOL = 2.0  # pixels — accounts for subpixel coord rounding
                            _identity_ok = (
                                bbox['x'] - _TOL <= x <= bbox['x'] + bbox['width'] + _TOL
                                and bbox['y'] - _TOL <= y <= bbox['y'] + bbox['height'] + _TOL
                            )
                    except Exception:
                        pass  # bounding_box() failed; fall through to slow path

                    if _identity_ok:
                        _best['semantic_match'] = True
                        _best['actual_text'] = actual_text
                        best_locator_obj = _best
                        _fast_path_resolved = True
                        logger.info(
                            f"   ✅ Semantically matched and identity verified: {_best['locator']}"
                        )
                    else:
                        logger.warning(
                            f"   ⚠️ Fast path: {_best['locator']!r} doesn't contain "
                            f"({x}, {y}) — checking each locator individually"
                        )
                        # _fast_path_resolved stays False; slow path will run below
                else:
                    # Element at (x, y) doesn't match expected_text — no locator for a
                    # different element can fix this; mark all and skip the slow path.
                    for loc in sorted_locators:
                        loc['semantic_match'] = False
                        loc['actual_text'] = actual_text
                    _fast_path_resolved = True

            if not _fast_path_resolved:
                if resolved_node is not None and resolved_node.children_nodes is not None:
                    # Identity-fail slow path: resolved_node already confirmed the element
                    # at (x, y) has the expected text, so semantic match is proven.
                    # We only need to find which unique locator resolves to that same element.
                    # One bounding_box() call per locator — cheaper than evaluate() + bounding_box().
                    for loc in sorted_locators:
                        _slow_bbox_ok = False
                        try:
                            _bb = await search_context.locator(loc['locator']).bounding_box()
                            if _bb is not None:
                                _TOL = 2.0
                                _slow_bbox_ok = (
                                    _bb['x'] - _TOL <= x <= _bb['x'] + _bb['width'] + _TOL
                                    and _bb['y'] - _TOL <= y <= _bb['y'] + _bb['height'] + _TOL
                                )
                        except Exception as _bbox_err:
                            logger.debug(f"   bounding_box() error for {loc['locator']!r}: {_bbox_err}")
                        loc['semantic_match'] = _slow_bbox_ok
                        loc['actual_text'] = actual_text  # already set from node check
                        if _slow_bbox_ok and best_locator_obj is None:
                            best_locator_obj = loc
                            logger.info(
                                f"   ✅ Spatially verified locator "
                                f"(semantic confirmed by node): {loc['locator']}"
                            )
                            break
                else:
                    # Standard slow path: no resolved node or CDP fallback node —
                    # evaluate each candidate locator directly for text content.
                    for loc in sorted_locators:
                        is_match, text = await validate_semantic_match(None, expected_text, page=search_context, locator=loc['locator'])
                        loc['semantic_match'] = is_match
                        loc['actual_text'] = text
                        if is_match and best_locator_obj is None:
                            best_locator_obj = loc
                            actual_text = text
                            logger.info(f"   ✅ Found semantically matching locator: {loc['locator']}")
                            break
            
            # If no semantic match found, DO NOT return wrong locator - return failure instead
            if best_locator_obj is None:
                logger.error(f"   ❌ SEMANTIC MISMATCH: No locator matched expected text '{expected_text}'")
                logger.error(f"   All {len(sorted_locators)} unique locators have wrong text content")
                
                # Return failure - do not give back a wrong locator
                return {
                    'element_id': element_id,
                    'description': element_description,
                    'found': False,
                    'error': f"Semantic mismatch: Expected '{expected_text}' but none of the {len(sorted_locators)} unique locators matched",
                    'semantic_match': False,
                    'expected_text': expected_text,
                    'candidates_found': len(sorted_locators),
                    'candidate_locators': [loc['locator'] for loc in sorted_locators[:3]]  # Top 3 for debugging
                }
        else:
            # No expected_text, just use first unique locator
            best_locator_obj = sorted_locators[0]
        
        if best_locator_obj:
            best_locator = best_locator_obj['locator']
            
            # Log final selected locator with complete details
            logger.info(f"")
            logger.info(f"{'='*80}")
            logger.info(f"✅ FINAL SELECTED LOCATOR for {element_id}")
            logger.info(f"{'='*80}")
            logger.info(f"   Locator: {best_locator}")
            logger.info(f"   Type: {best_locator_obj['type']}")
            logger.info(f"   Priority: {best_locator_obj['priority']} (1=best, 18=worst)")
            logger.info(f"   Strategy: {best_locator_obj['strategy']}")
            logger.info(f"   Validation Results:")
            logger.info(f"      - count: {best_locator_obj['count']}")
            logger.info(f"      - unique: {best_locator_obj['unique']}")
            logger.info(f"      - valid: {best_locator_obj['valid']}")
            logger.info(f"      - validated: {best_locator_obj['validated']}")
            logger.info(f"      - semantic_match: {semantic_match}")
            if expected_text:
                logger.info(f"      - expected_text: '{expected_text}'")
                logger.info(f"      - actual_text: '{actual_text[:50]}...' " if len(actual_text) > 50 else f"      - actual_text: '{actual_text}'")
            logger.info(f"      - validation_method: {best_locator_obj['validation_method']}")
            logger.info(f"   Total unique locators found: {len(unique_locators)}")
            logger.info(f"{'='*80}")
            logger.info(f"")
        else:
            best_locator = None
    else:
        best_locator = None
        
        # Log failure with detailed breakdown
        logger.error(f"")
        logger.error(f"{'='*80}")
        logger.error(f"❌ NO UNIQUE LOCATOR FOUND for {element_id}")
        logger.error(f"{'='*80}")

        # Log why - categorize failures
        non_unique = [loc for loc in validated_locators if loc.get(
            'validated') and loc.get('count', 0) > 1]
        not_found = [loc for loc in validated_locators if loc.get(
            'validated') and loc.get('count', 0) == 0]
        errors = [
            loc for loc in validated_locators if not loc.get('validated')]

        logger.error(f"   Failure Breakdown:")
        if non_unique:
            logger.error(f"      - {len(non_unique)} locators matched multiple elements (not unique)")
            for loc in non_unique[:3]:  # Show first 3
                logger.error(f"         • {loc['type']}: count={loc['count']}")
        if not_found:
            logger.error(f"      - {len(not_found)} locators found no elements")
            for loc in not_found[:3]:  # Show first 3
                logger.error(f"         • {loc['type']}: {loc['locator']}")
        if errors:
            logger.error(f"      - {len(errors)} locators had validation errors")
            for loc in errors[:3]:  # Show first 3
                logger.error(f"         • {loc['type']}: {loc.get('validation_error', 'Unknown error')}")
        
        logger.error(f"   Total strategies attempted: {len(validated_locators)}")
        logger.error(f"{'='*80}")
        logger.error(f"")

    # Step 6: Build result with complete validation data
    validation_summary = {
        'total_generated': len(validated_locators),
        'valid': sum(1 for loc in validated_locators if loc.get('valid')),
        'unique': sum(1 for loc in validated_locators if loc.get('unique')),
        'validated': sum(1 for loc in validated_locators if loc.get('validated')),
        'not_found': sum(1 for loc in validated_locators if loc.get('validated') and loc.get('count', 0) == 0),
        'not_unique': sum(1 for loc in validated_locators if loc.get('validated') and loc.get('count', 0) > 1),
        'errors': sum(1 for loc in validated_locators if not loc.get('validated')),
        'best_type': best_locator_obj['type'] if best_locator_obj else None,
        'best_strategy': best_locator_obj['strategy'] if best_locator_obj else None,
        'semantic_match': semantic_match,
        'validation_method': 'playwright'
    }
    
    result = {
        'element_id': element_id,
        'description': element_description,
        'found': best_locator is not None,
        'best_locator': best_locator,
        'stability': (
            best_locator_obj.get('stability', STABLE)
            if best_locator_obj else None
        ),
        'all_locators': validated_locators,
        'element_info': {
            'id': element_data['id'],
            'tagName': element_data['tagName'],
            'text': element_data['textContent'],
            'className': element_data['className'],
            'name': element_data['name'],
            'testId': element_data['dataTestId'],
            'actual_text': actual_text,  # Add actual text for debugging
        },
        'coordinates': element_data['coordinates'],
        'validation_summary': validation_summary,
        'semantic_match': semantic_match  # NEW: Flag indicating if actual text matches expected
    }
    
    # If semantic mismatch, add warning
    if expected_text and not semantic_match:
        result['semantic_warning'] = f"Expected '{expected_text}' but element contains '{actual_text}'"
    
    # Add validation data to the result itself for easy access
    if best_locator_obj:
        result['validated'] = True
        result['count'] = best_locator_obj.get('count', 1)
        result['unique'] = True
        result['valid'] = True
        result['validation_method'] = 'playwright'
        if best_locator_obj.get('stability', STABLE) != STABLE:
            logger.warning(
                f"   ⚠️ Emitting {best_locator_obj.get('stability')} locator "
                f"{best_locator} — no stable candidate validated"
            )
        # Approach metrics for pattern analysis (coordinate_fallback succeeded)
        result['approach_metrics'] = {
            **_approach_metrics_base,
            'locator_approach': 'coordinate_fallback',
            'fallback_depth': 7,
            'success': True,
        }
    else:
        result['validated'] = True  # Validation was attempted
        result['count'] = 0  # No unique locator found
        result['unique'] = False
        result['valid'] = False
        result['validation_method'] = 'playwright'
        # Approach metrics for pattern analysis (all approaches failed)
        result['approach_metrics'] = {
            **_approach_metrics_base,
            'locator_approach': 'coordinate_fallback',
            'fallback_depth': 7,
            'success': False,
        }
    
    # Log validation summary
    logger.info(f"")
    logger.info(f"📊 VALIDATION SUMMARY for {element_id}")
    logger.info(f"   Total strategies attempted: {validation_summary['total_generated']}")
    logger.info(f"   Valid (count=1): {validation_summary['valid']}")
    logger.info(f"   Unique (count=1): {validation_summary['unique']}")
    logger.info(f"   Not found (count=0): {validation_summary['not_found']}")
    logger.info(f"   Not unique (count>1): {validation_summary['not_unique']}")
    logger.info(f"   Validation errors: {validation_summary['errors']}")
    logger.info(f"   Successfully validated: {validation_summary['validated']}")
    logger.info(f"   Semantic match: {semantic_match}")
    if expected_text and not semantic_match:
        logger.warning(f"   ⚠️ SEMANTIC MISMATCH: Expected '{expected_text}', got '{actual_text[:50]}...'")
    if best_locator_obj:
        logger.info(f"   Best locator type: {validation_summary['best_type']}")
        logger.info(f"   Best strategy: {validation_summary['best_strategy']}")
    logger.info(f"")
    
    return result
