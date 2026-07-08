"""
Custom action registration for browser-use agent.

This module handles the registration of custom actions with the browser-use agent.
Custom actions allow the agent to call deterministic Python code during workflow execution,
bypassing the need for LLM calls for specific operations like locator validation.

Key Functions:
- register_custom_actions: Register custom actions with browser-use agent

The registration process:
1. Creates or retrieves the Tools instance from the agent
2. Defines parameter models for custom actions using Pydantic
3. Registers action handlers that wrap the actual implementation
4. Handles page object retrieval from browser_session via CDP
5. Converts results to ActionResult format for the agent

Playwright Connection Lifecycle (Day 04 — lazy, once-per-run):
- Playwright is connected lazily on the first find_unique_locator call within agent.run()
- The connection is reused for every subsequent call in the same run (one connect per workflow)
- Torn down by _teardown_playwright() called from the workflow finally block after agent.run() returns
- Each concurrent workflow has its own pw_cache closure — no shared state between workflows

Usage:
    from browser_service.agent.registration import register_custom_actions

    success = register_custom_actions(agent, page=None)
    if success:
        # Agent can now call find_unique_locator action
        pass
"""

import asyncio
import json
import logging
import re
from typing import Optional

from browser_service.locators.stability import (
    STABLE,
    is_dynamic_text,
    score_stability,
)

# Get logger
logger = logging.getLogger(__name__)

# Class tokens usable as a bare `.class` selector — anything with CSS meta
# characters (Tailwind `w-1/2`, `md:flex`) is skipped rather than escaped.
_CSS_IDENTIFIER_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_-]*$')


class _PlaywrightConnectionError(RuntimeError):
    """Raised exclusively by _ensure_playwright when the CDP session is dead or unreachable.

    Keeping this separate from RuntimeError prevents unrelated RuntimeErrors raised
    inside the find_unique_locator action body from being misidentified as dead-browser
    signals, which would terminate the agent prematurely with is_done=True.
    """


def _select_best_page(browser):
    """Return the most recently opened non-blank page across all CDP contexts, or None.

    Iterates contexts and pages in reverse (most-recently-opened last) so that the
    active tab is preferred when Chrome exposes an empty leading context or when a
    new tab was opened after the initial connect.
    """
    for ctx in reversed(browser.contexts):
        for pg in reversed(ctx.pages):
            if pg.url and pg.url not in ("about:blank", ""):
                return pg
    return None


def _extract_dom_node_attributes(dom_node) -> dict:
    """
    Extract standard attributes from a browser-use DOM node.
    
    This helper prevents code duplication across multiple locations
    where element attributes need to be extracted for locator generation.
    
    Args:
        dom_node: EnhancedDOMTreeNode from browser-use
        
    Returns:
        Dictionary with standard element attributes
    """
    attrs = dom_node.attributes if hasattr(dom_node, 'attributes') else {}
    # Carry WHICH test attribute produced the value — an element with only
    # data-test must not be looked up as [data-testid=...] (0 matches, hook lost).
    data_test_attr = (
        'data-test'
        if (not attrs.get('data-testid') and attrs.get('data-test'))
        else 'data-testid'
    )
    return {
        'tagName': dom_node.node_name.lower() if hasattr(dom_node, 'node_name') else '',
        'id': attrs.get('id', ''),
        'name': attrs.get('name', ''),
        'className': attrs.get('class', ''),
        'ariaLabel': attrs.get('aria-label', ''),
        'placeholder': attrs.get('placeholder', ''),
        'title': attrs.get('title', ''),
        'href': attrs.get('href', ''),
        'role': attrs.get('role', ''),
        'dataTestId': attrs.get('data-testid', '') or attrs.get('data-test', ''),
        'dataTestAttr': data_test_attr,
        'type': attrs.get('type', ''),  # For input elements
        'value': attrs.get('value', ''),  # Current value of input
        'xpath': dom_node.xpath if hasattr(dom_node, 'xpath') else '',
    }



def _detect_iframe_context(selector_map, coords: tuple) -> tuple:
    """
    Detect if element coordinates are inside an iframe's bounding box.
    
    This enables locator extraction for elements inside iframes by checking
    if the target coordinates fall within any iframe's bounds.
    
    Args:
        selector_map: Browser-use selector_map containing all elements
        coords: Tuple of (x, y) coordinates to check
        
    Returns:
        Tuple of (iframe_locator, iframe_id) if inside iframe, (None, None) otherwise.
        iframe_locator is the selector (e.g., 'iframe[id="main"]' or 'iframe[name="content"]')
    """
    if not selector_map or not coords:
        return None, None

    x, y = coords

    def _esc(value: str) -> str:
        """Escape \\ and " for a CSS attribute selector."""
        return value.replace('\\', '\\\\').replace('"', '\\"')

    def _attrs_of(node) -> dict:
        return node.attributes if hasattr(node, 'attributes') else {}

    # Collect iframes first (selector_map order = ordinal parity with the
    # pre-G6 behavior) so title/class uniqueness can be checked against
    # the other iframes browser-use indexed.
    iframes = [
        elem for elem in selector_map.values()
        if hasattr(elem, 'node_name') and elem.node_name.lower() == 'iframe'
    ]

    for iframe_ordinal, elem in enumerate(iframes):
        if not (hasattr(elem, 'absolute_position') and elem.absolute_position):
            continue
        pos = elem.absolute_position
        # Check if coordinates are within iframe bounds
        if not (pos.x <= x <= pos.x + pos.width and
                pos.y <= y <= pos.y + pos.height):
            continue

        attrs = _attrs_of(elem)
        iframe_id = attrs.get('id', '')
        iframe_name = attrs.get('name', '')
        iframe_title = (attrs.get('title', '') or '').strip()
        iframe_classes = (attrs.get('class', '') or '').split()
        others = [o for o in iframes if o is not elem]

        # Generate locator for the iframe using attribute selectors.
        # Cascade (G6): id → name → title → unique stable class → ordinal.
        # The ordinal is last because async third-party iframes (ASTPP:
        # #jsd-widget) shift iframe order between discovery and RF
        # runtime — `iframe >> nth=N` then points at a DIFFERENT frame.
        iframe_locator = None
        if iframe_id:
            iframe_locator = f'iframe[id="{_esc(iframe_id)}"]'
        elif iframe_name:
            iframe_locator = f'iframe[name="{_esc(iframe_name)}"]'
        else:
            title_taken_by_other = any(
                (_attrs_of(o).get('title', '') or '').strip() == iframe_title
                for o in others
            )
            if (iframe_title and not is_dynamic_text(iframe_title)
                    and not title_taken_by_other):
                # CKEditor 4 body: title="Rich Text Editor, {field}" —
                # deterministic per editor instance, no id/name.
                iframe_locator = f'iframe[title="{_esc(iframe_title)}"]'
            else:
                for cls in iframe_classes:
                    if not _CSS_IDENTIFIER_RE.match(cls):
                        continue  # CSS meta chars — not a bare .class token
                    if score_stability('class', cls) != STABLE:
                        continue  # init-order counters (cke_1) die next session
                    if any(cls in (_attrs_of(o).get('class', '') or '').split()
                           for o in others):
                        continue  # shared with another iframe — ambiguous
                    iframe_locator = f'iframe.{cls}'
                    break
            if iframe_locator is None:
                # Last resort: ordinal (0-indexed count of iframes).
                # Downstream marks the whole composite positional (B2).
                iframe_locator = f"iframe >> nth={iframe_ordinal}"

        logger.info(f"🖼️ IFRAME DETECTED: Element at ({x}, {y}) is inside {iframe_locator}")
        identifier = (iframe_id or iframe_name or iframe_title
                      or str(iframe_ordinal))
        return iframe_locator, identifier

    return None, None


def _find_smallest_containing_element(selector_map, coords, viewport_area, skip_tag=None):
    """
    Find the selector_map element with the smallest bounding box containing coords.

    Skips elements whose bounding box covers more than 80% of the viewport — these
    are page wrappers (<body>, #page-container) not specific interactive targets.
    When the LLM provides wrong coordinates, only wrappers match, so skipping them
    lets the text-based strategies in smart_locator.py take over.

    Args:
        selector_map: browser-use selector_map (idx → EnhancedDOMTreeNode)
        coords: (x, y) tuple in page-absolute pixels
        viewport_area: viewport_w * viewport_h, used for the wrapper threshold
        skip_tag: optional uppercase tag name to always skip (e.g. 'IFRAME' when
                  looking for an element nested inside an iframe — we want the
                  contained element, not the iframe itself)

    Returns:
        (idx, dom_node) of the best match, or (None, None) if no match found
    """
    if not selector_map or not coords:
        return None, None

    x, y = coords
    wrapper_threshold = viewport_area * 0.8
    skip_tag_upper = skip_tag.upper() if skip_tag else None
    best_match = (None, None)
    best_area = float('inf')

    for idx, elem in selector_map.items():
        if not (hasattr(elem, 'absolute_position') and elem.absolute_position):
            continue
        pos = elem.absolute_position
        if not (pos.x <= x <= pos.x + pos.width and pos.y <= y <= pos.y + pos.height):
            continue
        if skip_tag_upper and hasattr(elem, 'node_name') and elem.node_name.upper() == skip_tag_upper:
            continue
        area = pos.width * pos.height
        if area <= 0:
            continue
        if area > wrapper_threshold:
            logger.debug(
                f"   ⏭️ Skipping [{idx}]: covers {area / viewport_area:.0%} of viewport "
                f"(likely page wrapper)"
            )
            continue
        if area < best_area:
            best_area = area
            best_match = (idx, elem)

    return best_match


# ═══════════════════════════════════════════════════════════════════════════════
# CDP URL AND PAGE RETRIEVAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
# These helper functions consolidate the fallback strategy chains into
# maintainable, testable units. Each strategy is tried in priority order.

# CDP URL pattern: ws[s]://HOST:PORT/devtools/browser/UUID
# Supports: ws://, wss://, IPv4, IPv6, hostnames
_CDP_URL_PATTERN = re.compile(r'^wss?://[^\s/]+/devtools/browser/')


def _extract_cdp_host_port(cdp_url: str) -> str:
    """Extract host:port from CDP URL for cleaner logging."""
    if '/devtools/' in cdp_url:
        return cdp_url.split('/devtools/')[0]
    return cdp_url


def _get_cdp_url_from_session(browser_session) -> Optional[str]:
    """
    Get CDP URL from browser_session using multiple fallback strategies.

    This function is pure: it only reads from browser_session and returns the
    URL. It intentionally does NOT write to any module-level globals in
    cleanup.py — those globals are shared across concurrent tasks and the
    "last writer wins" semantics would cause one session's cleanup to target
    another session's browser. The primary cleanup path (browser_pid captured
    in workflow.py right after session.start()) is the only reliable mechanism
    for concurrent use.

    Strategies (in priority order):
    1. Direct cdp_url attribute
    2. cdp_client.url attribute
    3. Search all public attributes for WebSocket DevTools URL pattern

    Args:
        browser_session: The browser-use session object

    Returns:
        CDP URL string if found, None otherwise
    """
    if not browser_session:
        return None

    # Strategy 1: Direct cdp_url attribute (most common)
    if hasattr(browser_session, 'cdp_url'):
        try:
            cdp_url = browser_session.cdp_url
            if cdp_url and _CDP_URL_PATTERN.match(cdp_url):
                logger.info(f"✅ CDP URL from browser_session.cdp_url: {_extract_cdp_host_port(cdp_url)}")
                return cdp_url
        except Exception as e:
            logger.debug(f"Strategy 1 (cdp_url): {e}")

    # Strategy 2: cdp_client.url attribute
    if hasattr(browser_session, 'cdp_client'):
        try:
            cdp_client = browser_session.cdp_client
            if hasattr(cdp_client, 'url'):
                cdp_url = cdp_client.url
                if cdp_url and _CDP_URL_PATTERN.match(cdp_url):
                    logger.info(f"✅ CDP URL from cdp_client.url: {_extract_cdp_host_port(cdp_url)}")
                    return cdp_url
        except Exception as e:
            logger.debug(f"Strategy 2 (cdp_client.url): {e}")

    # Strategy 3: Search all public attributes for WebSocket DevTools URL
    logger.debug("🔍 Searching all attributes for CDP URL...")
    for attr in dir(browser_session):
        if attr.startswith('_'):
            continue
        try:
            value = getattr(browser_session, attr, None)
            if value and isinstance(value, str) and _CDP_URL_PATTERN.match(value):
                logger.info(f"✅ CDP URL found in attribute '{attr}': {_extract_cdp_host_port(value)}")
                return value
        except Exception:
            pass

    logger.warning("⚠️ Could not find CDP URL in browser_session")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTION HELPERS (B0–C) — module-level so unit tests can import directly.
# performed_actions and element_specs are passed as explicit parameters rather
# than captured from a closure: this keeps concurrency isolation intact (each
# workflow creates its own set/dict in register_custom_actions) while making
# the dependency on mutable state visible at the call site and testable without
# closure-piercing tricks.
# ═══════════════════════════════════════════════════════════════════════════════

_RF_SELECT_PREFIXES = frozenset({"label", "value", "text", "index"})


def _strip_rf_select_prefix(value: str) -> str:
    """Strip the Robot Framework Select Options By strategy prefix from a value string.

    RF encodes the selection strategy as the first whitespace-separated token:
        "label    default"       → "default"
        "value    some_val"      → "some_val"
        "text     My Option"     → "My Option"
        "index    0"             → "0"
        "United States"          → "United States"  (not an RF prefix — unchanged)
        "default"                → "default"         (single token — unchanged)

    Only strips when the first token is a known RF strategy keyword (label/value/text/index).
    This prevents multi-word option text like "United States" from being incorrectly truncated.
    Applied only to 'select' actions — never touches input/click values."""
    if not value:
        return value
    parts = value.split(None, 1)
    if len(parts) == 2 and parts[0].lower() in _RF_SELECT_PREFIXES:
        return parts[1].strip()
    return value


async def _wait_for_page_stability(active_page) -> None:
    """Wait for page to stabilize after a click/submit/select that may trigger navigation.

    domcontentloaded (not networkidle) — HTML parsed and DOM built is sufficient
    for the next find_unique_locator call. Times out silently if the page is
    already stable or no navigation occurred. asyncio.sleep(0) yields to the
    event loop first so any pending CDP navigation events are dispatched before
    wait_for_load_state checks the current load state — prevents the race where
    the CDP click fires but navigation hasn't started yet when we check."""
    try:
        await asyncio.sleep(0)
        await active_page.wait_for_load_state("domcontentloaded", timeout=5000)
    except Exception:
        pass



async def _dispatch_browser_use_event(
    browser_session, node, action: str, value: str, element_id: str, performed_actions: set
):
    """Attempt the interaction via browser-use's built-in event bus (preferred path).

    Returns (note, status) on success or not_applicable.
    Returns (None, None) as the universal fall-through signal — Playwright path retries.
    performed_actions.add() is called ONLY on confirmed success, never on fall-through."""
    from browser_use.browser.events import TypeTextEvent, ClickElementEvent, SelectDropdownOptionEvent
    try:
        if action in ("input", "type"):
            if not value:
                return "", "not_applicable"
            event = browser_session.event_bus.dispatch(TypeTextEvent(node=node, text=value, clear=True))
            await event
            await event.event_result(raise_if_any=True, raise_if_none=False)
            # TypeTextEvent types via CDP key events (character-by-character) — reliable if no exception raised.
            # Returns {input_x, input_y, actual_value} (0.12.6). Built-in concatenation-detection retry
            # self-corrects append-instead-of-replace without caller intervention. No explicit verification needed.
            performed_actions.add(element_id)
            return f"\n✅ AUTO-ACTION COMPLETE: Typed '{value}'", "auto_ok"

        elif action in ("click", "submit"):
            event = browser_session.event_bus.dispatch(ClickElementEvent(node=node))
            await event
            click_meta = await event.event_result(raise_if_any=True, raise_if_none=False)
            if isinstance(click_meta, dict) and "validation_error" in click_meta:
                # browser-use rejected click (file input or <select> element).
                # Fall through to Playwright which may handle these differently.
                return None, None
            performed_actions.add(element_id)
            return "\n✅ AUTO-ACTION COMPLETE: Clicked element", "auto_ok"

        elif action == "select":
            if not value:
                return "", "not_applicable"
            event = browser_session.event_bus.dispatch(SelectDropdownOptionEvent(node=node, text=value))
            await event
            sel = await event.event_result(raise_if_any=True, raise_if_none=False)
            if isinstance(sel, dict) and sel.get("success") == "true":
                performed_actions.add(element_id)
                return f"\n✅ AUTO-SELECT: Selected '{value}'", "auto_ok"
            # Any failure (Tom Select sibling pattern, option not found, unrecognised element, etc.)
            # — fall through to Playwright which has a wider selection strategy.
            # Do NOT add to performed_actions here; Playwright path will handle it.
            return None, None

        elif action in ("check", "uncheck"):
            # browser-use has no dedicated check/uncheck event — verified from source.
            # ClickElementEvent is a blind toggle with no state awareness (just clicks coordinates).
            # Fall through to Playwright's loc.check() / loc.uncheck() which are state-aware.
            return None, None

    except Exception as e:
        # Unexpected error in event path — log and fall through to Playwright.
        logger.warning(f"   AUTO-ACTION event path error for {element_id} ({action}): {e}")
        return None, None


async def _do_interaction_playwright(
    active_page, locator_str: str, action: str, value: str, element_id: str, performed_actions: set,
    dropdown_framework: str = "", select_id: str | None = None,
    datepicker_framework: str = "",
):
    """Playwright fallback with layered retry chains.

    Each action type runs through multiple strategies in order, stopping at first success.
    All strategies exhausted → auto_failed. performed_actions.add() only on confirmed success."""
    try:
        loc = active_page.locator(locator_str)
        if action in ("input", "type"):
            if not value:
                return "", "not_applicable"

            # Tier 0 (flatpickr): the widget's own API is the only
            # deterministic path — the input is readonly, so fill() below
            # would wait its full timeout and triple_click would open the
            # calendar overlay and leave it over the page. setDate parses
            # the value with the instance's own date format (lenient with
            # date-only values — verified live on ASTPP 2026-07-08).
            if datepicker_framework == "flatpickr":
                try:
                    diag = await loc.evaluate(
                        """(el, v) => {
                            try {
                                const fp = el._flatpickr;
                                if (!fp) return 'no_fp';
                                fp.setDate(v, true);
                                if (!fp.selectedDates || fp.selectedDates.length === 0) return 'no_date';
                                return 'ok';
                            } catch(e) {
                                return 'error:' + e.message;
                            }
                        }""",
                        value,
                    )
                    if diag == "ok":
                        performed_actions.add(element_id)
                        return f"\n✅ AUTO-ACTION COMPLETE: Set date '{value}' (flatpickr setDate JS)", "auto_ok"
                    logger.info(f"   ⚠️ flatpickr.tier0_js_failed: reason={diag} locator={locator_str!r} value={value!r}")
                except Exception as e:
                    logger.warning(f"   ⚠️ flatpickr.tier0_exception: locator={locator_str!r} error={e}")
                # Fall through to the generic input tiers — fail-open,
                # same contract as the Tom Select JS tiers.

            # Tier 1: fill() — fires input/change events
            try:
                await loc.fill(value, timeout=5000)
                try:
                    actual = await loc.input_value(timeout=2000)
                    if actual == value:
                        performed_actions.add(element_id)
                        return f"\n✅ AUTO-ACTION COMPLETE: Typed '{value}' (fill, verified)", "auto_ok"
                    # Mismatch — fall through to Tier 2
                except Exception:
                    # input_value() unsupported (e.g. contenteditable) — accept fill result
                    performed_actions.add(element_id)
                    return f"\n✅ AUTO-ACTION COMPLETE: Typed '{value}' (fill, unverified)", "auto_partial"
            except Exception:
                pass

            # Tier 2: triple_click + Control+a + press_sequentially — fires key events, better for reactive frameworks.
            # Control+a after triple_click ensures full selection on contenteditable: triple_click selects
            # only the clicked paragraph; Control+a selects all content across paragraphs. No-op on
            # <input>/<textarea> where triple_click already selects everything.
            try:
                await loc.triple_click(timeout=3000)
                await loc.press("Control+a")
                await loc.press_sequentially(value, delay=20)
                try:
                    actual = await loc.input_value(timeout=2000)
                    performed_actions.add(element_id)
                    return (
                        (f"\n✅ AUTO-ACTION COMPLETE: Typed '{value}' (key events, verified)", "auto_ok")
                        if actual == value
                        else (f"\n⚠️ AUTO-ACTION: Typed '{value}' but field shows '{actual}' (key events)", "auto_partial")
                    )
                except Exception:
                    performed_actions.add(element_id)
                    return f"\n✅ AUTO-ACTION COMPLETE: Typed '{value}' (key events, unverified)", "auto_partial"
            except Exception:
                pass

            # Tier 3: JS value assignment + dispatch events — last resort
            try:
                await loc.evaluate(
                    "(el, v) => { el.value = v; el.dispatchEvent(new Event('input', {bubbles: true})); el.dispatchEvent(new Event('change', {bubbles: true})); }",
                    value,
                )
                performed_actions.add(element_id)
                return f"\n✅ AUTO-ACTION COMPLETE: Typed '{value}' (JS, unverified)", "auto_partial"
            except Exception as e:
                return f"\n⚠️ AUTO-ACTION FAILED (input): all strategies exhausted — {e}", "auto_failed"

        elif action in ("click", "submit"):
            # Tier 1: standard click
            try:
                await loc.click(timeout=5000)
                performed_actions.add(element_id)
                return "\n✅ AUTO-ACTION COMPLETE: Clicked", "auto_ok"
            except Exception:
                pass

            # Tier 2: force click — bypasses visibility/actionability (element under overlay)
            try:
                await loc.click(force=True, timeout=5000)
                performed_actions.add(element_id)
                return "\n✅ AUTO-ACTION COMPLETE: Clicked (force)", "auto_ok"
            except Exception:
                pass

            # Tier 3: JS click — bypasses all Playwright checks and overlays
            try:
                await loc.evaluate("el => el.click()")
                performed_actions.add(element_id)
                return "\n✅ AUTO-ACTION COMPLETE: Clicked (JS)", "auto_ok"
            except Exception as e:
                return f"\n⚠️ AUTO-ACTION FAILED (click): all strategies exhausted — {e}", "auto_failed"

        elif action == "select":
            if not value:
                return "", "not_applicable"

            # Tom Select: JavaScript is the only reliable path.
            # The underlying <select> is hidden (display:none) so native select_option() always
            # fails. Run JS strategies before click-to-open — no wasted timeout attempts on
            # native options that are structurally guaranteed to fail.
            if dropdown_framework == "tom-select":
                # Tier 0 (preferred): direct getElementById when select_id is known.
                # Bypasses all DOM traversal — most reliable path, zero UI interaction needed.
                if select_id:
                    try:
                        diag = await active_page.evaluate(
                            """(args) => {
                                try {
                                    const select = document.getElementById(args.selectId);
                                    if (!select) return 'no_select';
                                    const ts = select.tomselect || select._tomSelect;
                                    if (!ts) return 'no_ts';
                                    const v = args.value;
                                    const opt = Array.from(select.options).find(
                                        o => o.text.trim() === v
                                          || o.value === v
                                          || o.text.trim().toLowerCase() === v.toLowerCase()
                                    );
                                    if (!opt) return 'no_opt';
                                    ts.setValue(opt.value, false);
                                    select.dispatchEvent(new Event('change', {bubbles: true}));
                                    return 'ok';
                                } catch(e) {
                                    return 'error:' + e.message;
                                }
                            }""",
                            {"selectId": select_id, "value": value},
                        )
                        if diag == "ok":
                            performed_actions.add(element_id)
                            return f"\n✅ AUTO-SELECT: Selected '{value}' (Tom Select JS via select_id)", "auto_ok"
                        else:
                            logger.info(f"   ⚠️ tom_select.tier0_js_failed: reason={diag} select_id={select_id!r} value={value!r}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ tom_select.tier0_exception: select_id={select_id!r} error={e}")

                # Tier 0b: locator-based traversal when select_id is unavailable.
                # Standard Tom Select DOM has <select> as the *previous sibling* of .ts-wrapper,
                # not a child. The old wrapper.querySelector('select') returned null for this
                # structure — fixed here by checking previousElementSibling first.
                try:
                    diag = await loc.evaluate(
                        """(el, targetText) => {
                            try {
                                const wrapper = el.closest('.ts-wrapper') || el.parentElement;
                                if (!wrapper) return 'no_wrapper';
                                let select = null;
                                if (wrapper.previousElementSibling && wrapper.previousElementSibling.tagName === 'SELECT')
                                    select = wrapper.previousElementSibling;
                                if (!select && wrapper.parentElement)
                                    select = wrapper.parentElement.querySelector('select');
                                if (!select) return 'no_select';
                                const ts = select.tomselect || select._tomSelect;
                                if (!ts) return 'no_ts';
                                const opt = Array.from(select.options).find(
                                    o => o.text.trim() === targetText
                                      || o.value === targetText
                                      || o.text.trim().toLowerCase() === targetText.toLowerCase()
                                );
                                if (!opt) return 'no_opt';
                                ts.setValue(opt.value, false);
                                select.dispatchEvent(new Event('change', {bubbles: true}));
                                return 'ok';
                            } catch(e) {
                                return 'error:' + e.message;
                            }
                        }""",
                        value,
                    )
                    if diag == "ok":
                        performed_actions.add(element_id)
                        return f"\n✅ AUTO-SELECT: Selected '{value}' (Tom Select JS via locator)", "auto_ok"
                    else:
                        logger.info(f"   ⚠️ tom_select.tier0b_js_failed: reason={diag} locator={locator_str!r} value={value!r}")
                except Exception as e:
                    logger.warning(f"   ⚠️ tom_select.tier0b_exception: locator={locator_str!r} error={e}")

                # Fall through to click-to-open as last resort for Tom Select.

            else:
                # Non-Tom-Select: try native select_option first.

                # Tier 1a: native <select> by visible label text
                try:
                    await loc.select_option(label=value, timeout=2000)
                    performed_actions.add(element_id)
                    return f"\n✅ AUTO-SELECT: Selected '{value}' (native label)", "auto_ok"
                except Exception:
                    pass

                # Tier 1b: native <select> by HTML value attribute
                try:
                    await loc.select_option(value=value, timeout=2000)
                    performed_actions.add(element_id)
                    return f"\n✅ AUTO-SELECT: Selected '{value}' (native value attr)", "auto_ok"
                except Exception:
                    pass

            # Tier 2: click-to-open — last resort for Tom Select, primary UI path for all others
            # (Select2, ARIA listbox, Bootstrap select, any custom dropdown).
            dropdown_opened = False
            try:
                await loc.click(timeout=3000)
                dropdown_opened = True

                # Adaptive wait: block until any dropdown container is actually visible.
                # Let wait_for_selector raise on timeout — the outer except catches it and
                # falls through to the final auto_failed return below.
                container_selectors = (
                    ".ts-dropdown-content, [role='listbox'], "
                    ".select2-results__options, .dropdown-menu"
                )
                await active_page.wait_for_selector(
                    container_selectors, state="visible", timeout=2000
                )

                parent = loc.locator("..")
                option_selectors = [
                    ".ts-dropdown-content",
                    "[role='listbox']",
                    ".select2-results__options",
                    ".dropdown-menu",
                ]

                # Exact match — parent-scoped first to avoid multi-dropdown ambiguity,
                # then page-global as fallback. .first.click() directly — no count() check
                # needed; absent/uninteractable raises and except continues to next strategy.
                for scope in [parent, active_page]:
                    for sel in option_selectors:
                        try:
                            await scope.locator(sel).get_by_text(value, exact=True).first.click(timeout=1500)
                            dropdown_opened = False
                            performed_actions.add(element_id)
                            return f"\n✅ AUTO-SELECT: Selected '{value}' (click-to-open, exact, {sel})", "auto_ok"
                        except Exception:
                            continue

                # Partial match — handles minor whitespace/casing differences in option text
                for scope in [parent, active_page]:
                    for sel in option_selectors:
                        try:
                            await scope.locator(sel).get_by_text(value).first.click(timeout=1500)
                            dropdown_opened = False
                            performed_actions.add(element_id)
                            return f"\n✅ AUTO-SELECT: Selected '{value}' (click-to-open, partial, {sel})", "auto_ok"
                        except Exception:
                            continue

            except Exception:
                pass
            finally:
                # Always close dropdown on exit — an open dropdown overlays the page and
                # blocks all subsequent element interactions.
                if dropdown_opened:
                    try:
                        await active_page.keyboard.press("Escape")
                        await active_page.wait_for_timeout(100)
                    except Exception:
                        pass

            return f"\n⚠️ AUTO-SELECT FAILED: all strategies exhausted for '{value}'", "auto_failed"

        elif action == "check":
            # State-aware: only clicks if not already checked
            try:
                await loc.check(timeout=5000)
                performed_actions.add(element_id)
                return "\n✅ AUTO-ACTION COMPLETE: Checked", "auto_ok"
            except Exception as e:
                return f"\n⚠️ AUTO-ACTION FAILED (check): {e}", "auto_failed"

        elif action == "uncheck":
            # State-aware: only clicks if currently checked
            try:
                await loc.uncheck(timeout=5000)
                performed_actions.add(element_id)
                return "\n✅ AUTO-ACTION COMPLETE: Unchecked", "auto_ok"
            except Exception as e:
                return f"\n⚠️ AUTO-ACTION FAILED (uncheck): {e}", "auto_failed"

    except Exception as e:
        return f"\n⚠️ AUTO-ACTION FAILED ({action}): {e}", "auto_failed"


async def _do_interaction(
    browser_session,
    active_page,
    locator_str: str,
    element_id: str,
    element_index: int | None,
    element_specs: dict,
    performed_actions: set,
    dropdown_framework: str = "",
    select_id: str | None = None,
    datepicker_framework: str = "",
):
    """Orchestrate the interaction for one element — event path first, Playwright fallback second.

    Returns (note, status, action, value). action and value are returned alongside note/status
    so the caller never needs to re-read element_specs independently (single source of truth)."""
    if element_id in performed_actions:
        return "", "not_applicable", "", ""
    spec = element_specs.get(element_id, {})
    action = spec.get("action", "get_text")
    value  = spec.get("value", "")
    if action not in ("input", "type", "click", "submit", "select", "check", "uncheck"):
        return "", "not_applicable", action, value

    # Tom Select: the ts-control is an <input>, so the agent labels it "input"/"type".
    # Typing into it filters the dropdown but never commits a selection.
    # Remap to "select" so the JS setValue() path runs instead of TypeTextEvent.
    if dropdown_framework == "tom-select" and action in ("input", "type"):
        action = "select"

    # Strip Robot Framework "label    X" / "value    X" strategy prefix so the
    # bare option text reaches JS comparisons and get_by_text() calls.
    if action == "select":
        raw_value = value
        value = _strip_rf_select_prefix(value)
        if value != raw_value:
            prefix = raw_value.split(None, 1)[0]
            logger.info(f"   🔧 RF prefix stripped for {element_id}: '{raw_value}' → '{value}' (prefix='{prefix}')")
        else:
            parts = raw_value.split(None, 1)
            if len(parts) == 2:
                logger.info(f"   ℹ️  select value for {element_id}: '{value}' (first token '{parts[0]}' not an RF prefix — kept as-is)")
            else:
                logger.debug(f"   ℹ️  select value for {element_id}: '{value}' (single token — no prefix possible)")

    # check/uncheck skip the event path entirely — browser-use has no state-aware check event.
    # Tom Select skips it too: SelectDropdownOptionEvent targets native <select> elements;
    # the indexed node is the ts-control <input>, so the event always returns (None, None).
    # Calling get_element_by_index() (a real CDP round-trip) only to get (None, None) from
    # _dispatch_browser_use_event is wasteful. Playwright's loc.check() / loc.uncheck() are
    # state-aware; go there directly without any event-path overhead.
    # flatpickr skips it for the same reason: TypeTextEvent "succeeds" against the
    # readonly input without changing widget state — only the Playwright path's
    # setDate JS tier actually sets the date.
    if (action not in ("check", "uncheck")
            and dropdown_framework != "tom-select"
            and datepicker_framework != "flatpickr"):
        if element_index is not None and browser_session is not None:
            try:
                node = await asyncio.wait_for(
                    browser_session.get_element_by_index(element_index),
                    timeout=10,
                )
                if node is not None:
                    note, status = await _dispatch_browser_use_event(
                        browser_session, node, action, value, element_id, performed_actions
                    )
                    if note is not None:  # None = fall-through signal; Playwright handles it
                        if action in ("click", "submit", "select") and status == "auto_ok":
                            await _wait_for_page_stability(active_page)
                        return note, status, action, value
            except Exception as e:
                logger.warning(f"   AUTO-ACTION event path failed for {element_id}: {e}")

    # Fallback: Playwright layered retry chain (always used for check/uncheck; fallback for others)
    note, status = await _do_interaction_playwright(
        active_page, locator_str, action, value, element_id, performed_actions,
        dropdown_framework=dropdown_framework, select_id=select_id,
        datepicker_framework=datepicker_framework,
    )
    if action in ("click", "submit", "select") and status == "auto_ok":
        await _wait_for_page_stability(active_page)
    return note, status, action, value


def register_custom_actions(agent, page=None, elements=None) -> bool:
    """
    Register custom actions with browser-use agent.

    This function registers the find_unique_locator custom action that allows
    the agent to call deterministic Python code for locator finding and validation.

    Args:
        agent: Browser-use Agent instance
        page: Unused — kept for API compatibility. The Playwright page is obtained
              lazily via CDP inside the handler (_ensure_playwright). Do not rely on
              this parameter.
        elements: Optional list of element specs [{"id": "elem_1", "action": "get_text", ...}].
                  When provided, completion is tracked via a closure dict. Once all expected
                  element IDs have been successfully processed, is_done=True is returned so the
                  browser-use agent loop terminates automatically — no LLM call to done() needed.

    Returns:
        bool: True if registration succeeded, False otherwise

    Phase: Custom Action Implementation
    Requirements: 3.1, 8.1, 9.1
    """
    try:
        logger.info("🔧 Registering custom actions with browser-use agent...")

        # Import required classes for custom action registration
        from browser_use.tools.service import Tools
        from browser_use.agent.views import ActionResult
        from pydantic import BaseModel, Field
        from typing import Literal

        # Import the action implementation
        from browser_service.agent.actions import find_unique_locator_action

        # ========================================
        # COMPLETION TRACKING (closure variables)
        # ========================================
        # Build a map of expected element IDs → action types from the elements list.
        # The inner action handler updates _completed_elements on each success and sets
        # is_done=True once every expected element ID has been processed, terminating
        # the browser-use agent loop without relying on the LLM to call done().
        _element_specs: dict = {}
        if elements:
            for _i, _elem in enumerate(elements):
                _eid = _elem.get("id", f"elem_unknown_{_i}")
                _element_specs[_eid] = {
                    "action": _elem.get("action", "get_text"),
                    "value": _elem.get("value", "") or "",
                }
        _expected_element_ids: set = set(_element_specs.keys())
        _total_expected: int = len(_expected_element_ids)
        _completed_elements: dict = {}  # element_id → best_locator (mutated by inner function)
        _performed_actions: set = set()  # idempotency guard — per register_custom_actions() call = per workflow

        # ========================================
        # PER-RUN PLAYWRIGHT CACHE (Change A — Day 04)
        # ========================================
        # One Playwright connection per agent.run() invocation, opened lazily on the
        # first find_unique_locator call and reused for every subsequent call in the same
        # run. Torn down by _teardown_playwright() in the workflow finally block.
        pw_cache: dict = {"instance": None, "browser": None, "page": None}

        async def _ensure_playwright(bs):
            """
            Lazy connect on first call; reuse on subsequent calls within the same run.

            Probe 20 (day-04-preflight) FAILED: session.reconnect() does not recover
            from BrowserStopEvent teardown (session was already destroyed before the
            probe could run). Per the spec FAIL path (option a), a stale connection
            mid-run is treated as fatal — _PlaywrightConnectionError is raised so the
            agent surfaces the error and downstream pipelines can retry the workflow.
            """
            if pw_cache["page"] is not None:
                # Health-check the cached connection before reusing it.
                # Primary: browser-use's CDP liveness flag (no round-trip).
                # Secondary: lightweight CDP round-trip to confirm Playwright side is alive.
                try:
                    if bs is not None and not bs.is_cdp_connected:
                        raise _PlaywrightConnectionError("session.is_cdp_connected is False")
                    await pw_cache["page"].evaluate("1")  # minimal CDP round-trip
                except Exception as exc:
                    logger.warning(f"Stale Playwright connection ({exc}); tearing down.")
                    # Probe 20 FAIL: reconnect() cannot recover from BrowserStopEvent
                    # teardown. Surface as fatal rather than attempting reconnect.
                    try:
                        if pw_cache["browser"] is not None:
                            await pw_cache["browser"].close()
                    except Exception as cleanup_exc:
                        logger.debug(f"browser.close() failed during stale-connection cleanup: {cleanup_exc}")
                    try:
                        if pw_cache["instance"] is not None:
                            await pw_cache["instance"].stop()
                    except Exception as cleanup_exc:
                        logger.debug(f"instance.stop() failed during stale-connection cleanup: {cleanup_exc}")
                    pw_cache.update({"instance": None, "browser": None, "page": None})
                    raise _PlaywrightConnectionError(
                        f"Playwright connection became stale mid-run ({exc}). "
                        "Probe 20 confirmed reconnect() cannot recover from "
                        "BrowserStopEvent teardown — surfacing error to agent."
                    ) from exc

                # Connection is alive. Re-select if the cached page became blank
                # (e.g. Chrome exposed an empty leading context on connect, or a new
                # tab was opened and the original page navigated away to about:blank).
                _cached_url = pw_cache["page"].url
                if not _cached_url or _cached_url == "about:blank":
                    _better = _select_best_page(pw_cache["browser"])
                    if _better is not None:
                        logger.info(
                            f"Playwright: re-selected active page "
                            f"({_cached_url!r} → {_better.url!r})"
                        )
                        pw_cache["page"] = _better
                return pw_cache["page"]

            # Fresh connect (first call this run).
            from playwright.async_api import async_playwright as _async_playwright
            cdp_url = _get_cdp_url_from_session(bs)
            if not cdp_url:
                raise _PlaywrightConnectionError("No CDP URL on browser_session — cannot connect Playwright")
            instance = await _async_playwright().start()
            try:
                connected = await instance.chromium.connect_over_cdp(cdp_url)
                # Select the most recently opened non-blank page. Chrome sometimes
                # exposes an empty leading context before the real tab is ready.
                page_obj = _select_best_page(connected)
                if page_obj is None:
                    # No non-blank page yet — fall back to first available page or fail.
                    contexts = connected.contexts
                    if not contexts or not contexts[0].pages:
                        raise _PlaywrightConnectionError(
                            f"CDP browser has no usable page "
                            f"(contexts={len(contexts)}, "
                            f"pages={len(contexts[0].pages) if contexts else 0})"
                        )
                    page_obj = contexts[0].pages[0]
                    logger.warning(
                        f"No non-blank page found; using contexts[0].pages[0] "
                        f"({page_obj.url!r})"
                    )
                try:
                    await page_obj.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception as e:
                    logger.warning(f"wait_for_load_state: {e}")
            except Exception:
                # connect_over_cdp (or page retrieval) failed — stop the instance we
                # already started so it doesn't leak, then re-raise to the caller.
                try:
                    await instance.stop()
                except Exception:
                    pass
                raise
            pw_cache.update({"instance": instance, "browser": connected, "page": page_obj})
            logger.info("✅ Playwright connected (lazy, once-per-run)")
            return page_obj

        async def _teardown_playwright():
            """Called from the workflow finally block when agent.run() returns."""
            try:
                if pw_cache["browser"] is not None:
                    await pw_cache["browser"].close()
            except Exception as e:
                logger.debug(f"Playwright browser close during teardown: {e}")
            try:
                if pw_cache["instance"] is not None:
                    await pw_cache["instance"].stop()
            except Exception as e:
                logger.debug(f"Playwright instance stop during teardown: {e}")
            pw_cache.update({"instance": None, "browser": None, "page": None})

        # Define parameter model for find_unique_locator action
        class FindUniqueLocatorParams(BaseModel):
            """Parameters for find_unique_locator custom action"""
            x: float = Field(description="X coordinate of element center")
            y: float = Field(description="Y coordinate of element center")
            element_id: str = Field(description="Element identifier (elem_1, elem_2, etc.)")
            element_description: str = Field(description="Human-readable description of element")
            expected_text: Optional[str] = Field(
                default=None,
                description="The ACTUAL visible text seen on the element (e.g., 'Submit', 'Nike Air Max 270'). Used for semantic validation to ensure we found the correct element."
            )
            candidate_locator: Optional[str] = Field(
                default=None,
                description="Optional candidate locator to validate first (e.g., 'id=search-input')"
            )
            element_index: Optional[int] = Field(
                default=None,
                description="Element index from browser state (e.g., 23 from '[23] Services'). When provided, we get the exact element from browser-use's DOM, ensuring precise locator generation. HIGHLY RECOMMENDED for accuracy."
            )
            is_collection: Optional[bool] = Field(
                default=None,
                description="Set to true if this element represents a COLLECTION (e.g., table rows, list items). When true, returns multi-element locator instead of single-element locator."
            )
            row_anchor_text: Optional[str] = Field(
                default=None,
                description=(
                    "When the target is a PER-ROW action inside a table or "
                    "repeated list (Edit/Delete/Download icon, row checkbox, "
                    "row link) and the task identifies WHICH row by its data, "
                    "pass that row-identifying text EXACTLY as visible in the "
                    "row. Example: task 'Edit customer 64625' -> "
                    "row_anchor_text='64625'. The engine anchors the locator "
                    "to the row containing this text instead of a brittle "
                    "positional index. Leave blank for elements that appear "
                    "once on the page (toolbar buttons, form fields)."
                )
            )
            # Vision-derived classification piggybacked on the per-step LLM
            # call. The locator pipeline corroborates this against a live
            # Playwright DOM probe before committing to a specialized
            # handler — the hint is one of two required sources of truth,
            # never unilateral. See docs/ELEMENT_TYPE_CLASSIFIER_ARCHITECTURE.md.
            element_type: Optional[Literal[
                "dropdown", "checkbox", "radio", "input", "button",
                "link", "image", "label", "text-area", "table",
                "file-upload", "date-picker", "other",
            ]] = Field(
                default=None,
                description=(
                    "Based on what you SEE in the screenshot at these "
                    "coordinates, which UI control type is this? Pick the "
                    "best match from the listed values. Only set this when "
                    "you are confident from visual inspection — leave blank "
                    "if uncertain. The deterministic locator code uses this "
                    "to route to a specialized handler."
                )
            )
            framework_hint: Optional[Literal[
                # Dropdown frameworks
                "tom-select", "select2", "kendo", "react-select",
                "vue-select", "ant-design", "material-ui",
                # Date-picker frameworks
                "flatpickr",
                # Table / grid frameworks
                "datatables", "ag-grid", "material-table", "react-table",
                # Generic — visible widget looks like a plain HTML control
                "native",
                # Vision sees a custom widget that doesn't match any of the above
                "other",
            ]] = Field(
                default=None,
                description=(
                    "If the visible widget matches a known UI framework's "
                    "appearance, name it. Applies to any specialized type "
                    "(dropdown, checkbox, radio, table, etc.) — examples: "
                    "Tom Select's pill-shaped tag input, Select2's caret, "
                    "Material-UI's floating-label inputs, DataTables' "
                    "search/pagination chrome, AG-Grid's cell editors. "
                    "Pick 'native' for plain HTML controls without a "
                    "framework wrapper. Leave blank if uncertain."
                )
            )

        # Get or create Tools instance from agent
        if not hasattr(agent, 'tools') or agent.tools is None:
            logger.info("   Creating new Tools instance for agent")
            tools = Tools()
            agent.tools = tools
        else:
            logger.info("   Using existing Tools instance from agent")
            tools = agent.tools

        # Register the find_unique_locator action
        @tools.registry.action(
            description="Find and validate unique locator for element at coordinates using 21 systematic strategies. "
                        "This action runs deterministically without LLM calls and validates all locators with Playwright. "
                        "Call this action after finding an element's coordinates to get a validated unique locator.",
            param_model=FindUniqueLocatorParams
        )
        async def find_unique_locator(
            params: FindUniqueLocatorParams,
            browser_session
        ) -> ActionResult:
            """
            Custom action wrapper that calls find_unique_locator_action.

            This function is called by the browser-use agent when it needs to find
            a unique locator for an element. It wraps the find_unique_locator_action
            function and returns results in ActionResult format.

            The browser_session parameter is provided by browser-use and contains
            the active browser context with the page that's currently open.
            """
            try:
                logger.info("🎯 Custom action 'find_unique_locator' called by agent")
                logger.info(f"   Element: {params.element_id} - {params.element_description}")
                logger.info(f"   Coordinates: ({params.x}, {params.y})")
                if params.expected_text:
                    logger.info(f"   Expected text: \"{params.expected_text}\"")
                
                # ALWAYS log element_index to debug what LLM is passing
                logger.info(f"   Element index: {params.element_index} (None means LLM did not provide it)")

                # ========================================
                # ELEMENT INDEX: Get element directly from browser-use DOM
                # ========================================
                # When element_index is provided, we can get the exact element from
                # browser-use's DOM state. This gives us:
                # 1. Accurate element attributes (id, class, text, aria-label, etc.)
                # 2. Confirmed bounding box coordinates (actual position, not LLM guess)
                # 3. Much higher accuracy for locator generation
                
                element_data_from_index = None
                confirmed_coords = None
                selector_map = None  # Will be populated either from element_index lookup or CDP fallback
                
                if params.element_index is not None and browser_session:
                    try:
                        logger.info(f"📋 Getting element [{params.element_index}] from browser-use DOM...")
                        
                        # ========================================
                        # Get selector_map from browser-use DOM watchdog
                        # ========================================
                        # When element_index is provided, it came from browser-use's snapshot,
                        # so the element WILL be in the selector_map (same data source).
                        if hasattr(browser_session, '_dom_watchdog') and browser_session._dom_watchdog:
                            watchdog = browser_session._dom_watchdog
                            if hasattr(watchdog, 'selector_map') and watchdog.selector_map:
                                selector_map = watchdog.selector_map
                                logger.info(f"   📊 Using selector_map: {len(selector_map)} elements")
                        
                        # Fallback to get_selector_map() if watchdog not available
                        if not selector_map:
                            selector_map = await browser_session.get_selector_map()
                            logger.info(f"   📊 Fallback to get_selector_map(): {len(selector_map) if selector_map else 0} elements")
                        
                        # Log diagnostics
                        if selector_map:
                            available_indices = sorted(selector_map.keys())
                            if available_indices:
                                logger.info(f"   📊 Index range: {min(available_indices)} - {max(available_indices)}")
                            # Log sample elements to verify table cells (td/th) are indexed
                            sample_types = {}
                            for idx in available_indices[:50]:
                                tag = selector_map[idx].node_name.upper() if hasattr(selector_map[idx], 'node_name') else '?'
                                sample_types[tag] = sample_types.get(tag, 0) + 1
                            logger.info(f"   📊 Element types in sample: {dict(sorted(sample_types.items(), key=lambda x: -x[1]))}")
                        
                        # Look up element from selector_map
                        dom_node = selector_map.get(params.element_index) if selector_map else None
                        
                        if dom_node:
                            logger.info(f"   ✅ Found element [{params.element_index}] in DOM")
                            
                            # Extract element attributes for locator generation
                            element_data_from_index = _extract_dom_node_attributes(dom_node)
                            
                            # Get text content from the element
                            if hasattr(dom_node, 'get_meaningful_text_for_llm'):
                                element_data_from_index['textContent'] = dom_node.get_meaningful_text_for_llm()
                            elif hasattr(dom_node, 'get_all_children_text'):
                                element_data_from_index['textContent'] = dom_node.get_all_children_text()
                            
                            # Get confirmed coordinates from bounding box
                            if hasattr(dom_node, 'absolute_position') and dom_node.absolute_position:
                                pos = dom_node.absolute_position
                                confirmed_coords = (
                                    int(pos.x + pos.width / 2),
                                    int(pos.y + pos.height / 2)
                                )
                                logger.info(f"   📍 Confirmed coordinates: {confirmed_coords} (from DOM bounding box)")
                            
                            logger.info(f"   📝 Element tag: <{element_data_from_index['tagName']}>")
                            if element_data_from_index.get('id'):
                                logger.info(f"   📝 Element id: {element_data_from_index['id']}")
                            if element_data_from_index.get('xpath'):
                                logger.info(f"   📝 Element xpath: {element_data_from_index['xpath']}")
                            if element_data_from_index.get('textContent'):
                                text_preview = element_data_from_index['textContent'][:50]
                                logger.info(f"   📝 Element text: \"{text_preview}...\"" if len(element_data_from_index.get('textContent', '')) > 50 else f"   📝 Element text: \"{element_data_from_index['textContent']}\"")
                        else:
                            logger.warning(f"   ⚠️ Element [{params.element_index}] not found in selector_map (available indices: {sorted(selector_map.keys()) if selector_map else 'none'})")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Could not get element by index: {e}")
                        logger.debug("   Full error:", exc_info=True)


                # ========================================
                # COORDINATE SCALING: Vision AI → Viewport Pixels
                # ========================================
                # Vision AI (Gemini) uses a normalized coordinate space [0-1000]
                # when identifying element positions from screenshots. This is
                # different from actual CSS pixel coordinates used by the DOM.
                #
                # WHY THIS IS NEEDED:
                # - element_index is only available for INTERACTABLE elements
                # - For non-interactable elements (table cells, text spans, labels),
                #   we rely on coordinate-based lookup using Vision AI coords
                # - Without scaling, coordinate lookups fail for table extraction
                #   (e.g., "get text from first row, second column")
                #
                # COORDINATE SOURCES:
                # - Vision AI (Gemini): Normalized [0-1000] space → NEEDS SCALING
                # - DOM element_index: CSS pixel coords → NO SCALING (handled above)
                #
                # Scaling formula: pixel_coord = (normalized_coord / 1000) * viewport_size
                GEMINI_COORD_SPACE = 1000
                DEFAULT_VIEWPORT = (1920, 1080)
                
                viewport_size = getattr(browser_session, '_original_viewport_size', None) or DEFAULT_VIEWPORT
                viewport_w, viewport_h = viewport_size
                
                scaled_x = int((params.x / GEMINI_COORD_SPACE) * viewport_w)
                scaled_y = int((params.y / GEMINI_COORD_SPACE) * viewport_h)
                
                logger.info(f"Coordinate scaling: ({params.x}, {params.y}) → ({scaled_x}, {scaled_y}) [0-1000 to {viewport_w}x{viewport_h}]")


                # ========================================
                # FALLBACK: Find element from selector_map using coordinates
                # ========================================
                # When element_index is NOT provided (which is typical for custom
                # dropdowns, dynamically loaded elements, or complex components),
                # we can still get element_data by finding which element's bounding box
                # contains the given coordinates. This is more accurate than coordinate-based
                # JavaScript extraction.
                if element_data_from_index is None and browser_session:
                    try:
                        logger.info(f"🔍 STEP A: Finding element at ({scaled_x}, {scaled_y}) from selector_map...")
                        selector_map = await browser_session.get_selector_map()
                        
                        if selector_map:
                            # Log element types to verify what's indexed
                            sample_types = {}
                            for idx, elem in list(selector_map.items())[:100]:
                                tag = elem.node_name.upper() if hasattr(elem, 'node_name') else '?'
                                sample_types[tag] = sample_types.get(tag, 0) + 1
                            logger.info(f"📊 Selector map has {len(selector_map)} elements")
                            logger.info(f"📊 Types: {dict(sorted(sample_types.items(), key=lambda x: -x[1]))}")
                            
                            viewport_area = viewport_w * viewport_h
                            idx, dom_node = _find_smallest_containing_element(
                                selector_map,
                                (scaled_x, scaled_y),
                                viewport_area,
                            )

                            if dom_node is not None:
                                logger.info(f"   ✅ Found element [{idx}] at coordinates!")
                                elem_tag = dom_node.node_name if hasattr(dom_node, 'node_name') else 'unknown'
                                logger.info(f"   📝 Element tag: <{elem_tag}>")
                                
                                # Extract element attributes
                                element_data_from_index = _extract_dom_node_attributes(dom_node)
                                
                                # Get text content
                                if hasattr(dom_node, 'get_meaningful_text_for_llm'):
                                    element_data_from_index['textContent'] = dom_node.get_meaningful_text_for_llm()
                                elif hasattr(dom_node, 'get_all_children_text'):
                                    element_data_from_index['textContent'] = dom_node.get_all_children_text()
                                
                                # Get confirmed coordinates from bounding box
                                if hasattr(dom_node, 'absolute_position') and dom_node.absolute_position:
                                    pos = dom_node.absolute_position
                                    confirmed_coords = (
                                        int(pos.x + pos.width / 2),
                                        int(pos.y + pos.height / 2)
                                    )
                                    logger.info(f"   📍 Confirmed coordinates: {confirmed_coords}")
                                
                                if element_data_from_index.get('id'):
                                    logger.info(f"   📝 Element id: {element_data_from_index['id']}")
                            else:
                                logger.warning(f"   ⚠️ No element found at ({scaled_x}, {scaled_y})")
                        else:
                            logger.warning(f"   ⚠️ selector_map is empty or None")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Could not find element by coordinates: {e}")
                        logger.debug("   Full error:", exc_info=True)

                # ========================================
                # FALLBACK-FIRST APPROACH for element_index
                # ========================================
                # IMPORTANT: Check params.element_index (what LLM provided), not element_data_from_index
                # (which may have been populated by coordinate lookup above)
                element_index_was_none = (params.element_index is None)
                
                if element_index_was_none and element_data_from_index is None:
                    # Coordinate lookup also failed - will rely on smart_locator.py fallbacks
                    logger.info(f"⚠️ element_index not provided AND coordinate lookup failed")
                    logger.info(f"   Will try TEXT-FIRST/SEMANTIC/COORDINATE strategies in smart_locator.py")
                    logger.info(f"   If all fail, will request LLM retry with element_index")
                    
                    # Fetch selector_map for iframe detection
                    try:
                        if hasattr(browser_session, '_cached_selector_map') and browser_session._cached_selector_map:
                            selector_map = browser_session._cached_selector_map
                            logger.info(f"   📊 Got selector_map: {len(selector_map)} elements")
                    except Exception as e:
                        logger.debug(f"   Could not get selector_map: {e}")




                # Lazy connect: one Playwright connection per agent.run(), reused across all calls.
                active_page = await _ensure_playwright(browser_session)

                final_x, final_y = scaled_x, scaled_y
                if confirmed_coords:
                    final_x, final_y = confirmed_coords
                    logger.info(f"Using DOM-confirmed coordinates: ({final_x}, {final_y})")
                else:
                    logger.info(f"Using scaled coordinates: ({final_x}, {final_y})")
                    
                    
                # ========================================
                # IFRAME DETECTION: Check if element is inside an iframe
                # ========================================
                iframe_context = None
                if selector_map:
                    iframe_context, _ = _detect_iframe_context(
                        selector_map, (final_x, final_y)
                    )
                    if iframe_context:
                        logger.info(f"   Element will be searched inside iframe: {iframe_context}")
                            
                        # ========================================
                        # IFRAME ELEMENT HANDLING (Optional Refinement)
                        # ========================================
                        # COORDINATE ASSUMPTION: Browser-use provides page-absolute coordinates
                        # via `absolute_position` which includes accumulated `total_frame_offset`
                        # from parent iframes (see browser_use/dom/service.py lines 530-535).
                        #
                        # EDGE CASES where this may fail silently:
                        # - Cross-origin iframes (may have iframe-relative coords)
                        # - Dynamically loaded content (not yet indexed)
                        #
                        # GRACEFUL FALLBACK: If bbox matching fails, find_unique_locator_action
                        # handles iframe_context properly with coordinate translation.
                        # This override is an OPTIMIZATION, not required for correctness.
                            
                        logger.info(f"🖼️ Iframe detected - attempting element lookup from selector_map")
                        logger.info(f"   📊 Selector map has {len(selector_map)} elements")
                            
                        try:
                            # Find element by coordinates in selector_map — skip the iframe
                            # itself (we want the contained element) and page wrappers
                            # (see _find_smallest_containing_element for wrapper rationale).
                            if selector_map:
                                viewport_area = viewport_w * viewport_h
                                idx, dom_node = _find_smallest_containing_element(
                                    selector_map,
                                    (final_x, final_y),
                                    viewport_area,
                                    skip_tag='IFRAME',
                                )

                                if dom_node is not None:
                                    logger.info(f"   ✅ Found element [{idx}] inside iframe!")
                                    elem_tag = dom_node.node_name if hasattr(dom_node, 'node_name') else 'unknown'
                                    logger.info(f"   📝 Element tag: <{elem_tag}>")
                                        
                                    # Update element_data_from_index with the correct element
                                    element_data_from_index = _extract_dom_node_attributes(dom_node)
                                    if hasattr(dom_node, 'get_meaningful_text_for_llm'):
                                        element_data_from_index['textContent'] = dom_node.get_meaningful_text_for_llm()
                                    elif hasattr(dom_node, 'get_all_children_text'):
                                        element_data_from_index['textContent'] = dom_node.get_all_children_text()
                                        
                                    logger.info(f"   📝 Element id: {element_data_from_index.get('id', 'N/A')}")
                                else:
                                    logger.info(f"   ℹ️ No bbox match found - will use find_unique_locator_action fallback")
                                    logger.debug(f"   (This is normal for cross-origin iframes or coord system mismatch)")
                            else:
                                logger.info(f"   ℹ️ No selector_map available - will use find_unique_locator_action fallback")
                        except Exception as e:
                            logger.warning(f"   ⚠️ Element lookup failed: {e}")
                            logger.debug("   Full error:", exc_info=True)
                    
                result = await find_unique_locator_action(
                    x=final_x,  # Use confirmed or scaled coordinates
                    y=final_y,  # Use confirmed or scaled coordinates
                    element_id=params.element_id,
                    element_description=params.element_description,
                    expected_text=params.expected_text,  # Pass expected_text for semantic validation
                    candidate_locator=params.candidate_locator,
                    element_data=element_data_from_index,  # Pass element attributes from DOM
                    page=active_page,
                    iframe_context=iframe_context,  # Pass iframe context if detected
                    is_collection=params.is_collection,  # Pass collection flag for multi-element detection
                    browser_session=browser_session,  # For resolved_node lookup (DELTA 1)
                    vision_type_hint=params.element_type,  # LLM's visual type classification (1 of 2 sources)
                    vision_framework_hint=params.framework_hint,  # LLM's framework guess (any specialized type)
                    row_anchor_text=params.row_anchor_text,  # Row-scoped rescue for per-row actions (G1)
                )

                # Convert result to ActionResult format
                action_result = None
                if result.get('found'):
                    best_locator = result.get('best_locator')

                    # Get validation data from result (not validation_summary)
                    validated = result.get('validated', False)
                    count = result.get('count', 0)
                    validation_method = result.get('validation_method', 'playwright')

                    # ========================================
                    # COMPLETION TRACKING
                    # ========================================
                    # Record this element as done. Use the element_id as the dict key so
                    # repeated calls for the same element overwrite rather than double-count.
                    _completed_elements[params.element_id] = best_locator

                    # Determine whether every expected element has been processed.
                    # Guard: _total_expected == 0 means elements=None/[] was passed, so we
                    # never auto-terminate (original behaviour preserved).
                    all_elements_done = (
                        _total_expected > 0
                        and set(_completed_elements.keys()) >= _expected_element_ids
                    )

                    # Reached only on locator-find success — best_locator is non-None by construction here.
                    # Unpack 4-tuple — action and value come from _do_interaction directly so this block
                    # never needs to re-read _element_specs. Single source of truth for action/value.
                    interaction_note, interaction_status, _action, _value = await _do_interaction(
                        browser_session, active_page, best_locator, params.element_id, params.element_index,
                        _element_specs, _performed_actions,
                        dropdown_framework=result.get("dropdown_framework") or "",
                        select_id=result.get("select_id") or None,
                        datepicker_framework=result.get("datepicker_framework") or "",
                    )
                    result["interaction_status"] = interaction_status
                    _idx = params.element_index

                    # Build display_note for the success message.
                    # On auto_failed with element_index: name the exact browser-use native action + index.
                    # On auto_failed without element_index: provide the validated locator.
                    # On success/not_applicable: pass interaction_note through unchanged.
                    if interaction_status == "auto_failed":
                        if _action in ("click", "submit"):
                            if _idx is not None:
                                display_note = (
                                    f"\n⚠️ Automated click failed — all strategies exhausted. "
                                    f"FALLBACK REQUIRED: Call click_element(index={_idx}). "
                                    f"After clicking, wait for page updates — this may navigate to a new page."
                                )
                            else:
                                display_note = (
                                    f"\n⚠️ Automated click failed — all strategies exhausted. "
                                    f"Validated locator: {best_locator} | Action required: click. "
                                    f"Assess the current page state and use any available browser-use action "
                                    f"to complete this click. If no available action can complete it, proceed "
                                    f"to the next element — Robot Framework will handle it at execution time."
                                )
                        elif _action in ("input", "type"):
                            if _idx is not None:
                                display_note = (
                                    f"\n⚠️ Automated input failed — all strategies exhausted. "
                                    f'FALLBACK REQUIRED: Call input_text(index={_idx}, text="{_value}").'
                                )
                            else:
                                display_note = (
                                    f"\n⚠️ Automated input failed — all strategies exhausted. "
                                    f'Validated locator: {best_locator} | Action required: input "{_value}". '
                                    f"Assess the current page state and use any available browser-use action "
                                    f"to type this value. If no available action can complete it, proceed "
                                    f"to the next element — Robot Framework will handle it at execution time."
                                )
                        elif _action == "select":
                            if _idx is not None:
                                display_note = (
                                    f"\n⚠️ Automated select failed — all strategies exhausted. "
                                    f'FALLBACK REQUIRED: Call select_dropdown_option(index={_idx}, text="{_value}").'
                                )
                            else:
                                display_note = (
                                    f"\n⚠️ Automated select failed — all strategies exhausted. "
                                    f'Validated locator: {best_locator} | Action required: select "{_value}". '
                                    f"Assess the current page state and use any available browser-use action "
                                    f"to select this option. If no available action can complete it, proceed "
                                    f"to the next element — Robot Framework will handle it at execution time."
                                )
                        else:
                            display_note = (
                                "\n⚠️ Automated interaction failed — all strategies exhausted. "
                                "Locator is valid. Proceed to the next element."
                            )
                    else:
                        display_note = interaction_note  # ✅ message, empty string, or "not_applicable"

                    # Interactions are now performed inside _do_interaction (see Change C above).
                    # SEQUENTIAL_PROCESSING_RULES no longer owns interaction execution.
                    if all_elements_done:
                        elements_found_json = json.dumps(
                            [
                                {"element_id": eid, "best_locator": loc,
                                 "found": True, "validated": True, "count": 1}
                                for eid, loc in _completed_elements.items()
                            ]
                        )
                        success_msg = (
                            f"✅ LOCATOR VALIDATED BY PLAYWRIGHT\n"
                            f"Element: {params.element_id}\n"
                            f"Locator: {best_locator}"
                            f"{display_note}\n"
                            f"All {_total_expected} locators found. "
                            f'Call done() with: {{"elements_found": {elements_found_json}}}'
                        )
                    else:
                        success_msg = (
                            f"✅ LOCATOR VALIDATED BY PLAYWRIGHT\n"
                            f"Element: {params.element_id}\n"
                            f"Locator: {best_locator}"
                            f"{display_note}\n"
                            f"Proceed to the next element."
                        )
                    # Keep long_term_memory minimal — it persists across every subsequent agent
                    # step. Embedding action guidance here would bias future decisions.
                    long_term = f"{params.element_id} validated = {best_locator}"

                    logger.info(f"✅ Custom action succeeded: {best_locator}")
                    logger.info(f"   Completion: {len(_completed_elements)}/{_total_expected} all_done={all_elements_done}")

                    # Log if this was a fallback success (no element_index provided)
                    if element_index_was_none:
                        logger.info(f"")
                        logger.info(f"{'='*80}")
                        logger.info(f"✅ FALLBACK SUCCESS: Locator found WITHOUT element_index")
                        logger.info(f"{'='*80}")
                        logger.info(f"   Element: {params.element_id}")
                        logger.info(f"   Locator: {best_locator}")
                        logger.info(f"   Method: {validation_method} (no LLM retry needed)")
                        logger.info(f"{'='*80}")
                        logger.info(f"")

                    action_result = ActionResult(
                        extracted_content=success_msg,
                        long_term_memory=long_term,
                        metadata=result,
                        is_done=all_elements_done,
                    )

                else:
                    # Error message for agent - CLEAR about failure
                    fallback_error = result.get('error', 'Could not find unique locator')
                    logger.error(f"❌ Custom action failed: {fallback_error}")
                    
                    # ========================================
                    # RETRY WITH element_index (only if it was originally None)
                    # ========================================
                    # If element_index was not provided AND fallback failed,
                    # request LLM to retry with the correct element_index.
                    # This handles interactable elements with stale DOM.
                    if element_index_was_none:
                        retry_msg = (
                            f"Fallback strategies failed for '{params.element_id}'. "
                            f"Reason: {fallback_error}. "
                            f"This may be an interactable element that requires element_index. "
                            f"Please look at the current DOM state, find the element described as "
                            f"'{params.element_description}', identify its index number "
                            f"(e.g., [42] for index 42), and call find_unique_locator again "
                            f"with element_index set to that number. "
                            f"Example: if you see '[2181] <input role=\"combobox\">',"
                            f" use element_index=2181"
                        )
                        logger.info(f"")
                        logger.info(f"{'='*80}")
                        logger.info(f"🔄 RETRY TRIGGERED: element_index was None and fallback failed")
                        logger.info(f"{'='*80}")
                        logger.info(f"   Element: {params.element_id}")
                        logger.info(f"   Description: {params.element_description}")
                        logger.info(f"   Fallback failure reason: {fallback_error}")
                        logger.info(f"   Action: Requesting LLM to retry with element_index")
                        logger.info(f"{'='*80}")
                        logger.info(f"")
                        action_result = ActionResult(
                            extracted_content=retry_msg,
                            error=retry_msg
                        )
                    else:
                        # element_index WAS provided but still failed
                        action_result = ActionResult(
                            error=f"FAILED: Could not find unique locator for {params.element_id}. Error: {fallback_error}. Try different coordinates or description.",
                            is_done=False  # Let agent try again with different approach
                        )

                return action_result

            except _PlaywrightConnectionError as e:
                # _PlaywrightConnectionError from _ensure_playwright means the browser
                # connection is dead (stale CDP or failed fresh connect). Surface as
                # is_done=True so the agent stops immediately rather than burning steps
                # retrying against a permanently unavailable browser.
                error_msg = f"Error in find_unique_locator custom action: {str(e)}"
                logger.error(f"❌ {error_msg}", exc_info=True)
                return ActionResult(error=error_msg, is_done=True)
            except Exception as e:
                error_msg = f"Error in find_unique_locator custom action: {str(e)}"
                logger.error(f"❌ {error_msg}", exc_info=True)
                return ActionResult(error=error_msg)

        # Expose teardown so the workflow finally block can close the shared connection.
        agent._pw_teardown = _teardown_playwright

        logger.info("✅ Custom action 'find_unique_locator' registered successfully")
        logger.info("   Agent can now call: find_unique_locator(x, y, element_id, element_description, expected_text, candidate_locator, element_index, is_collection)")
        return True

    except Exception as e:
        # Log error but don't crash - allow fallback to legacy workflow
        logger.error(f"❌ Failed to register custom actions: {str(e)}")
        logger.error("   Stack trace:", exc_info=True)
        logger.warning("⚠️ Continuing with legacy workflow (custom actions disabled)")
        return False



