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

# Get logger
logger = logging.getLogger(__name__)

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
    iframe_ordinal = 0
    
    for _idx, elem in selector_map.items():
        if hasattr(elem, 'node_name') and elem.node_name.lower() == 'iframe':
            if hasattr(elem, 'absolute_position') and elem.absolute_position:
                pos = elem.absolute_position
                # Check if coordinates are within iframe bounds
                if (pos.x <= x <= pos.x + pos.width and
                    pos.y <= y <= pos.y + pos.height):
                    # Get iframe identifier
                    attrs = elem.attributes if hasattr(elem, 'attributes') else {}
                    iframe_id = attrs.get('id', '')
                    iframe_name = attrs.get('name', '')
                    
                    # Generate locator for the iframe using attribute selectors
                    # Escape special characters to prevent selector injection
                    if iframe_id:
                        # Escape \ and " for CSS attribute selector
                        iframe_id_escaped = iframe_id.replace('\\', '\\\\').replace('"', '\\"')
                        iframe_locator = f'iframe[id="{iframe_id_escaped}"]'
                    elif iframe_name:
                        iframe_name_escaped = iframe_name.replace('\\', '\\\\').replace('"', '\\"')
                        iframe_locator = f'iframe[name="{iframe_name_escaped}"]'
                    else:
                        # Fallback: use ordinal-based selector (0-indexed count of iframes)
                        iframe_locator = f"iframe >> nth={iframe_ordinal}"
                    
                    logger.info(f"🖼️ IFRAME DETECTED: Element at ({x}, {y}) is inside {iframe_locator}")
                    return iframe_locator, iframe_id or iframe_name or str(iframe_ordinal)
            iframe_ordinal += 1
    
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


def register_custom_actions(agent, page=None, elements=None, workflow_id: str = "") -> bool:
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

        # Import the action implementation
        from browser_service.agent.actions import find_unique_locator_action

        # Import settings (with fallback for standalone mode)
        try:
            from src.backend.core.config import settings as _nl_settings
        except ImportError:
            _nl_settings = None

        # Resolve timeout (default matches NL repo config)
        if _nl_settings is not None and hasattr(_nl_settings, 'CUSTOM_ACTION_TIMEOUT'):
            custom_action_timeout = _nl_settings.CUSTOM_ACTION_TIMEOUT
        else:
            custom_action_timeout = 5

        # ========================================
        # COMPLETION TRACKING (closure variables)
        # ========================================
        # Build a map of expected element IDs → action types from the elements list.
        # The inner action handler updates _completed_elements on each success and sets
        # is_done=True once every expected element ID has been processed, terminating
        # the browser-use agent loop without relying on the LLM to call done().
        _element_specs: dict = {}  # element_id → action string
        if elements:
            for _i, _elem in enumerate(elements):
                _eid = _elem.get("id", f"elem_unknown_{_i}")
                _element_specs[_eid] = _elem.get("action", "get_text")
        _expected_element_ids: set = set(_element_specs.keys())
        _total_expected: int = len(_expected_element_ids)
        _completed_elements: dict = {}  # element_id → best_locator (mutated by inner function)

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
            mid-run is treated as fatal — RuntimeError is raised so the agent surfaces
            the error and downstream pipelines can retry the workflow.
            """
            if pw_cache["page"] is not None:
                # Health-check the cached connection before reusing it.
                # Primary: browser-use's CDP liveness flag (no round-trip).
                # Secondary: lightweight CDP round-trip to confirm Playwright side is alive.
                try:
                    if bs is not None and not bs.is_cdp_connected:
                        raise RuntimeError("session.is_cdp_connected is False")
                    await pw_cache["page"].evaluate("1")  # minimal CDP round-trip
                    return pw_cache["page"]
                except Exception as exc:
                    logger.warning(f"Stale Playwright connection ({exc}); tearing down.")
                    # Probe 20 FAIL: reconnect() cannot recover from BrowserStopEvent
                    # teardown. Surface as fatal rather than attempting reconnect.
                    try:
                        if pw_cache["browser"] is not None:
                            await pw_cache["browser"].close()
                    except Exception:
                        pass
                    try:
                        if pw_cache["instance"] is not None:
                            await pw_cache["instance"].stop()
                    except Exception:
                        pass
                    pw_cache.update({"instance": None, "browser": None, "page": None})
                    raise RuntimeError(
                        f"Playwright connection became stale mid-run ({exc}). "
                        "Probe 20 confirmed reconnect() cannot recover from "
                        "BrowserStopEvent teardown — surfacing error to agent."
                    )

            # Fresh connect (first call this run).
            from playwright.async_api import async_playwright as _async_playwright
            cdp_url = _get_cdp_url_from_session(bs)
            if not cdp_url:
                raise RuntimeError("No CDP URL on browser_session — cannot connect Playwright")
            instance = await _async_playwright().start()
            try:
                connected = await instance.chromium.connect_over_cdp(cdp_url)
                contexts = connected.contexts
                if not contexts or not contexts[0].pages:
                    raise RuntimeError(
                        f"CDP browser has no usable page "
                        f"(contexts={len(contexts)}, "
                        f"pages={len(contexts[0].pages) if contexts else 0})"
                    )
                page_obj = contexts[0].pages[0]
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
                                try:
                                    stable_hash = dom_node.compute_stable_hash()
                                except Exception as e:
                                    logger.debug(f"   ⚠️ compute_stable_hash failed: {e}")
                                else:
                                    logger.info(
                                        f"📊 LOCATOR_PROBE workflow_id={workflow_id or 'unknown'} "
                                        f"stable_hash={stable_hash} "
                                        f"element_id={params.element_id}"
                                    )
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

                try:
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
                    
                    result = await asyncio.wait_for(
                        find_unique_locator_action(
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
                        ),
                        timeout=custom_action_timeout
                    )
                except asyncio.TimeoutError:
                    # Handle timeout gracefully
                    timeout_msg = (
                        f"Custom action timed out after {custom_action_timeout} seconds "
                        f"for element {params.element_id}"
                    )
                    logger.error(f"⏱️ {timeout_msg}")
                    logger.error(f"   Element: {params.element_id} - {params.element_description}")
                    logger.error(f"   Coordinates: ({params.x}, {params.y})")

                    # Return error result
                    result = {
                        'element_id': params.element_id,
                        'description': params.element_description,
                        'found': False,
                        'error': timeout_msg,
                        'coordinates': {'x': params.x, 'y': params.y},
                        'validated': False,
                        'count': 0,
                        'unique': False,
                        'valid': False,
                        'validation_method': 'playwright'
                    }

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

                    # Success messages intentionally do NOT prescribe "move to next" —
                    # that would cause the LLM to skip the element's interactive action
                    # (click/input/submit) needed to advance the page. SEQUENTIAL_PROCESSING_RULES
                    # in the workflow prompt owns the per-element action flow. We only assert the
                    # locator is final (cost guard against re-finding the same locator).
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
                            f"Locator: {best_locator}\n"
                            f"Status: FINAL — do not call find_unique_locator again for this element.\n"
                            f"All {_total_expected} locators found. "
                            f'Call done() with: {{"elements_found": {elements_found_json}}}'
                        )
                    else:
                        success_msg = (
                            f"✅ LOCATOR VALIDATED BY PLAYWRIGHT\n"
                            f"Element: {params.element_id}\n"
                            f"Locator: {best_locator}\n"
                            f"Status: FINAL — do not call find_unique_locator again for this element.\n"
                            f"Next step: perform this element's action (input/click/submit) as specified "
                            f"in SEQUENTIAL_PROCESSING_RULES if interactive, otherwise proceed to the next element."
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

            except RuntimeError as e:
                # RuntimeError from _ensure_playwright means the browser connection is
                # dead (stale CDP or failed fresh connect). Surfacing as is_done=True
                # so the agent terminates immediately rather than burning remaining
                # steps retrying against a permanently unavailable browser.
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



