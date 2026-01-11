"""
Custom action registration for browser-use agent.

This module handles the registration of custom actions with the browser-use agent.
Custom actions allow the agent to call deterministic Python code during workflow execution,
bypassing the need for LLM calls for specific operations like locator validation.

Key Functions:
- register_custom_actions: Register custom actions with browser-use agent
- cleanup_playwright_cache: Clean up cached Playwright resources (call at workflow end)
- invalidate_playwright_cache: Force-invalidate cache (call when CDP URL changes)

The registration process:
1. Creates or retrieves the Tools instance from the agent
2. Defines parameter models for custom actions using Pydantic
3. Registers action handlers that wrap the actual implementation
4. Handles page object retrieval from browser_session via CDP
5. Converts results to ActionResult format for the agent

Playwright Cache Lifecycle:
- Cache is created on first custom action call per workflow
- Cache is reused across multiple custom action calls (performance optimization)
- Cache is invalidated if CDP URL changes (browser restart detection)
- Cache MUST be cleaned up at workflow end via cleanup_playwright_cache()
- Emergency cleanup via atexit handler prevents orphaned resources

Usage:
    from browser_service.agent.registration import register_custom_actions, cleanup_playwright_cache

    # Register custom actions with agent
    success = register_custom_actions(agent, page=None)
    if success:
        # Agent can now call find_unique_locator action
        pass
    
    # IMPORTANT: Always clean up at workflow end
    await cleanup_playwright_cache()
"""

import asyncio
import atexit
import logging
import re
from typing import Optional

# Get logger
logger = logging.getLogger(__name__)

# ========================================
# PLAYWRIGHT INSTANCE CACHE
# ========================================
# Module-level cache for Playwright instance (reuse across custom action calls)
# This significantly improves performance by avoiding repeated Playwright startup.
#
# IMPORTANT: This cache MUST be cleaned up when workflow completes.
# See cleanup_playwright_cache() for proper cleanup.
#
# THREAD SAFETY: _cache_lock protects concurrent access to cache variables.
# This prevents race conditions when multiple workflows run simultaneously.

_playwright_instance_cache = None
_connected_browser_cache = None
_cache_cdp_url = None
_cache_initialized = False  # Track if cache has ever been used
_cache_lock = asyncio.Lock()  # Thread-safe access to cache


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

def _sync_cleanup_playwright_cache():
    """
    Synchronous cleanup wrapper for atexit handler.
    
    This is registered with atexit to ensure cleanup happens even if
    the async cleanup_playwright_cache() is never called (e.g., crash/interrupt).
    
    Note: This is a best-effort cleanup. Some async resources may not be
    fully released in a sync context.
    """
    global _playwright_instance_cache, _connected_browser_cache, _cache_cdp_url, _cache_initialized
    
    if not _cache_initialized:
        return  # Nothing to clean up
    
    logger.info("🧹 ATEXIT: Running emergency Playwright cache cleanup...")
    
    try:
        # Note: We can't properly await async cleanup in atexit,
        # but we can at least clear the references to help GC
        if _connected_browser_cache:
            logger.info("   Clearing browser cache reference...")
            # Try to close if there's a sync close method
            if hasattr(_connected_browser_cache, 'close') and not asyncio.iscoroutinefunction(_connected_browser_cache.close):
                try:
                    _connected_browser_cache.close()
                except Exception:
                    pass
        
        if _playwright_instance_cache:
            logger.info("   Clearing Playwright cache reference...")
            # Playwright stop() is async, so we can't call it here
            # But clearing reference helps GC
        
        _playwright_instance_cache = None
        _connected_browser_cache = None
        _cache_cdp_url = None
        _cache_initialized = False
        
        logger.info("✅ ATEXIT: Cache references cleared")
    except Exception as e:
        logger.warning(f"⚠️ ATEXIT: Error during emergency cleanup: {e}")


# Register the sync cleanup with atexit (runs on normal Python exit)
atexit.register(_sync_cleanup_playwright_cache)


def invalidate_playwright_cache():
    """
    Force-invalidate the Playwright cache without closing resources.
    
    Call this when you detect that the cached CDP URL is stale (browser restarted)
    or when you want to force a fresh connection on the next custom action.
    
    Note: This does NOT close the existing resources - it just marks the cache
    as invalid so a new connection will be created. For proper cleanup,
    use cleanup_playwright_cache() instead.
    
    Returns:
        bool: True if cache was invalidated, False if cache was already empty
    """
    global _playwright_instance_cache, _connected_browser_cache, _cache_cdp_url
    
    had_cache = _playwright_instance_cache is not None or _connected_browser_cache is not None
    
    if had_cache:
        logger.info("🔄 Invalidating Playwright cache (forcing fresh connection on next use)")
        _playwright_instance_cache = None
        _connected_browser_cache = None
        _cache_cdp_url = None
    
    return had_cache


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
                _store_cdp_port_for_cleanup(cdp_url)
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
                    _store_cdp_port_for_cleanup(cdp_url)
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
                _store_cdp_port_for_cleanup(value)
                return value
        except Exception:
            pass
    
    logger.warning("⚠️ Could not find CDP URL in browser_session")
    return None


def _store_cdp_port_for_cleanup(cdp_url: str):
    """Store CDP port for cleanup module (best-effort)."""
    try:
        from browser_service.browser.cleanup import store_cdp_port
        store_cdp_port(cdp_url)
    except Exception as e:
        logger.debug(f"Could not store CDP port: {e}")


async def _get_active_page_from_browser(
    connected_browser, 
    browser_session, 
    fallback_page,
    created_new_instance: bool = False
):
    """
    Get active Playwright page using multiple fallback strategies.
    
    Strategies (in priority order):
    1. CDP browser contexts[0].pages[0] (primary for CDP connections)
    2. browser_session.get_pages()[0] (async method)
    3. browser_session.page (direct attribute)
    4. browser_session.get_current_page() (async method)
    5. browser_session.context.pages[0] (context attribute)
    6. browser_session.browser.contexts[0].pages[0] (nested browser access)
    7. Fallback page passed during registration
    
    Args:
        connected_browser: Playwright browser connected via CDP (may be None)
        browser_session: The browser-use session object
        fallback_page: Page object passed during registration (last resort)
        created_new_instance: Whether this is a freshly created CDP connection
        
    Returns:
        Active Playwright page object, or None if all strategies fail
    """
    active_page = None
    
    # ════════════════════════════════════════════════════════════════════
    # Strategy 1: CDP contexts (PRIMARY - use when we have CDP connection)
    # ════════════════════════════════════════════════════════════════════
    if connected_browser:
        try:
            contexts = connected_browser.contexts
            if contexts:
                logger.debug(f"CDP browser has {len(contexts)} context(s)")
                for idx, ctx in enumerate(contexts):
                    logger.debug(f"  Context[{idx}] has {len(ctx.pages)} page(s)")
                
                context = contexts[0]
                if context.pages:
                    active_page = context.pages[0]
                    page_url = active_page.url
                    
                    # Wait for DOM to be ready on new connections
                    if created_new_instance:
                        try:
                            await active_page.wait_for_load_state('domcontentloaded', timeout=5000)
                            logger.info("✅ Page DOM is ready for locator queries")
                        except Exception as e:
                            logger.warning(f"⚠️ Page load state wait: {e}")
                        logger.info(f"✅ Connected to page via CDP: {page_url}")
                    else:
                        logger.info(f"♻️ Reusing cached CDP page: {page_url}")
                    
                    return active_page
                else:
                    logger.debug("CDP context has no pages")
            else:
                logger.debug("CDP browser has no contexts")
        except Exception as e:
            logger.debug(f"Strategy 1 (CDP contexts): {e}")
    
    # ════════════════════════════════════════════════════════════════════
    # Strategies 2-6: browser_session fallbacks (when CDP not available)
    # ════════════════════════════════════════════════════════════════════
    
    # Strategy 2: get_pages() async method
    if hasattr(browser_session, 'get_pages'):
        try:
            pages = await browser_session.get_pages()
            if pages and len(pages) > 0:
                active_page = pages[0]
                logger.info(f"✅ Got page from browser_session.get_pages() ({len(pages)} total)")
                return active_page
        except Exception as e:
            logger.debug(f"Strategy 2 (get_pages): {e}")
    
    # Strategy 3: Direct .page attribute
    if hasattr(browser_session, 'page') and browser_session.page is not None:
        try:
            active_page = browser_session.page
            logger.info("✅ Got page from browser_session.page")
            return active_page
        except Exception as e:
            logger.debug(f"Strategy 3 (.page): {e}")
    
    # Strategy 4: get_current_page() async method
    if hasattr(browser_session, 'get_current_page'):
        try:
            active_page = await browser_session.get_current_page()
            if active_page:
                logger.info("✅ Got page from browser_session.get_current_page()")
                return active_page
        except Exception as e:
            logger.debug(f"Strategy 4 (get_current_page): {e}")
    
    # Strategy 5: context.pages attribute
    if hasattr(browser_session, 'context') and browser_session.context is not None:
        try:
            pages = browser_session.context.pages
            if pages and len(pages) > 0:
                active_page = pages[0]
                logger.info(f"✅ Got page from browser_session.context.pages ({len(pages)} total)")
                return active_page
        except Exception as e:
            logger.debug(f"Strategy 5 (context.pages): {e}")
    
    # Strategy 6: browser.contexts[0].pages
    if hasattr(browser_session, 'browser') and browser_session.browser is not None:
        try:
            contexts = browser_session.browser.contexts
            if contexts and len(contexts) > 0:
                pages = contexts[0].pages
                if pages and len(pages) > 0:
                    active_page = pages[0]
                    logger.info("✅ Got page from browser_session.browser.contexts[0].pages")
                    return active_page
        except Exception as e:
            logger.debug(f"Strategy 6 (browser.contexts): {e}")
    
    # ════════════════════════════════════════════════════════════════════
    # Strategy 7: Fallback page (last resort)
    # ════════════════════════════════════════════════════════════════════
    if fallback_page:
        logger.warning("⚠️ All page strategies failed, using fallback page")
        return fallback_page
    
    logger.error("❌ No page available - all strategies exhausted")
    return None

def register_custom_actions(agent, page=None) -> bool:
    """
    Register custom actions with browser-use agent.

    This function registers the find_unique_locator custom action that allows
    the agent to call deterministic Python code for locator finding and validation.

    The custom action will get the page object from browser_session during execution,
    ensuring we use the SAME browser that's already open. This is the key strategy:
    validate locators using the existing browser_use browser (no new instance needed).

    Args:
        agent: Browser-use Agent instance
        page: Optional Playwright page object (used as fallback if browser_session doesn't provide one)

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

        # Import settings
        from src.backend.core.config import settings

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
                            
                            # Find element whose bounding box contains the coordinates
                            best_match = None
                            best_area = float('inf')  # Prefer smaller (more specific) elements
                            
                            for idx, elem in selector_map.items():
                                if hasattr(elem, 'absolute_position') and elem.absolute_position:
                                    pos = elem.absolute_position
                                    # Check if coordinates are within bounding box
                                    if (pos.x <= scaled_x <= pos.x + pos.width and
                                        pos.y <= scaled_y <= pos.y + pos.height):
                                        area = pos.width * pos.height
                                        if area < best_area and area > 0:
                                            best_area = area
                                            best_match = (idx, elem)
                            
                            if best_match:
                                idx, dom_node = best_match
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




                # IMPORTANT: browser-use now uses CDP (Chrome DevTools Protocol) instead of Playwright
                # We need to connect to browser-use's browser via CDP to get a Playwright page for validation
                #
                # OPTIMIZATION: Reuse Playwright instance across custom action calls
                # Instead of creating a new instance every time, check cache first
                global _playwright_instance_cache, _connected_browser_cache, _cache_cdp_url, _cache_initialized
                
                active_page = None
                playwright_instance = None
                connected_browser = None
                created_new_instance = False

                try:
                    logger.info("🔍 Attempting to retrieve page from browser_session via CDP...")
                    logger.info(f"   browser_session type: {type(browser_session)}")

                    # ════════════════════════════════════════════════════════════════════
                    # GET CDP URL (using consolidated helper function)
                    # ════════════════════════════════════════════════════════════════════
                    cdp_url = _get_cdp_url_from_session(browser_session)

                    # ════════════════════════════════════════════════════════════════════
                    # CONNECT PLAYWRIGHT VIA CDP (with caching)
                    # ════════════════════════════════════════════════════════════════════
                    if cdp_url:
                        # Thread-safe cache access using lock
                        async with _cache_lock:
                            
                            # Check if we can reuse cached Playwright instance
                            if (_playwright_instance_cache and 
                                _connected_browser_cache and 
                                _cache_cdp_url == cdp_url):
                                logger.info("♻️  Reusing cached Playwright instance (performance optimization)")
                                playwright_instance = _playwright_instance_cache
                                connected_browser = _connected_browser_cache
                            else:
                                # Create new Playwright instance
                                try:
                                    from playwright.async_api import async_playwright

                                    logger.info("🔌 Connecting Playwright to browser-use's browser via CDP...")
                                    playwright_instance = await async_playwright().start()
                                    connected_browser = await playwright_instance.chromium.connect_over_cdp(cdp_url)
                                    
                                    # Cache for reuse (protected by lock)
                                    _playwright_instance_cache = playwright_instance
                                    _connected_browser_cache = connected_browser
                                    _cache_cdp_url = cdp_url
                                    _cache_initialized = True  # Mark cache as used for atexit cleanup
                                    created_new_instance = True
                                    logger.info("💾 Cached Playwright instance for reuse")

                                except Exception as e:
                                    logger.error(f"❌ Failed to connect Playwright via CDP: {e}")
                                    import traceback
                                    logger.debug(traceback.format_exc())
                    else:
                        # Log available attributes for debugging when CDP URL not found
                        logger.info(f"   browser_session attributes: {[attr for attr in dir(browser_session) if not attr.startswith('_')][:20]}")

                    # ════════════════════════════════════════════════════════════════════
                    # GET ACTIVE PAGE (using consolidated helper function)
                    # ════════════════════════════════════════════════════════════════════
                    active_page = await _get_active_page_from_browser(
                        connected_browser=connected_browser,
                        browser_session=browser_session,
                        fallback_page=page,
                        created_new_instance=created_new_instance
                    )

                except Exception as e:
                    logger.error(f"❌ Error getting page from browser_session: {e}", exc_info=True)
                    active_page = page  # Use the page passed during registration as fallback
                    if active_page:
                        logger.info(f"   Fallback page type: {type(active_page)}")


                # Unwrap browser-use Page wrapper to get actual Playwright page
                if active_page and not hasattr(active_page, 'locator'):
                    logger.warning(f"⚠️ Page object is a browser-use wrapper: {type(active_page)}")
                    logger.info("   Attempting to unwrap to get Playwright page...")

                    # browser-use wraps the Playwright page in browser_use.actor.page.Page
                    # Try multiple strategies to get the underlying Playwright page
                    playwright_page = None

                    # Strategy 1: Check for .page attribute
                    if hasattr(active_page, 'page') and active_page.page is not None:
                        playwright_page = active_page.page
                        logger.info("✅ Unwrapped page from wrapper.page")

                    # Strategy 2: Check for ._page attribute
                    elif hasattr(active_page, '_page') and active_page._page is not None:
                        playwright_page = active_page._page
                        logger.info("✅ Unwrapped page from wrapper._page")

                    # Strategy 3: Check for ._client attribute (CDP client)
                    elif hasattr(active_page, '_client') and active_page._client is not None:
                        # _client might be the CDP client, try to get page from it
                        client = active_page._client
                        if hasattr(client, 'page') and client.page is not None:
                            playwright_page = client.page
                            logger.info("✅ Unwrapped page from wrapper._client.page")
                        else:
                            logger.warning("   _client exists but has no page attribute")

                    # Strategy 4: Check for ._browser_session attribute
                    elif hasattr(active_page, '_browser_session') and active_page._browser_session is not None:
                        # Try to get page from the browser session
                        session = active_page._browser_session
                        if hasattr(session, 'page') and session.page is not None:
                            playwright_page = session.page
                            logger.info("✅ Unwrapped page from wrapper._browser_session.page")
                        elif hasattr(session, 'get_current_page'):
                            try:
                                playwright_page = await session.get_current_page()
                                logger.info("✅ Unwrapped page from wrapper._browser_session.get_current_page()")
                            except Exception as e:
                                logger.warning(f"   Failed to get page from _browser_session: {e}")

                    # Strategy 5: Use the wrapper directly if it has evaluate method
                    # browser-use Page wrapper might proxy Playwright methods
                    elif hasattr(active_page, 'evaluate'):
                        logger.info("⚠️ Using browser-use Page wrapper directly (has evaluate method)")
                        logger.info("   This wrapper might proxy Playwright methods")
                        playwright_page = active_page  # Use wrapper as-is

                    if playwright_page:
                        logger.info(f"   Playwright page type: {type(playwright_page)}")
                        active_page = playwright_page
                    else:
                        logger.error("❌ Could not unwrap browser-use Page wrapper!")
                        logger.error(f"   Wrapper attributes: {[attr for attr in dir(active_page) if not attr.startswith('__')][:20]}")
                        active_page = None

                # Final verification: ensure we have a page with required methods
                if active_page:
                    required_methods = ['locator', 'evaluate', 'evaluate_handle']
                    missing_methods = [m for m in required_methods if not hasattr(active_page, m)]

                    if missing_methods:
                        logger.error(f"❌ Page object is missing required methods: {missing_methods}")
                        logger.error(f"   Type: {type(active_page)}")
                        logger.error(f"   Available methods: {[attr for attr in dir(active_page) if not attr.startswith('_')][:30]}")
                        active_page = None
                    else:
                        logger.info(f"✅ Page object has all required methods: {required_methods}")
                        logger.info(f"   Page type: {type(active_page)}")
                        
                        # CRITICAL: Test page connectivity by running a simple evaluation
                        # This detects stale connections where the CDP page is no longer responsive
                        try:
                            test_result = await active_page.evaluate("() => document.body ? 'connected' : null")
                            if test_result != 'connected':
                                logger.warning("⚠️ Page connectivity test returned unexpected result, may be stale")
                        except Exception as connectivity_err:
                            logger.error(f"❌ Page connectivity test FAILED: {connectivity_err}")
                            logger.info("🔄 Invalidating stale cache and will retry with fresh connection...")
                            # Invalidate cache so next call creates fresh connection
                            _playwright_instance_cache = None
                            _connected_browser_cache = None
                            _cache_cdp_url = None
                            active_page = None  # Force retry

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
                        # Note: _iframe_id unpacked but unused (only iframe_context needed for locator)
                        iframe_context, _iframe_id = _detect_iframe_context(
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
                                # Find element by coordinates in selector_map
                                # (find smallest bounding box containing coords)
                                if selector_map:
                                    best_match = None
                                    best_area = float('inf')
                                    
                                    for idx, elem in selector_map.items():
                                        if hasattr(elem, 'absolute_position') and elem.absolute_position:
                                            pos = elem.absolute_position
                                            # Check if coordinates are within bounding box
                                            if (pos.x <= final_x <= pos.x + pos.width and
                                                pos.y <= final_y <= pos.y + pos.height):
                                                area = pos.width * pos.height
                                                # Skip iframe itself - we want element inside
                                                elem_tag = elem.node_name.upper() if hasattr(elem, 'node_name') else ''
                                                if elem_tag == 'IFRAME':
                                                    continue
                                                if area < best_area and area > 0:
                                                    best_area = area
                                                    best_match = (idx, elem)
                                    
                                    if best_match:
                                        idx, dom_node = best_match
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
                            is_collection=params.is_collection  # Pass collection flag for multi-element detection
                        ),
                        timeout=settings.CUSTOM_ACTION_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    # Handle timeout gracefully
                    timeout_msg = (
                        f"Custom action timed out after {settings.CUSTOM_ACTION_TIMEOUT} seconds "
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

                    # Success message for agent - CLEAR and UNAMBIGUOUS
                    # Include explicit confirmation that this is the CORRECT and FINAL locator
                    success_msg = (
                        "✅ SUCCESS - LOCATOR VALIDATED BY PLAYWRIGHT\n"
                        f"Element: {params.element_id}\n"
                        f"Locator: {best_locator}\n"
                        f"Validation Result: UNIQUE (count={count}, validated={validated})\n"
                        f"Method: {validation_method} (deterministic validation)\n"
                        "Status: COMPLETE AND CORRECT\n"
                        "This locator is guaranteed unique and valid.\n"
                        "Do NOT retry or attempt to find a different locator.\n"
                        "Move to the next element immediately."
                    )

                    logger.info(f"✅ Custom action succeeded: {best_locator}")
                    
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

                    # CRITICAL FIX: Do NOT set success=True when is_done=False
                    # ActionResult validation rule: success can only be True when is_done=True
                    # For regular actions that succeed, leave success as None (default)
                    action_result = ActionResult(
                        extracted_content=success_msg,
                        long_term_memory=f"✅ VALIDATED: {params.element_id} = {best_locator} (Playwright confirmed count=1, unique=True). This is the CORRECT locator. Do NOT retry.",
                        metadata=result,
                        is_done=False  # Don't mark as done, let agent continue with other elements
                        # success is None by default for successful actions that aren't done
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

                # Cleanup: Do NOT close cached Playwright instance (reused across calls)
                # Only clean up if we created a non-cached instance (shouldn't happen now)
                if connected_browser and connected_browser != _connected_browser_cache:
                    try:
                        logger.info("🧹 Cleaning up: Closing non-cached Playwright CDP connection...")
                        await connected_browser.close()
                        logger.info("✅ Playwright browser connection closed")
                    except Exception as e:
                        logger.warning(f"⚠️ Error closing Playwright browser: {e}")

                if playwright_instance and playwright_instance != _playwright_instance_cache:
                    try:
                        logger.info("🧹 Cleaning up: Stopping non-cached Playwright instance...")
                        await playwright_instance.stop()
                        logger.info("✅ Playwright instance stopped")
                    except Exception as e:
                        logger.warning(f"⚠️ Error stopping Playwright instance: {e}")
                
                # Cached instances are NOT cleaned up here - they persist for reuse
                # They will be cleaned up when workflow completes (see cleanup_playwright_cache)

                return action_result

            except Exception as e:
                error_msg = f"Error in find_unique_locator custom action: {str(e)}"
                logger.error(f"❌ {error_msg}", exc_info=True)

                # Cleanup on error - but only non-cached instances
                if connected_browser and connected_browser != _connected_browser_cache:
                    try:
                        await connected_browser.close()
                    except Exception:
                        pass
                if playwright_instance and playwright_instance != _playwright_instance_cache:
                    try:
                        await playwright_instance.stop()
                    except Exception:
                        pass

                return ActionResult(error=error_msg)

        logger.info("✅ Custom action 'find_unique_locator' registered successfully")
        logger.info("   Agent can now call: find_unique_locator(x, y, element_id, element_description, candidate_locator)")
        return True

    except Exception as e:
        # Log error but don't crash - allow fallback to legacy workflow
        logger.error(f"❌ Failed to register custom actions: {str(e)}")
        logger.error("   Stack trace:", exc_info=True)
        logger.warning("⚠️ Continuing with legacy workflow (custom actions disabled)")
        return False


async def cleanup_playwright_cache():
    """
    Clean up cached Playwright instance at the end of workflow.
    
    This should be called once after all custom actions have completed,
    not after each individual action call.
    
    Returns:
        bool: True if cleanup succeeded
    """
    global _playwright_instance_cache, _connected_browser_cache, _cache_cdp_url, _cache_initialized
    
    try:
        if _connected_browser_cache:
            try:
                logger.info("🧹 Workflow cleanup: Closing cached Playwright CDP connection...")
                await _connected_browser_cache.close()
                logger.info("✅ Cached Playwright browser connection closed")
            except Exception as e:
                logger.warning(f"⚠️ Error closing cached Playwright browser: {e}")

        if _playwright_instance_cache:
            try:
                logger.info("🧹 Workflow cleanup: Stopping cached Playwright instance...")
                await _playwright_instance_cache.stop()
                logger.info("✅ Cached Playwright instance stopped")
            except Exception as e:
                logger.warning(f"⚠️ Error stopping cached Playwright instance: {e}")
        
        # Clear cache and reset initialized flag
        _playwright_instance_cache = None
        _connected_browser_cache = None
        _cache_cdp_url = None
        _cache_initialized = False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during Playwright cache cleanup: {e}")
        return False

