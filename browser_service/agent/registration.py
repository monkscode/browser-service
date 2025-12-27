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
        'xpath': dom_node.xpath if hasattr(dom_node, 'xpath') else '',
    }


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
                
                if params.element_index is not None and browser_session:
                    try:
                        logger.info(f"📋 Getting element [{params.element_index}] from browser-use DOM...")
                        
                        # Debug: Log selector_map to verify table cells are indexed
                        selector_map = await browser_session.get_selector_map()
                        if selector_map:
                            available_indices = sorted(selector_map.keys())
                            logger.info(f"   📊 Selector map has {len(selector_map)} elements")
                            # Log sample elements to verify table cells (td/th) are indexed
                            sample_types = {}
                            for idx in available_indices[:50]:
                                tag = selector_map[idx].node_name.upper() if hasattr(selector_map[idx], 'node_name') else '?'
                                sample_types[tag] = sample_types.get(tag, 0) + 1
                            logger.info(f"   📊 Element types in sample: {dict(sorted(sample_types.items(), key=lambda x: -x[1]))}")
                        
                        # Use browser-use's get_element_by_index method (returns EnhancedDOMTreeNode | None)
                        dom_node = await browser_session.get_element_by_index(params.element_index)
                        
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
                        logger.debug(f"   Full error: ", exc_info=True)

                # ========================================
                # COORDINATE SCALING: LLM Screenshot → Viewport
                # ========================================
                # When llm_screenshot_size is set, browser-use resizes screenshots before
                # sending to the LLM. The LLM returns coordinates in the resized image space.
                # 
                # Built-in actions (like click) use _convert_llm_coordinates_to_viewport()
                # internally, but custom actions receive RAW coordinates from the LLM.
                # We must apply the same conversion here for our custom action.
                #
                # Example: If viewport is 1920x1080 and llm_screenshot_size is 1400x850
                # - LLM sees Solutions link at x=440 (in 1400px wide image)
                # - We scale: (440/1400) * 1920 = 604 (actual viewport x)
                
                scaled_x, scaled_y = params.x, params.y  # Default: no scaling
                
                # Debug: Check what attributes browser_session has
                has_llm_size = hasattr(browser_session, 'llm_screenshot_size') and browser_session.llm_screenshot_size
                has_viewport = hasattr(browser_session, '_original_viewport_size') and browser_session._original_viewport_size
                logger.info(f"   📊 browser_session.llm_screenshot_size: {getattr(browser_session, 'llm_screenshot_size', 'NOT SET')}")
                logger.info(f"   📊 browser_session._original_viewport_size: {getattr(browser_session, '_original_viewport_size', 'NOT SET')}")
                
                if has_llm_size:
                    if has_viewport:
                        original_width, original_height = browser_session._original_viewport_size
                        llm_width, llm_height = browser_session.llm_screenshot_size
                        
                        # Apply the same conversion as _convert_llm_coordinates_to_viewport
                        scaled_x = int((params.x / llm_width) * original_width)
                        scaled_y = int((params.y / llm_height) * original_height)
                        
                        logger.info(f"📐 COORDINATE SCALING applied:")
                        logger.info(f"   LLM screenshot: {llm_width}x{llm_height}")
                        logger.info(f"   Viewport: {original_width}x{original_height}")
                        logger.info(f"   Original: ({params.x}, {params.y}) → Scaled: ({scaled_x}, {scaled_y})")
                    else:
                        logger.info("   ⚠️ No _original_viewport_size - using coordinates as-is")
                else:
                    logger.info("   ⚠️ No llm_screenshot_size - using coordinates as-is")

                # ========================================
                # FALLBACK: Find element from selector_map using coordinates
                # ========================================
                # When element_index is NOT provided (which is typical for custom actions),
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
                            logger.info(f"   📊 Selector map has {len(selector_map)} interactive elements")
                            logger.info(f"   📊 Types: {dict(sorted(sample_types.items(), key=lambda x: -x[1])[:8])}")
                            
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
                                logger.info(f"   ✅ Found element [{idx}] in selector_map!")
                                
                                # Extract element attributes
                                element_data_from_index = _extract_dom_node_attributes(dom_node)
                                
                                # Get text content
                                if hasattr(dom_node, 'get_meaningful_text_for_llm'):
                                    element_data_from_index['textContent'] = dom_node.get_meaningful_text_for_llm()
                                elif hasattr(dom_node, 'get_all_children_text'):
                                    element_data_from_index['textContent'] = dom_node.get_all_children_text()
                                
                                # Extract confirmed_coords from bounding box
                                if hasattr(dom_node, 'absolute_position') and dom_node.absolute_position:
                                    pos = dom_node.absolute_position
                                    confirmed_coords = (
                                        int(pos.x + pos.width / 2),
                                        int(pos.y + pos.height / 2)
                                    )
                                    logger.info(f"   📍 Confirmed coordinates: {confirmed_coords}")
                                
                                logger.info(f"   📝 Found: <{element_data_from_index['tagName']}> xpath: {element_data_from_index.get('xpath', 'N/A')[:60]}")
                        
                        # ========================================
                        # STEP B: If not found in selector_map, search FULL DOM tree
                        # This finds ALL elements including non-interactive ones (table cells, etc.)
                        # 
                        # IMPORTANT: Browser validation showed that for fixed-position content (like tables),
                        # LLM coordinates often map DIRECTLY to viewport coords without scaling.
                        # We try RAW coords first, then SCALED as fallback.
                        # ========================================
                        if element_data_from_index is None:
                            try:
                                # Get full browser state with DOM tree
                                state = await browser_session.get_browser_state_summary(include_screenshot=False)

                                if state and state.dom_state and hasattr(state.dom_state, '_root') and state.dom_state._root:
                                    root = state.dom_state._root
                                    
                                    # Debug: count table nodes with bounds
                                    nodes_with_bounds = []
                                    def count_bounds(node, depth=0):
                                        if hasattr(node, 'original_node') and node.original_node:
                                            orig = node.original_node
                                            tag = orig.node_name if hasattr(orig, 'node_name') else '?'
                                            has_bounds = hasattr(orig, 'snapshot_node') and orig.snapshot_node and hasattr(orig.snapshot_node, 'bounds') and orig.snapshot_node.bounds
                                            if has_bounds and tag.lower() in ['td', 'th', 'tr', 'table']:
                                                b = orig.snapshot_node.bounds
                                                nodes_with_bounds.append(f"{tag}@({b.x:.0f},{b.y:.0f} {b.width:.0f}x{b.height:.0f})")
                                        if hasattr(node, 'children'):
                                            for child in node.children:
                                                count_bounds(child, depth + 1)
                                    count_bounds(root)
                                    if nodes_with_bounds:
                                        logger.info(f"   📊 Table elements with bounds: {nodes_with_bounds[:10]}")
                                    
                                    # Recursive function to find element at coordinates in full DOM tree
                                    def find_element_at_coords(node, x, y, depth=0):
                                        """
                                        Recursively search DOM tree for SMALLEST element containing coordinates.
                                        Returns tuple of (element, area) or (None, inf).
                                        """
                                        best = None
                                        best_area = float('inf')
                                        
                                        if hasattr(node, 'original_node') and node.original_node:
                                            orig = node.original_node
                                            if hasattr(orig, 'snapshot_node') and orig.snapshot_node:
                                                bounds = getattr(orig.snapshot_node, 'bounds', None)
                                                if bounds and hasattr(bounds, 'x') and bounds.width > 0 and bounds.height > 0:
                                                    if (bounds.x <= x <= bounds.x + bounds.width and
                                                        bounds.y <= y <= bounds.y + bounds.height):
                                                        area = bounds.width * bounds.height
                                                        if area > 0:
                                                            best = orig
                                                            best_area = area
                                        
                                        if hasattr(node, 'children') and node.children:
                                            for child in node.children:
                                                child_result, child_area = find_element_at_coords(child, x, y, depth + 1)
                                                if child_result and child_area < best_area:
                                                    best = child_result
                                                    best_area = child_area
                                        
                                        return best, best_area
                                    
                                    # Try ALL coordinate strategies and collect candidates
                                    candidates = []
                                    
                                    # Strategy 1: RAW LLM coordinates
                                    raw_x, raw_y = params.x, params.y
                                    logger.info(f"🔍 STEP B.1: Trying RAW LLM coords ({raw_x}, {raw_y})...")
                                    node1, area1 = find_element_at_coords(root, raw_x, raw_y)
                                    if node1 and hasattr(node1, 'node_name') and node1.node_name.lower() not in ['html', 'body']:
                                        candidates.append(('RAW', node1, area1, raw_x, raw_y))
                                    
                                    # Strategy 2: SCALED viewport coordinates
                                    logger.info(f"🔍 STEP B.2: Trying SCALED coords ({scaled_x}, {scaled_y})...")
                                    node2, area2 = find_element_at_coords(root, scaled_x, scaled_y)
                                    if node2 and hasattr(node2, 'node_name') and node2.node_name.lower() not in ['html', 'body']:
                                        candidates.append(('SCALED', node2, area2, scaled_x, scaled_y))
                                    
                                    # Strategy 3: OFFSET-based for CENTERED content
                                    if has_llm_size and has_viewport:
                                        llm_width, llm_height = browser_session.llm_screenshot_size
                                        viewport_width, viewport_height = browser_session._original_viewport_size
                                        offset_x = (viewport_width - llm_width) / 2
                                        offset_result_x = params.x + offset_x
                                        offset_result_y = params.y  # Keep Y raw
                                        logger.info(f"🔍 STEP B.3: Trying OFFSET coords ({offset_result_x:.0f}, {offset_result_y:.0f})...")
                                        node3, area3 = find_element_at_coords(root, offset_result_x, offset_result_y)
                                        if node3 and hasattr(node3, 'node_name') and node3.node_name.lower() not in ['html', 'body']:
                                            candidates.append(('OFFSET', node3, area3, offset_result_x, offset_result_y))
                                    
                                    # Pick best candidate: prefer SMALLEST element whose text matches expected_text
                                    # (Parent divs may also contain the text, so we need the smallest/most specific match)
                                    found_node = None
                                    found_area = float('inf')
                                    best_strategy = None
                                    text_match_found = False
                                    text_match_area = float('inf')
                                    
                                    for strategy, node, area, x, y in candidates:
                                        # Check if text matches expected_text
                                        node_text = ''
                                        if hasattr(node, 'get_meaningful_text_for_llm'):
                                            node_text = node.get_meaningful_text_for_llm() or ''
                                        elif hasattr(node, 'get_all_children_text'):
                                            node_text = node.get_all_children_text() or ''
                                        
                                        text_matches = params.expected_text and params.expected_text.lower() in node_text.lower()
                                        tag = node.node_name.lower() if hasattr(node, 'node_name') else 'unknown'
                                        
                                        if text_matches:
                                            logger.info(f"   📝 {strategy} at ({x:.0f}, {y:.0f}): <{tag}> area={area:.0f} - text contains '{params.expected_text}'")
                                            # Pick SMALLEST element that matches (most specific)
                                            if area < text_match_area:
                                                text_match_area = area
                                                found_node = node
                                                found_area = area
                                                best_strategy = strategy
                                                text_match_found = True
                                        elif not text_match_found:
                                            # No text match yet, track smallest element as fallback
                                            if area < found_area:
                                                found_node = node
                                                found_area = area
                                                best_strategy = strategy
                                    
                                    if best_strategy:
                                        logger.info(f"   🎯 Using {best_strategy} strategy (area={found_area:.0f})")
                                    
                                    if found_node:
                                        tag_name = found_node.node_name.lower() if hasattr(found_node, 'node_name') else ''
                                        logger.info(f"   ✅ Found <{tag_name}> in full DOM tree!")
                                        
                                        # Extract element attributes from full DOM node
                                        element_data_from_index = {
                                            'tagName': tag_name,
                                            'id': found_node.attributes.get('id', '') if hasattr(found_node, 'attributes') and found_node.attributes else '',
                                            'name': found_node.attributes.get('name', '') if hasattr(found_node, 'attributes') and found_node.attributes else '',
                                            'className': found_node.attributes.get('class', '') if hasattr(found_node, 'attributes') and found_node.attributes else '',
                                            'ariaLabel': found_node.attributes.get('aria-label', '') if hasattr(found_node, 'attributes') and found_node.attributes else '',
                                            'placeholder': found_node.attributes.get('placeholder', '') if hasattr(found_node, 'attributes') and found_node.attributes else '',
                                            'title': found_node.attributes.get('title', '') if hasattr(found_node, 'attributes') and found_node.attributes else '',
                                            'href': found_node.attributes.get('href', '') if hasattr(found_node, 'attributes') and found_node.attributes else '',
                                            'role': found_node.attributes.get('role', '') if hasattr(found_node, 'attributes') and found_node.attributes else '',
                                            'dataTestId': (found_node.attributes.get('data-testid', '') or found_node.attributes.get('data-test', '')) if hasattr(found_node, 'attributes') and found_node.attributes else '',
                                            'xpath': found_node.xpath if hasattr(found_node, 'xpath') else '',
                                        }
                                        
                                        # Get text content
                                        if hasattr(found_node, 'get_meaningful_text_for_llm'):
                                            element_data_from_index['textContent'] = found_node.get_meaningful_text_for_llm()
                                        elif hasattr(found_node, 'get_all_children_text'):
                                            element_data_from_index['textContent'] = found_node.get_all_children_text()
                                        
                                        # Extract confirmed_coords from snapshot bounds
                                        if hasattr(found_node, 'snapshot_node') and found_node.snapshot_node:
                                            bounds = getattr(found_node.snapshot_node, 'bounds', None)
                                            if bounds and hasattr(bounds, 'x'):
                                                confirmed_coords = (
                                                    int(bounds.x + bounds.width / 2),
                                                    int(bounds.y + bounds.height / 2)
                                                )
                                                logger.info(f"   📍 Confirmed coordinates: {confirmed_coords}")
                                        
                                        if element_data_from_index.get('xpath'):
                                            logger.info(f"   📝 XPath: {element_data_from_index['xpath'][:80]}")
                                        if element_data_from_index.get('textContent'):
                                            text = element_data_from_index['textContent'][:40]
                                            logger.info(f"   📝 Text: \"{text}\"")
                                    else:
                                        logger.warning(f"   ⚠️ No element found in full DOM tree at SCALED coords ({scaled_x}, {scaled_y})")
                                else:
                                    logger.warning(f"   ⚠️ Could not access full DOM tree (_root is not available)")
                            except Exception as e:
                                logger.warning(f"   ⚠️ Full DOM tree search failed: {e}")
                        
                        # Log final status
                        if element_data_from_index is None:
                            logger.warning(f"   ⚠️ No element found at coordinates ({scaled_x}, {scaled_y}) in either selector_map or full DOM")
                            
                    except Exception as e:
                        logger.warning(f"   ⚠️ DOM-based element search failed: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())




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

                    # Strategy 1: Connect via CDP (browser-use's current architecture)
                    # Get CDP URL from browser_session
                    cdp_url = None

                    # Try session.cdp_url
                    if hasattr(browser_session, 'cdp_url'):
                        try:
                            cdp_url = browser_session.cdp_url
                            if cdp_url:
                                logger.info(f"✅ Found CDP URL from browser_session.cdp_url: {cdp_url}")
                                # Store CDP port for cleanup (it may be None at cleanup time)
                                try:
                                    from browser_service.browser.cleanup import store_cdp_port
                                    store_cdp_port(cdp_url)
                                except Exception as store_err:
                                    logger.warning(f"Failed to store CDP port: {store_err}")
                        except Exception as e:
                            logger.debug(f"Error accessing browser_session.cdp_url: {e}")

                    # Try cdp_client.url
                    if not cdp_url and hasattr(browser_session, 'cdp_client'):
                        try:
                            cdp_client = browser_session.cdp_client
                            if hasattr(cdp_client, 'url'):
                                cdp_url = cdp_client.url
                                if cdp_url:
                                    logger.info(f"✅ Found CDP URL from cdp_client.url: {cdp_url}")
                        except Exception as e:
                            logger.debug(f"Error accessing cdp_client.url: {e}")

                    # Search all attributes for CDP URL
                    if not cdp_url:
                        logger.info("🔍 Searching for CDP URL in browser_session attributes...")
                        for attr in dir(browser_session):
                            if not attr.startswith('_'):
                                try:
                                    value = getattr(browser_session, attr, None)
                                    if value and isinstance(value, str) and 'ws://' in value and 'devtools' in value:
                                        cdp_url = value
                                        logger.info(f"✅ Found CDP URL in {attr}: {cdp_url}")
                                        break
                                except Exception:
                                    pass

                    # If we have CDP URL, connect Playwright to browser-use's browser
                    if cdp_url:
                        # Thread-safe cache access using lock
                        async with _cache_lock:
                            # Global declaration must come first in Python
                            global _playwright_instance_cache, _connected_browser_cache, _cache_cdp_url, _cache_initialized
                            
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

                        # Get the active page from browser-use's browser
                        if connected_browser:
                            try:
                                # DEBUG: Log all available contexts and pages
                                logger.info(f"   DEBUG: connected_browser has {len(connected_browser.contexts)} context(s)")
                                for idx, ctx in enumerate(connected_browser.contexts):
                                    logger.info(f"   DEBUG: Context[{idx}] has {len(ctx.pages)} page(s)")
                                    for pidx, pg in enumerate(ctx.pages):
                                        try:
                                            pg_url = pg.url
                                            logger.info(f"   DEBUG:   Page[{pidx}] URL: {pg_url}")
                                        except Exception as pe:
                                            logger.info(f"   DEBUG:   Page[{pidx}] URL error: {pe}")

                                if connected_browser.contexts:
                                    context = connected_browser.contexts[0]
                                    if not created_new_instance:
                                        logger.info(f"♻️  Using {len(connected_browser.contexts)} cached context(s)")
                                    else:
                                        logger.info(f"✅ Found {len(connected_browser.contexts)} context(s)")

                                    if context.pages:
                                        active_page = context.pages[0]
                                        page_url = active_page.url  # Note: .url is a property, not a method
                                        
                                        # CRITICAL: Wait for page to be ready before using it for locator queries
                                        # This prevents stale page issues where the DOM isn't yet accessible
                                        if created_new_instance:
                                            try:
                                                # Wait for DOM to be accessible (commit = first paint, fastest)
                                                await active_page.wait_for_load_state('domcontentloaded', timeout=5000)
                                                logger.info("✅ Page DOM is ready for locator queries")
                                            except Exception as load_wait_err:
                                                logger.warning(f"⚠️ Page load state wait timed out: {load_wait_err}")
                                                # Continue anyway - may still work
                                        
                                        if created_new_instance:
                                            logger.info("✅ Connected to browser-use's page via CDP!")
                                            logger.info(f"   Page URL: {page_url}")
                                            logger.info(f"   Page type: {type(active_page)}")
                                        else:
                                            logger.info(f"♻️  Reusing page at {page_url}")
                                    else:
                                        logger.warning("⚠️ Context has no pages")
                                else:
                                    logger.warning("⚠️ Browser has no contexts")
                            except Exception as e:
                                logger.error(f"❌ Error accessing browser contexts/pages: {e}")
                    else:
                        logger.warning("⚠️ Could not find CDP URL in browser_session")
                        logger.info(f"   browser_session attributes: {[attr for attr in dir(browser_session) if not attr.startswith('_')][:20]}")

                    # Strategy 2: Try get_pages() method (fallback)
                    if active_page is None and hasattr(browser_session, 'get_pages'):
                        try:
                            pages = await browser_session.get_pages()
                            if pages and len(pages) > 0:
                                active_page = pages[0]
                                logger.info("✅ Got Playwright page from browser_session.get_pages()[0]")
                                logger.info(f"   Page type: {type(active_page)}")
                                logger.info(f"   Total pages: {len(pages)}")
                            else:
                                logger.warning("⚠️ browser_session.get_pages() returned empty list")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to get pages: {e}")

                    # Strategy 2: Direct access to .page attribute
                    if active_page is None and hasattr(browser_session, 'page') and browser_session.page is not None:
                        active_page = browser_session.page
                        logger.info("✅ Got active page from browser_session.page")
                        logger.info(f"   Page type: {type(active_page)}")

                    # Strategy 3: Try get_current_page() method
                    if active_page is None and hasattr(browser_session, 'get_current_page'):
                        active_page = await browser_session.get_current_page()
                        logger.info("✅ Got active page from browser_session.get_current_page()")
                        logger.info(f"   Page type: {type(active_page)}")

                    # Strategy 3: Try context.pages
                    elif hasattr(browser_session, 'context') and browser_session.context is not None:
                        pages = browser_session.context.pages
                        if pages and len(pages) > 0:
                            active_page = pages[0]  # Get the first (usually only) page
                            logger.info("✅ Got active page from browser_session.context.pages[0]")
                            logger.info(f"   Page type: {type(active_page)}")
                            logger.info(f"   Total pages: {len(pages)}")
                        else:
                            logger.warning("⚠️ browser_session.context.pages is empty")

                    # Strategy 4: Try browser.contexts[0].pages
                    elif hasattr(browser_session, 'browser') and browser_session.browser is not None:
                        contexts = browser_session.browser.contexts
                        if contexts and len(contexts) > 0:
                            pages = contexts[0].pages
                            if pages and len(pages) > 0:
                                active_page = pages[0]
                                logger.info("✅ Got active page from browser_session.browser.contexts[0].pages[0]")
                                logger.info(f"   Page type: {type(active_page)}")
                            else:
                                logger.warning("⚠️ browser_session.browser.contexts[0].pages is empty")
                        else:
                            logger.warning("⚠️ browser_session.browser.contexts is empty")

                    # Fallback: Use page passed during registration
                    if active_page is None:
                        logger.warning("⚠️ All page retrieval strategies failed")
                        logger.warning("⚠️ Falling back to page passed during registration")
                        active_page = page
                        if active_page:
                            logger.info(f"   Fallback page type: {type(active_page)}")
                        else:
                            logger.error("❌ Fallback page is also None!")

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

                # Call the actual implementation with the active page
                # Wrap in timeout protection using CUSTOM_ACTION_TIMEOUT from config
                try:
                    # Determine final coordinates to use:
                    # Priority 1: confirmed_coords from element_index (most accurate)
                    # Priority 2: scaled coordinates from LLM (fallback)
                    final_x, final_y = scaled_x, scaled_y
                    if confirmed_coords:
                        final_x, final_y = confirmed_coords
                        logger.info(f"📍 Using CONFIRMED coordinates: ({final_x}, {final_y})")
                    else:
                        logger.info(f"📍 Using SCALED coordinates: ({final_x}, {final_y})")
                    
                    result = await asyncio.wait_for(
                        find_unique_locator_action(
                            x=final_x,  # Use confirmed or scaled coordinates
                            y=final_y,  # Use confirmed or scaled coordinates
                            element_id=params.element_id,
                            element_description=params.element_description,
                            expected_text=params.expected_text,  # Pass expected_text for semantic validation
                            candidate_locator=params.candidate_locator,
                            element_data=element_data_from_index,  # Pass element attributes from DOM
                            page=active_page
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
                    error_msg = result.get('error', 'Could not find unique locator')
                    logger.error(f"❌ Custom action failed: {error_msg}")

                    action_result = ActionResult(
                        error=f"FAILED: Could not find unique locator for {params.element_id}. Error: {error_msg}. Try different coordinates or description.",
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

