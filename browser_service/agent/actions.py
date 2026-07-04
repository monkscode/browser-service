"""
Custom action implementations for browser-use agent.

This module provides custom actions that the browser-use agent can call during workflow execution.
Custom actions allow the agent to invoke deterministic Python code for locator finding and validation,
bypassing the need for LLM calls for these operations.

Key Functions:
- find_unique_locator_action: Find and validate unique locator for element at coordinates

The custom action uses Playwright's built-in validation methods to ensure locators are unique
and valid before returning them to the agent.

Usage:
    from browser_service.agent.actions import find_unique_locator_action

    # Called by agent during workflow execution
    result = await find_unique_locator_action(
        x=450.5,
        y=320.8,
        element_id="elem_1",
        element_description="Search input box",
        expected_text="Search",
        candidate_locator="id=search-input",
        page=playwright_page
    )
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

# Get logger
logger = logging.getLogger(__name__)


def _log_success_result(element_id: str, result: Dict[str, Any]) -> None:
    """Log successful locator finding result with detailed information."""
    best_locator = result.get('best_locator')
    validation_summary = result.get('validation_summary', {})

    logger.info("")
    logger.info(f"{'='*80}")
    logger.info(f"✅ CUSTOM ACTION SUCCEEDED for {element_id}")
    logger.info(f"{'='*80}")
    logger.info(f"   Best Locator: {best_locator}")
    logger.info(f"   Locator Type: {validation_summary.get('best_type', 'unknown')}")
    logger.info(f"   Strategy: {validation_summary.get('best_strategy', 'unknown')}")
    logger.info("   Validation Results:")
    logger.info(f"      - validated: {result.get('validated', False)}")
    logger.info(f"      - count: {result.get('count', 0)}")
    logger.info(f"      - unique: {result.get('unique', False)}")
    logger.info(f"      - valid: {result.get('valid', False)}")
    logger.info(f"      - validation_method: {result.get('validation_method', 'unknown')}")
    logger.info("   Validation Summary:")
    logger.info(f"      - total_strategies: {validation_summary.get('total_generated', 0)}")
    logger.info(f"      - valid: {validation_summary.get('valid', 0)}")
    logger.info(f"      - unique: {validation_summary.get('unique', 0)}")
    logger.info(f"      - not_found: {validation_summary.get('not_found', 0)}")
    logger.info(f"      - not_unique: {validation_summary.get('not_unique', 0)}")
    logger.info(f"      - errors: {validation_summary.get('errors', 0)}")
    logger.info(f"{'='*80}")
    logger.info("")


def _log_failure_result(element_id: str, element_description: str, x: float, y: float, result: Dict[str, Any]) -> None:
    """Log failed locator finding result with detailed error information."""
    error = result.get('error', 'Unknown error')
    validation_summary = result.get('validation_summary', {})

    logger.error("")
    logger.error(f"{'='*80}")
    logger.error(f"❌ CUSTOM ACTION FAILED for {element_id}")
    logger.error(f"{'='*80}")
    logger.error(f"   Error: {error}")
    logger.error(f"   Element ID: {element_id}")
    logger.error(f"   Description: {element_description}")
    logger.error(f"   Coordinates: ({x}, {y})")
    if validation_summary:
        logger.error("   Validation Summary:")
        logger.error(f"      - total_strategies: {validation_summary.get('total_generated', 0)}")
        logger.error(f"      - valid: {validation_summary.get('valid', 0)}")
        logger.error(f"      - not_found: {validation_summary.get('not_found', 0)}")
        logger.error(f"      - not_unique: {validation_summary.get('not_unique', 0)}")
        logger.error(f"      - errors: {validation_summary.get('errors', 0)}")
    logger.error(f"{'='*80}")
    logger.error("")


async def find_unique_locator_action(
    x: float,
    y: float,
    element_id: str,
    element_description: str,
    expected_text: Optional[str] = None,
    candidate_locator: Optional[str] = None,
    element_data: Optional[Dict[str, Any]] = None,  # Element attributes from browser-use DOM
    page=None,
    iframe_context: Optional[str] = None,  # Iframe locator if element is inside an iframe
    is_collection: Optional[bool] = None,  # Collection flag for multi-element detection
    browser_session=None,  # BrowserSession for resolved_node lookup in smart_locator
    vision_type_hint: Optional[str] = None,  # LLM's visual type classification (any specialized type)
    vision_framework_hint: Optional[str] = None,  # LLM's framework guess (any specialized type)
) -> Dict[str, Any]:
    """
    Custom action that agent can call to find and validate unique locator.
    ALL validation done with Playwright - no JavaScript needed.
    Runs deterministically (no LLM calls).

    This function is registered with browser-use and callable by the agent.

    Comprehensive error handling includes:
    - Input validation (page object, coordinates, element_id)
    - Specific exception handling (TimeoutError, CancelledError, RuntimeError, ValueError)
    - Structured error results with error_type and full context
    - Detailed logging with element_id, coordinates, and error messages
    - SEMANTIC VALIDATION: Compares expected_text against actual element text

    Args:
        x: X coordinate of element center
        y: Y coordinate of element center
        element_id: Element identifier (elem_1, elem_2, etc.)
        element_description: Human-readable description
        expected_text: The actual visible text AI sees on the element (e.g., "Submit", "Nike Air Max 270").
                      Used for semantic validation to ensure we found the CORRECT element.
        candidate_locator: Optional locator suggested by agent (e.g., "id=search")
        page: Playwright page object
        iframe_context: Optional iframe locator (e.g., 'iframe[id=\"main\"]') if element is inside an iframe.
                       When provided, locator searches will be performed inside the iframe context.

    Returns:
        Dict with validated locator or error:
        {
            'element_id': str,
            'description': str,
            'found': bool,
            'best_locator': str | None,
            'all_locators': List[Dict],
            'element_info': Dict,
            'coordinates': Dict,
            'validation_summary': Dict,
            'error': str | None,  # Only present if error occurred
            'error_type': str | None,  # Type of error (e.g., 'TimeoutError', 'PageObjectError')
            'validated': bool,
            'count': int,
            'unique': bool,
            'valid': bool,
            'semantic_match': bool,  # NEW: True if actual text matches expected_text
            'validation_method': str
        }

    Phase: Error Handling and Logging
    Requirements: 8.2, 8.4, 9.1
    """
    # Import config here to avoid circular imports
    from browser_service.config import config

    # Budget for the strategy cascade — CUSTOM_ACTION_TIMEOUT env var, default 5
    custom_action_timeout = config.locator.custom_action_timeout

    logger.info(f"🎯 Custom Action: find_unique_locator called for {element_id}")
    logger.info(f"   Description: {element_description}")
    logger.info(f"   Coordinates: ({x}, {y})")
    if expected_text:
        logger.info(f"   Expected text: \"{expected_text}\"")
    if element_data:
        logger.info(f"   Element data from index: tag=<{element_data.get('tagName', '?')}>, text=\"{element_data.get('textContent', '')[:30]}...\"")
    if candidate_locator:
        logger.info(f"   Candidate locator: {candidate_locator}")
    if iframe_context:
        logger.info(f"   🖼️ Iframe context: {iframe_context}")
    if is_collection:
        logger.info(f"   📋 Collection mode: {is_collection}")

    # Helper function to create structured error result
    def create_error_result(error_type: str, error_message: str, additional_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a structured error result with complete validation data."""
        result = {
            'element_id': element_id,
            'description': element_description,
            'found': False,
            'error': error_message,
            'error_type': error_type,
            'coordinates': {'x': x, 'y': y},
            'validated': False,
            'count': 0,
            'unique': False,
            'valid': False,
            'validation_method': 'playwright'
        }
        if additional_context:
            result.update(additional_context)
        return result

    try:
        # ========================================
        # VALIDATION: Input Parameters
        # ========================================

        # Validate page object
        if page is None:
            error_msg = "Page object is None - cannot validate locators"
            logger.error(f"❌ {error_msg}")
            logger.error(f"   Element ID: {element_id}")
            logger.error(f"   Description: {element_description}")
            logger.error(f"   Coordinates: ({x}, {y})")
            return create_error_result('PageObjectError', error_msg)

        # Validate coordinates
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            error_msg = f"Invalid coordinates: x={x} (type={type(x).__name__}), y={y} (type={type(y).__name__})"
            logger.error(f"❌ {error_msg}")
            logger.error(f"   Element ID: {element_id}")
            return create_error_result('InvalidCoordinatesError', error_msg)

        if x < 0 or y < 0:
            error_msg = f"Negative coordinates not allowed: x={x}, y={y}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"   Element ID: {element_id}")
            return create_error_result('InvalidCoordinatesError', error_msg)

        # Validate element_id
        if not element_id or not isinstance(element_id, str):
            error_msg = f"Invalid element_id: {element_id} (type={type(element_id).__name__})"
            logger.error(f"❌ {error_msg}")
            return create_error_result('InvalidElementIdError', error_msg)

        # ========================================
        # STEP 1: Validate Candidate Locator (if provided)
        # ========================================

        if candidate_locator:
            logger.info("")
            logger.info("🔍 VALIDATING CANDIDATE LOCATOR")
            logger.info(f"   Locator: {candidate_locator}")
            logger.info("   Method: Playwright page.locator().count()")
            # Scope candidate validation to the iframe when one is detected.
            # Mirrors the search_context pattern in Step 2 (smart locator path).
            # Without this, page.locator() would search the main frame only,
            # causing silent false-rejection (count==0) or false-acceptance
            # (count==1 matching an unrelated main-frame element with the same selector).
            search_root = page.frame_locator(iframe_context) if iframe_context else page

            try:
                # Validate candidate locator syntax
                if not isinstance(candidate_locator, str) or not candidate_locator.strip():
                    logger.warning(f"⚠️ Invalid candidate locator format: {candidate_locator}")
                    logger.info("🔄 Continuing with smart locator finder...")
                else:
                    # Use shared conversion function from browser_service.locators
                    from browser_service.locators import convert_to_playwright_locator
                    
                    playwright_locator, was_converted = convert_to_playwright_locator(candidate_locator)
                    
                    if was_converted:
                        logger.info(f"   Converted to Playwright format: {playwright_locator}")

                    # DEBUG: Log page state before locator call
                    try:
                        page_url = page.url
                        logger.info(f"   DEBUG: Page URL before locator: {page_url}")
                        logger.info(f"   DEBUG: Locator being used: '{playwright_locator}'")
                    except Exception as debug_e:
                        logger.warning(f"   DEBUG: Could not get page URL: {debug_e}")

                    # Try to validate with Playwright
                    count = await search_root.locator(playwright_locator).count()
                    logger.info(f"   DEBUG: page.locator('{playwright_locator}').count() returned: {count}")

                    # Log detailed validation results
                    is_unique = (count == 1)
                    is_valid = (count == 1)

                    logger.info("   Validation Results:")
                    logger.info(f"      - count: {count}")
                    logger.info(f"      - unique: {is_unique}")
                    logger.info(f"      - valid: {is_valid}")
                    logger.info("      - validated: True")
                    logger.info("      - validation_method: playwright")

                    if count == 1:
                        # Close the "unique but semantically wrong" hole (probe 06).
                        # Validate the RESOLVED element — what playwright_locator actually
                        # resolves to on the page — not the intended element from element_data
                        # (audit Issue 2b). Guard with expected_text so we accept on uniqueness
                        # alone when no semantic hint is available (audit Issue 5).
                        _semantic_ok = True
                        if expected_text:
                            from browser_service.locators import validate_semantic_match
                            _semantic_ok, _observed = await validate_semantic_match(
                                None, expected_text, page=search_root, locator=playwright_locator
                            )
                            if not _semantic_ok:
                                logger.info(
                                    f"⚠️ Candidate locator UNIQUE but semantically wrong "
                                    f"(expected={expected_text!r}, observed={_observed!r}); "
                                    f"falling through to smart locator finder."
                                )

                        if _semantic_ok:
                            # Candidate is valid, unique, and semantically correct.
                            # Use the converted locator for Browser Library compatibility.
                            final_locator = playwright_locator

                            logger.info("")
                            logger.info(f"{'='*80}")
                            logger.info("✅ CANDIDATE LOCATOR IS UNIQUE AND SEMANTIC MATCH - Using directly!")
                            logger.info(f"{'='*80}")
                            logger.info("   Skipping 21 strategies (not needed)")
                            logger.info(f"   Original: {candidate_locator}")
                            if was_converted:
                                logger.info(f"   Converted: {final_locator} (Browser Library compatible)")
                            logger.info("   Type: candidate")
                            logger.info("   Priority: 0 (agent-provided)")
                            logger.info(f"{'='*80}")
                            logger.info("")

                            locator_lower = final_locator.lower().lstrip()

                            # Extract element metadata from DOM data (available via element_data param)
                            elem_tag = ''
                            elem_has_text = False
                            elem_data_available = False
                            if element_data:
                                elem_tag = element_data.get('tagName', '').lower()
                                text_content = element_data.get('textContent', '') or element_data.get('text', '')
                                elem_has_text = bool(text_content and text_content.strip())
                                elem_data_available = True

                            return {
                                'element_id': element_id,
                                'description': element_description,
                                'found': True,
                                'best_locator': final_locator,  # Use converted locator
                                'all_locators': [{
                                    'type': 'candidate',
                                    'locator': final_locator,  # Use converted locator
                                    'priority': 0,
                                    'strategy': 'Agent-provided candidate (converted for Browser Library)' if was_converted else 'Agent-provided candidate',
                                    'count': count,
                                    'unique': True,
                                    'valid': True,
                                    'validated': True,
                                    'validation_method': 'playwright'
                                }],
                                'element_info': {},
                                'coordinates': {'x': x, 'y': y},
                                'validation_summary': {
                                    'total_generated': 1,
                                    'valid': 1,
                                    'unique': 1,
                                    'validated': 1,
                                    'not_found': 0,
                                    'not_unique': 0,
                                    'errors': 0,
                                    'best_type': 'candidate',
                                    'best_strategy': 'Agent-provided candidate',
                                    'validation_method': 'playwright'
                                },
                                # Add validation data at result level
                                'validated': True,
                                'count': count,
                                'unique': True,
                                'valid': True,
                                'validation_method': 'playwright',
                                # Per-element approach metrics for pattern analysis
                                'approach_metrics': {
                                    'locator_approach': 'actions_candidate',
                                    'fallback_depth': 0,  # Best case - candidate worked
                                    'success': True,
                                    'element_tag': elem_tag,
                                    'has_id': (
                                        locator_lower.startswith('#')
                                        or '[id=' in locator_lower
                                        or (
                                            'id=' in locator_lower
                                            and not any(k in locator_lower for k in ('data-testid=', 'data-test=', 'data-qa='))
                                        )
                                    ),
                                    'has_text_content': elem_has_text,
                                    'element_data_available': elem_data_available,
                                    'is_collection': is_collection is True,
                                    'is_in_iframe': bool(iframe_context),
                                }
                            }
                    elif count > 1:
                        logger.info(f"   ⚠️ Candidate locator NOT UNIQUE (matches {count} elements)")
                        logger.info("   🔄 Continuing with smart locator finder to find unique locator...")
                    else:  # count == 0
                        logger.info("   ⚠️ Candidate locator NOT FOUND (matches 0 elements)")
                        logger.info("   🔄 Continuing with smart locator finder to find valid locator...")

            except ValueError as e:
                # Invalid locator syntax
                logger.warning(f"⚠️ Candidate locator has invalid syntax: {e}")
                logger.warning(f"   Locator: {candidate_locator}")
                logger.info("🔄 Continuing with smart locator finder...")

            except asyncio.TimeoutError as e:
                # Playwright timeout during validation
                logger.warning(f"⚠️ Candidate locator validation timed out: {e}")
                logger.warning(f"   Locator: {candidate_locator}")
                logger.info("🔄 Continuing with smart locator finder...")

            except RuntimeError as e:
                # Playwright runtime errors (including invalid CSS selectors)
                error_str = str(e).lower()
                if 'not a valid selector' in error_str or 'invalid selector' in error_str:
                    logger.warning(f"⚠️ Candidate locator has invalid CSS syntax: {e}")
                    logger.warning(f"   Locator: {candidate_locator}")
                    logger.warning("   Note: This often happens with numeric IDs (e.g., #123)")
                    logger.info("🔄 Continuing with smart locator finder (will use [id='...'] syntax)...")
                else:
                    logger.warning(f"⚠️ Candidate locator validation failed with RuntimeError: {e}")
                    logger.warning(f"   Locator: {candidate_locator}")
                    logger.info("🔄 Continuing with smart locator finder...")

            except Exception as e:
                # Generic error during candidate validation
                logger.warning(f"⚠️ Candidate locator validation failed: {type(e).__name__}: {e}")
                logger.warning(f"   Locator: {candidate_locator}")
                logger.info("🔄 Continuing with smart locator finder...")

        # ========================================
        # STEP 2: Call Smart Locator Finder
        # ========================================

        logger.info("🔍 Calling smart_locator_finder with 21 strategies...")

        # Import smart_locator_finder from browser_service.locators
        try:
            from browser_service.locators import find_unique_locator_at_coordinates
        except ImportError as e:
            error_msg = f"Failed to import smart_locator from browser_service.locators: {e}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"   Element ID: {element_id}")
            logger.error("   This is a critical error - smart_locator module is required")
            return create_error_result('ImportError', error_msg)

        # Call smart locator finder with timeout protection
        try:
            # Create search context based on iframe detection
            # If element is inside an iframe, use frame_locator for all searches
            if iframe_context:
                logger.info(f"🖼️ Creating frame context: page.frame_locator('{iframe_context}')")
                search_context = page.frame_locator(iframe_context)
            else:
                search_context = page
            
            # Per-element latency telemetry (bench harness). The timer line is
            # emitted on the timeout path too, so every element yields a sample.
            _locator_timer_start = time.monotonic()
            result = await asyncio.wait_for(
                find_unique_locator_at_coordinates(
                    page=page,
                    search_context=search_context,  # Either page or frame_locator
                    iframe_context=iframe_context,  # For composite locator generation
                    x=x,
                    y=y,
                    element_id=element_id,
                    element_description=element_description,
                    expected_text=expected_text,  # Pass expected_text for semantic validation
                    library_type=config.robot_library,  # Use configured library type
                    element_data=element_data,  # Pass element attributes from browser-use DOM
                    is_collection=is_collection,  # Pass collection flag for multi-element detection
                    browser_session=browser_session,  # For resolved_node lookup (DELTA 1)
                    vision_type_hint=vision_type_hint,  # LLM's visual classification (1 of 2 sources)
                    vision_framework_hint=vision_framework_hint,  # LLM's framework guess
                ),
                timeout=custom_action_timeout
            )

            duration_ms = (time.monotonic() - _locator_timer_start) * 1000.0
            logger.info(
                f"LOCATOR_TIMER element_id={element_id} "
                f"duration_ms={duration_ms:.1f} found={bool(result.get('found'))}"
            )
            # Only enrich an EXISTING approach_metrics dict — failure results may
            # not carry one, and inventing it here would corrupt pattern analysis.
            if isinstance(result.get('approach_metrics'), dict):
                result['approach_metrics']['duration_ms'] = round(duration_ms, 1)

            # Log the result with detailed information
            if result.get('found'):
                _log_success_result(element_id, result)
            else:
                _log_failure_result(element_id, element_description, x, y, result)

            return result

        except asyncio.TimeoutError:
            duration_ms = (time.monotonic() - _locator_timer_start) * 1000.0
            logger.info(
                f"LOCATOR_TIMER element_id={element_id} "
                f"duration_ms={duration_ms:.1f} found=False"
            )
            # Handle timeout gracefully
            timeout_msg = f"Smart locator finder timed out after {custom_action_timeout} seconds"
            logger.error(f"⏱️ {timeout_msg}")
            logger.error(f"   Element ID: {element_id}")
            logger.error(f"   Description: {element_description}")
            logger.error(f"   Coordinates: ({x}, {y})")
            logger.error("   This may indicate a complex page or slow network")

            return create_error_result('TimeoutError', timeout_msg, {
                'timeout_seconds': custom_action_timeout
            })

        except asyncio.CancelledError:
            # Task was cancelled (e.g., browser closed)
            cancel_msg = "Smart locator finder was cancelled (browser may have closed)"
            logger.error(f"🚫 {cancel_msg}")
            logger.error(f"   Element ID: {element_id}")
            logger.error(f"   Coordinates: ({x}, {y})")

            return create_error_result('CancelledError', cancel_msg)

        except RuntimeError as e:
            # Runtime errors (e.g., event loop issues, browser closed)
            runtime_msg = f"Runtime error in smart locator finder: {str(e)}"
            logger.error(f"❌ {runtime_msg}")
            logger.error(f"   Element ID: {element_id}")
            logger.error(f"   Coordinates: ({x}, {y})")
            logger.error("   This may indicate the browser was closed or the page navigated away")
            logger.error("   Stack trace:", exc_info=True)

            return create_error_result('RuntimeError', runtime_msg)

        except Exception as e:
            # Catch any other errors from smart_locator_finder
            finder_error_msg = f"Smart locator finder raised {type(e).__name__}: {str(e)}"
            logger.error(f"❌ {finder_error_msg}")
            logger.error(f"   Element ID: {element_id}")
            logger.error(f"   Coordinates: ({x}, {y})")
            logger.error("   Stack trace:", exc_info=True)

            return create_error_result(type(e).__name__, finder_error_msg)

    except asyncio.TimeoutError:
        # Top-level timeout (shouldn't happen, but handle it)
        timeout_msg = "Custom action timed out at top level"
        logger.error(f"⏱️ {timeout_msg}")
        logger.error(f"   Element ID: {element_id}")
        logger.error(f"   Coordinates: ({x}, {y})")

        return create_error_result('TimeoutError', timeout_msg)

    except asyncio.CancelledError:
        # Top-level cancellation
        cancel_msg = "Custom action was cancelled"
        logger.error(f"🚫 {cancel_msg}")
        logger.error(f"   Element ID: {element_id}")
        logger.error(f"   Coordinates: ({x}, {y})")

        return create_error_result('CancelledError', cancel_msg)

    except KeyboardInterrupt:
        # User interrupted execution
        interrupt_msg = "Custom action interrupted by user"
        logger.error(f"⚠️ {interrupt_msg}")
        logger.error(f"   Element ID: {element_id}")

        return create_error_result('KeyboardInterrupt', interrupt_msg)

    except Exception as e:
        # Catch-all for any unexpected errors
        error_msg = f"Unexpected error in find_unique_locator_action: {type(e).__name__}: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.error(f"   Element ID: {element_id}")
        logger.error(f"   Description: {element_description}")
        logger.error(f"   Coordinates: ({x}, {y})")
        if candidate_locator:
            logger.error(f"   Candidate locator: {candidate_locator}")
        logger.error("   Stack trace:", exc_info=True)

        return create_error_result(type(e).__name__, error_msg)
