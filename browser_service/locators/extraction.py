"""
Element Attribute Extraction Module

This module provides the complete locator extraction and validation pipeline.
It combines element attribute extraction, locator generation, and Playwright
validation to find the best unique locator for a given element.

Pipeline:
1. Extract element attributes from coordinates using minimal JavaScript
2. Generate candidate locators from attributes (priority-ordered)
3. Validate each locator using Playwright API
4. Select the best unique locator (count=1, correct element)

This approach uses minimal JavaScript (< 50 lines) for attribute extraction,
then relies on Playwright's Python API for validation. This is much cleaner
and more maintainable than generating large JavaScript validation code.
"""

from typing import Dict, Any, Optional
import logging

from .generation import generate_locators_from_attributes
from .validation import validate_locator_playwright

logger = logging.getLogger(__name__)


async def extract_element_attributes(page, coords: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """
    Extract element attributes using minimal JavaScript.
    This is like inspecting an element in F12 DevTools.

    Args:
        page: Playwright page object
        coords: Dictionary with 'x' and 'y' coordinates

    Returns:
        Dictionary with element attributes or None if not found.
        Attributes include:
        - id: Element ID
        - name: Element name attribute
        - testId: data-testid attribute
        - ariaLabel: aria-label attribute
        - role: ARIA role
        - title: Title attribute
        - placeholder: Placeholder text
        - type: Input type
        - tagName: HTML tag name
        - className: CSS classes
        - text: Text content (first 100 chars)
        - href: Link href
        - visible: Whether element is visible
        - boundingBox: Element position and size

    Example:
        >>> attrs = await extract_element_attributes(page, {'x': 100, 'y': 200})
        >>> attrs['tagName']
        'button'
        >>> attrs['id']
        'submit-btn'
    """
    try:
        # Minimal JavaScript to get element attributes
        # Enhanced: 
        # 1. Shadow DOM support - pierces through shadow roots for modern web apps
        # 2. Checkbox/radio input detection for label-adjacent scenarios
        element_info = await page.evaluate("""
            (coords) => {
                // Shadow DOM piercing function - for Material UI, Salesforce Lightning, etc.
                function elementFromPointDeep(x, y, root = document) {
                    let element = root.elementFromPoint(x, y);
                    if (!element) return null;
                    while (element && element.shadowRoot) {
                        const innerElement = element.shadowRoot.elementFromPoint(x, y);
                        if (!innerElement || innerElement === element) break;
                        element = innerElement;
                    }
                    return element;
                }
                
                let el = elementFromPointDeep(coords.x, coords.y);
                if (!el || el.tagName === 'HTML' || el.tagName === 'BODY') {
                    return null;
                }

                // Helper function to extract attributes from an element
                const extractAttrs = (element) => ({
                    // Primary identifiers (highest priority)
                    id: element.id || null,
                    name: element.name || null,
                    testId: element.dataset?.testid || null,

                    // Semantic attributes
                    ariaLabel: element.getAttribute('aria-label') || null,
                    role: element.getAttribute('role') || null,
                    title: element.title || null,
                    placeholder: element.placeholder || null,
                    type: element.type || null,

                    // Structure
                    tagName: element.tagName.toLowerCase(),
                    className: element.className || null,

                    // Content
                    text: element.textContent?.trim().slice(0, 100) || null,
                    href: element.href || null,

                    // Visibility
                    visible: element.offsetParent !== null,

                    // Position (for verification)
                    boundingBox: {
                        x: element.getBoundingClientRect().x,
                        y: element.getBoundingClientRect().y,
                        width: element.getBoundingClientRect().width,
                        height: element.getBoundingClientRect().height
                    }
                });

                // If element is already an input (checkbox/radio), return it directly
                if (el.tagName === 'INPUT' && (el.type === 'checkbox' || el.type === 'radio')) {
                    return extractAttrs(el);
                }

                // If element is a LABEL, find its associated input
                if (el.tagName === 'LABEL') {
                    const forId = el.getAttribute('for');
                    if (forId) {
                        const input = document.getElementById(forId);
                        if (input && (input.type === 'checkbox' || input.type === 'radio')) {
                            return extractAttrs(input);
                        }
                    }
                    // Check for nested input
                    const nestedInput = el.querySelector('input[type="checkbox"], input[type="radio"]');
                    if (nestedInput) {
                        return extractAttrs(nestedInput);
                    }
                }

                // SMART DETECTION: If clicked on text near a checkbox/radio, find the actual input
                // This handles cases where checkbox text is not wrapped in a <label>
                const parent = el.parentElement;
                if (parent) {
                    // Look for checkbox/radio inputs in the same parent container
                    const inputs = parent.querySelectorAll('input[type="checkbox"], input[type="radio"]');
                    if (inputs.length > 0) {
                        // Find the closest input to the click point
                        let closestInput = null;
                        let closestDistance = Infinity;
                        
                        inputs.forEach(input => {
                            const rect = input.getBoundingClientRect();
                            const inputCenterX = rect.x + rect.width / 2;
                            const inputCenterY = rect.y + rect.height / 2;
                            const distance = Math.sqrt(
                                Math.pow(coords.x - inputCenterX, 2) + 
                                Math.pow(coords.y - inputCenterY, 2)
                            );
                            // Only consider inputs within 200px (reasonable proximity for label-input pairs)
                            if (distance < closestDistance && distance < 200) {
                                closestDistance = distance;
                                closestInput = input;
                            }
                        });
                        
                        if (closestInput) {
                            return extractAttrs(closestInput);
                        }
                    }
                    
                    // Also check grandparent (for deeper nesting like <form><div>text<input></div></form>)
                    const grandparent = parent.parentElement;
                    if (grandparent) {
                        const gpInputs = grandparent.querySelectorAll('input[type="checkbox"], input[type="radio"]');
                        if (gpInputs.length > 0) {
                            let closestInput = null;
                            let closestDistance = Infinity;
                            
                            gpInputs.forEach(input => {
                                const rect = input.getBoundingClientRect();
                                const inputCenterX = rect.x + rect.width / 2;
                                const inputCenterY = rect.y + rect.height / 2;
                                const distance = Math.sqrt(
                                    Math.pow(coords.x - inputCenterX, 2) + 
                                    Math.pow(coords.y - inputCenterY, 2)
                                );
                                if (distance < closestDistance && distance < 200) {
                                    closestDistance = distance;
                                    closestInput = input;
                                }
                            });
                            
                            if (closestInput) {
                                return extractAttrs(closestInput);
                            }
                        }
                    }
                }

                // No nearby input found, return original element's attributes
                return extractAttrs(el);
            }
        """, coords)

        return element_info

    except Exception as e:
        logger.error(f"Error extracting element attributes: {e}")
        return None


async def extract_and_validate_locators(
    page,
    element_description: str,
    element_coords: Dict[str, float],
    library_type: str = "browser"
) -> Dict[str, Any]:
    """
    Complete locator extraction and validation pipeline.
    Uses Playwright's built-in methods - no massive JavaScript!

    CRITICAL: Only returns locators with count=1 as valid (unique locators only).

    Args:
        page: Playwright page object
        element_description: Description of the element (for logging)
        element_coords: {x, y} coordinates from browser-use vision
        library_type: "browser" or "selenium"

    Returns:
        Dictionary with extraction results:
        - found: Whether a valid locator was found
        - best_locator: The best unique locator string
        - all_locators: List of all validated locators
        - unique_locators: List of unique locators (count=1)
        - element_info: Element attributes
        - validation_summary: Summary of validation results
        - validated: Whether validation was performed
        - count: Count for best locator (1 if unique)
        - unique: Whether best locator is unique
        - valid: Whether best locator is valid
        - validation_method: Always "playwright"
        - error: Error message if extraction failed

    Example:
        >>> result = await extract_and_validate_locators(
        ...     page, "Search button", {'x': 100, 'y': 200}, "browser"
        ... )
        >>> result['found']
        True
        >>> result['best_locator']
        'id=search-btn'
        >>> result['unique']
        True
    """
    logger.info(f"🔍 Extracting locators for: {element_description}")
    logger.info(
        f"   Coordinates: ({element_coords['x']}, {element_coords['y']})")

    # Step 1: Extract element attributes using minimal JavaScript
    element_attrs = await extract_element_attributes(page, element_coords)

    if not element_attrs:
        logger.error("❌ Could not find element at coordinates")
        return {
            'found': False,
            'error': 'Element not found at coordinates'
        }

    logger.info(
        f"   Found element: <{element_attrs['tagName']}> \"{element_attrs.get('text', '')[:50]}\"")

    # Step 2: Generate locators from attributes (in Python, not JavaScript!)
    locators = generate_locators_from_attributes(element_attrs, library_type)

    if not locators:
        logger.warning(
            "⚠️ No locators could be generated from element attributes")
        return {
            'found': False,
            'error': 'No locators could be generated',
            'element_info': element_attrs
        }

    logger.info(f"   Generated {len(locators)} candidate locators")

    # Step 3: Validate each locator using Playwright validation
    validated_locators = []
    for loc in locators:
        validation = await validate_locator_playwright(
            page,
            loc['locator'],
            element_coords
        )

        # Merge validation results into locator dict
        loc.update(validation)

        if loc.get('validated'):
            validated_locators.append(loc)
            if loc.get('valid'):
                # valid=True means count=1 (unique)
                status = "✅ UNIQUE"
            else:
                # valid=False means count>1 or count=0
                count = loc.get('count', 0)
                if count > 1:
                    status = f"⚠️ NOT UNIQUE ({count} matches)"
                else:
                    status = "❌ NOT FOUND"
            correct = "✅" if loc.get(
                'correct_element') else "⚠️ Different element"
            logger.info(
                f"   {loc['type']}: {loc['locator']} → {status}, {correct}")
        else:
            logger.warning(f"   ❌ {loc['type']}: {loc['locator']} → VALIDATION FAILED")

    # Step 4: Filter and sort - ONLY unique locators (count=1) are considered valid
    unique_locators = [loc for loc in validated_locators if loc.get(
        'unique') and loc.get('correct_element')]
    valid_locators = [loc for loc in validated_locators if loc.get('valid')]

    # Step 5: Select best locator
    best_locator = None
    if unique_locators:
        # Prefer unique locators that match the correct element, sorted by priority
        best_locator = sorted(unique_locators, key=lambda x: x['priority'])[0]
        logger.info(
            f"✅ Best locator: {best_locator['locator']} (unique, correct element)")
    elif valid_locators:
        # Fallback to any valid locator
        best_locator = sorted(valid_locators, key=lambda x: x['priority'])[0]
        logger.warning(
            f"⚠️ Best locator: {best_locator['locator']} (valid but not unique or wrong element)")

    result = {
        'found': best_locator is not None,
        'best_locator': best_locator['locator'] if best_locator else None,
        'all_locators': validated_locators,
        'unique_locators': unique_locators,
        'element_info': element_attrs,
        'validation_summary': {
            'total_generated': len(locators),
            'valid': len(valid_locators),
            'unique': len(unique_locators),
            'best_type': best_locator['type'] if best_locator else None,
            'validation_method': 'playwright'
        }
    }

    # Add validation data to the result itself for easy access
    if best_locator:
        result['validated'] = True
        result['count'] = best_locator.get('count', 1)
        result['unique'] = best_locator.get('unique', True)
        result['valid'] = best_locator.get('valid', True)
        result['validation_method'] = 'playwright'
    else:
        result['validated'] = True  # Validation was attempted
        result['count'] = 0  # No unique locator found
        result['unique'] = False
        result['valid'] = False
        result['validation_method'] = 'playwright'

    return result
