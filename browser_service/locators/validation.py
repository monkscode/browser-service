"""
Locator Validation Module

This module validates locators using Playwright's built-in API methods.
It checks if a locator uniquely identifies an element on the page.

CRITICAL VALIDATION RULE:
- A locator is ONLY valid if count=1 (unique)
- If count>1, the locator matches multiple elements and is NOT usable
- If count=0, the locator doesn't match any elements

The validation process:
1. Count matches using Playwright's locator.count() method
2. Get element details for the first match
3. Check visibility using is_visible()
4. Verify coordinates match the expected element (if provided)
5. Return validation results with valid=True only if count==1

This is equivalent to testing a selector in the browser's F12 Console.
"""

from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# LOCATOR CONVERSION FUNCTIONS
# =============================================================================

# Playwright's built-in selector engines that work natively
PLAYWRIGHT_NATIVE_ENGINES = frozenset(['id', 'text', 'css', 'xpath', 'role'])


def is_already_playwright_selector(locator: str) -> bool:
    """
    Check if a locator is already in a Playwright-compatible format.
    
    Playwright-compatible formats include:
    - Attribute selectors: [name="q"], [placeholder="Search"]
    - CSS selectors with brackets: a[title="..."], input[type="text"]
    - Class selectors: div.class
    - ID selectors: #myid
    - Child selectors: div > span
    - Sibling selectors: div + p, div ~ p
    - Pseudo selectors: a:hover, div::before
    - Descendant selectors: div span
    
    Args:
        locator: The locator string to check
        
    Returns:
        True if the locator is already Playwright-compatible, False otherwise
    """
    if not locator or not isinstance(locator, str):
        return False
    
    locator = locator.strip()
    
    # If it starts with '[', it's definitely an attribute selector
    if locator.startswith('['):
        return True
    
    # If it contains '[', it's a CSS selector with attribute (e.g., a[href="..."])
    if '[' in locator:
        return True
    
    # For other CSS characters, we need to check if they appear BEFORE any '='
    # This prevents false positives like placeholder=john.doe@email.com
    # where the '.' is in the VALUE, not the selector part
    
    if '=' in locator:
        # Split into prefix (selector part) and value
        prefix_part = locator.split('=', 1)[0]
        
        # Check for CSS characters in the PREFIX only
        # Space in prefix = descendant selector (div span=...) - unlikely but check
        # . in prefix = class selector (div.class=...) - unlikely but check
        # # in prefix = ID selector (#myid=...) - unlikely but check  
        # >, +, ~, : in prefix = combinators/pseudo
        css_chars_in_prefix = any(c in prefix_part for c in '.#>+~: ')
        
        return css_chars_in_prefix
    else:
        # No '=' sign - check if it looks like a CSS selector
        # Could be: tag name, .class, #id, div > span, etc.
        return any(c in locator for c in '.#>+~: ')


def convert_to_playwright_locator(locator: str) -> Tuple[str, bool]:
    """
    Convert various locator formats to Playwright-compatible format.
    
    This function handles conversion of shorthand locator formats to formats
    that Playwright understands natively. It's a pure function with no side effects.
    
    Supported conversions:
    - name=value → [name='value']
    - placeholder=value → [placeholder='value']
    - aria-label=value → [aria-label='value']
    - data-testid=value → [data-testid='value']
    - title=value → [title='value']
    - link=text → text=text
    - tag=button → button
    - All other attr=value → [attr='value']
    
    Playwright-native engines (id, text, css, xpath, role) are preserved as-is.
    Note: id= is now a native Playwright engine and handles numeric IDs correctly.
    
    Args:
        locator: The locator string to convert
        
    Returns:
        Tuple of (converted_locator, was_converted):
        - converted_locator: The Playwright-compatible locator string
        - was_converted: True if conversion was performed, False if passed through as-is
        
    Examples:
        >>> convert_to_playwright_locator("name=q")
        ("[name='q']", True)
        >>> convert_to_playwright_locator("id=search")
        ("id=search", False)  # Playwright native engine - no conversion
        >>> convert_to_playwright_locator("id=123")
        ("id=123", False)  # Playwright id= handles numeric IDs natively
        >>> convert_to_playwright_locator("[name='q']")
        ("[name='q']", False)
        >>> convert_to_playwright_locator("placeholder=Search products")
        ("[placeholder='Search products']", True)
    """
    if not locator or not isinstance(locator, str):
        return (locator or "", False)
    
    locator = locator.strip()
    
    # If it's already a CSS/Playwright selector, return as-is
    if is_already_playwright_selector(locator):
        return (locator, False)
    
    # If no '=', it's likely a tag name or already valid
    if '=' not in locator:
        return (locator, False)
    
    # Split into prefix and value
    prefix, value = locator.split('=', 1)
    prefix_lower = prefix.lower().strip()
    prefix_original = prefix.strip()  # Keep original case for attribute names
    value = value.strip()
    
    # Remove surrounding quotes if present (handles both single and double quotes)
    # e.g., title="Premium Sports..." → Premium Sports...
    # e.g., title='Premium Sports...' → Premium Sports...
    if len(value) >= 2:
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
    
    # Handle empty value after quote stripping (Issue #4 fix)
    if not value:
        # Return original locator - empty value is likely intentional or an error
        # Let Playwright handle the validation
        return (locator, False)
    
    # Escape special characters for CSS attribute selector
    # Order matters: escape backslashes FIRST, then single quotes
    # 1. Backslash is CSS escape character - must double it
    # 2. Single quote would terminate our string - must escape it
    value_escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    
    # Check if it's a Playwright-native engine
    if prefix_lower in PLAYWRIGHT_NATIVE_ENGINES:
        # These work natively in Playwright - don't convert
        return (locator, False)
    
    # Special conversions
    if prefix_lower == 'link':
        # link=text → text=text for Playwright
        return (f"text={value}", True)
    
    if prefix_lower == 'tag':
        # tag=button → button for Playwright (CSS tag selector)
        return (value, True)
    
    # ALL other formats are treated as HTML attribute selectors
    # This covers:
    # - Standard HTML: placeholder, name, title, type, class, href, src, alt, value, for
    # - ARIA: aria-label, aria-labelledby, aria-describedby, aria-expanded, etc.
    # - Testing: data-testid, data-test, data-cy, data-qa, data-automation
    # - Frameworks: ng-model (Angular), v-model (Vue), formControlName, etc.
    # Convert to CSS attribute selector: [attr='value']
    return (f"[{prefix_original}='{value_escaped}']", True)


# =============================================================================
# LOCATOR VALIDATION FUNCTIONS
# =============================================================================


async def validate_locator_playwright(
    page,
    locator: str,
    expected_coords: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Validate a locator using Playwright's built-in methods.
    This is exactly like testing a selector in F12 Console.

    CRITICAL: Only locators with count=1 are marked as valid=True (unique locators only).

    Args:
        page: Playwright page object
        locator: The locator to validate (e.g., "id=search", "text=Login")
        expected_coords: Optional {x, y} to verify we found the right element

    Returns:
        Dictionary with validation results:
        - valid: True only if count==1 (unique locator)
        - unique: True if count==1
        - count: Number of elements matching the locator
        - validated: True if validation was performed
        - validation_method: Always "playwright"
        - is_visible: Whether the element is visible
        - correct_element: Whether coordinates match (if expected_coords provided)
        - element_info: Details about the element (tag, id, class, text, etc.)
        - bounding_box: Element's position and size
        - error: Error message if validation failed

    Example:
        >>> result = await validate_locator_playwright(page, "id=search-btn")
        >>> result['valid']  # True if exactly 1 match
        True
        >>> result['count']  # Number of matches
        1
    """
    try:
        # Step 1: Count matches (like F12: document.querySelectorAll().length)
        count = await page.locator(locator).count()

        if count == 0:
            return {
                'valid': False,
                'unique': False,
                'count': 0,
                'validated': True,
                'validation_method': 'playwright',
                'error': 'Locator does not match any elements'
            }

        # Step 2: Get first element details (like F12: inspect element)
        element_info = await page.locator(locator).first.evaluate("""
            (el) => ({
                tag: el.tagName.toLowerCase(),
                id: el.id || null,
                className: el.className || null,
                text: el.textContent?.trim().slice(0, 100) || null,
                visible: el.offsetParent !== null,
                boundingBox: {
                    x: el.getBoundingClientRect().x,
                    y: el.getBoundingClientRect().y,
                    width: el.getBoundingClientRect().width,
                    height: el.getBoundingClientRect().height
                }
            })
        """)

        # Step 3: Check visibility (like F12: computed styles)
        is_visible = await page.locator(locator).first.is_visible()

        # Step 4: Get bounding box
        bounding_box = element_info['boundingBox']

        # Step 5: Verify it's the correct element (if coords provided)
        correct_element = True
        if expected_coords and bounding_box:
            # Check if expected coords are within the element's bounding box
            x_match = (bounding_box['x'] <= expected_coords['x'] <=
                       bounding_box['x'] + bounding_box['width'])
            y_match = (bounding_box['y'] <= expected_coords['y'] <=
                       bounding_box['y'] + bounding_box['height'])
            correct_element = x_match and y_match

        # CRITICAL: Only mark as valid if count == 1 (unique locator)
        return {
            'valid': count == 1,  # Only unique locators are valid
            'unique': count == 1,
            'count': count,
            'validated': True,
            'validation_method': 'playwright',
            'is_visible': is_visible,
            'correct_element': correct_element,
            'element_info': element_info,
            'bounding_box': bounding_box
        }

    except Exception as e:
        logger.error(f"Error validating locator '{locator}': {e}")
        return {
            'valid': False,
            'unique': False,
            'count': 0,
            'validated': False,
            'validation_method': 'playwright',
            'error': str(e)
        }
