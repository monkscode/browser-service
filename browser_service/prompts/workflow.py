"""
Workflow Prompt Builder

This module builds workflow prompts for the browser automation agent.
The prompts guide the agent through the process of:
1. Navigating to target URLs
2. Finding elements using vision
3. Extracting and validating locators
4. Returning structured results

The module supports two workflow modes:
- Custom Action Mode: Uses find_unique_locator action with Playwright validation
- Legacy Mode: Uses JavaScript-based validation (backward compatibility)

Prompt Structure:
- User goal and context
- Step-by-step workflow instructions
- Element list with descriptions
- Custom action documentation (if enabled)
- Example workflows
- Critical rules and completion criteria
"""

from typing import List, Dict, Any


def build_workflow_prompt(
    user_query: str,
    url: str,
    elements: List[Dict[str, Any]],
    library_type: str = "browser",
    include_custom_action: bool = True
) -> str:
    """
    Build workflow prompt for browser-use agent.

    The agent will:
    1. Navigate to the URL
    2. Find each element using vision
    3. Get element coordinates
    4. Call find_unique_locator custom action (if enabled) OR use JavaScript validation (legacy)

    Args:
        user_query: User's goal for the workflow
        url: Target URL to navigate to
        elements: List of elements to find, each with 'id', 'description', and optional 'action'
        library_type: Robot Framework library type - "browser" (Browser Library/Playwright)
                     or "selenium" (SeleniumLibrary)
        include_custom_action: If True, include custom action instructions;
                              if False, use legacy JavaScript validation

    Returns:
        Formatted prompt string for the agent

    Raises:
        ValueError: If elements list is empty or URL is invalid

    Example:
        >>> elements = [
        ...     {"id": "elem_1", "description": "Search input box", "action": "input"},
        ...     {"id": "elem_2", "description": "Search button", "action": "click"}
        ... ]
        >>> prompt = build_workflow_prompt(
        ...     user_query="Find search elements",
        ...     url="https://example.com",
        ...     elements=elements,
        ...     library_type="browser",
        ...     include_custom_action=True
        ... )
    """

    # Input validation
    if not elements:
        raise ValueError("Elements list cannot be empty")
    
    # Element limit safeguard - prevents LLM context overflow and excessively long workflows
    MAX_ELEMENTS = 50
    if len(elements) > MAX_ELEMENTS:
        raise ValueError(f"Too many elements ({len(elements)}). Maximum allowed is {MAX_ELEMENTS}.")
    
    if not url or not url.strip():
        raise ValueError("URL cannot be empty")
    
    # Ensure URL has protocol
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = f'https://{url}'
    
    # Sanitize user_query to prevent prompt injection
    # Replace newlines with spaces, limit length
    user_query = user_query.replace('\n', ' ').replace('\r', ' ').strip()
    if len(user_query) > 500:
        user_query = user_query[:500] + '...'

    # Build element list with validation
    element_list = []
    for idx, elem in enumerate(elements):
        elem_id = elem.get('id', f'elem_unknown_{idx}')  # Default with index if missing
        elem_desc = elem.get('description', 'No description provided')  # Default description
        elem_action = elem.get('action', 'get_text')  # Default to get_text if missing
        
        # Sanitize description to prevent prompt issues
        elem_desc = elem_desc.replace('\n', ' ').replace('\r', ' ').strip()
        if len(elem_desc) > 200:
            elem_desc = elem_desc[:200] + '...'
        
        element_list.append(f"   - {elem_id}: {elem_desc} (action: {elem_action})")

    elements_str = "\n".join(element_list)

    if include_custom_action:
        # NEW WORKFLOW: Use custom action for locator finding
        prompt = f"""
You are completing a web automation workflow.

USER'S GOAL: {user_query}

WORKFLOW STEPS:
1. Navigate to {url}
2. Find each element listed below using your vision
3. For EACH element, call the find_unique_locator action to get a validated unique locator

ELEMENTS TO FIND:
{elements_str}

═══════════════════════════════════════════════════════════════════
CUSTOM ACTION: find_unique_locator
═══════════════════════════════════════════════════════════════════

This action finds and validates unique locators for web elements using 21 systematic strategies.
It uses Playwright validation to ensure every locator is unique (count=1).

PARAMETERS:
  • x (float, required): X coordinate of element center
  • y (float, required): Y coordinate of element center
  • element_id (str, required): Element identifier from the list above (e.g., "elem_1")
  • element_description (str, required): USE THE EXACT DESCRIPTION from ELEMENTS TO FIND above!
    ⚠️ This MUST match the description from the list (e.g., "all visible rows in the table body")
    ⚠️ DO NOT rewrite or simplify the description - it's used for collection detection!
    ⚠️ Preserve keywords like "all", "rows", "cells", "items", "each" - these are CRITICAL!
  • expected_text (str, optional but HIGHLY RECOMMENDED): The ACTUAL visible text you see on the element.
    This is CRITICAL for validation - we use it to verify we found the RIGHT element.
    Examples: "Submit", "Add to Cart", "Nike Air Max 270", "Search"
    ⚠️ For buttons/links: Use the exact button/link text you see
    ⚠️ For inputs: Use the placeholder or label text if visible
    ⚠️ For product names: Use the actual product name text you see
  • element_index (int, ★★★ REQUIRED FOR ACCURACY ★★★): The element INDEX from the DOM state.
    ⚠️ ALWAYS PROVIDE THIS for EVERY element - it is the MOST ACCURATE METHOD!
    ⚠️ When you see [49] <td>John</td>, set element_index=49
    ⚠️ When you see [23] <a>Services</a>, set element_index=23
    ⚠️ This works for ALL elements including table cells, buttons, inputs, links, etc.
    ⚠️ We extract all element attributes (id, class, xpath) from this index automatically
    ⚠️ Without element_index, we fall back to less accurate coordinate-based approaches
  • candidate_locator (str, optional): Your suggested locator if you can identify one
    Examples: "id=search-input", "data-testid=login-btn", "name=username"

⚠️ CRITICAL - YOU MUST CALL THIS ACTION:
  • You MUST call find_unique_locator for EVERY element in the list above
  • Call it IMMEDIATELY after you've identified the element using your vision
  • Call it IMMEDIATELY after you've obtained the element's center coordinates
  • The custom action handles ALL validation automatically

HOW IT WORKS:
  1. If you provide a candidate_locator, the action validates it first with Playwright
  2. If the candidate is unique (count=1), it returns immediately - FAST!
  3. If the candidate is not unique or not provided, it tries 21 strategies:
     - Priority 1: id, data-testid, name (most stable)
     - Priority 2: aria-label, placeholder, title (semantic)
     - Priority 3: text content, role (content-based)
     - Priority 4-21: CSS and XPath strategies (fallbacks)
  4. Each strategy is validated with Playwright to ensure count=1
  5. Returns the first unique locator found

WHAT YOU RECEIVE:
The action returns a validated result with these fields:
  • validated: true (always - validation was performed)
  • count: 1 (guaranteed - only unique locators are returned)
  • unique: true (guaranteed - count equals 1)
  • valid: true (guaranteed - locator is usable)
  • best_locator: "id=search-input" (the validated locator string)
  • validation_method: "playwright" (how it was validated)
  • element_id: "elem_1" (matches your input)
  • found: true (element was found and locator extracted)

IMPORTANT - NO VALIDATION NEEDED FROM YOU:
  ✓ The action handles ALL validation using Playwright
  ✓ You do NOT need to check if the locator is unique
  ✓ You do NOT need to count elements
  ✓ You do NOT need to execute JavaScript
  ✓ Simply call the action and trust the validated result

═══════════════════════════════════════════════════════════════════
EXAMPLE WORKFLOW
═══════════════════════════════════════════════════════════════════

Scenario: Search for shoes on Flipkart and get first product name
Elements: elem_1 (search box, action=input), elem_2 (product name, action=get_text)

Step 1: Navigate to {url}

Step 2: Find elem_1 (search box) using vision
  → Element: "Search input box"
  → Coordinates: x=450.5, y=320.8

Step 3: Call find_unique_locator for elem_1
  find_unique_locator(
      x=450.5,
      y=320.8,
      element_id="elem_1",
      element_description="Search input box",
      expected_text="Search for products, brands and more",  ← ACTUAL placeholder text you see!
      candidate_locator="name=q"
  )
  → Result: {{"element_id": "elem_1", "best_locator": "[name='q']", "validated": true, "count": 1}}
  (Note: name=q is automatically converted to [name='q'] for Playwright compatibility)

Step 4: PERFORM ACTION for elem_1 (action=input)
  → Type "shoes" into the search box
  → Press Enter
  → Wait for search results to load

Step 5: Find elem_2 (product name) using vision on the CURRENT page (results page)
  → Element: "First product name in search results"
  → Coordinates: x=320.5, y=450.2

Step 6: Call find_unique_locator for elem_2
  find_unique_locator(
      x=320.5,
      y=450.2,
      element_id="elem_2",
      element_description="First product name in search results",
      expected_text="Nike Air Max 270"  ← ACTUAL product name text you see on screen!
  )
  → Result: {{"element_id": "elem_2", "best_locator": "[data-testid='product-title']", "validated": true, "count": 1, "semantic_match": true}}

Step 7: Store result (action=get_text means extract locator only, no interaction)

Step 8: Call done() with all validated results

KEY POINT: Elements are processed IN ORDER. elem_1's action (input) caused a page change,
so elem_2 is naturally found on the new page. No explicit phase separation needed.

═══════════════════════════════════════════════════════════════════
CRITICAL INSTRUCTIONS
═══════════════════════════════════════════════════════════════════

✓ MUST call find_unique_locator for EVERY element in the list
✓ MUST provide accurate coordinates (x, y) from your vision
✓ MUST provide expected_text - the ACTUAL visible text you see on the element
  (This is CRITICAL - it prevents finding the wrong element!)
✓ SHOULD provide candidate_locator if you can identify id, data-testid, or name
✓ MUST NOT validate locators yourself - the action does this
✓ MUST NOT execute JavaScript to check uniqueness - the action does this
✓ MUST NOT use querySelector, querySelectorAll, or execute_js for validation
✓ MUST NOT retry or check count - the action guarantees count=1
✓ ONLY call done() when ALL elements have validated results from the action

⛔ FORBIDDEN ACTIONS:
  • DO NOT call execute_js with querySelector to validate locators
  • DO NOT try to count elements yourself
  • DO NOT check if locators are unique yourself
  • The find_unique_locator action does ALL validation for you!

⛔ STRICT SCOPE - DO NOT OVER-COMPLETE THE TASK:
  • ONLY process the elements listed in ELEMENTS TO FIND above
  • ONLY perform actions explicitly requested in the USER'S GOAL
  • DO NOT add extra elements not in the list (e.g., if asked to fill username, don't also fill password)
  • DO NOT infer or guess additional steps (e.g., don't click Submit unless asked)
  • DO NOT "complete" forms or workflows beyond what was explicitly requested
  • If the user asks to "type X into field Y", ONLY type X into field Y - nothing more
  • Treat the ELEMENTS TO FIND list as EXHAUSTIVE - there are no hidden elements to find

═══════════════════════════════════════════════════════════════════
SEQUENTIAL ELEMENT PROCESSING
═══════════════════════════════════════════════════════════════════

Process elements IN THE ORDER THEY ARE LISTED. For each element:

1. Find the element using your vision
2. Get element coordinates (x, y)
3. Call find_unique_locator to extract and validate the locator
4. Based on the element's action field, decide what to do next:

   ACTION BEHAVIORS:
   • action='input': Type the specified text, press Enter, wait for page updates
   • action='click': Click the element, wait for page updates
   • action='submit': Click the element (submits form), wait for page updates
   • action='get_text', 'get_attribute', or any other: Just store the locator (no interaction)
   • action is missing/null: Just store the locator (no interaction)

5. Move to the NEXT element in the list

⚠️ IMPORTANT:
  • Process elements sequentially in the order given
  • Interactive actions (input/click/submit) may change the page
  • Subsequent elements will be found on whatever page is currently displayed
  • Wait for page loads/updates after interactive actions before moving to next element
  • The Step Planner has already ordered elements correctly for the workflow

⚠️ IMPORTANT - NUMERIC IDs:
  • If you find an element with ID starting with a number (e.g., id="892238219")
  • DO NOT try to use querySelector('#892238219') - this is INVALID CSS
  • INSTEAD: Call find_unique_locator with candidate_locator="id=892238219"
  • The custom action will handle numeric IDs correctly using [id="..."] syntax

⚠️ EDGE CASE HANDLING:
  • If an element cannot be found, record it as {{"element_id": "...", "found": false, "error": "Element not visible/not found"}}
  • Continue processing remaining elements (don't stop the entire workflow)
  • If an interactive element fails, still try to process result elements on the current page
  • If all elements have same action type (all interactive or all result), still process in order
  • Empty descriptions are handled (just use coordinates and candidate locator)
  • Missing element IDs will be assigned default values (elem_unknown_0, elem_unknown_1, etc.)
  • Special characters in descriptions are automatically sanitized

⚠️ CHECKBOX/RADIO HANDLING:
  • When clicking on checkboxes or radio buttons, provide the LABEL TEXT as expected_text
  • Examples: "checkbox 1", "remember me", "agree to terms", "male", "female"
  • The system will automatically find the actual <input> element, not the text label
  • Do NOT click directly on checkboxes - just call find_unique_locator with correct expected_text
  • The returned locator will point to the actual checkbox/radio input element
  • This ensures Robot Framework can use proper keywords like "Check Checkbox"

COMPLETION CRITERIA:
  • ALL elements in ELEMENTS TO FIND must have validated locators from find_unique_locator
  • ONLY elements in ELEMENTS TO FIND should have their actions performed
  • DO NOT process any elements not in the ELEMENTS TO FIND list
  • Each result must have: validated=true, count=1, unique=true, valid=true
  • Call done() IMMEDIATELY after processing ALL listed elements - do not continue
  • If the list has 1 element, process that 1 element and call done()
  • If the list has 3 elements, process those 3 elements and call done()
  • The number of elements you process MUST match the number in ELEMENTS TO FIND

Your final done() call MUST include the complete JSON with all elements_found data!
DO NOT extract text content - just return the validated locators!
"""
    else:
        # LEGACY WORKFLOW: Use JavaScript validation (backward compatibility)
        prompt = f"""
You are completing a web automation workflow.

USER'S GOAL: {user_query}

WORKFLOW STEPS:
1. Navigate to {url}
2. Find each element listed below using your vision
3. For EACH element, return its center coordinates (x, y)

ELEMENTS TO FIND:
{elements_str}

CRITICAL INSTRUCTIONS:
1. Use your vision to identify each element on the page
2. For EACH element, use execute_js to get its DOM ID and coordinates
3. Execute this JavaScript for each element you find:
   ```javascript
   (function() {{
     const element = document.querySelector('YOUR_SELECTOR_HERE');
     if (element) {{
       const rect = element.getBoundingClientRect();
       const domId = element.id || '';
       const domName = element.name || '';
       const domClass = element.className || '';
       const domTestId = element.getAttribute('data-testid') || '';

       // VALIDATE LOCATORS: Check uniqueness
       const locators = [];

       // Check ID locator
       if (domId) {{
         // Always use attribute selector for IDs (handles numeric IDs correctly)
         const idCount = document.querySelectorAll(`[id="${{domId}}"]`).length;
         locators.push({{
           type: 'id',
           locator: `id=${{domId}}`,
           count: idCount,
           unique: idCount === 1,
           validated: true,
           note: 'Using [id="..."] selector (works with numeric IDs)'
         }});
       }}

       // Check name locator
       if (domName) {{
         const nameCount = document.querySelectorAll(`[name="${{domName}}"]`).length;
         locators.push({{
           type: 'name',
           locator: `name=${{domName}}`,
           count: nameCount,
           unique: nameCount === 1,
           validated: true
         }});
       }}

       // Check data-testid locator
       if (domTestId) {{
         const testIdCount = document.querySelectorAll(`[data-testid="${{domTestId}}"]`).length;
         locators.push({{
           type: 'data-testid',
           locator: `data-testid=${{domTestId}}`,
           count: testIdCount,
           unique: testIdCount === 1,
           validated: true
         }});
       }}

       // Check CSS class locator
       if (domClass) {{
         const firstClass = domClass.split(' ')[0];
         const tagName = element.tagName.toLowerCase();
         const cssCount = document.querySelectorAll(`${{tagName}}.${{firstClass}}`).length;
         locators.push({{
           type: 'css-class',
           locator: `${{tagName}}.${{firstClass}}`,
           count: cssCount,
           unique: cssCount === 1,
           validated: true
         }});
       }}

       return JSON.stringify({{
         element_id: "REPLACE_WITH_ELEM_ID_FROM_LIST",
         found: true,
         coordinates: {{ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 }},
         element_type: element.tagName.toLowerCase(),
         visible_text: element.textContent.trim().substring(0, 100),
         dom_id: domId,
         dom_attributes: {{
           id: domId,
           name: domName,
           class: domClass,
           'data-testid': domTestId
         }},
         locators: locators
       }});
     }}
     return JSON.stringify({{ element_id: "REPLACE_WITH_ELEM_ID", found: false }});
   }})()
   ```

4. **CRITICAL VALIDATION STEP:** After executing JavaScript for each element, CHECK the locators:
   - Look at the "locators" array in the JavaScript result
   - Find locators where "unique": true AND "count": 1
   - If NO unique locator found for an element, try a DIFFERENT selector and execute JavaScript again
   - Keep trying different selectors until you find a unique locator (count=1)

5. ONLY call done() when ALL elements have at least ONE unique locator (count=1)
   ```json
   {{
     "workflow_completed": true,
     "elements_found": [
       {{ "element_id": "elem_1", "found": true, "coordinates": {{"x": 450, "y": 320}}, "dom_id": "search-input", ... }},
       {{ "element_id": "elem_2", "found": true, "coordinates": {{"x": 650, "y": 520}}, "dom_id": "product-link", ... }}
     ]
   }}
   ```

CRITICAL RULES:
- You MUST execute JavaScript for EACH element to get its DOM attributes
- You MUST CHECK if locators are unique (count=1) in the JavaScript result
- If a locator is NOT unique (count>1), try a DIFFERENT selector (more specific)
- ONLY call done() when ALL elements have at least ONE unique locator
- You MUST include the element_id from the list above in each result
- You MUST call done() with the complete JSON structure
- DO NOT just say "I found it" - you MUST return the structured JSON
- The JSON MUST include all elements from the list above

UNIQUENESS REQUIREMENT:
- A locator is ONLY valid if count=1 (unique)
- If count>1, the locator matches multiple elements and is NOT usable
- You MUST find a unique locator for each element before calling done()
- Try more specific selectors: id > data-testid > name > specific CSS > XPath

Your final done() call MUST include the complete JSON with all elements_found data!
REMEMBER: ONLY call done() when ALL elements have at least ONE unique locator (count=1)!
"""

    return prompt.strip()
