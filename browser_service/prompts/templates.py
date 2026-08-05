"""
Prompt Templates

This module contains reusable prompt fragments and templates used across
workflow and system prompts. These constants help maintain consistency
and make it easier to update common instructions.

Template Variables:
- {url}: Target URL to navigate to
- {user_query}: User's goal or objective
- {elements_str}: Formatted list of elements to find
- {elem_id}: Element identifier (e.g., "elem_1")
- {elem_desc}: Element description
- {elem_action}: Action to perform on element

Usage:
    Import these constants in workflow.py or system.py to build prompts
    with consistent messaging and formatting.
"""

# Custom Action Documentation
CUSTOM_ACTION_HEADER = """
═══════════════════════════════════════════════════════════════════
CUSTOM ACTION: find_unique_locator
═══════════════════════════════════════════════════════════════════

This action finds and validates unique locators for web elements using 21 systematic strategies.
It uses Playwright validation to ensure every locator is unique (count=1).
"""

CUSTOM_ACTION_PARAMETERS = """
PARAMETERS:
  • x (float, required): X coordinate of element center
  • y (float, required): Y coordinate of element center
  • element_id (str, required): Element identifier from the list above (e.g., "elem_1")
  • element_description (str, required): Human-readable description of the element
  • candidate_locator (str, optional): Your suggested locator if you can identify one
    Examples: "id=search-input", "data-testid=login-btn", "name=username"
"""

CUSTOM_ACTION_HOW_IT_WORKS = """
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
"""

CUSTOM_ACTION_RETURN_VALUE = """
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
"""

CUSTOM_ACTION_NO_VALIDATION_NEEDED = """
IMPORTANT - NO VALIDATION NEEDED FROM YOU:
  ✓ The action handles ALL validation using Playwright
  ✓ You do NOT need to check if the locator is unique
  ✓ You do NOT need to count elements
  ✓ You do NOT need to execute JavaScript
  ✓ Simply call the action and trust the validated result
"""

# Critical Instructions
CRITICAL_MUST_CALL_ACTION = """
⚠️ CRITICAL - YOU MUST CALL THIS ACTION:
  • You MUST call find_unique_locator for EVERY element in the list above
  • Call it IMMEDIATELY after you've identified the element using your vision
  • Call it IMMEDIATELY after you've obtained the element's center coordinates
  • The custom action handles ALL validation automatically
"""

FORBIDDEN_ACTIONS = """
⛔ FORBIDDEN ACTIONS:
  • DO NOT call execute_js with querySelector to validate locators
  • DO NOT try to count elements yourself
  • DO NOT check if locators are unique yourself
  • DO NOT extract text content from elements - just find the locators
  • DO NOT use querySelector after getting the locator - just return it
  • The find_unique_locator action does ALL validation for you!
"""

YOUR_ONLY_JOB = """
⚠️ IMPORTANT - YOUR ONLY JOB:
  • Find elements and get their validated locators
  • DO NOT extract text, click, or interact with elements
  • DO NOT verify the locator works by using it
  • Just call find_unique_locator and store the result
  • The locators will be used later in Robot Framework tests
"""

NUMERIC_IDS_WARNING = """
⚠️ IMPORTANT - NUMERIC IDs:
  • If you find an element with ID starting with a number (e.g., id="892238219")
  • DO NOT try to use querySelector('#892238219') - this is INVALID CSS
  • INSTEAD: Call find_unique_locator with candidate_locator="id=892238219"
  • The custom action will handle numeric IDs correctly using [id="..."] syntax
  • DO NOT try to extract text using the locator - just return the locator itself
"""

DO_NOT_EXTRACT_TEXT = """
⚠️ CRITICAL - DO NOT EXTRACT TEXT:
  • After getting the locator from find_unique_locator, DO NOT use it
  • DO NOT call execute_js to extract text using the locator
  • DO NOT verify the locator by using querySelector
  • Just store the locator and move to the next element
  • The locators will be used in Robot Framework tests, not by you
"""

# Completion Criteria
COMPLETION_CRITERIA_CUSTOM_ACTION = """
COMPLETION CRITERIA:
  • ALL elements must have validated results from find_unique_locator action
  • Each result must have: validated=true, count=1, unique=true, valid=true
  • Call done() with complete JSON structure containing all results
  • DO NOT extract text or interact with elements - just return the locators
"""

COMPLETION_CRITERIA_LEGACY = """
COMPLETION CRITERIA:
- ONLY call done() when ALL elements have unique locators (count=1)
- Include success=True if all elements have unique locators
- Include success=False if you cannot find unique locators for some elements
"""

# Workflow Steps
WORKFLOW_STEPS_CUSTOM_ACTION = """
WORKFLOW STEPS:
1. Navigate to {url}
2. Find each element listed below using your vision
3. For EACH element, call the find_unique_locator action to get a validated unique locator
"""

WORKFLOW_STEPS_LEGACY = """
WORKFLOW STEPS:
1. Navigate to {url}
2. Find each element listed below using your vision
3. For EACH element, return its center coordinates (x, y)
"""

# Example JSON Structures
EXAMPLE_RESULT_JSON = """
{
  "element_id": "elem_1",
  "found": true,
  "best_locator": "id=search-input",
  "validated": true,
  "count": 1,
  "unique": true,
  "valid": true,
  "validation_method": "playwright"
}
"""

EXAMPLE_WORKFLOW_COMPLETION_JSON = """
{
  "workflow_completed": true,
  "results": [
    {
      "element_id": "elem_1",
      "found": true,
      "best_locator": "id=search-input",
      "validated": true,
      "count": 1,
      "unique": true
    },
    {
      "element_id": "elem_2",
      "found": true,
      "best_locator": "data-testid=product-card",
      "validated": true,
      "count": 1,
      "unique": true
    }
  ]
}
"""

# Validation Rules
UNIQUENESS_REQUIREMENT = """
UNIQUENESS REQUIREMENT:
- A locator is ONLY valid if count=1 (unique)
- If count>1, the locator matches multiple elements and is NOT usable
- You MUST find a unique locator for each element before calling done()
- Try more specific selectors: id > data-testid > name > specific CSS > XPath
"""

# Locator Priority Order
LOCATOR_PRIORITY_ORDER = """
LOCATOR PRIORITY ORDER:
1. id (most stable, unique identifier)
2. data-testid (designed for testing)
3. name (semantic attribute)
4. aria-label (accessibility attribute)
5. placeholder (form field hint)
6. title (tooltip text)
7. text content (visible text)
8. role (Playwright-specific)
9. CSS class (lower priority, may not be unique)
10. XPath (fallback for complex cases)
"""

# System Prompt Fragments
VERIFICATION_RULES = """
⚠️ CRITICAL VERIFICATION RULES:
   1. When find_unique_locator returns validated=true, the locator is UNIQUE (count=1)
   2. You MUST verify the locator points to the CORRECT element (matches description)
   3. If locator is unique BUT wrong element → Try again with different coordinates
   4. If locator is unique AND correct element → Mark SUCCESS, move to next element
   5. Maximum 2 retries per element - if still wrong, mark as failed and move on
"""

RETRY_LOGIC = """
RETRY LOGIC:
- Retry 1: If locator is unique but wrong element, try different coordinates
- Retry 2: If still wrong, try one more time with more accurate coordinates
- After 2 retries: Mark element as failed, move to next element
- This prevents infinite loops while allowing correction of coordinate errors
"""

VALIDATION_GUARANTEE = """
VALIDATION GUARANTEE:
- The find_unique_locator action validates UNIQUENESS (count=1) using Playwright
- It does NOT validate CORRECTNESS (whether it's the right element)
- You MUST verify the locator points to the element matching the description
- If unique but wrong element: Your coordinates were off, retry with better coordinates
- If unique and correct element: Success, move to next element
- Maximum 2 retries per element to avoid infinite loops
"""

# ═══════════════════════════════════════════════════════════════════════════════
# EXTENDED WORKFLOW TEMPLATES (Extracted from workflow.py for maintainability)
# ═══════════════════════════════════════════════════════════════════════════════

# Extended Parameters - includes element_index, expected_text, is_collection
CUSTOM_ACTION_PARAMETERS_EXTENDED = """
PARAMETERS:
  • x (float, required): X coordinate of element center
  • y (float, required): Y coordinate of element center
  • element_id (str, required): Element identifier from the list above (e.g., "elem_1")
  • element_description (str, required): USE THE EXACT DESCRIPTION from ELEMENTS TO FIND above!
    ⚠️ This MUST match the description from the list (e.g., "all visible rows in the table body")
    ⚠️ DO NOT rewrite or simplify the description - it's used for collection detection!
    ⚠️ Preserve keywords like "all", "rows", "cells", "items", "each" - these are CRITICAL!
  • expected_text (str, optional): The ACTUAL visible text you see on the element.
    ⚠️ PROVIDE THIS whenever the element has visible text, a label, or a placeholder.
    Only omit for icon-only elements that have absolutely no visible text or label.
    Examples: "Submit", "Add to Cart", "Nike Air Max 270", "Search"
    ⚠️ For buttons/links: Use the exact button/link text you see
    ⚠️ For inputs: Use the placeholder or label text if visible
    ⚠️ For product names: Use the actual product name text you see
  • element_index (int, required): The element INDEX from the DOM state — the DOM
    element list is your PRIMARY grounding for every element.
    ⚠️ ALWAYS PROVIDE THIS for EVERY element - it is the MOST ACCURATE METHOD!
    ⚠️ When you see [49] <td>John</td>, set element_index=49
    ⚠️ When you see [23] <a>Services</a>, set element_index=23
    ⚠️ This works for ALL elements including table cells, buttons, inputs, links, etc.
    ⚠️ We extract all element attributes (id, class, xpath) from this index automatically
    ⚠️ Without element_index, we fall back to less accurate coordinate-based approaches
  • candidate_locator (str, optional): Your suggested locator if you can identify one
    Examples: "id=search-input", "data-testid=login-btn", "name=username"
  • is_collection (bool, ★ REQUIRED FOR COLLECTIONS ★): Set to true if element has (loop: FOR)
    ⚠️ When you see "loop: FOR" in ELEMENTS TO FIND, you MUST set is_collection=true
    ⚠️ This ensures we return a multi-element locator (e.g., .rt-tr-group) not single-element
    ⚠️ Examples: table rows, list items, all cells in a column
    ⚠️ A single control inside ONE row (an Edit/Delete/Download icon, a row checkbox) is
    NOT a collection even though it sits in a table - leave is_collection unset and pass
    row_anchor_text instead (see below)
  • row_anchor_text (str, ★ REQUIRED FOR ROW-SPECIFIC ACTIONS ★): The row-identifying data
    when the target is a PER-ROW control inside a table or repeated list.
    ⚠️ Tables repeat the same Edit/Delete/Download icons on EVERY row - only the row's
    DATA distinguishes them. Without this, the locator may act on the WRONG row later!
    ⚠️ Use the row-identifying datum FROM THE TASK (account number, customer name, invoice
    id) exactly as it appears in the row's text.
    ⚠️ Example: task says "Edit customer 64625", the Edit icon is in the row showing 64625
    → find_unique_locator(..., row_anchor_text="64625")
    ⚠️ Also for row checkboxes and row links: "Select the row for John Smith"
    → row_anchor_text="John Smith"
    ⚠️ Leave blank for elements that appear ONCE on the page (toolbar buttons, form fields,
    menu links) - this parameter is ONLY for controls repeated across rows/items.
"""

# Example workflow demonstrating the complete flow
EXAMPLE_WORKFLOW_TEMPLATE = """
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

Step 4: PERFORM ACTION for elem_1 (action=input, value="shoes")
  → Type "shoes" (the VALUE from elem_1's spec) into the search box
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
"""

# Critical instructions checklist
CRITICAL_INSTRUCTIONS_CHECKLIST = """
═══════════════════════════════════════════════════════════════════
CRITICAL INSTRUCTIONS
═══════════════════════════════════════════════════════════════════

✓ MUST call find_unique_locator for EVERY element in the list
✓ MUST provide accurate coordinates (x, y) from your vision
✓ MUST provide expected_text when the element has visible text, a label, or a placeholder
  (This is CRITICAL — it prevents finding the wrong element! Omit only for icon-only elements.)
✓ SHOULD provide candidate_locator if you can identify id, data-testid, or name
✓ MUST NOT validate locators yourself - the action does this
✓ MUST NOT execute JavaScript to check uniqueness - the action does this
✓ MUST NOT use querySelector, querySelectorAll, or execute_js for validation
✓ MUST NOT retry or check count - the action guarantees count=1
✓ ONLY call done() when ALL elements have validated results from the action
"""

# Strict scope rules to prevent over-completion
STRICT_SCOPE_RULES = """
⛔ STRICT SCOPE - DO NOT OVER-COMPLETE THE TASK:
  • ONLY process the elements listed in ELEMENTS TO FIND above
  • ONLY perform actions explicitly requested in the USER'S GOAL
  • DO NOT add extra elements not in the list (e.g., if asked to fill username, don't also fill password)
  • DO NOT infer or guess additional steps (e.g., don't click Submit unless asked)
  • DO NOT "complete" forms or workflows beyond what was explicitly requested
  • If the user asks to "type X into field Y", ONLY type X into field Y - nothing more
  • Treat the ELEMENTS TO FIND list as EXHAUSTIVE - there are no hidden elements to find
"""

# Sequential processing rules with action behaviors
SEQUENTIAL_PROCESSING_RULES = """
═══════════════════════════════════════════════════════════════════
SEQUENTIAL ELEMENT PROCESSING
═══════════════════════════════════════════════════════════════════

Process elements IN THE ORDER THEY ARE LISTED. For each element:

1. Find the element using your vision
2. Get element coordinates (x, y)
3. Call find_unique_locator to extract and validate the locator
4. find_unique_locator performs the interaction automatically — check the result:

   ACTION BEHAVIORS — interactions are performed automatically by find_unique_locator:
   • input/type: auto-filled using layered strategies (fill → key events → JS). "✅" = succeeded.
   • click/submit: auto-clicked using layered strategies (click → force → JS). Page stability
     is awaited automatically after click/submit/select — multi-page flows are handled. "✅" = succeeded.
   • check: checkbox/radio checked using state-aware Playwright loc.check() — only acts if not
     already checked. "✅" = succeeded.
   • uncheck: checkbox unchecked using state-aware Playwright loc.uncheck() — only acts if
     currently checked. "✅" = succeeded.
   • select: auto-selected using framework-aware layered strategies. Tom Select: JS API via
     select_id (getElementById) → JS sibling traversal → click-to-open. All others: native label
     → native value → click-to-open. Handles <select>, ARIA, Tom Select, Select2, Bootstrap.
     "✅" = succeeded.
   • get_text/get_attribute/other: locator stored only, no interaction performed.

   If "⚠️ Automated ... failed" with "FALLBACK REQUIRED: Call ...":
     An element index is available. Execute the specified browser-use native action BEFORE
     proceeding to the next element:
     • click(index=N)                                — for click/submit fallback
     • input(index=N, text="...")                    — for input/type fallback
     • select_dropdown(index=N, text="...")          — for select fallback
     After the fallback action completes (success or failure), proceed to the next element.
     Do NOT call find_unique_locator again for this element.

   If "⚠️ Automated ... failed" with "Validated locator: ... | Action required: ...":
     No element index is available (table row, dynamic DOM, non-interactive element).
     The validated locator is provided — use it to understand which element needs the action.
     Assess the current page state and use any available browser-use action to complete the
     interaction. Do NOT attempt to find or guess an index — it may not exist in the DOM.
     If no available action can complete it, proceed to the next element —
     Robot Framework will handle the interaction at execution time.

5. Move to the NEXT element in the list

⚠️ IMPORTANT:
  • Process elements sequentially in the order given
  • DO NOT re-perform actions that already show "✅". Call find_unique_locator ONE ELEMENT AT A TIME.
  • Always provide element_index — without it the browser-use event path is skipped
    (TypeTextEvent provides concatenation-detection for reactive inputs; ClickElementEvent
    uses CDP coordinates for reliable click targeting).
  • Subsequent elements will be found on whatever page is currently displayed
  • The Step Planner has already ordered elements correctly for the workflow
"""

# Edge case handling including checkboxes
EDGE_CASE_HANDLING = """
⚠️ EDGE CASE HANDLING:
  • NEVER record an element as found: false without having called find_unique_locator
    for it at least once. When find_unique_locator fails, a screenshot of the current
    page is attached to your NEXT message (when available) — re-examine the failed
    element in it, correct element_index/coordinates, and retry before giving up.
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
"""

# Extended completion criteria with element count verification
COMPLETION_CRITERIA_EXTENDED = """
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
