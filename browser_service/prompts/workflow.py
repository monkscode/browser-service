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

from typing import List, Dict, Any, Optional

# Import reusable prompt templates
from browser_service.prompts.templates import (
    CUSTOM_ACTION_HEADER,
    CUSTOM_ACTION_PARAMETERS_EXTENDED,
    CUSTOM_ACTION_HOW_IT_WORKS,
    CUSTOM_ACTION_RETURN_VALUE,
    CUSTOM_ACTION_NO_VALIDATION_NEEDED,
    EXAMPLE_WORKFLOW_TEMPLATE,
    CRITICAL_INSTRUCTIONS_CHECKLIST,
    FORBIDDEN_ACTIONS,
    STRICT_SCOPE_RULES,
    SEQUENTIAL_PROCESSING_RULES,
    NUMERIC_IDS_WARNING,
    EDGE_CASE_HANDLING,
    COMPLETION_CRITERIA_EXTENDED,
)


def build_workflow_prompt(
    user_query: str,
    url: str,
    elements: List[Dict[str, Any]],
    library_type: str = "browser",
    include_custom_action: bool = True,
    client_hints: Optional[List[str]] = None
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
        client_hints: Optional list of application-specific hints/context to include
                     in the prompt. These help the LLM understand application-specific
                     behavior (e.g., slow loading times, sidebar behavior).

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
        elem_value = elem.get('value', '')  # Get value for input actions
        elem_loop_type = elem.get('loop_type', None)  # Get loop type for collection detection
        
        # Sanitize description to prevent prompt issues
        elem_desc = elem_desc.replace('\n', ' ').replace('\r', ' ').strip()
        if len(elem_desc) > 200:
            elem_desc = elem_desc[:200] + '...'
        
        # Format element based on action type and loop_type
        if elem_value and elem_action in ['input', 'type']:
            # Input actions with value (CRITICAL for credentials/search terms)
            element_list.append(f"   - {elem_id}: {elem_desc} (action: {elem_action}, value: \"{elem_value}\")")
        elif elem_loop_type:
            # Collection elements (loop: FOR indicates multi-element)
            element_list.append(f"   - {elem_id}: {elem_desc} (action: {elem_action}, loop: {elem_loop_type})")
        else:
            element_list.append(f"   - {elem_id}: {elem_desc} (action: {elem_action})")

    elements_str = "\n".join(element_list)

    # Build client hints section if provided
    client_hints_section = ""
    if client_hints:
        hints_text = "\n".join(f"• {hint}" for hint in client_hints)
        client_hints_section = f"""
═══════════════════════════════════════════════════════════════════
APPLICATION-SPECIFIC HINTS:
═══════════════════════════════════════════════════════════════════
{hints_text}
"""

    if include_custom_action:
        # NEW WORKFLOW: Use custom action for locator finding
        # Build prompt using templates for maintainability
        prompt = f"""You are completing a web automation workflow.
{client_hints_section}
USER'S GOAL: {user_query}

WORKFLOW STEPS:
1. Navigate to {url}
2. Find each element listed below using your vision
3. For EACH element, call the find_unique_locator action to get a validated unique locator

ELEMENTS TO FIND:
{elements_str}
{CUSTOM_ACTION_HEADER}
{CUSTOM_ACTION_PARAMETERS_EXTENDED}
{CUSTOM_ACTION_HOW_IT_WORKS}
{CUSTOM_ACTION_RETURN_VALUE}
{CUSTOM_ACTION_NO_VALIDATION_NEEDED}
{EXAMPLE_WORKFLOW_TEMPLATE.format(url=url)}
{CRITICAL_INSTRUCTIONS_CHECKLIST}
{FORBIDDEN_ACTIONS}
{STRICT_SCOPE_RULES}
{SEQUENTIAL_PROCESSING_RULES}
{NUMERIC_IDS_WARNING}
{EDGE_CASE_HANDLING}
{COMPLETION_CRITERIA_EXTENDED}
"""
    else:
        # LEGACY WORKFLOW: Use JavaScript validation (backward compatibility)
        prompt = f"""
You are completing a web automation workflow.
{client_hints_section}
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
