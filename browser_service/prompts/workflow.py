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
    WORKFLOW_STEPS_LEGACY,
    COMPLETION_CRITERIA_LEGACY,
    UNIQUENESS_REQUIREMENT,
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
        prompt = f"""You are completing a web automation workflow.
{client_hints_section}
USER'S GOAL: {user_query}
{WORKFLOW_STEPS_LEGACY.format(url=url)}
ELEMENTS TO FIND:
{elements_str}
{UNIQUENESS_REQUIREMENT}
{COMPLETION_CRITERIA_LEGACY}
"""
    return prompt.strip()
