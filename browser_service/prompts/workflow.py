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

from typing import Any, Dict, List, Optional

# Import reusable prompt templates
from browser_service.prompts.templates import (
    COMPLETION_CRITERIA_EXTENDED,
    COMPLETION_CRITERIA_LEGACY,
    CRITICAL_INSTRUCTIONS_CHECKLIST,
    CUSTOM_ACTION_HEADER,
    CUSTOM_ACTION_HOW_IT_WORKS,
    CUSTOM_ACTION_NO_VALIDATION_NEEDED,
    CUSTOM_ACTION_PARAMETERS_EXTENDED,
    CUSTOM_ACTION_RETURN_VALUE,
    EDGE_CASE_HANDLING,
    EXAMPLE_WORKFLOW_TEMPLATE,
    FORBIDDEN_ACTIONS,
    NUMERIC_IDS_WARNING,
    SEQUENTIAL_PROCESSING_RULES,
    STRICT_SCOPE_RULES,
    UNIQUENESS_REQUIREMENT,
    WORKFLOW_STEPS_LEGACY,
)


def _sanitize_prompt_field(value: Any) -> str:
    """Flatten an interpolated value to one line.

    Every field below is authored by an LLM (the plan) or echoed from user
    input, so a newline in one would otherwise inject prompt structure.
    """
    return str(value or "").replace("\n", " ").replace("\r", " ").strip()


def _render_element_lines(idx: int, elem: Dict[str, Any]) -> List[str]:
    """The prompt lines for ONE element, in order.

    Extracted from build_workflow_prompt's loop when the orphan-navigation
    line was added: that function was already at the configured cognitive
    complexity ceiling, so the branch had nowhere to go.

    Usually one line. Two when the element carries ``navigate_before`` — an
    ORPHAN navigation, one the plan states explicitly and no element's action
    triggers. Elements are processed in order and the agent normally changes
    pages as a side effect of the previous element's action (see
    EXAMPLE_WORKFLOW_TEMPLATE: "elem_1's action caused a page change, so
    elem_2 is naturally found on the new page"), a model with no way to
    express "now go here". Without this line the destination reaches the agent
    only through the goal prose, which it follows most of the time but not all
    of it.

    MEASURED 2026-08-05 over the 2026-07-16..2026-08-05 logs, on the
    population this line actually targets — the elements the caller stamps
    with ``navigate_before``, each matched against the page its validation ran
    on. Every workflow self-reports its side, because a rendered line is in
    the captured prompt:

        browser-use 0.12.6, no line       6 / 82 wrong page   7.3%
        browser-use 0.13.7, no line       2 / 55 wrong page   3.6%
        browser-use 0.13.7, line rendered 0 / 18 wrong page   0.0%

    Three things follow, and none of them is "fixed":

    - The failure is NOT extinct. It reproduces on 0.13.7, most recently
      2026-08-04, and both 0.13.7 occurrences ended in a failed run.
      Wrong-page runs pass 4/8; right-page runs 122/163. An earlier note here
      claimed 0 across 07-31, 08-04 and 08-05 — that was a wider population,
      and it is wrong for this one.
    - 0.13.7 did not end it. 7.3% -> 3.6% is Fisher two-sided p = 0.475, i.e.
      noise. Do not repeat the "coincided with the upgrade" causation claim.
    - This line has NOT been shown to change any outcome. 0/18 against a 3.6%
      baseline expects 0.65 failures; P(zero | no effect) = 0.51, Fisher
      p = 1.000. The agent reached the right page unaided in 53 of 55 runs, so
      the line is redundant roughly 96% of the time.

    REMOVAL CONDITION, so the deferral is explicit rather than permanent:
    re-measure once >= 80 validations have run WITH the line rendered (81
    gives 95% confidence of catching a 3.6% rate; about 18 render per pass of
    the private corpus, so four or five of them). Then decide either way:

    - >= 1 wrong-page validation → the line does not prevent the failure.
      Delete it, and ``navigate_before`` with it.
    - 0 wrong-page validations → it has cleared the bar it shipped on. Record
      the count here and drop this condition.

    Until that count is reached this is a corner-case guard on an unproven
    effect, kept because the failure still occurs and still costs a run. The
    entire population is one query shape from one corpus — no query in the
    30-query suite produces a mid-sequence navigation, so only private-corpus
    runs advance the count.

    ``navigate_before`` is optional in both directions — an element without it
    renders byte-identically to before the key existed, and a service that
    predates the key simply ignores it.
    """
    elem_id = elem.get("id", f"elem_unknown_{idx}")  # Default with index if missing
    elem_action = elem.get("action", "get_text")  # Default to get_text if missing
    elem_value = elem.get("value", "")  # Get value for input actions
    elem_loop_type = elem.get("loop_type", None)  # Get loop type for collection detection

    # .get with a default, NOT `or` — an element carrying description="" kept
    # the empty string before this was extracted, and must keep it.
    elem_desc = _sanitize_prompt_field(elem.get("description", "No description provided"))
    if len(elem_desc) > 200:
        elem_desc = elem_desc[:200] + "..."

    lines: List[str] = []

    elem_navigate = _sanitize_prompt_field(elem.get("navigate_before"))
    if elem_navigate:
        lines.append(f"   → Then navigate to {elem_navigate}")

    # Format element based on action type and loop_type
    if elem_value and elem_action in ["input", "type"]:
        # Input actions with value (CRITICAL for credentials/search terms)
        lines.append(f'   - {elem_id}: {elem_desc} (action: {elem_action}, value: "{elem_value}")')
    elif elem_loop_type:
        # Collection elements (loop: FOR indicates multi-element)
        lines.append(f"   - {elem_id}: {elem_desc} (action: {elem_action}, loop: {elem_loop_type})")
    else:
        lines.append(f"   - {elem_id}: {elem_desc} (action: {elem_action})")

    return lines


def build_workflow_prompt(
    user_query: str,
    url: str,
    elements: List[Dict[str, Any]],
    include_custom_action: bool = True,
    client_hints: Optional[List[str]] = None,
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
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    # Sanitize user_query to prevent prompt injection
    # Replace newlines with spaces, limit length
    user_query = user_query.replace("\n", " ").replace("\r", " ").strip()
    if len(user_query) > 500:
        user_query = user_query[:500] + "..."

    # Build element list with validation
    element_list = []
    for idx, elem in enumerate(elements):
        element_list.extend(_render_element_lines(idx, elem))

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
