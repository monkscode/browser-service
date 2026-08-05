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
from typing import Any, Dict, Optional

from browser_service.locators.classifier import classify_element_type
from browser_service.locators.stability import classify_locator

# Get logger
logger = logging.getLogger(__name__)


# One evaluate() on the element the candidate ACTUALLY resolves to. Keys are
# named to match element_data so the same describe/classify helpers consume
# either shape unchanged.
#
# id/tagName/className are typeof-guarded, not read bare. HTMLFormElement has
# [OverrideBuiltins], so a named control shadows the same-named property:
# <form><input name="id"> makes el.id return the INPUT NODE, which serialises
# as the string 'ref: <Node>' and would read as a PROVABLE mismatch — a
# spurious reject on a shape (<input name="id">) that is ordinary in CRUD
# forms. Falling back to '' makes it UNKNOWN instead, which _identity_mismatch
# already treats as "no evidence". className carried this guard from the start
# (SVG className is an SVGAnimatedString); id and tagName need it too.
#
# tagName is lowercased to match every other element_info producer — the full
# path's extraction JS lowercases explicitly and browser-use's element_data is
# lowercase. nlrf falls back to element_info['tagName'] for element_type, so an
# UPPERCASE value here would change assembler prompt text on this path alone.
_RESOLVED_ELEMENT_JS = """el => ({
    id: typeof el.id === 'string' ? el.id : '',
    tagName: typeof el.tagName === 'string' ? el.tagName.toLowerCase() : '',
    textContent: (el.textContent || '').trim().slice(0, 500),
    className: typeof el.className === 'string' ? el.className : '',
    ariaInvalid: el.getAttribute('aria-invalid') || '',
    parentClassName: (el.parentElement && typeof el.parentElement.className === 'string')
        ? el.parentElement.className : '',
    name: el.getAttribute('name') || '',
    dataTestId: el.getAttribute('data-testid') || '',
    role: el.getAttribute('role') || '',
    type: el.getAttribute('type') || ''
})"""


# The read runs immediately after count()==1, so the element exists and the
# evaluate is ~2ms. This bound only engages when the node vanishes in between
# — a race in which Playwright's default would WAIT 30 seconds, six times the
# entire cascade budget (custom_action_timeout, 5s), on a code path that sits
# OUTSIDE the asyncio.wait_for guarding STEP 2 and is therefore unbounded
# otherwise. Timing out is safe by construction: it yields None, which is
# "unknown", which leaves the pre-existing accept untouched.
_RESOLVED_READ_TIMEOUT_MS = 2000


async def _read_resolved_element(
    search_root,
    locator: str,
    timeout_ms: int = _RESOLVED_READ_TIMEOUT_MS,
    first_only: bool = False,
) -> Optional[Dict[str, Any]]:
    """Read the element ``locator`` resolves to — what we are about to accept.

    Returns None when the element cannot be read (detached node, exotic
    selector engine, or the bounded wait expiring). Callers must treat None as
    "unknown", never as "mismatch": this read exists to make the accept honest,
    and must never turn a working accept into a found=false.

    ``first_only`` scopes the read to the first match. A many-match locator is
    a Playwright strict-mode violation for ``Locator.evaluate``, so the
    collection accept — which is asking about a locator that SHOULD match many
    — would otherwise get None back and read it as "unknown", i.e. accept
    unchecked. That would silently disarm the q08 identity guard.
    """
    try:
        target = search_root.locator(locator)
        if first_only:
            target = target.nth(0)
        resolved = await target.evaluate(_RESOLVED_ELEMENT_JS, timeout=timeout_ms)
    except Exception as e:
        logger.warning(f"   ⚠️ Could not read resolved element for '{locator}': {e}")
        return None
    # Anything but a dict is unusable evidence — treat it as unknown rather
    # than letting it reach the describe/classify helpers, where an unexpected
    # shape would raise inside the accept and drop a valid candidate.
    return resolved if isinstance(resolved, dict) else None


def _identity_mismatch(
    element_data: Optional[Dict[str, Any]],
    resolved: Optional[Dict[str, Any]],
    *,
    compare_id: bool = True,
) -> str:
    """Reason string when ``resolved`` is PROVABLY not the indexed element.

    q08: element_data was <select id="dropdown"> while the agent's candidate
    `select#dropdown option[selected='selected']` resolved to an <option>.
    The semantic check could not catch it — Chromium moves the `selected`
    content attribute onto the chosen option, so the resolved element really
    did read "Option 2". Only identity separates the two.

    Deliberately conservative — two provable signals, nothing inferred. An
    absent element_data, an unreadable element, or an empty id on either side
    yields "" (accept unchanged); an empty id is not evidence of a different
    element, only of a missing attribute.

    ``compare_id=False`` for the COLLECTION accept, where the id signal is not
    merely weaker but unsound. That caller reads ``.nth(0)`` while the agent
    indexes whichever member it clicked, and the members of a keyed list carry
    distinct ids by construction — so a server-rendered table keyed by database
    id mismatches on every row but the first. The reject then falls through to
    the cascade, which returns the ANCESTOR and reinstates the unvalidated
    child selector that accept exists to eliminate. Tag stays compared on both
    paths: that is the half with evidence behind it (the bench rejected
    `article.product_pod` as "indexed <a> but resolves to <article>").
    """
    if not element_data or not resolved:
        return ""

    indexed_tag = (element_data.get("tagName") or "").lower()
    resolved_tag = (resolved.get("tagName") or "").lower()
    if indexed_tag and resolved_tag and indexed_tag != resolved_tag:
        return f"indexed <{indexed_tag}> but candidate resolves to <{resolved_tag}>"

    if compare_id:
        indexed_id = (element_data.get("id") or "").strip()
        resolved_id = (resolved.get("id") or "").strip()
        if indexed_id and resolved_id and indexed_id != resolved_id:
            return f"indexed id '{indexed_id}' but candidate resolves to id '{resolved_id}'"

    return ""


def _locator_has_id(locator: str) -> bool:
    """The ``approach_metrics.has_id`` signal for a candidate-path accept.

    Both accept branches report the same locator shapes, so both must answer
    this the same way — nlrf's tools/analyze_locator_patterns.py counts the
    field across approaches, and two definitions make that count meaningless.

    Reads the LOCATOR, not element_data: on this path the question is whether
    the address we are about to ship is id-anchored. The data-* exclusions stop
    `[data-testid="x"]` scoring as an id purely because it contains "id=".
    """
    low = (locator or "").lower().lstrip()
    return (
        low.startswith("#")
        or "[id=" in low
        or (
            "id=" in low
            and not any(k in low for k in ("data-testid=", "data-test=", "data-qa="))
        )
    )


def _candidate_element_info(
    element_data: Optional[Dict[str, Any]], source: str = "candidate_element_data"
) -> Dict[str, Any]:
    """element_info for a candidate-path accept (E1).

    The fast path used to return element_info={} — element_classes /
    aria_invalid / parent_classes and the tagName routing fallback all
    vanished downstream whenever the agent's proposed locator validated.
    Tolerant .get: element_data producers vary in which keys they carry.

    Fed the RESOLVED element where one could be read, so the payload
    describes the element actually being returned rather than the element
    the agent indexed (q08: those were a <select> and an <option>).
    """
    if not element_data:
        return {}
    return {
        "id": element_data.get("id", ""),
        "tagName": element_data.get("tagName", ""),
        "text": element_data.get("textContent", "") or element_data.get("text", ""),
        "className": element_data.get("className", ""),
        "ariaInvalid": element_data.get("ariaInvalid", ""),
        "parentClassName": element_data.get("parentClassName", ""),
        "name": element_data.get("name", ""),
        "testId": element_data.get("dataTestId", ""),
        "source": source,
    }


def _candidate_tier0_stamp(
    element_data: Optional[Dict[str, Any]], element_description: str
) -> Dict[str, Any]:
    """Classifier stamp for a candidate-path accept — Tier-0 DOM evidence only.

    The full path corroborates classifier verdicts with a DOM probe before
    any handler commits; the candidate path has no such probe (it reads the
    resolved element once, but never probes), so the
    stamp is gated to Tier-0 verdicts carrying attribute evidence beyond the
    bare tag name (type= / className: / role= signals). Bare-tagName verdicts
    (select/tr/li) stay unstamped — nlrf's tagName fallback already routes
    those, and stamping tr/li would put 'table-row'/'list-item' into
    dropdown_framework and fire the DROPDOWN block on collection steps.
    Vision hints never feed this stamp (they are only corroborated on the
    probe path).

    select_id: only when the element is itself a <select> with an id — no
    derivation from input ids, no probe.
    """
    if not element_data:
        return {}

    stamp: Dict[str, Any] = {}
    if (element_data.get("tagName") or "").lower() == "select" and element_data.get("id"):
        stamp["select_id"] = element_data["id"]

    type_info = classify_element_type(element_data, element_description)
    if "tier:0" not in type_info.signals:
        return stamp
    if not any(s.startswith(("type=", "className:", "role=")) for s in type_info.signals):
        return stamp
    # role=combobox/listbox is on the approved skip list: a
    # dropdown/combobox-input stamp matches no composer TYPE rule, while
    # the tagName fallback (div/input/span/button) routes TYPE 2/3.
    if type_info.primary_type == "dropdown" and type_info.framework == "combobox-input":
        return stamp

    stamp["element_type"] = type_info.primary_type
    stamp["classifier_confidence"] = type_info.confidence
    stamp["classifier_signals"] = list(type_info.signals)
    if type_info.primary_type == "dropdown":
        stamp["dropdown_framework"] = type_info.framework or ""
    elif type_info.primary_type == "date-picker":
        stamp["datepicker_framework"] = type_info.framework or ""
        # Explicit empty: 'flatpickr' in dropdown_framework would misroute
        # the Assembler's dropdown table (Task D guard — mirrors the
        # date_picker handler).
        stamp["dropdown_framework"] = ""
    return stamp


def _log_success_result(element_id: str, result: Dict[str, Any]) -> None:
    """Log successful locator finding result with detailed information."""
    best_locator = result.get("best_locator")
    validation_summary = result.get("validation_summary", {})

    logger.info("")
    logger.info(f"{'=' * 80}")
    logger.info(f"✅ CUSTOM ACTION SUCCEEDED for {element_id}")
    logger.info(f"{'=' * 80}")
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
    logger.info(f"{'=' * 80}")
    logger.info("")


def _log_failure_result(
    element_id: str, element_description: str, x: float, y: float, result: Dict[str, Any]
) -> None:
    """Log failed locator finding result with detailed error information."""
    error = result.get("error", "Unknown error")
    validation_summary = result.get("validation_summary", {})

    logger.error("")
    logger.error(f"{'=' * 80}")
    logger.error(f"❌ CUSTOM ACTION FAILED for {element_id}")
    logger.error(f"{'=' * 80}")
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
    logger.error(f"{'=' * 80}")
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
    vision_type_hint: Optional[
        str
    ] = None,  # LLM's visual type classification (any specialized type)
    vision_framework_hint: Optional[str] = None,  # LLM's framework guess (any specialized type)
    row_anchor_text: Optional[str] = None,  # Row-identifying datum from the QA step (G1/Task B)
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
        logger.info(f'   Expected text: "{expected_text}"')
    if element_data:
        logger.info(
            f'   Element data from index: tag=<{element_data.get("tagName", "?")}>, text="{(element_data.get("textContent") or "")[:30]}..."'
        )
    if candidate_locator:
        logger.info(f"   Candidate locator: {candidate_locator}")
    if iframe_context:
        logger.info(f"   🖼️ Iframe context: {iframe_context}")
    if is_collection:
        logger.info(f"   📋 Collection mode: {is_collection}")

    # Helper function to create structured error result
    def create_error_result(
        error_type: str, error_message: str, additional_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a structured error result with complete validation data."""
        result = {
            "element_id": element_id,
            "description": element_description,
            "found": False,
            "error": error_message,
            "error_type": error_type,
            "coordinates": {"x": x, "y": y},
            "validated": False,
            "count": 0,
            "unique": False,
            "valid": False,
            "validation_method": "playwright",
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
            return create_error_result("PageObjectError", error_msg)

        # Validate coordinates
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            error_msg = f"Invalid coordinates: x={x} (type={type(x).__name__}), y={y} (type={type(y).__name__})"
            logger.error(f"❌ {error_msg}")
            logger.error(f"   Element ID: {element_id}")
            return create_error_result("InvalidCoordinatesError", error_msg)

        if x < 0 or y < 0:
            error_msg = f"Negative coordinates not allowed: x={x}, y={y}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"   Element ID: {element_id}")
            return create_error_result("InvalidCoordinatesError", error_msg)

        # Validate element_id
        if not element_id or not isinstance(element_id, str):
            error_msg = f"Invalid element_id: {element_id} (type={type(element_id).__name__})"
            logger.error(f"❌ {error_msg}")
            return create_error_result("InvalidElementIdError", error_msg)

        # ========================================
        # ROW-ANCHOR EXPECTED-TEXT CORRECTION (ASTPP gate q02)
        # ========================================
        # The agent sometimes passes the row anchor datum as expected_text
        # for a per-row control whose own DOM text disagrees — the candidate
        # semantic check and the cascade's text-first step would then both
        # target the anchor CELL. Correct before either runs.
        from browser_service.locators import correct_expected_text_for_row_anchor

        expected_text, _anchor_corrected = correct_expected_text_for_row_anchor(
            expected_text, row_anchor_text, element_data
        )
        if _anchor_corrected:
            logger.info(
                f"   📌 ROW-ANCHOR CORRECTION: expected_text was the anchor datum "
                f"'{row_anchor_text}'; using element's own text '{expected_text}' "
                f"(signal: row-anchor-corrects-expected-text)"
            )

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

                    playwright_locator, was_converted = convert_to_playwright_locator(
                        candidate_locator
                    )

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
                    logger.info(
                        f"   DEBUG: page.locator('{playwright_locator}').count() returned: {count}"
                    )

                    # Log detailed validation results
                    is_unique = count == 1
                    is_valid = count == 1

                    logger.info("   Validation Results:")
                    logger.info(f"      - count: {count}")
                    logger.info(f"      - unique: {is_unique}")
                    logger.info(f"      - valid: {is_valid}")
                    logger.info("      - validated: True")
                    logger.info("      - validation_method: playwright")

                    if count == 1:
                        # q02 guard: a unique candidate that targets the row
                        # anchor datum itself while the indexed element's own
                        # text disagrees is the anchor CELL, not the row's
                        # control. Reject into the cascade (which row-scopes
                        # correctly) — the semantic check can't stand in when
                        # expected_text is absent.
                        from browser_service.locators import candidate_targets_row_anchor

                        _anchor_norm = " ".join(str(row_anchor_text or "").split())
                        _elem_text_norm = " ".join(
                            str(
                                (element_data or {}).get("textContent")
                                or (element_data or {}).get("text")
                                or ""
                            ).split()
                        )
                        _anchor_reject = bool(
                            _anchor_norm
                            and _elem_text_norm
                            and _elem_text_norm != _anchor_norm
                            and candidate_targets_row_anchor(playwright_locator, row_anchor_text)
                        )
                        if _anchor_reject:
                            logger.info(
                                f"   ⛔ ROW-ANCHOR CANDIDATE REJECT: '{playwright_locator}' "
                                f"targets the anchor datum '{row_anchor_text}' but the "
                                f"element's own text is '{_elem_text_norm}' — continuing "
                                f"with smart locator finder "
                                f"(signal: row-anchor-rejects-candidate)"
                            )

                        # q08 identity guard: read the element the candidate
                        # actually resolves to. Two jobs — reject it when it is
                        # provably not the element the agent indexed, and
                        # describe what we accept instead of what was indexed.
                        # The semantic check cannot stand in here: an <option>
                        # under a <select> can carry exactly the expected text.
                        _resolved_element = None
                        _identity_reason = ""
                        if not _anchor_reject:
                            _resolved_element = await _read_resolved_element(
                                search_root, playwright_locator
                            )
                            _identity_reason = _identity_mismatch(element_data, _resolved_element)
                            if _identity_reason:
                                logger.info(
                                    f"   ⛔ CANDIDATE IDENTITY REJECT: '{playwright_locator}' "
                                    f"— {_identity_reason}; continuing with smart locator "
                                    f"finder (signal: candidate-resolves-to-different-element)"
                                )

                        # Close the "unique but semantically wrong" hole (probe 06).
                        # Validate the RESOLVED element — what playwright_locator actually
                        # resolves to on the page — not the intended element from element_data
                        # (audit Issue 2b). Guard with expected_text so we accept on uniqueness
                        # alone when no semantic hint is available (audit Issue 5).
                        _semantic_ok = not _anchor_reject and not _identity_reason
                        if _semantic_ok and expected_text:
                            from browser_service.locators import validate_semantic_match

                            # accept_empty_interactive (q05 guard i): a unique
                            # candidate with an EMPTY semantic surface cannot
                            # contradict expected_text — the agent sometimes
                            # attaches the label's text to a surface-less input
                            # (broken <label for=>), and the veto killed the
                            # correct locator. Opt-in here only.
                            _semantic_ok, _observed = await validate_semantic_match(
                                None,
                                expected_text,
                                page=search_root,
                                locator=playwright_locator,
                                accept_empty_interactive=True,
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
                            logger.info(f"{'=' * 80}")
                            logger.info(
                                "✅ CANDIDATE LOCATOR IS UNIQUE AND SEMANTIC MATCH - Using directly!"
                            )
                            logger.info(f"{'=' * 80}")
                            logger.info("   Skipping 21 strategies (not needed)")
                            logger.info(f"   Original: {candidate_locator}")
                            if was_converted:
                                logger.info(
                                    f"   Converted: {final_locator} (Browser Library compatible)"
                                )
                            logger.info("   Type: candidate")
                            logger.info("   Priority: 0 (agent-provided)")
                            logger.info(f"{'=' * 80}")
                            logger.info("")

                            # Describe the element being RETURNED. The identity
                            # gate above has already rejected anything provably
                            # different, so the resolved read is both safe and
                            # fresher than element_data (q08 / E1).
                            _describes = _resolved_element or element_data
                            _info_source = (
                                "candidate_resolved_element"
                                if _resolved_element
                                else "candidate_element_data"
                            )

                            # Extract element metadata from DOM data
                            elem_tag = ""
                            elem_has_text = False
                            elem_data_available = bool(element_data)
                            if _describes:
                                elem_tag = (_describes.get("tagName") or "").lower()
                                text_content = _describes.get("textContent", "") or _describes.get(
                                    "text", ""
                                )
                                elem_has_text = bool(text_content and text_content.strip())

                            # Acceptance unchanged (a volatile candidate that
                            # falls through could end in found=false, which
                            # must not rise) — but the payload reports the
                            # tier honestly so nlrf can warn or heal (E1).
                            candidate_stability = classify_locator(final_locator)
                            if candidate_stability != "stable":
                                logger.warning(
                                    f"   ⚠️ Agent-provided candidate is {candidate_stability}: "
                                    f"{final_locator} — accepted, reported for healing"
                                )

                            # E1 (Option B): the accept must not starve the
                            # composer/idiom routing downstream — copy the DOM
                            # evidence and stamp Tier-0 verdicts.
                            candidate_stamp = _candidate_tier0_stamp(
                                _describes, element_description
                            )
                            if candidate_stamp:
                                logger.info(f"   🏷️ Candidate Tier-0 stamp: {candidate_stamp}")

                            return {
                                **candidate_stamp,
                                "element_id": element_id,
                                "description": element_description,
                                "found": True,
                                "best_locator": final_locator,  # Use converted locator
                                "stability": candidate_stability,
                                "all_locators": [
                                    {
                                        "type": "candidate",
                                        "locator": final_locator,  # Use converted locator
                                        "priority": 0,
                                        "strategy": "Agent-provided candidate (converted for Browser Library)"
                                        if was_converted
                                        else "Agent-provided candidate",
                                        "count": count,
                                        "unique": True,
                                        "valid": True,
                                        "validated": True,
                                        "validation_method": "playwright",
                                        "stability": candidate_stability,
                                    }
                                ],
                                "element_info": _candidate_element_info(_describes, _info_source),
                                "coordinates": {"x": x, "y": y},
                                "validation_summary": {
                                    "total_generated": 1,
                                    "valid": 1,
                                    "unique": 1,
                                    "validated": 1,
                                    "not_found": 0,
                                    "not_unique": 0,
                                    "errors": 0,
                                    "best_type": "candidate",
                                    "best_strategy": "Agent-provided candidate",
                                    "validation_method": "playwright",
                                },
                                # Add validation data at result level
                                "validated": True,
                                "count": count,
                                "unique": True,
                                "valid": True,
                                "validation_method": "playwright",
                                # Per-element approach metrics for pattern analysis
                                "approach_metrics": {
                                    "locator_approach": "actions_candidate",
                                    "fallback_depth": 0,  # Best case - candidate worked
                                    "success": True,
                                    "element_tag": elem_tag,
                                    "has_id": _locator_has_id(final_locator),
                                    "has_text_content": elem_has_text,
                                    "element_data_available": elem_data_available,
                                    "is_collection": is_collection is True,
                                    "is_in_iframe": bool(iframe_context),
                                },
                            }
                    elif count > 1 and is_collection:
                        # A COLLECTION request asked for many, so many matches
                        # is the answer — not a failure. Rejecting the agent's
                        # own address here is what forced the traversal to
                        # return an ANCESTOR of the indexed element (`ol > li`
                        # for an indexed <a>), leaving the assembler to invent
                        # an unvalidated child selector to get back down. Both
                        # captured collection failures start there:
                        #   31a66a98  Get Attribute <li> title -> AttributeError
                        #   q06 rep2  tbody tr -> div[role=gridcell] -> timeout
                        #
                        # The unique path's identity guard cannot stand in: its
                        # read raises strict mode on a many-match locator and
                        # returns None, which means "unknown -> accept". Read
                        # the FIRST match instead and apply the same provable
                        # tag/id comparison. First match only, matching that
                        # guard's strength — a collection may legitimately mix
                        # element kinds, and over-rejecting costs a good
                        # locator.
                        #
                        # TAG ONLY. The id half of the comparison is unsound
                        # here: the read is pinned to .nth(0) while the agent
                        # indexes whichever member it clicked, and a keyed
                        # list's members carry distinct ids by construction.
                        _first = await _read_resolved_element(
                            search_root, playwright_locator, first_only=True
                        )
                        _collection_reason = _identity_mismatch(
                            element_data, _first, compare_id=False
                        )
                        if _collection_reason:
                            logger.info(
                                f"   ⛔ COLLECTION CANDIDATE IDENTITY REJECT: "
                                f"'{playwright_locator}' — {_collection_reason}; continuing "
                                f"with smart locator finder (signal: "
                                f"collection-candidate-resolves-to-different-element)"
                            )
                        else:
                            final_locator = playwright_locator
                            logger.info(
                                f"   ✅ COLLECTION CANDIDATE ACCEPTED: '{final_locator}' "
                                f"matches {count} elements and resolves to the indexed "
                                f"element (signal: collection-candidate-accepted)"
                            )
                            _describes = _first or element_data
                            _info_source = (
                                "candidate_resolved_element" if _first else "candidate_element_data"
                            )
                            elem_tag = ""
                            elem_has_text = False
                            if _describes:
                                elem_tag = (_describes.get("tagName") or "").lower()
                                _text = _describes.get("textContent", "") or _describes.get(
                                    "text", ""
                                )
                                elem_has_text = bool(_text and _text.strip())
                            collection_stability = classify_locator(final_locator)
                            return {
                                # element_type LAST-writes 'collection': nlrf
                                # routes the assembler's FOR-loop block on it
                                # (tasks._needs_loop). Without it a
                                # single-element keyword gets pointed at a
                                # many-match locator and Browser raises strict
                                # mode at run time, which --dryrun cannot see.
                                **_candidate_tier0_stamp(_describes, element_description),
                                "element_type": "collection",
                                "element_id": element_id,
                                "description": element_description,
                                "found": True,
                                "best_locator": final_locator,
                                "stability": collection_stability,
                                "all_locators": [
                                    {
                                        "type": "collection",
                                        "locator": final_locator,
                                        "priority": 0,
                                        "strategy": "Agent-provided collection candidate",
                                        "count": count,
                                        "unique": False,
                                        "valid": True,
                                        "validated": True,
                                        "validation_method": "playwright",
                                        "stability": collection_stability,
                                    }
                                ],
                                "element_info": _candidate_element_info(_describes, _info_source),
                                "coordinates": {"x": x, "y": y},
                                "validation_summary": {
                                    "total_generated": 1,
                                    "valid": 1,
                                    "unique": 0,
                                    "validated": 1,
                                    "not_found": 0,
                                    "not_unique": 1,
                                    "errors": 0,
                                    "best_type": "collection",
                                    "best_strategy": "Agent-provided collection candidate",
                                    "validation_method": "playwright",
                                },
                                "validated": True,
                                "count": count,
                                "unique": False,
                                "valid": True,
                                "validation_method": "playwright",
                                "approach_metrics": {
                                    "locator_approach": "actions_candidate_collection",
                                    "fallback_depth": 0,
                                    "success": True,
                                    "element_tag": elem_tag,
                                    "has_id": _locator_has_id(final_locator),
                                    "has_text_content": elem_has_text,
                                    "element_data_available": bool(element_data),
                                    "is_collection": True,
                                    "is_in_iframe": bool(iframe_context),
                                },
                            }
                    elif count > 1:
                        logger.info(f"   ⚠️ Candidate locator NOT UNIQUE (matches {count} elements)")
                        logger.info(
                            "   🔄 Continuing with smart locator finder to find unique locator..."
                        )
                    else:  # count == 0
                        logger.info("   ⚠️ Candidate locator NOT FOUND (matches 0 elements)")
                        logger.info(
                            "   🔄 Continuing with smart locator finder to find valid locator..."
                        )

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
                if "not a valid selector" in error_str or "invalid selector" in error_str:
                    logger.warning(f"⚠️ Candidate locator has invalid CSS syntax: {e}")
                    logger.warning(f"   Locator: {candidate_locator}")
                    logger.warning("   Note: This often happens with numeric IDs (e.g., #123)")
                    logger.info(
                        "🔄 Continuing with smart locator finder (will use [id='...'] syntax)..."
                    )
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
            return create_error_result("ImportError", error_msg)

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
                    element_data=element_data,  # Pass element attributes from browser-use DOM
                    is_collection=is_collection,  # Pass collection flag for multi-element detection
                    browser_session=browser_session,  # For resolved_node lookup (DELTA 1)
                    vision_type_hint=vision_type_hint,  # LLM's visual classification (1 of 2 sources)
                    vision_framework_hint=vision_framework_hint,  # LLM's framework guess
                    row_anchor_text=row_anchor_text,  # Row-scoped rescue for per-row actions (G1)
                ),
                timeout=custom_action_timeout,
            )

            duration_ms = (time.monotonic() - _locator_timer_start) * 1000.0
            logger.info(
                f"LOCATOR_TIMER element_id={element_id} "
                f"duration_ms={duration_ms:.1f} found={bool(result.get('found'))}"
            )
            # Only enrich an EXISTING approach_metrics dict — failure results may
            # not carry one, and inventing it here would corrupt pattern analysis.
            if isinstance(result.get("approach_metrics"), dict):
                result["approach_metrics"]["duration_ms"] = round(duration_ms, 1)

            # Log the result with detailed information
            if result.get("found"):
                _log_success_result(element_id, result)
            else:
                _log_failure_result(element_id, element_description, x, y, result)

            return result

        except asyncio.TimeoutError:
            duration_ms = (time.monotonic() - _locator_timer_start) * 1000.0
            logger.info(
                f"LOCATOR_TIMER element_id={element_id} duration_ms={duration_ms:.1f} found=False"
            )
            # Handle timeout gracefully
            timeout_msg = f"Smart locator finder timed out after {custom_action_timeout} seconds"
            logger.error(f"⏱️ {timeout_msg}")
            logger.error(f"   Element ID: {element_id}")
            logger.error(f"   Description: {element_description}")
            logger.error(f"   Coordinates: ({x}, {y})")
            logger.error("   This may indicate a complex page or slow network")

            return create_error_result(
                "TimeoutError", timeout_msg, {"timeout_seconds": custom_action_timeout}
            )

        except asyncio.CancelledError:
            # Task was cancelled (e.g., browser closed)
            cancel_msg = "Smart locator finder was cancelled (browser may have closed)"
            logger.error(f"🚫 {cancel_msg}")
            logger.error(f"   Element ID: {element_id}")
            logger.error(f"   Coordinates: ({x}, {y})")

            return create_error_result("CancelledError", cancel_msg)

        except RuntimeError as e:
            # Runtime errors (e.g., event loop issues, browser closed)
            runtime_msg = f"Runtime error in smart locator finder: {str(e)}"
            logger.error(f"❌ {runtime_msg}")
            logger.error(f"   Element ID: {element_id}")
            logger.error(f"   Coordinates: ({x}, {y})")
            logger.error("   This may indicate the browser was closed or the page navigated away")
            logger.error("   Stack trace:", exc_info=True)

            return create_error_result("RuntimeError", runtime_msg)

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

        return create_error_result("TimeoutError", timeout_msg)

    except asyncio.CancelledError:
        # Top-level cancellation
        cancel_msg = "Custom action was cancelled"
        logger.error(f"🚫 {cancel_msg}")
        logger.error(f"   Element ID: {element_id}")
        logger.error(f"   Coordinates: ({x}, {y})")

        return create_error_result("CancelledError", cancel_msg)

    except KeyboardInterrupt:
        # User interrupted execution
        interrupt_msg = "Custom action interrupted by user"
        logger.error(f"⚠️ {interrupt_msg}")
        logger.error(f"   Element ID: {element_id}")

        return create_error_result("KeyboardInterrupt", interrupt_msg)

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
