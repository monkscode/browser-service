"""
Action fitness for an accepted locator (E2a, 2026-09-15 locator evaluation).

The cascade validated uniqueness and text only. A unique, text-matching
element can still be unable to perform the step: Fill Text on a <div>
(flipkart u02: "Element is not an <input>"), Fill Text on the iframe that
hosts an editor (21 corpus failures on iframe[id="mce_0_ifr"]), a read of an
input that is empty although the agent said it shows "buy milk" (todomvc
u10). This module answers one question — can the resolved element perform
this action — from one bounded evaluate() on it.

Provable signals only, the contract of the identity guard in
agent/actions.py: an unreadable element, a missing action or an unknown
action is "" (fits), never a misfit. The rules mirror Playwright's own
actionability checks, because the generated test runs on Playwright:

- fill (input/type): an <input> of a fillable type, a <textarea> or
  [contenteditable]; not disabled, not readonly. A <label> fits — Playwright
  fills its associated control.
- click/submit/select/check/uncheck: enabled — not :disabled and not inside
  [aria-disabled="true"] (Playwright's "enabled" definition).
- get_text/get_attribute: deliberately narrow. nlrf maps every locate-only
  keyword to get_text (Wait For Elements State, Get Element Count, Upload
  File By Selector, Hover, Get Attribute — element_identification.py
  _ACTION_EXACT), so "must show text" would reject legitimate targets. The
  only misfit is a form control with an EMPTY live value that the agent
  claims shows expected_text, when that text is not its placeholder,
  aria-label, title or label.

Referenced by: locators/smart_locator.py, agent/actions.py, locators/read_target.py
Depends on: (none — Playwright objects are passed in)
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Shorter than actions._RESOLVED_READ_TIMEOUT_MS (2s): every fit read runs
# straight after count()==1, inside the 5s cascade budget, and one call can
# read at several sites.
FIT_READ_TIMEOUT_MS = 1000

READ_ACTIONS = frozenset({"get_text", "get_attribute"})
_FILL_ACTIONS = frozenset({"input", "type"})
_ENABLED_ACTIONS = frozenset({"click", "submit", "select", "check", "uncheck"})
_UNFILLABLE_INPUT_TYPES = frozenset(
    {"checkbox", "radio", "file", "button", "submit", "reset", "image", "hidden"}
)
_FORM_CONTROL_TAGS = frozenset({"input", "textarea", "select"})

FIT_FACTS_JS = """el => {
    const tag = typeof el.tagName === 'string' ? el.tagName.toLowerCase() : '';
    const labels = el.labels
        ? Array.from(el.labels).map(l => (l.textContent || '').trim()).join(' ')
        : '';
    return {
        tag: tag,
        type: (el.getAttribute('type') || '').toLowerCase(),
        disabled: !!(el.matches && el.matches(':disabled')),
        ariaDisabled: !!(el.closest && el.closest('[aria-disabled="true"]')),
        readOnly: el.readOnly === true,
        contentEditable: el.isContentEditable === true,
        value: typeof el.value === 'string' ? el.value : '',
        placeholder: el.getAttribute('placeholder') || '',
        ariaLabel: el.getAttribute('aria-label') || '',
        title: el.getAttribute('title') || '',
        labelText: labels.slice(0, 500)
    };
}"""


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split()).casefold()


def action_misfit(
    action: Optional[str],
    facts: Optional[Dict[str, Any]],
    expected_text: Optional[str] = None,
) -> str:
    """Reason the element described by ``facts`` cannot perform ``action``.

    "" when it can, or when that cannot be established (no action, no facts,
    unreadable tag, an action with no rule).
    """
    if not action or not isinstance(facts, dict):
        return ""
    tag = facts.get("tag") or ""
    if not tag:
        return ""

    if action in _FILL_ACTIONS:
        if tag == "label":
            return ""
        editable = (
            (tag == "input" and (facts.get("type") or "") not in _UNFILLABLE_INPUT_TYPES)
            or tag == "textarea"
            or bool(facts.get("contentEditable"))
        )
        if not editable:
            kind = f" (type={facts.get('type')})" if tag == "input" else ""
            return f"<{tag}>{kind} cannot be filled"
        if facts.get("disabled") or facts.get("ariaDisabled"):
            return f"<{tag}> is disabled"
        if facts.get("readOnly"):
            return f"<{tag}> is read-only"
        return ""

    if action in _ENABLED_ACTIONS:
        if facts.get("disabled"):
            return f"<{tag}> is disabled"
        if facts.get("ariaDisabled"):
            return f"<{tag}> is inside [aria-disabled=true]"
        return ""

    if action in READ_ACTIONS:
        needle = _norm(expected_text)
        if not needle or tag not in _FORM_CONTROL_TAGS or _norm(facts.get("value")):
            return ""
        for field in ("placeholder", "ariaLabel", "title", "labelText"):
            if needle in _norm(facts.get(field)):
                return ""
        return f"<{tag}> is empty — nothing to read as {expected_text!r}"

    return ""


async def read_fit_facts(
    search_root, locator: str, timeout_ms: int = FIT_READ_TIMEOUT_MS
) -> Optional[Dict[str, Any]]:
    """One bounded evaluate() on the element ``locator`` resolves to.

    None on any failure (detached node, strict-mode many-match, exotic
    selector engine, timeout) — callers treat None as unknown.
    """
    try:
        facts = await search_root.locator(locator).evaluate(FIT_FACTS_JS, timeout=timeout_ms)
    except Exception as e:
        logger.warning(f"   ⚠️ Could not read fitness facts for '{locator}': {e}")
        return None
    return facts if isinstance(facts, dict) else None


async def check_action_fit(
    search_root,
    locator: str,
    action: Optional[str],
    expected_text: Optional[str] = None,
) -> str:
    """'' when the element at ``locator`` can perform ``action`` or when that
    cannot be established; otherwise why it cannot. Reads nothing when
    ``action`` is falsy — the flag-off path stays untouched."""
    if not action:
        return ""
    return action_misfit(action, await read_fit_facts(search_root, locator), expected_text)
