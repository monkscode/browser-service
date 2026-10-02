"""
The frame prefix of a locator inside an iframe (browser-service #35).

A locator validated inside a frame only works in the generated test when it
names the frame: ``iframe[id="x"] >>> <locator>`` (Browser Library's frame
piercing). Without it the test looks in the main page — it times out, or, when
the main page has a same-selector element, passes on the WRONG element. Bench
q09 shipped ``p[contenteditable='true']`` bare 5 times in 18 runs on
2026-10-02 because the agent-candidate exit never added the frame.

The rule used to be written four times as a nested function in
smart_locator.py, applied to a result by a fifth, and stripped by hand in
agent/actions.py — and every exit decided for itself whether to call one.
It lives here once:

- add_frame / strip_frame: the string rule and its inverse.
- apply_frame_to_result: prefix the best locator AND every ``all_locators``
  entry (tasks/workflow.py's re-ranker and id check only choose among entries,
  they never build a string), mark the result positional when the frame hop
  is ordinal, and record the frame under ``iframe_context``.
- ensure_frame_on_result: the gate at the exit of find_unique_locator_action.
  Every result leaves through it, so a new exit cannot forget the frame. It
  only ever ADDS a missing frame: with no frame, on a not-found result, or
  when nothing is bare, the same object comes back unchanged.

Not handled: a frame inside a frame (the detector returns one frame).

Referenced by: locators/smart_locator.py, agent/actions.py
Depends on: locators/stability.py
"""

from typing import Any, Optional

from browser_service.locators.stability import POSITIONAL, is_positional_locator

# Browser Library's frame-piercing separator: <frame selector> >>> <locator>.
FRAME_SEPARATOR = " >>> "


def _carries_frame(locator: str, iframe_context: str) -> bool:
    """True when ``locator`` starts with the full hop, ``<frame> >>> ``.

    Starting with the frame's text is not enough: an <iframe> nested in the
    frame can have an in-frame locator that begins with — or equals — the
    host frame's own selector (``iframe.editor-preview`` inside
    ``iframe.editor``). Read as "already framed", it would ship bare and
    resolve, at page level, to the host frame.
    """
    return locator.startswith(f"{iframe_context}{FRAME_SEPARATOR}")


def add_frame(locator: str, iframe_context: Optional[str]) -> str:
    """``locator`` addressed through ``iframe_context``; idempotent."""
    if iframe_context and not _carries_frame(locator, iframe_context):
        return f"{iframe_context}{FRAME_SEPARATOR}{locator}"
    return locator


def strip_frame(locator: str, iframe_context: Optional[str]) -> str:
    """``locator`` without its ``iframe_context`` hop — the form a
    ``page.frame_locator(iframe_context)`` search root can resolve."""
    if iframe_context and _carries_frame(locator, iframe_context):
        return locator[len(iframe_context) + len(FRAME_SEPARATOR) :]
    return locator


def _is_locator(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _entries(result: dict) -> list[dict]:
    """The all_locators entries, tolerating a missing or malformed list."""
    entries = result.get("all_locators")
    return [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []


def apply_frame_to_result(result: dict, iframe_context: Optional[str]) -> dict:
    """Apply the frame to best_locator and every entry of all_locators."""
    if not iframe_context:
        return result

    if _is_locator(result.get("best_locator")):
        result["best_locator"] = add_frame(result["best_locator"], iframe_context)

    entries = _entries(result)
    for loc in entries:
        if _is_locator(loc.get("locator")):
            loc["locator"] = add_frame(loc["locator"], iframe_context)

    # An ordinal iframe hop (iframe >> nth=N >>> ...) encodes DOM order:
    # the whole composite is positional even when the inner locator is
    # stable (B2). Row-anchored results are exempt from the whole-string
    # check — their >> nth=0 is the containment collapse (structural,
    # parent-first document order); for those only the hop itself can
    # make the composite positional.
    best = result.get("best_locator") or ""
    positional_scope = iframe_context if result.get("row_anchored") else best
    if best and is_positional_locator(positional_scope):
        result["stability"] = POSITIONAL
        for loc in entries:
            loc["stability"] = POSITIONAL

    result["iframe_context"] = iframe_context
    return result


def ensure_frame_on_result(result: Any, iframe_context: Optional[str]) -> Any:
    """The exit gate: add the frame to a found result that lacks it.

    Returns ``result`` itself, unchanged, when there is no frame, when the
    result is not a found result, or when neither the best locator nor any
    alternative is bare — so the exits that already carry the frame are
    provably unaffected.
    """
    if not iframe_context or not isinstance(result, dict) or not result.get("found"):
        return result

    locators = [result.get("best_locator")] + [e.get("locator") for e in _entries(result)]
    if all(_carries_frame(loc, iframe_context) for loc in locators if isinstance(loc, str) and loc):
        return result

    return apply_frame_to_result(result, iframe_context)
