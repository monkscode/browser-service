"""
Locator stability scoring (Task 10, E1).

The pipeline validates candidates against the live page ("unique right
now"); the generated Robot test runs in a fresh browser session, minutes
to months later.  This module scores locator raw material for the gap
between those two moments:

- ``stable``     — hand-authored ids/names/classes and test attributes;
                   expected to survive a fresh session.
- ``volatile``   — session-generated values: framework id counters
                   (``ext-gen1042``, ``ember472``, ``tomselect-2``, ...),
                   UUID/hash shapes, timestamp/counter suffixes.  Valid at
                   discovery, dead on the next page load.
- ``positional`` — locators that encode today's DOM order (``>> nth=``,
                   ``:nth-child()``, numeric XPath predicates, ordinal
                   group indexes).  Worse than volatile: on reorder they
                   silently hit a *different* element instead of failing.

Scoring rules are deliberately conservative — evidence from the
2026-07-06 log/bench audit:

- Bare all-digit ids stay ``stable``: ``id=880667900`` is GitHub's repo
  database id and passed bench q01 3/3 in fresh RF sessions.  Digit-heavy
  does not mean session-volatile (narrowed from the analysis doc's
  >=40%-digit rule, which would have demoted the most-used id in our logs).
- Short ordinal suffixes stay ``stable``: ``dt-search-0`` (DataTables,
  24+ emissions, never failed) is deterministic per page.
- ``tomselect-N`` is ``volatile``: the original production incident
  (35 log hits); its hard-coded suppression in handlers/dropdown.py
  folds into this scorer.

Consumers: STEP-0/STEP-3 candidate ordering in smart_locator.py, the
PHASE-2 re-ranker and priority forcer in tasks/workflow.py, handler
result assembly, and the ``stability`` result-payload field read by
nlrf (warnings, healing).

Depends on:
    - (none — stdlib only)
"""

import re

STABLE = "stable"
VOLATILE = "volatile"
POSITIONAL = "positional"

# Framework-generated id shapes.  Each pattern is anchored at the start of
# the value and requires the generated shape (prefix + counter), so
# hand-authored ids that merely share a prefix ("mat-icon-button",
# "embermail") are not flagged.
_FRAMEWORK_ID_PATTERNS = (
    re.compile(r"ext-gen\d+"),           # ExtJS generated element
    re.compile(r"ext-comp-\d+"),         # ExtJS component
    re.compile(r"ember\d+"),             # Ember view id
    re.compile(r"mat-[a-z][a-z-]*-\d+"),  # Angular Material (mat-input-5)
    re.compile(r"select2-"),             # Select2 (random middle segments)
    re.compile(r"gwt-uid-\d+"),          # GWT
    re.compile(r"tomselect-\d+"),        # Tom Select init-order counter
    re.compile(r"cke_\d+"),              # CKEditor 4 init-order counter (G6);
                                         # name-derived cke_editor1 /
                                         # cke_wysiwyg_frame stay stable
    re.compile(r"radix-"),               # Radix UI generated id
    re.compile(r":r[0-9a-z]+:"),         # React useId
)

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
# Long pure-hex value with at least one digit (md5/sha fragments, cache
# busters).  The digit requirement keeps ordinary long words out.
_HEX_HASH_RE = re.compile(r"(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{16,}$")

# Word-like prefix glued to a >=4-digit tail: timestamps and counters
# ("field-1749283746", "session_1699999999").  A bare all-digit value has
# no \D and never matches — that is the narrowed digit rule.
_DIGIT_SUFFIX_RE = re.compile(r".*\D\d{4,}$")

# --- positional locator shapes -------------------------------------------
_NTH_ENGINE_RE = re.compile(r">>\s*nth=")               # Playwright nth engine
_NTH_CSS_RE = re.compile(r":nth-(?:child|of-type)\(")   # CSS structural
_XPATH_GROUP_INDEX_RE = re.compile(r"\)\[\d+\]")        # (//...)[N]
_XPATH_STEP_INDEX_RE = re.compile(r"\[\d+\]")           # //div[1]/...

# --- dynamic-text shapes ---------------------------------------------------
_TRAILING_COUNT_RE = re.compile(r"\(\s*\d+[^)]*\)\s*$")   # "Cart (3 items)"
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SLASH_DATE_RE = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}")
_CLOCK_TIME_RE = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?!\d)")
_BARE_NUMBER_RE = re.compile(r"[$€£]?\s?\d[\d,.]*\+?$")

# --- embedded-value extraction for whole-locator classification -----------
_LOCATOR_ID_VALUE_RES = (
    re.compile(r"(?:^|\s|=)#([^\s>\[.:,\"']+)"),   # #value / css=#value
    re.compile(r"^id=(.+)$"),                       # id=value
    re.compile(r"\[id=[\"']([^\"']+)[\"']\]"),      # [id="value"]
)
_LOCATOR_TEXT_VALUE_RES = (
    re.compile(r"text=[\"']([^\"']*)[\"']"),        # text="..."
    re.compile(r"text\(\)\s*=\s*'([^']*)'"),        # xpath text()='...'
    re.compile(r"contains\(text\(\),\s*'([^']*)'"),  # xpath contains(text(),..)
)

_RANK = {STABLE: 0, VOLATILE: 1, POSITIONAL: 2}


def score_stability(attr_name: str, value: str) -> str:
    """
    Classify an attribute VALUE as ``stable`` or ``volatile``.

    Applied uniformly to id, name, class and data-testid values —
    ``attr_name`` is part of the contract for future per-attribute rules
    but does not change the verdict today.  Position is a property of a
    locator, not of a value, so this never returns ``positional``.
    """
    if not value:
        return STABLE

    for pattern in _FRAMEWORK_ID_PATTERNS:
        if pattern.match(value):
            return VOLATILE

    if _UUID_RE.fullmatch(value) or _HEX_HASH_RE.fullmatch(value):
        return VOLATILE

    if _DIGIT_SUFFIX_RE.fullmatch(value):
        return VOLATILE

    return STABLE


def is_positional_locator(locator: str) -> bool:
    """
    True when the locator encodes DOM order: ``>> nth=``, CSS
    ``:nth-child()``/``:nth-of-type()``, XPath group indexes ``(...)[N]``,
    or bare numeric XPath step predicates ``//div[1]``.

    Numeric step predicates are only checked on XPath-shaped locators;
    CSS/attribute predicates like ``[data-x="1"]`` carry quotes and never
    match the bare-number pattern.
    """
    if not locator:
        return False

    if _NTH_ENGINE_RE.search(locator) or _NTH_CSS_RE.search(locator):
        return True

    is_xpath = (
        locator.startswith("xpath=")
        or locator.startswith("//")
        or locator.startswith("(//")
    )
    if is_xpath and (
        _XPATH_GROUP_INDEX_RE.search(locator)
        or _XPATH_STEP_INDEX_RE.search(locator)
    ):
        return True

    return False


def is_dynamic_text(text: str) -> bool:
    """
    True when visible text looks data-bound and likely to differ at RF
    runtime: trailing parenthesised counts ("Cart (3 items)"), dates,
    clock times, or bare numbers/prices.  Ordinary labels with embedded
    digits ("Q1 2025 report") stay static.
    """
    if not text:
        return False
    stripped = text.strip()
    if _TRAILING_COUNT_RE.search(stripped):
        return True
    if _ISO_DATE_RE.search(stripped) or _SLASH_DATE_RE.search(stripped):
        return True
    if _CLOCK_TIME_RE.search(stripped):
        return True
    if _BARE_NUMBER_RE.fullmatch(stripped):
        return True
    return False


def classify_locator(locator: str) -> str:
    """
    Classify a finished locator string for payload assembly sites that
    did not build it from a known attribute (text-first, semantic,
    accessibility, handler results).

    Position dominates: a positional hop makes the whole locator
    positional even if every embedded value is stable.
    """
    if not locator:
        return STABLE

    if is_positional_locator(locator):
        return POSITIONAL

    for pattern in _LOCATOR_ID_VALUE_RES:
        match = pattern.search(locator)
        if match and score_stability("id", match.group(1)) == VOLATILE:
            return VOLATILE

    for pattern in _LOCATOR_TEXT_VALUE_RES:
        match = pattern.search(locator)
        if match and is_dynamic_text(match.group(1)):
            return VOLATILE

    return STABLE


def stability_rank(tier: str) -> int:
    """Sort key: stable(0) < volatile(1) < positional(2); unknown last."""
    return _RANK.get(tier, len(_RANK) + 1)
