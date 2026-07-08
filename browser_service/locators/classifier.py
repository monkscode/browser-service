"""
Element Type Classifier
=======================

Tiered DOM-first classifier that determines an element's type and
framework before specialized strategy dispatch. Two-source-of-truth
design: the classifier produces the FIRST source (DOM signals + optional
vision hint piggybacked from browser-use). The SECOND source — a
Playwright DOM probe in :mod:`dom_probe` — runs in the dispatcher to
corroborate before any specialized handler executes.

Tier 0 — pure DOM rules (deterministic, microseconds). HTML-semantic;
         vision hints can never override these (a textarea IS a textarea).
Tier 1 — multi-signal voting across tagName / role / className /
         description / vision hint. Hint counts as a strong (+3) vote
         when present.

The vision hint comes from browser-use's per-step LLM call (free
piggyback via ``FindUniqueLocatorParams.element_type``). It is one
voting signal among many — the dispatcher's DOM probe must agree
before a specialized handler runs.

Design contract: see `docs/ELEMENT_TYPE_CLASSIFIER_ARCHITECTURE.md` Section 5
+ the two-source-of-truth design discussion.

Referenced by:
    - browser_service.locators.smart_locator (dispatcher)
    - browser_service.locators.handlers.* (consume ElementTypeInfo)
    - browser_service.locators.dom_probe (shares framework patterns)

Depends on:
    - stdlib only
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ElementTypeInfo:
    """
    Classification verdict for an element.

    Returned by `classify_element_type()` and consumed by the dispatcher in
    `smart_locator.py` plus the per-type handlers. The `confidence` and
    `signals` fields are load-bearing — handlers use confidence to decide
    whether to commit to specialized paths or hedge, and signals is the
    debug breadcrumb trail used to diagnose misclassifications in production.

    Attributes:
        primary_type: One of "dropdown", "collection", "checkbox", "radio",
            "file-upload", "date-picker", "input", "button", "link",
            "label", "image", "text", "unknown".
        framework: For dropdowns: "tom-select", "select2", "kendo",
            "react-select", "vue-select", "ant-design", "material-ui",
            "native", "combobox-input". For collections: "table-row",
            "list-item", "". For checkbox/radio: "native", "custom",
            "toggle". Empty string when no framework signal was found.
        confidence: "high" — multi-signal DOM agreement (e.g., tagName=select
            + role=combobox). "medium" — one strong DOM signal OR
            description+role agreement. "low" — description-only OR
            vision-assisted OR conflicting signals.
        signals: Debug trace of what fed the verdict, including which tier
            produced it. Example: ["tagName=div", "className: ts-wrapper",
            "desc:dropdown", "tier:0", "vote:dropdown=high"].
    """

    primary_type: str
    framework: str = ""
    confidence: str = "low"
    signals: list[str] = field(default_factory=list)


# Framework className patterns. Order matters: the first match wins, so the
# more specific patterns (e.g., 'vs__dropdown-toggle') should appear before
# more general ones (e.g., a plain 'select2' substring).
_DROPDOWN_FRAMEWORK_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tom-select", ("ts-wrapper", "ts-control")),
    ("select2", ("select2",)),
    ("kendo", ("k-dropdown", "k-combobox", "k-multiselect")),
    ("react-select", ("react-select",)),
    ("vue-select", ("vs__dropdown-toggle",)),
    ("ant-design", ("ant-select",)),
    # className lowercased before matching, so 'MuiSelect-root' compares as
    # 'muiselect-root' and 'MuiAutocomplete' as 'muiautocomplete'.
    ("material-ui", ("muiselect-root", "muiautocomplete")),
)

_NAV_PREFIXES: tuple[str, ...] = (
    "nav-", "menu-", "tab-", "breadcrumb-", "pagination-", "dropdown-",
)

_DROPDOWN_DESC_KEYWORDS: tuple[str, ...] = (
    "dropdown", "select", "combobox", "multiselect", "picker", "chooser",
)

_COLLECTION_DESC_KEYWORDS: tuple[str, ...] = (
    "rows", "items", "all ", "each", "every", "list of",
    "visible rows", "table rows", "filtered",
    "cells", "column cell", "column cells",
    "results table", "data table",
)

_COLLECTION_CLASS_TOKENS: frozenset[str] = frozenset({
    "row", "item", "card", "entry", "record",
    "tr-group", "list-item", "grid-item",
})

_CHECKBOX_DESC_KEYWORDS: tuple[str, ...] = (
    "checkbox", "tick", "toggle", "switch",
)

_RADIO_DESC_KEYWORDS: tuple[str, ...] = (
    "radio button", "radio option",
)

# Deliberately NOT including bare "import": list-page toolbars carry
# Import buttons that merely navigate (ASTPP #import) — a file-upload
# hunt there wastes a probe and risks anchoring an unrelated input.
_FILE_UPLOAD_DESC_KEYWORDS: tuple[str, ...] = (
    "upload", "choose file", "browse", "attach", "file input",
)

_TIER1_HIGH_WEIGHT = 3    # tagName / role / vision hint (each a strong signal)
_TIER1_MEDIUM_WEIGHT = 2  # className
_TIER1_LOW_WEIGHT = 1     # description

_CONFIDENCE_HIGH_THRESHOLD = 4
_CONFIDENCE_MEDIUM_THRESHOLD = 3

# Vision hints map to our internal primary_type vocabulary. The action
# schema lets the LLM pick from a slightly wider list (button / link /
# image / etc.) for general classification; we only consume the values
# that map to a specialized handler.
_VISION_HINT_TO_TYPE: dict[str, str] = {
    "dropdown": "dropdown",
    "checkbox": "checkbox",
    "radio": "radio",
    "table": "collection",
    "file-upload": "file-upload",
    # The following hints don't trigger specialized routing today; they
    # propagate as informational signals only.
    "input": "input",
    "text-area": "input",
    "button": "button",
    "link": "link",
    "image": "image",
    "label": "label",
    "other": "",
}


def map_vision_hint(hint: str) -> str:
    """Return the internal primary_type for a vision hint, or '' if unmapped."""
    return _VISION_HINT_TO_TYPE.get(hint.lower().strip(), "")


def classify_element_type(
    element_data: dict,
    element_description: str,
    vision_type_hint: Optional[str] = None,
    vision_framework_hint: Optional[str] = None,
) -> ElementTypeInfo:
    """
    Classify an element's type and (where applicable) framework.

    Runs Tier 0 DOM rules first (HTML-deterministic — vision hints can
    NEVER override these); on no Tier 0 match, runs Tier 1 multi-signal
    voting. The vision hint is one voting signal among many; the
    dispatcher's DOM probe (in :mod:`dom_probe`) corroborates before
    any specialized handler runs.

    Args:
        element_data: Dict with element attributes from browser-use DOM.
            Recognized keys: tagName, className, role, type, id, name.
        element_description: Human-readable description from the planner.
            Used as a low-weight signal in Tier 1 voting.
        vision_type_hint: Optional LLM-derived classification piggybacked
            via ``FindUniqueLocatorParams.element_type``. Mapped to our
            internal type vocabulary via ``_VISION_HINT_TO_TYPE``.
            Counts as a strong (+3) Tier 1 vote when present. NEVER
            overrides Tier 0 deterministic rules — when it conflicts
            with a Tier 0 hit, the conflict is logged in signals but
            the DOM verdict stands.
        vision_framework_hint: Optional LLM-derived framework name for
            specialized types. Used to seed the framework field when
            DOM signals don't reveal one — pending probe corroboration.

    Returns:
        ElementTypeInfo with primary_type, framework, confidence, signals.
        Returns primary_type="unknown" when no signal fires — caller
        should fall through to the generic 21-strategy path.
    """
    element_data = element_data or {}

    tag = (element_data.get("tagName") or "").lower()
    classes_lower = (element_data.get("className") or "").lower()
    role = (element_data.get("role") or "").lower()
    input_type = (element_data.get("type") or "").lower()
    desc = (element_description or "").lower()
    class_tokens = classes_lower.split()
    has_nav_class = any(
        c.startswith(_NAV_PREFIXES) for c in class_tokens
    )

    # Normalize the vision hint to our internal type vocabulary up
    # front. An unmapped hint becomes "" (effectively absent).
    vision_hint_normalized = ""
    if vision_type_hint:
        vision_hint_normalized = _VISION_HINT_TO_TYPE.get(
            vision_type_hint.lower().strip(), ""
        )

    framework_hint = (vision_framework_hint or "").lower().strip()

    # ===== Tier 0 — pure DOM rules =====
    tier0 = _tier0_dom_rules(
        tag=tag, input_type=input_type, classes_lower=classes_lower,
        role=role, has_nav_class=has_nav_class,
    )
    if tier0 is not None:
        # HTML-deterministic verdict. The vision hint NEVER overrides
        # Tier 0 — but we annotate the signal trail so downstream debug
        # can see whether sources agreed. Conflicts in particular are
        # high-value telemetry: vision says X, DOM says Y → either
        # browser-use hallucinated or coords landed on a wrong node.
        if vision_hint_normalized:
            if vision_hint_normalized == tier0.primary_type:
                tier0.signals.append(
                    f"vision-hint-agree:{vision_hint_normalized}"
                )
            else:
                tier0.signals.append(
                    f"vision-hint-conflict:hint={vision_hint_normalized},"
                    f"dom={tier0.primary_type}"
                )
        if framework_hint and not tier0.framework:
            # DOM didn't reveal a framework but vision named one.
            # Seed the framework field — the dispatcher's DOM probe will
            # confirm or reject before any framework-specific handler
            # commits to it.
            tier0.framework = framework_hint
            tier0.signals.append(f"framework-hint:{framework_hint}")
        return tier0

    # ===== Tier 1 — multi-signal voting =====
    return _tier1_vote(
        tag=tag,
        role=role,
        classes_lower=classes_lower,
        class_tokens=class_tokens,
        has_nav_class=has_nav_class,
        desc=desc,
        vision_hint=vision_hint_normalized,
        framework_hint=framework_hint,
    )


def _tier0_dom_rules(
    tag: str,
    input_type: str,
    classes_lower: str,
    role: str,
    has_nav_class: bool,
) -> Optional[ElementTypeInfo]:
    """
    Pure DOM rule check. Returns an ``ElementTypeInfo`` on a hit, or
    ``None`` to fall through to Tier 1 voting. HTML-semantic — never
    influenced by vision hints.
    """
    # Rule 1: <select>
    if tag == "select":
        return ElementTypeInfo(
            "dropdown", "native", "high",
            ["tagName=select", "tier:0"],
        )

    # Rules 2-5: <input type=...>
    if tag == "input":
        if input_type == "checkbox":
            return ElementTypeInfo(
                "checkbox", "native", "high",
                ["tagName=input", "type=checkbox", "tier:0"],
            )
        if input_type == "radio":
            return ElementTypeInfo(
                "radio", "native", "high",
                ["tagName=input", "type=radio", "tier:0"],
            )
        if input_type == "file":
            return ElementTypeInfo(
                "file-upload", "native", "high",
                ["tagName=input", "type=file", "tier:0"],
            )
        if input_type == "date":
            return ElementTypeInfo(
                "date-picker", "native", "high",
                ["tagName=input", "type=date", "tier:0"],
            )

    # Rule 6: <tr>
    if tag == "tr":
        return ElementTypeInfo(
            "collection", "table-row", "high",
            ["tagName=tr", "tier:0"],
        )

    # Rule 7: <li> NOT in nav context
    if tag == "li" and not has_nav_class:
        return ElementTypeInfo(
            "collection", "list-item", "medium",
            ["tagName=li", "tier:0"],
        )

    # Rules 8-14: framework className patterns
    for framework, patterns in _DROPDOWN_FRAMEWORK_PATTERNS:
        for pattern in patterns:
            if pattern in classes_lower:
                return ElementTypeInfo(
                    "dropdown", framework, "high",
                    [f"className:{pattern}", "tier:0"],
                )

    # Rules 15-17: role-based
    if role in ("combobox", "listbox"):
        return ElementTypeInfo(
            "dropdown", "combobox-input", "high",
            [f"role={role}", "tier:0"],
        )
    if role == "checkbox":
        return ElementTypeInfo(
            "checkbox", "custom", "high",
            ["role=checkbox", "tier:0"],
        )
    if role == "radio":
        return ElementTypeInfo(
            "radio", "custom", "high",
            ["role=radio", "tier:0"],
        )
    if role == "switch":
        return ElementTypeInfo(
            "checkbox", "toggle", "high",
            ["role=switch", "tier:0"],
        )

    return None


def _tier1_vote(
    tag: str,
    role: str,
    classes_lower: str,
    class_tokens: list[str],
    has_nav_class: bool,
    desc: str,
    vision_hint: str = "",
    framework_hint: str = "",
) -> ElementTypeInfo:
    """
    Multi-signal voting fallback when Tier 0 rules don't fire.

    Voting weights (per spec §5 + two-source-of-truth design):
        tagName indicator → +3 (high)
        role indicator    → +3 (high)
        vision hint       → +3 (high) — strong signal but never overrides
                                        Tier 0; corroborated by DOM probe
                                        in the dispatcher
        className pattern → +2 (medium)
        description hint  → +1 (low)

    Confidence thresholds:
        max >= 4 → high  — needs ≥2 sources to agree (e.g., vision hint
                           + DOM signal). Single +3 vote alone caps at
                           medium so the dispatcher knows to require
                           probe corroboration with extra care.
        max >= 3 → medium
        max >= 1 → low (signal conflict or sole low-weight signal)
        max == 0 → primary_type=unknown, confidence=low
    """
    votes: dict[str, int] = {
        "dropdown": 0,
        "collection": 0,
        "checkbox": 0,
        "radio": 0,
        "file-upload": 0,
    }
    signal_log: list[str] = []

    # tagName signals (+3) — Tier 0 already caught select/input/tr/li.
    # The remaining tag-based collection hint is <option> / <dt> / <dd>.
    if tag in ("option", "dt", "dd"):
        votes["collection"] += _TIER1_HIGH_WEIGHT
        signal_log.append(f"tagName={tag}=>collection(+{_TIER1_HIGH_WEIGHT})")

    # role signals (+3) — Tier 0 already caught combobox/listbox/switch/etc.
    # Remaining role-based collection hints: row/grid always; listitem only outside nav context.
    if role in ("row", "grid") or (role == "listitem" and not has_nav_class):
        votes["collection"] += _TIER1_HIGH_WEIGHT
        signal_log.append(f"role={role}=>collection(+{_TIER1_HIGH_WEIGHT})")

    # className signals (+2) — collection patterns with nav-prefix exclusion.
    if not has_nav_class:
        for cls in class_tokens:
            if cls in _COLLECTION_CLASS_TOKENS:
                votes["collection"] += _TIER1_MEDIUM_WEIGHT
                signal_log.append(
                    f"className:{cls}=>collection(+{_TIER1_MEDIUM_WEIGHT})"
                )
                break

    # className signals for dropdown / checkbox / radio that didn't match
    # the Tier 0 framework set — weak class hints only (e.g., "dropdown" or
    # "select" appearing as a class token).
    if any(kw in classes_lower for kw in ("dropdown", "select", "combobox")):
        votes["dropdown"] += _TIER1_MEDIUM_WEIGHT
        signal_log.append(f"className:dropdown-hint(+{_TIER1_MEDIUM_WEIGHT})")
    if any(c.startswith(("checkbox-", "switch-", "toggle-")) for c in class_tokens):
        votes["checkbox"] += _TIER1_MEDIUM_WEIGHT
        signal_log.append(f"className:checkbox-pattern(+{_TIER1_MEDIUM_WEIGHT})")
    if any(c.startswith("radio-") for c in class_tokens):
        votes["radio"] += _TIER1_MEDIUM_WEIGHT
        signal_log.append(f"className:radio-pattern(+{_TIER1_MEDIUM_WEIGHT})")

    # description signals (+1) — lowest weight; a single description hint
    # alone produces "low" confidence so the dispatcher can let the handler
    # decide whether to commit.
    if any(kw in desc for kw in _DROPDOWN_DESC_KEYWORDS):
        votes["dropdown"] += _TIER1_LOW_WEIGHT
        signal_log.append(f"desc:dropdown(+{_TIER1_LOW_WEIGHT})")
    if any(kw in desc for kw in _COLLECTION_DESC_KEYWORDS):
        votes["collection"] += _TIER1_LOW_WEIGHT
        signal_log.append(f"desc:collection(+{_TIER1_LOW_WEIGHT})")
    if any(kw in desc for kw in _CHECKBOX_DESC_KEYWORDS):
        votes["checkbox"] += _TIER1_LOW_WEIGHT
        signal_log.append(f"desc:checkbox(+{_TIER1_LOW_WEIGHT})")
    if any(kw in desc for kw in _RADIO_DESC_KEYWORDS):
        votes["radio"] += _TIER1_LOW_WEIGHT
        signal_log.append(f"desc:radio(+{_TIER1_LOW_WEIGHT})")
    if any(kw in desc for kw in _FILE_UPLOAD_DESC_KEYWORDS):
        votes["file-upload"] += _TIER1_LOW_WEIGHT
        signal_log.append(f"desc:file-upload(+{_TIER1_LOW_WEIGHT})")

    # vision hint signal (+3) — strong, but treated as ONE source of truth.
    # The dispatcher requires DOM probe corroboration before any specialized
    # handler runs. When the hint is the SOLE signal, max_vote == 3 → confidence
    # = "medium" (handlers know to be more conservative). When the hint also
    # agrees with at least one other signal (className / description / role),
    # max_vote >= 4 → "high".
    if vision_hint and vision_hint in votes:
        votes[vision_hint] += _TIER1_HIGH_WEIGHT
        signal_log.append(
            f"vision-hint:{vision_hint}(+{_TIER1_HIGH_WEIGHT})"
        )

    max_vote = max(votes.values())
    signal_log.append("tier:1")

    if max_vote == 0:
        # No DOM/desc/hint signal fired. The vision hint, if set, was already
        # voted above and would have moved max_vote off zero — so this branch
        # genuinely means "we know nothing." Caller falls through to generic.
        signal_log.append("no-signals")
        # Even with no vote winner, propagate the framework hint so the
        # dispatcher's discovery-mode probe still has a target to confirm.
        return ElementTypeInfo("unknown", framework_hint, "low", signal_log)

    sorted_votes = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    top_type, top_score = sorted_votes[0]
    second_score = sorted_votes[1][1] if len(sorted_votes) > 1 else 0

    # Conflict (top two tied with non-trivial votes) — drop confidence to low
    # so handlers know to hedge.
    if top_score == second_score and top_score >= _TIER1_MEDIUM_WEIGHT:
        confidence = "low"
        signal_log.append(f"vote:conflict({top_type}={top_score},tied)")
    elif max_vote >= _CONFIDENCE_HIGH_THRESHOLD:
        confidence = "high"
        signal_log.append(f"vote:{top_type}={top_score}")
    elif max_vote >= _CONFIDENCE_MEDIUM_THRESHOLD:
        confidence = "medium"
        signal_log.append(f"vote:{top_type}={top_score}")
    else:
        confidence = "low"
        signal_log.append(f"vote:{top_type}={top_score}")

    # Seed the framework field from the vision hint. The dispatcher's DOM
    # probe will confirm or reject before any framework-specific handler
    # commits to it — never trusted unilaterally.
    framework = framework_hint if framework_hint else ""
    if framework:
        signal_log.append(f"framework-hint:{framework}")

    return ElementTypeInfo(top_type, framework, confidence, signal_log)
