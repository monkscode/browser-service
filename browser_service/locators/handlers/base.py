"""
Shared helpers for handler modules.

``build_locator_result`` is the canonical factory for the standard locator
result dict that every handler returns on success.  All required fields and
invariant values are set here; handler-specific extras are merged in via the
three extension points documented on the function.

Referenced by:
    - browser_service.locators.handlers.dropdown
    - browser_service.locators.handlers.checkbox
    - browser_service.locators.handlers.collection

Depends on:
    - browser_service.locators.stability
"""

from browser_service.locators.stability import classify_locator


def build_locator_result(
    *,
    element_id: str,
    description: str,
    best_locator: str,
    element_type: str,
    strategy_name: str,
    classifier_confidence,
    classifier_signals,
    unique: bool = True,
    count: int = 1,
    stability: str | None = None,
    all_locator_extra: dict | None = None,
    validation_summary_extra: dict | None = None,
    **top_level_extra,
) -> dict:
    """
    Build the standard locator result dict shared across handler modules.

    Parameters
    ----------
    element_id, description, best_locator, element_type, strategy_name :
        Required identity and routing fields present on every result.
    classifier_confidence, classifier_signals :
        Passed through verbatim from ``ElementTypeInfo``; ``signals`` is
        coerced to ``list`` so callers may pass any iterable.
    unique :
        Whether the locator matches exactly one element.  Propagated into
        both the top-level result and ``all_locators[0]``.  ``validation_summary
        ["unique"]`` is set to ``1`` when True, ``0`` when False.
    count :
        Element count — set at both the top-level result dict and inside
        ``all_locators[0]``.  Single-element handlers pass the default (1).
        Collection handlers pass the actual collection count; their subsequent
        ``result["count"] = n`` after this call is a harmless no-op overwrite.
    stability :
        Stability tier ("stable" | "volatile" | "positional", E1).  When
        omitted, the finished locator is classified — ordinal strategies
        like ``(//div[...])[N]`` come back "positional", embedded
        framework ids "volatile".  Set at the top level and inside
        ``all_locators[0]``.
    all_locator_extra :
        Extra fields merged into ``all_locators[0]`` (e.g. ``quality_score``).
    validation_summary_extra :
        Extra fields merged into ``validation_summary``
        (e.g. ``multi_element``, ``collection_count``).
    **top_level_extra :
        Handler-specific top-level fields
        (e.g. ``dropdown_framework``, ``select_id``, ``framework``,
        ``quality_score``, ``element_info``).  Merged into the result dict
        **before** the invariant base keys so that base values always win
        over any accidental name collision in the extras.
    """
    resolved_stability = stability or classify_locator(best_locator)

    all_locator_entry: dict = {
        "type": element_type,
        "locator": best_locator,
        "priority": 0,
        "strategy": strategy_name,
        "count": count,
        "unique": unique,
        "valid": True,
        "validation_method": "playwright",
        "stability": resolved_stability,
    }
    if all_locator_extra:
        all_locator_entry.update(all_locator_extra)

    validation_summary: dict = {
        "total_generated": 1,
        "valid": 1,
        "unique": 1 if unique else 0,
        "best_type": element_type,
        "best_strategy": strategy_name,
        "validation_method": "playwright",
    }
    if validation_summary_extra:
        validation_summary.update(validation_summary_extra)

    # top_level_extra is spread first so that invariant base keys always win
    # over any accidental same-name key in the caller's extras.
    return {
        **top_level_extra,
        "element_id": element_id,
        "description": description,
        "found": True,
        "best_locator": best_locator,
        "element_type": element_type,
        "stability": resolved_stability,
        "count": count,
        "unique": unique,
        "valid": True,
        "validated": True,
        "all_locators": [all_locator_entry],
        "validation_summary": validation_summary,
        "validation_method": "playwright",
        "semantic_match": True,
        "classifier_confidence": classifier_confidence,
        "classifier_signals": list(classifier_signals),
    }
