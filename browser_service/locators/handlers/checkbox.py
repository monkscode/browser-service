"""
Checkbox / radio / toggle / switch element locator generation.

Dispatched from ``smart_locator.py`` when the classifier verdict is
``primary_type in {"checkbox", "radio"}``. Routes on
``type_info.framework`` (``native`` for ``<input type="checkbox|radio">``,
``custom`` for ``role="checkbox|radio"`` widgets, ``toggle`` for
``role="switch"``).

Strategy order:

  1. Direct attribute anchors from ``element_data`` — id / name+value /
     name. Most stable when classifier hit Tier 0.
  2. ``find_checkbox_or_radio_by_label(...)`` — relocated from
     ``smart_locator.py``. Walks ``<label>`` / nested input / adjacent
     input patterns from the visible label text. Re-
     exported so the existing text-first call site in
     ``_find_element_by_expected_text`` keeps working unchanged.
  3. Custom-widget anchors for ``role=checkbox|radio|switch`` —
     aria-label, id, or name attribute lookup.
  4. Always-fallback contract: return ``None`` so the orchestrator runs
     the generic 21-strategy.

Public functions:
    find_locator(...) -> Optional[dict]
        Standard handler entry point — see Section 7 of
        docs/ELEMENT_TYPE_CLASSIFIER_ARCHITECTURE.md.
    find_checkbox_or_radio_by_label(page, label_text) -> Optional[dict]
        Legacy helper preserved for the text-first call site at
        smart_locator.py L910.

Referenced by:
    - browser_service.locators.smart_locator (dispatcher + back-compat shim)

Depends on:
    - re, typing.Optional (stdlib), structlog
    - browser_service.locators.classifier.ElementTypeInfo (type hint only)
"""

import structlog
from typing import TYPE_CHECKING, Optional

from .base import build_locator_result

if TYPE_CHECKING:
    from ..classifier import ElementTypeInfo

logger = structlog.get_logger(__name__)


# ----------------------------------------------------------------------
# Public entry point — handler dispatcher
# ----------------------------------------------------------------------


async def find_locator(
    page,
    element_data: dict,
    type_info: "ElementTypeInfo",
    element_id: str,
    element_description: str,
    expected_text: Optional[str],
    search_context,
    iframe_context: Optional[str],
    confirmed_coords: Optional[tuple],
) -> Optional[dict]:
    """
    Checkbox / radio / toggle handler entry point.

    Routes on ``type_info.framework``:
      - ``native`` → ``<input type="checkbox|radio">``: prefer
        ``element_data`` id/name; fall back to label-text search.
      - ``custom`` → ``role="checkbox|radio"``: prefer aria-label-anchored
        locator; fall back to label-text search.
      - ``toggle`` → ``role="switch"``: same as custom.

    Returns a result dict with ``element_type`` matching the classifier
    primary_type when a strategy succeeds; returns ``None`` otherwise.
    Always-fallback contract: never raises.
    """
    framework = type_info.framework
    primary_type = type_info.primary_type  # "checkbox" or "radio"
    label = (expected_text or element_description or "").strip()

    # ---- Strategy 1: element_data attribute anchors ----
    attr_result = await _try_element_data_attrs(
        element_data=element_data or {},
        primary_type=primary_type,
        framework=framework,
        search_context=search_context,
    )
    if attr_result:
        return _build_result(
            locator=attr_result["locator"],
            strategy_name=attr_result["strategy"],
            element_id=element_id,
            element_description=element_description,
            type_info=type_info,
            element_type=attr_result.get("element_type", primary_type),
        )

    # ---- Strategy 2: label-text search (legacy logic, native widgets) ----
    if label and framework == "native":
        legacy = await find_checkbox_or_radio_by_label(search_context, label)
        if legacy:
            return _build_result(
                locator=legacy["locator"],
                strategy_name="Label-text search (native checkbox/radio)",
                element_id=element_id,
                element_description=element_description,
                type_info=type_info,
                element_type=legacy.get("element_type", primary_type),
            )

    # ---- Strategy 3: custom-widget role + aria-label anchor ----
    if framework in ("custom", "toggle") and label:
        role_for_widget = (
            "switch" if framework == "toggle" else primary_type
        )
        custom_result = await _try_custom_widget_by_label(
            search_context=search_context,
            role=role_for_widget,
            label=label,
        )
        if custom_result:
            return _build_result(
                locator=custom_result["locator"],
                strategy_name=custom_result["strategy"],
                element_id=element_id,
                element_description=element_description,
                type_info=type_info,
                element_type=primary_type,
            )

    logger.info("checkbox.all_strategies_exhausted", primary_type=primary_type, framework=framework)
    return None


# ----------------------------------------------------------------------
# Strategy 1 — element_data attribute anchors
# ----------------------------------------------------------------------


async def _try_element_data_attrs(
    element_data: dict,
    primary_type: str,
    framework: str,
    search_context,
) -> Optional[dict]:
    """
    Build a locator from id / name+value / name in element_data and
    validate it returns exactly one element.
    """
    el_id = (element_data.get("id") or "").strip()
    el_name = (element_data.get("name") or "").strip()
    el_value = (element_data.get("value") or "").strip()

    candidates: list[tuple[str, str]] = []
    if el_id:
        candidates.append((f"id={el_id}", "id-anchored"))
    if el_name and el_value and primary_type == "radio":
        candidates.append((
            f'input[type="radio"][name="{el_name}"][value="{el_value}"]',
            "name+value (radio)",
        ))
    if el_name:
        if framework == "native":
            candidates.append((
                f'input[type="{primary_type}"][name="{el_name}"]',
                "name-anchored",
            ))
        elif framework == "toggle":
            candidates.append((
                f'[role="switch"][name="{el_name}"]',
                "role+name-anchored",
            ))
        else:
            candidates.append((
                f'[role="{primary_type}"][name="{el_name}"]',
                "role+name-anchored",
            ))

    for locator, name in candidates:
        if await _locator_unique(search_context, locator):
            logger.info("checkbox.strategy1_succeeded", strategy=name, locator=locator)
            return {"locator": locator, "strategy": name, "element_type": primary_type}

    return None


# ----------------------------------------------------------------------
# Strategy 3 — custom-widget anchors (role=checkbox|radio|switch)
# ----------------------------------------------------------------------


async def _try_custom_widget_by_label(
    search_context, role: str, label: str
) -> Optional[dict]:
    """
    Try aria-label-anchored locator, then a Playwright role= selector
    scoped to the visible label.
    """
    candidates: list[tuple[str, str]] = [
        (f'[role="{role}"][aria-label="{label}"]',
         f"role={role}+aria-label"),
        (f'role={role}[name="{label}"]',
         f"role={role}+accessible-name"),
    ]
    for locator, name in candidates:
        if await _locator_unique(search_context, locator):
            logger.info("checkbox.strategy3_succeeded", strategy=name, locator=locator)
            return {"locator": locator, "strategy": name}
    return None


async def _locator_unique(search_context, locator: str) -> bool:
    """Return True iff ``locator`` matches exactly one element."""
    try:
        count = await search_context.locator(locator).count()
    except Exception as e:
        logger.warning("checkbox.locator_validation_failed", error=str(e))
        return False
    return count == 1


# ----------------------------------------------------------------------
# Result construction
# ----------------------------------------------------------------------


def _build_result(
    locator: str,
    strategy_name: str,
    element_id: str,
    element_description: str,
    type_info: "ElementTypeInfo",
    element_type: str,
) -> dict:
    """Standard handler result-dict shape — delegates to base.build_locator_result."""
    return build_locator_result(
        element_id=element_id,
        description=element_description,
        best_locator=locator,
        element_type=element_type,
        strategy_name=strategy_name,
        classifier_confidence=type_info.confidence,
        classifier_signals=type_info.signals,
        framework=type_info.framework,
    )


# ======================================================================
# Shared label-walk finder — used by find_locator() Strategy 2 above and
# by the evidence-gated text-first detour in smart_locator.py
# (_find_element_by_expected_text). Exact label match is tried before
# substring; there is deliberately NO positional (nth) fallback.
# ======================================================================


async def find_checkbox_or_radio_by_label(
    page, label_text: str
) -> Optional[dict]:
    """
    Find a checkbox or radio input element associated with the given
    label text.

    This handles multiple scenarios:
      1. ``<label for="id">text</label> <input id="id" type="checkbox">``
      2. ``<label><input type="checkbox"> text</label>``
      3. ``<input type="checkbox"> text`` (no label, adjacent text)

    Args:
        page: Playwright page object.
        label_text: The visible text near the checkbox/radio.

    Returns:
        Dict with 'locator' and 'element_type' if found, None otherwise.
    """
    if not label_text:
        return None

    text = label_text.strip()
    logger.info("checkbox.finder_start", label=text)

    # Strategy 1: <label> with matching text → use its 'for' attribute.
    # Exact match first — has-text is a substring match, so "Option 1"
    # would hit "Option 10" when that label comes first in the DOM.
    try:
        label_locator = None
        for candidate in (
            f'label:text-is("{text}")',
            f'label:has-text("{text}")',
        ):
            if await page.locator(candidate).count() >= 1:
                label_locator = candidate
                break

        if label_locator:
            for_attr = await page.locator(label_locator).first.get_attribute(
                "for"
            )

            if for_attr:
                input_locator = f'input[id="{for_attr}"]'
                input_count = await page.locator(input_locator).count()

                if input_count == 1:
                    input_type = await page.locator(
                        input_locator
                    ).first.get_attribute("type")
                    if input_type in ("checkbox", "radio"):
                        final_locator = f"id={for_attr}"
                        logger.info("checkbox.found_via_label_for", input_type=input_type, locator=final_locator)
                        return {
                            "locator": final_locator,
                            "element_type": input_type,
                        }
            else:
                # No 'for' attribute — check for nested input inside label.
                try:
                    nested_checkbox = (
                        f'{label_locator} >> input[type="checkbox"]'
                    )
                    if await page.locator(nested_checkbox).count() == 1:
                        checkbox_id = await page.locator(
                            nested_checkbox
                        ).first.get_attribute("id")
                        checkbox_name = await page.locator(
                            nested_checkbox
                        ).first.get_attribute("name")

                        if checkbox_id:
                            final_locator = f"id={checkbox_id}"
                        elif checkbox_name:
                            final_locator = f'input[type="checkbox"][name="{checkbox_name}"]'
                        else:
                            final_locator = nested_checkbox

                        logger.info("checkbox.found_nested_in_label", input_type="checkbox", locator=final_locator)
                        return {
                            "locator": final_locator,
                            "element_type": "checkbox",
                        }

                    nested_radio = (
                        f'{label_locator} >> input[type="radio"]'
                    )
                    if await page.locator(nested_radio).count() == 1:
                        radio_id = await page.locator(
                            nested_radio
                        ).first.get_attribute("id")
                        radio_name = await page.locator(
                            nested_radio
                        ).first.get_attribute("name")
                        radio_value = await page.locator(
                            nested_radio
                        ).first.get_attribute("value")

                        if radio_id:
                            final_locator = f"id={radio_id}"
                        elif radio_name and radio_value:
                            final_locator = (
                                f'input[type="radio"][name="{radio_name}"]'
                                f'[value="{radio_value}"]'
                            )
                        elif radio_name:
                            final_locator = f'input[type="radio"][name="{radio_name}"]'
                        else:
                            final_locator = nested_radio

                        logger.info("checkbox.found_nested_in_label", input_type="radio", locator=final_locator)
                        return {
                            "locator": final_locator,
                            "element_type": "radio",
                        }
                except Exception as e:
                    logger.info("checkbox.nested_input_error", error=str(e))
    except Exception as e:
        logger.info("checkbox.label_search_error", error=str(e))

    # Strategy 2: text element with adjacent checkbox/radio.
    try:
        adjacent_patterns = [
            f'input[type="checkbox"]:left-of(:text("{text}"):visible)',
            f'input[type="radio"]:left-of(:text("{text}"):visible)',
            f':text("{text}") >> xpath=preceding-sibling::input[@type="checkbox"]',
            f':text("{text}") >> xpath=preceding-sibling::input[@type="radio"]',
        ]

        for pattern in adjacent_patterns:
            try:
                count = await page.locator(pattern).count()
                if count == 1:
                    element = page.locator(pattern).first
                    input_type = await element.get_attribute("type")
                    input_id = await element.get_attribute("id")
                    input_name = await element.get_attribute("name")
                    input_value = await element.get_attribute("value")

                    if input_id:
                        final_locator = f"id={input_id}"
                    elif input_name and input_value:
                        final_locator = (
                            f'input[type="{input_type}"][name="{input_name}"]'
                            f'[value="{input_value}"]'
                        )
                    elif input_name:
                        final_locator = f'input[type="{input_type}"][name="{input_name}"]'
                    else:
                        continue

                    logger.info("checkbox.found_adjacent", input_type=input_type, locator=final_locator)
                    return {
                        "locator": final_locator,
                        "element_type": input_type,
                    }
            except Exception:
                pass
    except Exception as e:
        logger.info("checkbox.adjacent_search_error", error=str(e))

    logger.info("checkbox.finder_not_found", label=text)
    return None
