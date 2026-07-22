"""
Date-picker handler (Task D / G4).

Real sites render date filters as READONLY flatpickr inputs (ASTPP:
``#customer_cdr_from_date``, class ``flatpickr-input``, instance on
``el._flatpickr``). ``Fill Text`` on a readonly input waits for
editability forever — the generated test dies with a timeout at that
step, every run. The only robust path is the widget's own API:
``el._flatpickr.setDate(value, true)`` — one ``Evaluate JavaScript``
line in the generated code (verified live on the ASTPP site 2026-07-08,
including a date-only value against its 'Y-m-d H:i:S' format).

The handler's job is the Assembler contract, not a new locator cascade:
resolve the input that CARRIES the flatpickr instance (the candidate
itself, or the probe's nearest-container anchor when vision clicked a
calendar toggle button), emit its most stable locator, and stamp
``element_type="date-picker"`` + top-level
``datepicker_framework="flatpickr"`` (the dropdown_framework/select_id
pipe precedent — ``element_info`` is never forwarded by the nlrf
identify agent).

Commits ONLY on flatpickr. Native ``input[type=date]`` falls through to
the generic path unchanged — plain ``Fill Text`` already works there
(demote-never-delete: no behavior change without evidence of need).

    id → input.flatpickr-input[name=...] → sole input.flatpickr-input →
    anchor XPath (positional-scored last resort)

Referenced by:
    - browser_service.locators.smart_locator (dispatcher)

Depends on:
    - browser_service.locators.handlers.base (result factory)
    - browser_service.locators.stability (id stability check)
"""

import logging
from typing import Optional

from browser_service.locators.handlers.base import build_locator_result
from browser_service.locators.stability import STABLE, score_stability

logger = logging.getLogger(__name__)


async def find_locator(
    page,
    element_data: dict,
    type_info,
    element_id: str,
    element_description: str,
    expected_text: Optional[str],
    search_context,
    iframe_context: Optional[str],
    confirmed_coords: Optional[tuple],
    probe_result: Optional[dict] = None,
) -> Optional[dict]:
    """
    Date-picker handler entry point.

    Resolves the flatpickr INPUT to target: from ``element_data`` when
    the clicked element already is the input, otherwise from the probe's
    ``anchor_xpath`` (the nearest-container scan result). Returns the
    standard handler payload, or ``None`` to fall through to the
    generic path. Always-fallback contract: never raises.
    """
    element_data = element_data or {}

    # Only flatpickr has a specialized emission today. Native
    # input[type=date] works with plain Fill Text — generic path
    # unchanged.
    framework = (type_info.framework or "").lower()
    if framework != "flatpickr":
        logger.info(
            "date_picker.framework_not_specialized framework=%s — falling through",
            framework or "(none)",
        )
        return None

    anchor_xpath = (probe_result or {}).get("anchor_xpath", "")

    input_id = ""
    input_name = ""

    is_direct_input = (
        (element_data.get("tagName") or "").lower() == "input"
        and "flatpickr-input"
        in ((element_data.get("className") or "").lower().split())
    )
    if is_direct_input:
        input_id = (element_data.get("id") or "").strip()
        input_name = (element_data.get("name") or "").strip()

    if anchor_xpath and not (input_id or input_name):
        try:
            anchor_locator = search_context.locator(f"xpath={anchor_xpath}")
            input_id = (await anchor_locator.get_attribute("id")) or ""
            input_name = (await anchor_locator.get_attribute("name")) or ""
        except Exception as e:
            logger.info("date_picker.anchor_attr_read_failed error=%s", e)

    if not (input_id or input_name or anchor_xpath):
        logger.info("date_picker.no_input_resolved — falling through")
        return None

    # Candidate order mirrors the generic cascade: authored id, then
    # name scoped to the flatpickr class (ASTPP's from/to inputs SHARE
    # name="callstart[]" — the uniqueness check below rejects that),
    # then structural uniqueness, then the probe's positional xpath as
    # a marked last resort.
    candidates: list[tuple[str, str]] = []
    if input_id and score_stability("id", input_id) == STABLE:
        candidates.append((f"id={input_id}", "flatpickr-input id"))
    elif input_id:
        candidates.append((f"id={input_id}", "flatpickr-input id (volatile)"))
    if input_name:
        escaped_name = input_name.replace('\\', '\\\\').replace('"', '\\"')
        candidates.append((
            f'input.flatpickr-input[name="{escaped_name}"]',
            "flatpickr-input name",
        ))
    candidates.append(("input.flatpickr-input", "sole flatpickr input on page"))
    if anchor_xpath:
        candidates.append((f"xpath={anchor_xpath}", "probe anchor xpath"))

    chosen = None
    for locator, strategy in candidates:
        try:
            if await search_context.locator(locator).count() == 1:
                chosen = (locator, strategy)
                break
        except Exception as e:
            logger.info(
                "date_picker.candidate_probe_failed locator=%s error=%s",
                locator, e,
            )

    if chosen is None:
        logger.info("date_picker.no_unique_candidate — falling through")
        return None

    locator, strategy = chosen

    # readonly is informational: the setDate idiom works either way, but
    # downstream warnings/healing want to know why Fill Text was not used.
    readonly = False
    try:
        readonly = (
            await search_context.locator(locator).get_attribute("readonly")
        ) is not None
    except Exception:
        pass

    logger.info(
        "date_picker.resolved locator=%s strategy=%s readonly=%s",
        locator, strategy, readonly,
    )
    return build_locator_result(
        element_id=element_id,
        description=element_description,
        best_locator=locator,
        element_type="date-picker",
        strategy_name=strategy,
        classifier_confidence=type_info.confidence,
        classifier_signals=type_info.signals,
        datepicker_framework="flatpickr",
        # Explicit empty: _attach_classifier_metadata setdefault would
        # otherwise stamp dropdown_framework="flatpickr" and misroute the
        # Assembler's dropdown table.
        dropdown_framework="",
        element_info={
            "date_input": True,
            "readonly": readonly,
            "source": "date-picker-handler",
        },
    )
