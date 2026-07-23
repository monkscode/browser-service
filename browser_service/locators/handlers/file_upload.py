"""
File-upload handler (Task E / G5).

Real sites hide ``<input type="file">`` behind a styled button (ASTPP:
``#customer_import_mapper`` is display:none, the visible control an
unlabelled Browse button). Clicking that button in generated RF code
opens the browser's NATIVE file dialog — un-automatable, the test hangs
until timeout. The only robust upload path is ``Upload File By
Selector`` aimed at the input itself; hidden inputs are legal targets
for it (Playwright ``setInputFiles`` semantics).

So unlike every other handler, the correct element to RETURN is not the
element vision clicked: it is the file input the dispatcher's DOM probe
anchored via its nearest-container scan (``anchor_xpath``). This module
turns that anchor into the most stable locator available:

    id → input[type=file][name=...] → unique input[type=file] →
    anchor XPath (positional-scored last resort — demote, never delete)

The payload carries ``element_type="file-upload"`` (the Assembler's
routing key: emit Upload File By Selector, never Click) and
``element_info.hidden_input`` so downstream knows the visibility state.

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
    File-upload handler entry point.

    Resolves the file INPUT to target: from ``element_data`` when the
    clicked element already is the input, otherwise from the probe's
    ``anchor_xpath`` (the nearest-container scan result). Returns the
    standard handler payload, or ``None`` to fall through to the
    generic path. Always-fallback contract: never raises.
    """
    element_data = element_data or {}
    anchor_xpath = (probe_result or {}).get("anchor_xpath", "")

    input_id = ""
    input_name = ""

    is_direct_input = (element_data.get("tagName") or "").lower() == "input" and (
        element_data.get("type") or ""
    ).lower() == "file"
    if is_direct_input:
        input_id = (element_data.get("id") or "").strip()
        input_name = (element_data.get("name") or "").strip()

    if anchor_xpath and not (input_id or input_name):
        try:
            anchor_locator = search_context.locator(f"xpath={anchor_xpath}")
            input_id = (await anchor_locator.get_attribute("id")) or ""
            input_name = (await anchor_locator.get_attribute("name")) or ""
        except Exception as e:
            logger.info("file_upload.anchor_attr_read_failed error=%s", e)

    if not (input_id or input_name or anchor_xpath):
        logger.info("file_upload.no_input_resolved — falling through")
        return None

    # Candidate order mirrors the generic cascade: authored id, then
    # name, then structural uniqueness, then the probe's positional
    # xpath as a marked last resort.
    candidates: list[tuple[str, str]] = []
    if input_id and score_stability("id", input_id) == STABLE:
        candidates.append((f"id={input_id}", "file-input id"))
    elif input_id:
        candidates.append((f"id={input_id}", "file-input id (volatile)"))
    if input_name:
        escaped_name = input_name.replace("\\", "\\\\").replace('"', '\\"')
        candidates.append(
            (
                f'input[type="file"][name="{escaped_name}"]',
                "file-input name",
            )
        )
    candidates.append(('input[type="file"]', "sole file input on page"))
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
                "file_upload.candidate_probe_failed locator=%s error=%s",
                locator,
                e,
            )

    if chosen is None:
        logger.info("file_upload.no_unique_candidate — falling through")
        return None

    locator, strategy = chosen

    # Visibility is informational: Upload File By Selector works on
    # hidden inputs, but downstream warnings/healing want to know.
    hidden = False
    try:
        hidden = not await search_context.locator(locator).is_visible()
    except Exception:
        pass

    logger.info(
        "file_upload.resolved locator=%s strategy=%s hidden=%s",
        locator,
        strategy,
        hidden,
    )
    return build_locator_result(
        element_id=element_id,
        description=element_description,
        best_locator=locator,
        element_type="file-upload",
        strategy_name=strategy,
        classifier_confidence=type_info.confidence,
        classifier_signals=type_info.signals,
        element_info={
            "file_input": True,
            "hidden_input": hidden,
            "source": "file-upload-handler",
        },
    )
