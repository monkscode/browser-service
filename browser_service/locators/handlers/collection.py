"""
Collection element detection and locator generation.

Handles repeatable collections — table rows (`<tr>`), list items (`<li>`),
cards, and other class-pattern collections — for which the locator pipeline
must produce a multi-element selector that Robot Framework iterates over.

Public functions:
    find_locator(...) -> Optional[dict]                                             (async)
        Standard handler entry point — see Section 7 of
        docs/ELEMENT_TYPE_CLASSIFIER_ARCHITECTURE.md.
    _is_collection_element(element_data, element_description) -> bool
    _extract_collection_class(element_data) -> Optional[str]
    _find_collection_locator(page, element_data, collection_class) -> Optional[str]   (async)
    _find_collection_by_text_traversal(page, expected_text) -> Optional[dict]          (async)

Referenced by:
    - browser_service.locators.smart_locator (dispatcher routes
      primary_type=="collection" here; re-exports legacy helpers for
      back-compat with tests)

Depends on:
    - re, typing.Optional (stdlib), structlog
    - Playwright page object (passed in; not imported)
    - browser_service.locators.classifier.ElementTypeInfo (type hint only)
"""

import re
import structlog
from typing import TYPE_CHECKING, Optional

from .base import build_locator_result

if TYPE_CHECKING:
    from ..classifier import ElementTypeInfo

logger = structlog.get_logger(__name__)


def _is_collection_element(element_data: dict, element_description: str) -> bool:
    """
    Detect if element is part of a repeatable collection.
    Works generically without hardcoding specific library classes.

    Detection methods:
    1. Semantic description keywords (rows, items, all, each)
    2. Standard HTML collection tags (tr, li, option)
    3. Common class patterns (row, item, card, entry)

    Args:
        element_data: Dict with element attributes from browser-use DOM
        element_description: Human-readable description from planner

    Returns:
        True if element appears to be part of a collection
    """
    tag = element_data.get('tagName', '').lower()
    class_name = element_data.get('className', '').lower()
    desc = element_description.lower()

    # Method 1: Semantic description keywords
    # Note: LLM may modify descriptions, so we check for various patterns
    desc_keywords = [
        'rows', 'items', 'all ', 'each', 'every', 'list of',
        'visible rows', 'table rows', 'filtered',
        'cells', 'column cell', 'column cells',  # Table column patterns
        'results table', 'data table',  # Table context patterns
    ]
    if any(kw in desc for kw in desc_keywords):
        logger.info("collection.detected_by_description", element_description=element_description)
        return True

    # Method 2: Standard HTML collection tags
    if tag in ['tr', 'li', 'option', 'dt', 'dd']:
        logger.info("collection.detected_by_tag", tag=tag)
        return True

    # Method 3: Common class patterns (generic, not library-specific).
    # Per-token exact match (not substring) so 'nav-item' does not match 'item'.
    # Nav-prefix exclusion: navigation/menu/breadcrumb/etc. items are UI chrome,
    # not data collections — never treat them as collections even if a token matches.
    collection_patterns = {'row', 'item', 'card', 'entry', 'record', 'tr-group', 'list-item', 'grid-item'}
    nav_prefixes = ('nav-', 'menu-', 'tab-', 'breadcrumb-', 'pagination-', 'dropdown-')
    for cls in class_name.split():
        if any(cls.startswith(prefix) for prefix in nav_prefixes):
            continue
        if cls in collection_patterns:
            logger.info("collection.detected_by_class", cls=cls)
            return True

    return False


def _extract_collection_class(element_data: dict) -> Optional[str]:
    """
    Find the most specific class that identifies collection items.

    SMART APPROACH:
    1. Prioritize classes containing semantic patterns (row, item, tr, card, etc.)
    2. Skip short/cryptic utility classes using pattern detection (not hardcoded lists)
    3. Return None if no suitable class found (better to fail than use wrong class)

    Args:
        element_data: Dict with element attributes

    Returns:
        Most appropriate class for collection matching, or None if not suitable
    """

    class_name = element_data.get('className', '')
    if not class_name:
        return None

    classes = class_name.split()

    # PRIORITY 1: Classes containing semantic collection patterns
    # These ARE the actual collection classes we want
    collection_patterns = ['row', 'item', 'tr', 'card', 'entry', 'record', 'group', 'cell', 'list']
    for cls in classes:
        cls_lower = cls.lower()
        for pattern in collection_patterns:
            if pattern in cls_lower:
                logger.info("collection.class_extracted", cls=cls, pattern=pattern)
                return cls

    # PRIORITY 2: Skip utility-like classes using pattern detection
    # Utility classes typically: short (<=5 chars) OR follow letter-number pattern (mt-4, px-12)
    for cls in classes:
        # Skip if too short (likely utility: mt-4, p-2, d-flex are ~5 chars or less)
        if len(cls) <= 5:
            continue
        # Skip if matches pattern: 1-4 letters + hyphen + number (e.g., mt-4, px-12, col-6)
        if re.match(r'^[a-z]{1,4}-\d+$', cls.lower()):
            continue
        # Skip if matches pattern: single letter + hyphen (e.g., d-flex, m-auto)
        if re.match(r'^[a-z]-', cls.lower()):
            continue
        # This looks like a meaningful class name
        logger.info("collection.class_component_like", cls=cls)
        return cls

    # No suitable class found - better to return None than use wrong class
    logger.info("collection.no_class_found", class_name=class_name)
    return None


async def _find_collection_locator(page, element_data: dict, collection_class: str) -> Optional[str]:
    """
    Build a locator that matches all items in the collection.
    Tries multiple container strategies to find the most reliable locator.

    Args:
        page: Playwright page object
        element_data: Dict with element attributes
        collection_class: The class that identifies collection items

    Returns:
        Multi-element locator string, or None if not found
    """
    tag = element_data.get('tagName', '').lower()

    # Strategy 1: Standard HTML table - use tbody tr
    if tag == 'tr':
        candidates = [
            'tbody tr',
            'table tr:not(:first-child)',  # Skip header row
            'tbody > tr'
        ]
        for locator in candidates:
            try:
                count = await page.locator(locator).count()
                if count > 1:
                    logger.info("collection.table_row_locator", locator=locator, count=count)
                    return locator
            except Exception:
                continue

    # Strategy 2: Standard HTML list - use ul/ol li
    if tag == 'li':
        candidates = ['ul li', 'ol li', 'ul > li', 'ol > li']
        for locator in candidates:
            try:
                count = await page.locator(locator).count()
                if count > 1:
                    logger.info("collection.list_item_locator", locator=locator, count=count)
                    return locator
            except Exception:
                continue

    # Strategy 3: Class-based locator with common container patterns
    container_patterns = [
        f'.{collection_class}',  # Token-exact class selector
    ]

    # Try adding common parent containers
    parent_containers = ['tbody', '.table-body', '.list', '.grid', '.container',
                        '[class*="body"]', '[class*="content"]', '[class*="list"]']

    for parent in parent_containers:
        container_patterns.append(f'{parent} .{collection_class}')

    for locator in container_patterns:
        try:
            count = await page.locator(locator).count()
            if count > 1:
                logger.info("collection.class_locator_found", locator=locator, count=count)
                return locator
        except Exception:
            continue

    # Fallback: Just the class (might include non-data rows)
    fallback = f'.{collection_class}'
    try:
        count = await page.locator(fallback).count()
        if count > 1:
            logger.info("collection.fallback_locator", locator=fallback, count=count)
            return fallback
    except Exception:
        pass

    return None


async def _find_collection_by_text_traversal(page, expected_text: str) -> Optional[dict]:
    """
    Find collection (table rows, list items) by using expected_text as a beacon.

    This is the PRIMARY method for finding collections. It works by:
    1. Finding an element containing the expected_text
    2. Traversing UP to find the row/item container
    3. Looking for siblings with similar structure
    4. Generating a collection locator for all matching elements

    This approach works even when element_index is invalid (common for non-interactive elements).

    Args:
        page: Playwright page object
        expected_text: Text that should be in one of the collection items (e.g., "Cierra")

    Returns:
        Dict with 'locator', 'count', 'row_class' if found, None otherwise
    """
    if not expected_text or len(expected_text.strip()) < 2:
        return None

    text = expected_text.strip()
    logger.info("collection.text_traversal_start", text=text)

    try:
        # Step 1: Find the beacon element. Prefer an exact text match — a
        # substring beacon can start the walk from the wrong instance (A4).
        # browser-use truncates long expected_text with a literal ellipsis
        # ("A Light in the ..."), so a truncated beacon prefix-matches via
        # the substring engine instead.
        needle = text
        truncated = needle.endswith("...") or needle.endswith("…")
        if truncated:
            needle = needle.rstrip(".…").strip()
            if not needle:
                logger.info("collection.beacon_empty_after_truncation", text=text)
                return None

        text_locator = None
        if not truncated and '"' not in needle:
            exact = page.locator(f'text="{needle}"').first
            if await exact.count() > 0:
                text_locator = exact

        if text_locator is None:
            text_locator = page.locator(f"text={needle}").first
        count = await text_locator.count()

        if count == 0:
            logger.info("collection.text_not_found", text=text)
            return None

        # Step 2: Traverse UP to find the row container using JavaScript.
        # Semantic class matching is per hyphen/underscore SEGMENT — the old
        # substring regex made 'flex-grow-1' a row container ("g-r-o-w").
        # Framework layout classes (row, col-*, flex utilities …) are never
        # emitted and never count toward sibling similarity (A4).
        row_info = await text_locator.evaluate("""
            (el) => {
                const isLayoutClass = (cls) => {
                    const c = cls.toLowerCase();
                    return ['row', 'rows', 'container', 'grid', 'flex', 'wrapper'].includes(c) ||
                        /^col(-|$)/.test(c) ||
                        /^(flex|align|justify|order|offset)-/.test(c) ||
                        /^[a-z]{1,2}-/.test(c);
                };
                const SEMANTIC = ['row', 'tr', 'item', 'record', 'entry', 'card', 'group'];
                const hasSemanticSegment = (cls) =>
                    cls.toLowerCase().split(/[-_]/).some(seg => SEMANTIC.includes(seg));
                const cssSafe = (s) => /^[A-Za-z][A-Za-z0-9_-]*$/.test(s);

                let current = el;
                while (current && current.parentElement) {
                    current = current.parentElement;
                    const tag = current.tagName.toLowerCase();
                    if (tag === 'body' || tag === 'html') break;
                    const className = (typeof current.className === 'string' ? current.className : '') || '';
                    const classes = className.split(/\\s+/).filter(c => c.length > 0);
                    const role = current.getAttribute('role') || '';

                    const structural = tag === 'tr' || tag === 'li' ||
                        role === 'row' || role === 'listitem';
                    if (!structural && !classes.some(hasSemanticSegment)) continue;

                    // A non-structural element with nothing but layout
                    // classes has no semantic anchor - not a data row.
                    const meaningful = classes.filter(c => !isLayoutClass(c));
                    if (!structural && meaningful.length === 0) continue;

                    const parent = current.parentElement;
                    if (!parent) continue;

                    // Collection = >1 siblings sharing the row's meaningful
                    // class set. Same tag alone is not enough - a lone
                    // styled banner next to unrelated same-tag panels is
                    // not a collection.
                    const similar = Array.from(parent.children).filter(s =>
                        s.tagName === current.tagName &&
                        meaningful.every(c => s.classList.contains(c))
                    );
                    if (similar.length <= 1) continue;

                    // Emitted class: semantic first, then any meaningful
                    // class long enough to be a component name.
                    const bestClass =
                        meaningful.find(c => hasSemanticSegment(c) && cssSafe(c)) ||
                        meaningful.find(c => c.length > 5 && cssSafe(c)) ||
                        null;

                    // Scope anchor: the row's own parent (id > class > tag).
                    let parentAnchor = null;
                    const ptag = parent.tagName.toLowerCase();
                    if (parent.id && cssSafe(parent.id)) {
                        parentAnchor = '#' + parent.id;
                    } else {
                        const pcn = (typeof parent.className === 'string' ? parent.className : '') || '';
                        const parentClass = pcn.split(/\\s+/).filter(c => c.length > 0)
                            .find(c => !isLayoutClass(c) && cssSafe(c));
                        parentAnchor = parentClass ? '.' + parentClass :
                            (ptag !== 'body' && ptag !== 'html' ? ptag : null);
                    }

                    return {
                        tag: tag,
                        className: bestClass,
                        allClasses: className,
                        parentAnchor: parentAnchor,
                        siblingCount: similar.length,
                        role: role,
                        parentTag: ptag
                    };
                }
                return null;
            }
        """)

        if row_info:
            logger.info("collection.row_container_found", tag=row_info["tag"], cls=row_info.get("className") or "")
            logger.info("collection.sibling_count", count=row_info["siblingCount"])

            # Generate collection locator, scoped to the row's own parent —
            # never a bare page-global class (A4: `.row` matched every grid
            # row on the page, chrome included).
            tag = row_info.get('tag', '')
            best_class = row_info.get('className')
            anchor = row_info.get('parentAnchor')

            if best_class:
                suffix = f"{tag}.{best_class}"
            elif tag in ('tr', 'li'):
                suffix = tag
            elif row_info.get('role') == 'row':
                suffix = '[role="row"]'
            else:
                logger.info("collection.indeterminate_locator", row_info=str(row_info))
                return None

            if anchor:
                locator = f"{anchor} > {suffix}"
            elif best_class or row_info.get('role') == 'row':
                locator = suffix
            elif tag == 'tr':
                locator = 'tbody tr'
            else:  # tag == 'li'
                locator = 'ul li, ol li'

            # Validate the locator
            try:
                count = await page.locator(locator).count()
                if count >= 1:
                    logger.info("collection.traversal_locator_validated", locator=locator, count=count)
                    return {
                        'locator': locator,
                        'count': count,
                        'row_class': row_info.get('className'),
                        'tag': row_info.get('tag'),
                        'source': 'text_traversal'
                    }
                else:
                    logger.info("collection.locator_zero_matches", locator=locator)
            except Exception as e:
                logger.info("collection.traversal_validation_failed", error=str(e))
        else:
            logger.info("collection.traversal_no_row_container")

        return None

    except Exception as e:
        logger.warning("collection.text_traversal_failed", error=str(e))
        return None


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
    Collection handler entry point — dispatched from smart_locator when
    classifier returns primary_type="collection".

    Tries text-traversal first (uses expected_text as a beacon to find the
    real row container — robust when element_data is from a wrong outer
    container). Falls back to element_data class extraction. Returns None
    on failure so the caller falls through to the generic 21-strategy.

    The dropdown veto (Bug 3) is preserved here even though Tier 0 of the
    classifier catches most Tom Select cases by class. When browser-use
    coords land on a Bootstrap `.row` outer wrapper instead of the
    `.ts-control` input itself, the classifier will return "collection";
    this guard prevents that misclassification from producing a `.row`
    locator for what is actually a dropdown.
    """
    # Bug 3 guard: dropdown signals override collection routing.
    # Local import avoids the cycle smart_locator <-> handlers.collection.
    from ..smart_locator import is_dropdown_element

    if is_dropdown_element(element_data, element_description):
        logger.info("collection.handler_dropdown_veto")
        return None

    # PRIMARY METHOD: text-traversal using expected_text as a beacon.
    if expected_text:
        text_result = await _find_collection_by_text_traversal(
            search_context, expected_text
        )

        if (
            text_result
            and text_result.get("locator")
            and text_result.get("count", 0) > 1
        ):
            collection_locator = text_result["locator"]
            count = text_result["count"]

            logger.info("collection.locator_found_text_traversal", locator=collection_locator, count=count)
            logger.info("collection.skip_semantic_validation")

            result = build_locator_result(
                element_id=element_id,
                description=element_description,
                best_locator=collection_locator,
                element_type="collection",
                strategy_name="Text-traversal collection locator",
                classifier_confidence=type_info.confidence,
                classifier_signals=type_info.signals,
                unique=False,
                count=count,
                all_locator_extra={"quality_score": 90},
                validation_summary_extra={
                    "multi_element": True,
                    "collection_count": count,
                },
                quality_score=90,
                element_info={
                    "tagName": text_result.get("tag", ""),
                    "className": text_result.get("row_class", ""),
                    "collection_class": text_result.get("row_class", ""),
                    "source": "text_traversal",
                },
            )
            result["count"] = count
            return result
        elif text_result:
            logger.info(
                "collection.traversal_single_match_rejected",
                count=text_result.get("count", 0),
            )
        else:
            logger.info("collection.traversal_miss_fallback_to_element_data")

    # FALLBACK METHOD: element_data class extraction.
    collection_class = _extract_collection_class(element_data)
    if not collection_class:
        logger.info("collection.no_class_extracted")
        return None

    logger.info("collection.class_from_element_data", collection_class=collection_class)

    collection_locator = await _find_collection_locator(
        search_context, element_data, collection_class
    )
    if not collection_locator:
        return None

    try:
        count = await search_context.locator(collection_locator).count()
    except Exception as e:
        logger.warning("collection.locator_validation_failed", error=str(e))
        return None

    if count <= 1:
        return None

    # Validate that matched elements contain expected_text.
    if expected_text:
        text_found = False
        try:
            for i in range(min(count, 3)):
                el_text = (
                    await search_context.locator(collection_locator)
                    .nth(i)
                    .text_content()
                ) or ""
                if expected_text.lower() in el_text.lower():
                    text_found = True
                    break
        except Exception:
            pass

        if not text_found:
            logger.warning("collection.locator_text_mismatch", locator=collection_locator, expected_text=expected_text)
            logger.warning("collection.locator_rejected_wrong_elements")
            return None

    logger.info("collection.locator_found", locator=collection_locator, count=count)

    result = build_locator_result(
        element_id=element_id,
        description=element_description,
        best_locator=collection_locator,
        element_type="collection",
        strategy_name="Element-data collection locator",
        classifier_confidence=type_info.confidence,
        classifier_signals=type_info.signals,
        unique=False,
        count=count,
        all_locator_extra={"quality_score": 85},
        validation_summary_extra={
            "multi_element": True,
            "collection_count": count,
        },
        quality_score=85,
        element_info={
            "tagName": element_data.get("tagName", ""),
            "className": element_data.get("className", ""),
            "collection_class": collection_class,
            "source": "element_data_collection",
        },
    )
    result["count"] = count
    return result
