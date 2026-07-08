"""
Playwright DOM Structural Probe
===============================

Asks the live page DOM whether a candidate element actually has the
structural pattern of a specialized type (dropdown / checkbox / radio /
collection) — independent of any LLM classification or planner
description.

This is the second source of truth in the two-source verification model
(see ``docs/ELEMENT_TYPE_CLASSIFIER_ARCHITECTURE.md`` §5 + the
two-source-of-truth design discussion). The first source is the
classifier's verdict (Tier 0 + Tier 1 + optional vision hint from
browser-use). The second source — this probe — runs a JS function in
the page context that walks the DOM around the candidate element and
returns a structural confirmation.

When both sources agree, we commit to the specialized handler. When
they disagree, we fall through to the generic 21-strategy path. No
specialized handler runs on a unilateral classification.

Two operating modes:
  - **Confirmation**: classifier says ``dropdown`` → probe verifies
    structural signals exist before the dropdown handler runs.
  - **Discovery**: classifier says ``unknown`` but vision hint suggests
    ``dropdown`` → probe runs anyway; if structure is found, the verdict
    is promoted from ``unknown`` to the suspected type.

Cost: one ``page.evaluate`` call per probe — typically 1–5ms even on
heavy pages. No network, no LLM, no $.

Walk scope (intentionally wide for completeness, structurally bounded
to avoid cross-element false positives):

  1. Self + ALL descendants (depth-unlimited)
  2. ALL ancestors up to ``<body>`` (cap depth=8 for safety)
  3. Each ancestor's immediate prev/next siblings + their descendants
  4. ``aria-controls`` / ``aria-labelledby`` / ``aria-describedby``
     target lookups
  5. ``<label for=X>`` backlinks when the candidate has an ``id``
  6. Open shadow roots on any visited element (Web Components like
     ``<fancy-dropdown>`` whose ``role=combobox`` lives inside the
     shadow root). Closed shadow roots are still opaque by design.
  7. ``elementFromPoint(x±10, y±10)`` for coord-jitter robustness

Return shape (always a dict, never raises):
    {
        "confirmed": bool,           # at least one type-specific signal fired
        "framework": str,            # framework name when detected, else ""
        "signals": list[str],        # debug trail: ["self:role=combobox", ...]
        "anchor_xpath": str,         # XPath of the element that confirmed
        "anchor_tag": str,           # lowercased tagName of the anchor
    }

The ``anchor_xpath`` lets handlers re-target when the probe found the
structure on an ancestor / sibling instead of the original click node
(Bug 3 pattern: coords land on outer ``.row``, structure lives on the
``.ts-wrapper`` adjacent to ancestor's sibling).

Referenced by:
    - browser_service.locators.smart_locator (dispatcher invokes probe
      between classifier verdict and specialized handler dispatch)

Depends on:
    - Playwright Page (passed in by caller)
    - browser_service.locators.classifier (consumes the same framework
      patterns to keep the two modules in sync)
"""

import json
import logging
from typing import Any, Optional

from .classifier import _DROPDOWN_FRAMEWORK_PATTERNS

logger = logging.getLogger(__name__)


# Allowed suspected_type values. ``probe_specialized_type`` returns the
# empty result for anything else — no JS is even executed.
_SUPPORTED_TYPES: frozenset[str] = frozenset({
    "dropdown", "checkbox", "radio", "collection", "file-upload",
    "date-picker",
})

# Result shape returned when the probe cannot run (no page, bad type,
# JS error). Dispatcher treats unconfirmed as "trust the classifier on
# its own" — same default behavior as today.
_UNCONFIRMED: dict[str, Any] = {
    "confirmed": False,
    "framework": "",
    "signals": [],
    "anchor_xpath": "",
    "anchor_tag": "",
}


# JS framework-pattern table — derived from classifier's
# _DROPDOWN_FRAMEWORK_PATTERNS so we don't maintain two lists. JSON-
# encoded once at import time.
_FRAMEWORK_PATTERNS_JSON = json.dumps([
    [name, list(patterns)]
    for name, patterns in _DROPDOWN_FRAMEWORK_PATTERNS
])


# Single JS function. Takes (coords, suspectedType, frameworkPatterns,
# candidateXPath?). When candidateXPath is provided, the candidate is
# located by XPath (more reliable than coords for elements browser-use
# already identified). When only coords are provided, falls back to
# elementFromPoint + jitter.
_PROBE_JS = r"""
(args) => {
    const { coords, suspectedType, frameworkPatterns, candidateXPath } = args;

    // ---- Locate candidate element ----
    let candidate = null;

    if (candidateXPath) {
        try {
            const r = document.evaluate(
                candidateXPath, document, null,
                XPathResult.FIRST_ORDERED_NODE_TYPE, null
            );
            candidate = r.singleNodeValue;
        } catch (e) { /* XPath failed; fall through */ }
    }

    if (!candidate && coords) {
        candidate = document.elementFromPoint(coords.x, coords.y);
        // Coord jitter ±10px in 4 directions
        if (!candidate) {
            const offsets = [[-10,0],[10,0],[0,-10],[0,10],[-10,-10],[10,10]];
            for (const [dx, dy] of offsets) {
                candidate = document.elementFromPoint(coords.x+dx, coords.y+dy);
                if (candidate) break;
            }
        }
    }

    if (!candidate) {
        return { confirmed: false, framework: '', signals: ['no-element'],
                 anchor_xpath: '', anchor_tag: '' };
    }

    // ---- file-upload: nearest-container input[type=file] scan (G5) ----
    // The generic walk below adds ancestor NODES without their child
    // trees, so a file input that is the candidate's SIBLING (the
    // standard styled-button shape) is invisible to it — while the
    // ancestor-sibling rings WOULD see a NEIGHBORING widget's input and
    // false-positive. File upload needs same-container semantics: climb
    // at most 4 scopes (self + 3 ancestors), never scan body-wide. The
    // anchor is the input itself so the handler can emit ITS locator —
    // hidden file inputs are legal Upload File By Selector targets.
    if (suspectedType === 'file-upload') {
        let hit = null;
        let scopeLabel = '';
        if (candidate.tagName === 'INPUT' &&
            (candidate.type === 'file' || candidate.getAttribute('type') === 'file')) {
            hit = candidate;
            scopeLabel = 'self';
        } else {
            let scope = candidate;
            for (let d = 0; d < 4 && scope && scope !== document.body; d++) {
                const found = scope.querySelector &&
                    scope.querySelector('input[type="file"]');
                if (found) { hit = found; scopeLabel = 'container[' + d + ']'; break; }
                scope = scope.parentElement;
            }
        }
        if (!hit) {
            return { confirmed: false, framework: '',
                     signals: ['file-upload:no-input-in-container'],
                     anchor_xpath: '', anchor_tag: '' };
        }
        return { confirmed: true, framework: 'native',
                 signals: [scopeLabel + ':tag:input[type=file]'],
                 anchor_xpath: xpathOf(hit), anchor_tag: 'input' };
    }

    // ---- date-picker: nearest-container date-input scan (G4) ----
    // Same-container semantics as the file-upload scan above: flatpickr
    // often hides behind a calendar toggle BUTTON (wrap mode), so the
    // input to anchor is the candidate's SIBLING — invisible to the
    // generic walk, while ancestor-sibling rings would false-positive on
    // a NEIGHBORING widget's input. Climb at most 4 scopes, never scan
    // body-wide. The anchor is the input itself: flatpickr stamps class
    // 'flatpickr-input' + property el._flatpickr on the REAL input (the
    // only legal setDate target); input[type=date] is plain native.
    if (suspectedType === 'date-picker') {
        function isDateInput(n) {
            if (!n || n.tagName !== 'INPUT') return false;
            const c = ((n.className && n.className.toString &&
                        n.className.toString()) || '').toLowerCase();
            return !!n._flatpickr ||
                   c.split(/\s+/).indexOf('flatpickr-input') !== -1 ||
                   n.type === 'date' || n.getAttribute('type') === 'date';
        }
        let hit = null;
        let scopeLabel = '';
        if (isDateInput(candidate)) {
            hit = candidate;
            scopeLabel = 'self';
        } else {
            let scope = candidate;
            for (let d = 0; d < 4 && scope && scope !== document.body; d++) {
                const found = scope.querySelector &&
                    scope.querySelector('input.flatpickr-input, input[type="date"]');
                if (found) { hit = found; scopeLabel = 'container[' + d + ']'; break; }
                scope = scope.parentElement;
            }
        }
        if (!hit) {
            return { confirmed: false, framework: '',
                     signals: ['date-picker:no-input-in-container'],
                     anchor_xpath: '', anchor_tag: '' };
        }
        const hitCls = ((hit.className && hit.className.toString &&
                         hit.className.toString()) || '').toLowerCase();
        const isFlatpickr = !!hit._flatpickr ||
            hitCls.split(/\s+/).indexOf('flatpickr-input') !== -1;
        return { confirmed: true,
                 framework: isFlatpickr ? 'flatpickr' : 'native',
                 signals: [scopeLabel + (isFlatpickr
                     ? ':flatpickr-input' : ':tag:input[type=date]')],
                 anchor_xpath: xpathOf(hit), anchor_tag: 'input' };
    }

    // ---- Build candidate set: self + descendants + ancestors +
    // ancestor siblings (+ their descendants) + aria-targets + label-for
    // + open shadow roots ----
    const visited = new Set();
    const queue = [];
    // Bound shadow-root recursion. A Web Component graph can chain
    // hosts arbitrarily; cap depth so a pathological page can't make
    // the probe pathologically slow.
    const SHADOW_DEPTH_CAP = 4;

    function add(el, label, shadowDepth) {
        if (!el || el.nodeType !== 1 || visited.has(el)) return;
        visited.add(el);
        queue.push({ el, label });
        // Pierce open shadow roots when present. Closed shadow roots
        // return null on .shadowRoot and are intentionally opaque
        // (browser security boundary).
        if (el.shadowRoot && shadowDepth < SHADOW_DEPTH_CAP) {
            for (const c of el.shadowRoot.children) {
                addAllInTree(c, `${label}.shadow`, shadowDepth + 1);
            }
        }
    }

    function addAllInTree(el, label, shadowDepth) {
        // Walk an element + its descendants, descending into any open
        // shadow roots encountered. Used wherever the previous version
        // used querySelectorAll('*'); querySelectorAll doesn't pierce
        // shadow roots, so we recurse by hand via children.
        if (!el || el.nodeType !== 1) return;
        add(el, label, shadowDepth);
        for (const c of el.children) {
            addAllInTree(c, `${label}.descendant`, shadowDepth);
        }
    }

    // Self + ALL descendants (with shadow piercing)
    addAllInTree(candidate, 'self', 0);

    // Ancestors up to body, with sibling rings
    let cur = candidate.parentElement;
    let depth = 0;
    while (cur && cur !== document.body && depth < 8) {
        add(cur, `ancestor[${depth}]`, 0);
        // Immediate prev/next siblings + their full descendant trees
        for (const sib of [cur.previousElementSibling, cur.nextElementSibling]) {
            if (!sib) continue;
            addAllInTree(sib, `ancestor[${depth}].sib`, 0);
        }
        cur = cur.parentElement;
        depth++;
    }

    // aria-controls / aria-labelledby / aria-describedby target lookups
    for (const attr of ['aria-controls', 'aria-labelledby', 'aria-describedby']) {
        const v = candidate.getAttribute && candidate.getAttribute(attr);
        if (!v) continue;
        for (const id of v.split(/\s+/)) {
            const t = id && document.getElementById(id);
            if (t) add(t, `aria-target:${attr}`, 0);
        }
    }

    // label[for=X] backlinks when candidate has an id
    if (candidate.id) {
        try {
            const labels = document.querySelectorAll(
                'label[for="' + (CSS && CSS.escape ? CSS.escape(candidate.id) : candidate.id) + '"]'
            );
            for (const l of labels) {
                add(l, 'label-for', 0);
                // Also check what the label is structurally near
                let lp = l.parentElement;
                if (lp) addAllInTree(lp, 'label-for.parent', 0);
            }
        } catch (e) { /* invalid CSS escape; skip */ }
    }

    // ---- Type-specific signal checks ----
    function checksFor(el, type) {
        const out = [];
        const role = el.getAttribute && el.getAttribute('role');
        const tag = el.tagName;
        const cls = (el.className && el.className.toString && el.className.toString().toLowerCase()) || '';

        if (type === 'dropdown') {
            if (tag === 'SELECT') out.push({ signal: 'tag:select', framework: 'native' });
            if (role === 'combobox') out.push({ signal: 'role:combobox', framework: '' });
            if (role === 'listbox') out.push({ signal: 'role:listbox', framework: '' });
            if (el.hasAttribute && el.hasAttribute('aria-haspopup')) {
                const v = el.getAttribute('aria-haspopup');
                if (['listbox', 'menu', 'tree', 'true', ''].includes(v)) {
                    out.push({ signal: `aria-haspopup=${v||'true'}`, framework: '' });
                }
            }
            if (el.hasAttribute && el.hasAttribute('aria-controls')) {
                const tid = el.getAttribute('aria-controls');
                const t = tid && document.getElementById(tid);
                if (t) {
                    const tr = t.getAttribute && t.getAttribute('role');
                    const tcls = (t.className && t.className.toString && t.className.toString().toLowerCase()) || '';
                    if (tr === 'listbox' || tr === 'menu' ||
                        tcls.includes('dropdown-menu') || tcls.includes('ts-dropdown')) {
                        out.push({ signal: 'aria-controls→popup', framework: '' });
                    }
                }
            }
            // Framework class scan (uses table passed in from Python)
            for (const [framework, patterns] of frameworkPatterns) {
                for (const p of patterns) {
                    if (cls.indexOf(p) !== -1) {
                        out.push({ signal: `class:${p}`, framework });
                        break;
                    }
                }
            }
        } else if (type === 'checkbox') {
            if (tag === 'INPUT' && (el.type === 'checkbox' || el.getAttribute('type') === 'checkbox')) {
                out.push({ signal: 'tag:input[type=checkbox]', framework: 'native' });
            }
            if (role === 'checkbox') out.push({ signal: 'role:checkbox', framework: 'custom' });
            if (role === 'switch') out.push({ signal: 'role:switch', framework: 'toggle' });
            if (el.hasAttribute && el.hasAttribute('aria-checked')) {
                out.push({ signal: 'aria-checked', framework: '' });
            }
        } else if (type === 'radio') {
            if (tag === 'INPUT' && (el.type === 'radio' || el.getAttribute('type') === 'radio')) {
                out.push({ signal: 'tag:input[type=radio]', framework: 'native' });
            }
            if (role === 'radio') out.push({ signal: 'role:radio', framework: 'custom' });
            if (role === 'radiogroup') out.push({ signal: 'role:radiogroup', framework: '' });
        } else if (type === 'collection') {
            if (tag === 'TR') out.push({ signal: 'tag:tr', framework: 'table-row' });
            if (tag === 'LI') out.push({ signal: 'tag:li', framework: 'list-item' });
            if (tag === 'TBODY' || tag === 'THEAD') out.push({ signal: `tag:${tag.toLowerCase()}`, framework: '' });
            if (role === 'row') out.push({ signal: 'role:row', framework: 'table-row' });
            if (role === 'listitem') {
                const navPrefixes = ['nav-', 'menu-', 'tab-', 'breadcrumb-', 'pagination-', 'dropdown-'];
                const liCls = ((el.className && el.className.toString && el.className.toString()) || '').toLowerCase();
                const isNavLi = liCls.split(/\s+/).some(c => navPrefixes.some(p => c.startsWith(p)));
                if (!isNavLi) out.push({ signal: 'role:listitem', framework: 'list-item' });
            }
            if (role === 'grid' || role === 'gridcell') out.push({ signal: `role:${role}`, framework: '' });
        }
        return out;
    }

    // ---- Run checks across all candidates, gather signals ----
    const allSignals = [];
    let framework = '';
    let anchor = null;
    let anchorLabel = '';

    for (const { el, label } of queue) {
        const matches = checksFor(el, suspectedType);
        for (const m of matches) {
            allSignals.push(`${label}:${m.signal}`);
            if (m.framework && !framework) framework = m.framework;
            // Prefer a wrapper/ancestor anchor over the self anchor: the outer
            // element is usually the better re-anchor target for handler
            // re-targeting. Upgrade if self matched first and a non-self fires later.
            if (!anchor || (anchorLabel.startsWith('self') && !label.startsWith('self'))) {
                anchor = el;
                anchorLabel = label;
            }
        }
    }

    // ---- Build XPath of anchor for handler re-targeting ----
    function xpathOf(el) {
        if (!el || el === document.body) return '';
        const parts = [];
        while (el && el !== document.body && el.parentElement) {
            const tag = el.tagName.toLowerCase();
            // Position among same-tag siblings
            let idx = 1, n = 0, found = false;
            for (const c of el.parentElement.children) {
                if (c.tagName.toLowerCase() === tag) {
                    n++;
                    if (c === el) { idx = n; found = true; }
                }
            }
            parts.unshift(`${tag}[${idx}]`);
            el = el.parentElement;
        }
        return '/html/body/' + parts.join('/');
    }

    return {
        confirmed: allSignals.length > 0,
        framework,
        signals: allSignals.slice(0, 30),  // cap log noise
        anchor_xpath: xpathOf(anchor),
        anchor_tag: anchor ? anchor.tagName.toLowerCase() : '',
    };
}
"""


async def probe_specialized_type(
    page,
    suspected_type: str,
    coords: Optional[tuple[float, float]] = None,
    candidate_xpath: Optional[str] = None,
) -> dict[str, Any]:
    """
    Ask the live page whether structural signals for ``suspected_type``
    exist on or around the candidate element.

    At least one of ``coords`` or ``candidate_xpath`` must be provided.
    When both are provided, ``candidate_xpath`` is preferred (more
    reliable than coords; coords is fallback for jitter cases).

    Args:
        page: Playwright Page (page-level, not frame_locator — page is
            needed for ``page.evaluate``). Iframe-scoped probes should
            use the iframe's own page reference if available, otherwise
            the probe runs against the top page.
        suspected_type: One of "dropdown", "checkbox", "radio",
            "collection". Other values return ``_UNCONFIRMED`` without
            running any JS.
        coords: ``(x, y)`` of the click point from browser-use. Used
            via ``elementFromPoint`` (with ±10px jitter on miss).
        candidate_xpath: Optional precise XPath of the candidate element
            (preferred when known — e.g., from ``element_data['xpath']``).

    Returns:
        Result dict per module docstring. Never raises.
    """
    if suspected_type not in _SUPPORTED_TYPES:
        return dict(_UNCONFIRMED)

    if not coords and not candidate_xpath:
        return dict(_UNCONFIRMED)

    args = {
        "coords": ({"x": float(coords[0]), "y": float(coords[1])}
                   if coords else None),
        "suspectedType": suspected_type,
        "frameworkPatterns": json.loads(_FRAMEWORK_PATTERNS_JSON),
        "candidateXPath": candidate_xpath or None,
    }

    try:
        result = await page.evaluate(_PROBE_JS, args)
    except Exception as e:
        logger.warning(
            "   ⚠️  DOM probe failed for %s: %s: %s",
            suspected_type, type(e).__name__, e,
        )
        return dict(_UNCONFIRMED)

    if not isinstance(result, dict):
        return dict(_UNCONFIRMED)

    # Defensive: guarantee the keys our caller expects exist even if the
    # JS shape drifts.
    return {
        "confirmed": bool(result.get("confirmed")),
        "framework": str(result.get("framework") or ""),
        "signals": list(result.get("signals") or []),
        "anchor_xpath": str(result.get("anchor_xpath") or ""),
        "anchor_tag": str(result.get("anchor_tag") or ""),
    }
