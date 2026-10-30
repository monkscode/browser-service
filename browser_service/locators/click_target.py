"""
The one control a click on a dead container means (bs #63, owner-approved 2026-10-30).

The agent sometimes points at a container (a menu ``<li>``, a table ``<td>`` or ``<tr>``)
whose centre is not on the link inside it: a click at the centre lands on dead space and does
nothing, while the generated ``Click`` on the container's locator does the same. This module
answers one question from one bounded read on the pointed element: is there exactly ONE control
inside it that the step names, that is visible and uncovered, and that does not hold the
container's centre? If so it returns that control, with the element data and the centre point
the two click sites need; in every other case it returns None and the caller changes nothing.

Found can never fall: any doubt (not a click, a control or a container with its own listener, a
collection, a row-anchored or text-less step, no xpath, a stale or ambiguous xpath, two candidate
controls, a control that hides its value, a failed read) is None. A name is matched by whole
words (``\\p{M}`` keeps Devanagari words whole), a name that only names an action on the row
("Delete Alice" for a row Alice) never answers for the row, and the innermost match wins.

Referenced by: agent/actions.py, agent/registration.py
Depends on: locators/action_fit.py, locators/hidden_value.py
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from browser_service.locators.action_fit import FIT_READ_TIMEOUT_MS
from browser_service.locators.hidden_value import HIDES_VALUE_JS

logger = logging.getLogger(__name__)

# The same eight tags as TAGS in INNER_CONTROL_JS: a pointed control is never re-aimed.
CONTROL_TAGS = frozenset(
    {"a", "button", "input", "select", "textarea", "label", "summary", "option"}
)

# The proven rule (bench/private/bs63/stress_opus/). HIDES(c) is resolved below, so the block
# stays comparable byte for byte with the plan.
INNER_CONTROL_JS = r"""
(el, {needle, step}) => {
    const TAGS = new Set(['a', 'button', 'input', 'select', 'textarea', 'label', 'summary', 'option']);
    if (TAGS.has(el.tagName.toLowerCase())) return null;
    const CONTROL = 'a, button, input:not([type="hidden"]), select, textarea, summary, [onclick], [role="button"], '
        + '[role="link"], [role="menuitem"], [role="tab"], [role="option"], [role="checkbox"], [role="radio"], '
        + '[role="switch"]';
    const r = el.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    const at = el.ownerDocument.elementFromPoint(cx, cy);
    if (el.hasAttribute('onclick') || el.getAttribute('tabindex') === '0' || getComputedStyle(el).cursor === 'pointer'
        || (at && !at.closest(CONTROL) && getComputedStyle(at).cursor === 'pointer')) return null;
    const norm = s => (s || '').split(/\s+/).filter(Boolean).join(' ').toLowerCase();
    const shown = c => { const b = c.getBoundingClientRect(), s = getComputedStyle(c);
        return b.width > 1 && b.height > 1 && s.visibility !== 'hidden'
            && c.checkVisibility({opacityProperty: true, visibilityProperty: true, contentVisibilityAuto: true}); };
    const name = c => [c.innerText || c.textContent || '', c.getAttribute('aria-label') || '',
        c.getAttribute('title') || '',
        ['button', 'submit', 'reset'].includes(c.type) ? (c.value || '') : '']
        .concat(Array.from(c.querySelectorAll('img[alt]')).map(i => i.alt)).join(' ');
    const words = s => ' ' + (s || '').toLowerCase().replace(/[^\p{L}\p{M}\p{N}]+/gu, ' ').trim() + ' ';
    const shownText = e => norm(e.innerText || e.textContent || '');
    const dbl = s => ' ' + s.split(' ').join('  ') + ' ';
    const times = (s, n) => dbl(s).split(dbl(n)).length - 1;
    const first = c => { const g = el.ownerDocument.createRange(); g.setStart(el, 0); g.setEndBefore(c);
        return !/\p{L}/u.test(g.toString()); };
    const carries = c => { const w = words(name(c));
        return ((norm(name(c)) === needle || (words(needle).trim() !== '' && w.includes(words(needle))))
                && times(shownText(el), needle) <= times(shownText(c), needle))
            || (w.trim() !== '' && words(needle).includes(w) && words(step).includes(w) && first(c)); };
    const hits = Array.from(el.querySelectorAll(CONTROL))
        .filter(c => !HIDES(c) && shown(c) && carries(c));
    const inner = hits.filter(c => !hits.some(o => o !== c && c.contains(o)));
    if (inner.length !== 1) return null;
    const c = inner[0], box = c.getBoundingClientRect();
    const top = el.ownerDocument.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
    if (top && !c.contains(top)) return null;
    return Array.from(c.getClientRects()).some(b => cx >= b.left && cx <= b.right && cy >= b.top && cy <= b.bottom)
        ? null : c;
}
""".replace("HIDES(c)", HIDES_VALUE_JS + "(c)")

# The control's element data: the keys of registration._extract_dom_node_attributes plus
# textContent and hasClickListener, the xpath absolute as browser-use writes it.
ELEMENT_DATA_JS = """
c => {
    const parts = [];
    for (let n = c; n && n.nodeType === 1; n = n.parentElement) {
        let i = 1;
        for (let s = n.previousElementSibling; s; s = s.previousElementSibling) {
            if (s.tagName === n.tagName) i++;
        }
        parts.unshift(n.tagName.toLowerCase() + '[' + i + ']');
    }
    const a = k => c.getAttribute(k) || '';
    const p = c.parentElement;
    const h = HIDES(c);
    return {
        tagName: c.tagName.toLowerCase(),
        id: c.id || '',
        name: a('name'),
        className: typeof c.className === 'string' ? c.className : '',
        ariaLabel: a('aria-label'),
        ariaInvalid: a('aria-invalid'),
        parentClassName: p && typeof p.className === 'string' ? p.className : '',
        placeholder: a('placeholder'),
        title: a('title'),
        href: h ? '' : a('href'),
        role: a('role'),
        dataTestId: a('data-testid') || a('data-test'),
        dataTestAttr: (!a('data-testid') && a('data-test')) ? 'data-test' : 'data-testid',
        type: a('type'),
        value: h ? '' : a('value'),
        xpath: parts.join('/'),
        backendNodeId: null,
        hasClickListener: false,
        textContent: h ? '' : (c.innerText || '').trim().slice(0, 300)
    };
}
""".replace("HIDES(c)", HIDES_VALUE_JS + "(c)")


@dataclass(frozen=True)
class InnerControl:
    """The control found inside a dead container."""

    element_data: Dict[str, Any]  # same keys as _extract_dom_node_attributes + textContent
    x: float  # page coordinates of the control's centre (bounding_box, so a frame is included)
    y: float
    handle: Any  # the control's ElementHandle

    async def is_answer(self, search_root, locator: str) -> bool:
        """True only when ``locator`` (frame prefix already stripped) resolves to exactly this control."""
        try:
            loc = search_root.locator(locator)
            if await loc.count() != 1:
                return False
            return bool(
                await loc.evaluate("(el, o) => el === o", self.handle, timeout=FIT_READ_TIMEOUT_MS)
            )
        except Exception:  # a navigation destroys the handle's context: not the answer
            return False


async def find_inner_control(
    search_root,
    *,
    action: Optional[str],
    expected_text: Optional[str],
    is_collection: Optional[bool],
    element_data: Optional[Dict[str, Any]],
    description: Optional[str],
    row_anchor_text: Optional[str],
) -> Optional[InnerControl]:
    """The one control inside the pointed container that the step names, or None.

    ``element_data`` None or without an xpath = no re-aim (a bare candidate is not a door);
    ``description`` is the step's description, the second witness for a name; ``row_anchor_text``
    set = the agent named a row, which says nothing about the target.
    """
    if action != "click" or is_collection or row_anchor_text:
        return None
    needle = " ".join((expected_text or "").split()).lower()
    if not needle or not element_data or not element_data.get("xpath"):
        return None
    if str(element_data.get("tagName") or "").lower() in CONTROL_TAGS or element_data.get(
        "hasClickListener"
    ):
        return None
    try:
        pointed = search_root.locator(f"xpath={element_data['xpath']}")
        if await pointed.count() != 1:
            return None
        found = await pointed.evaluate_handle(
            INNER_CONTROL_JS,
            {"needle": needle, "step": description or ""},
            timeout=FIT_READ_TIMEOUT_MS,
        )
        handle = found.as_element()
        if handle is None:
            return None
        box = await handle.bounding_box()
        if not box:
            return None
        data = await handle.evaluate(ELEMENT_DATA_JS)
        return InnerControl(
            element_data=data,
            x=box["x"] + box["width"] / 2,
            y=box["y"] + box["height"] / 2,
            handle=handle,
        )
    except Exception as exc:
        logger.warning("click target read failed, the click stays where it was: %s", exc)
        return None
