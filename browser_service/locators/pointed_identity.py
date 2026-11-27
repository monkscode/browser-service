"""
The pointed element (point-then-prove D1): what the agent pointed at, read from the live page.

The target is the indexed node (its browser-use xpath) or the element under the agent's point —
never "the smallest indexed box around the point", which on TodoMVC was the whole app root.
``box_is_hit`` tells such a box from the hit: the box must be the hit itself or the nearest
control at or above it.

``proven`` (D2) is the identity gate's test: a locator ships for a performed step only when it
lands on the target or on one allowlisted relation of it (``RELATION_JS``).

Referenced by: agent/registration.py, agent/actions.py
Depends on: locators/smart_locator.py (the open-shadow-aware hit-test helper),
    locators/frame_locator.py, locators/validation.py
"""

from browser_service.locators.frame_locator import strip_frame
from browser_service.locators.smart_locator import _SHADOW_DOM_HELPER_JS
from browser_service.locators.validation import convert_to_playwright_locator

XPATH_TIMEOUT_MS = 1000

_HIT_JS = f"""
(args) => {{
    {_SHADOW_DOM_HELPER_JS}
    return getElementFromPointWithShadow(document, args.x, args.y);
}}
"""

# A point on these names no element: empty page, or a frame bs did not enter.
_NO_TARGET_JS = "el => ['body', 'html', 'iframe'].includes(el.localName)"

# The nearest control at or above ``el``: the element itself, then up to 6 ancestors, crossing
# open shadow hosts. null when none.
_NEAREST_CONTROL_JS = """
    function nearestControl(el) {
        const TAGS = ['a', 'button', 'input', 'select', 'textarea', 'label', 'summary', 'option'];
        const ROLES = ['button', 'link', 'checkbox', 'radio', 'switch', 'tab', 'menuitem',
            'menuitemcheckbox', 'menuitemradio', 'option', 'combobox', 'textbox', 'searchbox',
            'slider', 'spinbutton', 'treeitem'];
        for (let level = 0; el && level <= 6; level++) {
            const role = (el.getAttribute('role') || '').trim().toLowerCase();
            const editable = el.getAttribute('contenteditable');
            if (TAGS.includes(el.localName) || ROLES.includes(role)
                || (editable !== null && editable.toLowerCase() !== 'false')) {
                return el;
            }
            el = el.parentElement || (el.parentNode instanceof ShadowRoot ? el.parentNode.host : null);
        }
        return null;
    }
"""

_BOX_IS_HIT_JS = f"""
(hit, box) => {{
    {_NEAREST_CONTROL_JS}
    return hit === box || nearestControl(hit) === box;
}}
"""

# D2: how the locator's one match ``el`` relates to the target ``t``; "" = no allowlisted
# relation. A detached node throws (a re-render between the hit-test and this read): unreadable.
RELATION_JS = f"""
(el, t) => {{
    {_NEAREST_CONTROL_JS}
    if (!el.isConnected || !t.isConnected) throw new Error('detached');
    if (el === t) return 'same';
    if (nearestControl(t) === el) return 'control-above';
    // Playwright fills and clicks a label's control, so a label and its control are one.
    if ((el.localName === 'label' && el.control === t)
        || (t.localName === 'label' && t.control === el)) return 'only-control';
    // The one control of its kind beside the target. A container is never proven, and neither
    // is the body: an overlay's common ancestor with any page control is the body.
    if (nearestControl(el) !== el) return '';
    const above = [];
    for (let n = t, d = 0; n && d <= 3; n = n.parentElement, d++) above.push(n);
    let lca = null;
    for (let n = el, d = 0; n && d <= 3 && !lca; n = n.parentElement, d++) {{
        if (above.includes(n)) lca = n;
    }}
    if (!lca || ['body', 'html'].includes(lca.localName)) return '';
    const kind = (n) => [n.localName, typeof n.type === 'string' ? n.type : '',
        (n.getAttribute('role') || '').trim().toLowerCase()].join('|');
    const mine = kind(el);
    const same = [lca, ...lca.querySelectorAll(el.localName)].filter((n) => kind(n) === mine);
    return same.length === 1 ? 'only-control' : '';
}}
"""

# A collection: the target is one of the matches or holds one (crossing open shadow hosts).
_HOLDS_MATCH_JS = """
(t, matches) => {
    if (!t.isConnected) throw new Error('detached');
    return matches.some((m) => {
        for (let n = m; n; n = n.parentNode instanceof ShadowRoot ? n.parentNode.host : n.parentNode) {
            if (n === t) return true;
        }
        return false;
    });
}
"""

# How a refusal names an element: <tag id=… type=… role=… class=…>, attributes only.
_DESCRIBE_JS = """
el => {
    const bits = [el.localName];
    for (const name of ['id', 'type', 'role', 'class']) {
        const value = (el.getAttribute(name) || '').trim().replace(/\\s+/g, ' ');
        if (value) bits.push(name + '=' + value.slice(0, 40));
    }
    return '<' + bits.join(' ') + '>';
}
"""


async def _kept(handle):
    """``handle`` unless it names no element (none, body, html, an iframe)."""
    if handle is None or await handle.evaluate(_NO_TARGET_JS):
        return None
    return handle


async def _hit_at(page, iframe_context: str | None, point: tuple[float, float]):
    """The element under ``point`` (viewport pixels), inside the frame with frame-relative
    coordinates; None for nothing, body, html or an iframe. Raises when the page cannot be read."""
    frame, (x, y) = page, point
    if iframe_context:
        iframe = page.locator(iframe_context)
        box = await iframe.bounding_box(timeout=XPATH_TIMEOUT_MS)
        frame = await (await iframe.element_handle(timeout=XPATH_TIMEOUT_MS)).content_frame()
        if box is None or frame is None:
            raise RuntimeError(f"cannot enter the frame {iframe_context}")
        x, y = x - box["x"], y - box["y"]
    found = await frame.evaluate_handle(_HIT_JS, {"x": x, "y": y})
    return await _kept(found.as_element())


async def pointed_target(
    page,
    iframe_context: str | None,
    element_data: dict | None,
    by_index: bool,
    point: tuple[float, float] | None,
):
    """The element the agent pointed at, as an ElementHandle, or None. Never raises.

    With ``by_index`` and an xpath: the element at that xpath in the frame root. Otherwise, or
    when the xpath fails: the open-shadow-aware element under ``point`` (viewport pixels),
    hit-tested inside the frame with frame-relative coordinates. ``point`` None = no fallback.
    """
    try:
        return await pointed_target_or_raise(page, iframe_context, element_data, by_index, point)
    except Exception:
        return None


async def pointed_target_or_raise(
    page,
    iframe_context: str | None,
    element_data: dict | None,
    by_index: bool,
    point: tuple[float, float] | None,
):
    """``pointed_target``, except that a page the point cannot be read on raises: the identity
    gate tells "unreadable" from "no element"."""
    xpath = (element_data or {}).get("xpath")
    if by_index and xpath:
        root = page.frame_locator(iframe_context) if iframe_context else page
        try:
            handle = await root.locator(f"xpath={xpath}").element_handle(timeout=XPATH_TIMEOUT_MS)
            return await _kept(handle)
        except Exception:  # unresolvable xpath (svg, shadow, re-render): the point decides
            pass
    return None if point is None else await _hit_at(page, iframe_context, point)


async def proven(
    page, iframe_context: str | None, locator: str, target, is_collection: bool
) -> tuple[bool, str]:
    """Does ``locator`` land on ``target`` (D2)? ``(True, relation)`` or ``(False, reason)``.

    The frame hop is stripped and the locator counted in the frame. A collection is proven when
    the target is one of its matches or holds one; a single result when it matches exactly one
    element with a ``RELATION_JS`` relation. The reason names both elements by tag and box
    centre; any read error, a detached node included, is ``(False, "unreadable")``. Never raises.
    """
    try:
        root = page.frame_locator(iframe_context) if iframe_context else page
        selector = convert_to_playwright_locator(strip_frame(locator, iframe_context))[0]
        matches = root.locator(selector)
        count = await matches.count()
        found = f"{count} elements"
        if is_collection:
            if count >= 1 and await target.evaluate(
                _HOLDS_MATCH_JS, await matches.element_handles()
            ):
                return True, "collection"
        elif count == 1:
            element = await matches.element_handle(timeout=XPATH_TIMEOUT_MS)
            relation = await element.evaluate(RELATION_JS, target)
            if relation:
                return True, relation
            found = await _named(element)
        return False, f"finds {found}; the element you pointed at is {await _named(target)}"
    except Exception:
        return False, "unreadable"


async def _named(handle) -> str:
    """``<tag id=… type=… role=… class=…> at (x, y)``: the box centre in viewport pixels tells two
    alike elements apart (no text or value is read). No box (hidden): the tag alone."""
    name = await handle.evaluate(_DESCRIBE_JS)
    box = await handle.bounding_box()
    if box is None:
        return name
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    return f"{name} at ({round(x)}, {round(y)})"


async def box_is_hit(box, hit) -> bool:
    """True when ``box`` is ``hit`` or the nearest control at or above it.

    Both handles must live in one frame; an unreadable pair raises, and the caller decides.
    """
    return bool(await hit.evaluate(_BOX_IS_HIT_JS, box))
