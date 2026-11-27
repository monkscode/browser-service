"""
The pointed element (point-then-prove D1): what the agent pointed at, read from the live page.

The target is the indexed node (its browser-use xpath) or the element under the agent's point —
never "the smallest indexed box around the point", which on TodoMVC was the whole app root.
``box_is_hit`` tells such a box from the hit: the box must be the hit itself or the nearest
control at or above it.

Referenced by: agent/registration.py, agent/actions.py
Depends on: locators/smart_locator.py (the open-shadow-aware hit-test helper)
"""

from browser_service.locators.smart_locator import _SHADOW_DOM_HELPER_JS

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


async def _kept(handle):
    """``handle`` unless it names no element (none, body, html, an iframe)."""
    if handle is None or await handle.evaluate(_NO_TARGET_JS):
        return None
    return handle


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
        xpath = (element_data or {}).get("xpath")
        if by_index and xpath:
            root = page.frame_locator(iframe_context) if iframe_context else page
            try:
                handle = await root.locator(f"xpath={xpath}").element_handle(
                    timeout=XPATH_TIMEOUT_MS
                )
                return await _kept(handle)
            except Exception:  # unresolvable xpath (svg, shadow, re-render): the point decides
                pass
        if point is None:
            return None
        frame, (x, y) = page, point
        if iframe_context:
            iframe = page.locator(iframe_context)
            box = await iframe.bounding_box(timeout=XPATH_TIMEOUT_MS)
            frame = await (await iframe.element_handle(timeout=XPATH_TIMEOUT_MS)).content_frame()
            x, y = x - box["x"], y - box["y"]
        found = await frame.evaluate_handle(_HIT_JS, {"x": x, "y": y})
        return await _kept(found.as_element())
    except Exception:
        return None


async def box_is_hit(box, hit) -> bool:
    """True when ``box`` is ``hit`` or the nearest control at or above it.

    Both handles must live in one frame; an unreadable pair raises, and the caller decides.
    """
    return bool(await hit.evaluate(_BOX_IS_HIT_JS, box))
