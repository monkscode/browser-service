"""
Read-step identity check for the agent-candidate path (wrong-element fix, K v1).

When the candidate identity guard (actions._identity_mismatch) rejects a unique
agent candidate on a READ step, the index — not the candidate — can be the
wrong evidence. u07: the agent indexed the sidebar "Dashboard" link while its
candidate named the page's <h6> heading; the reject sent the cascade back to
the link, and the test passed while verifying the wrong element.

The call site may accept such a candidate only when one live read proves all
five clauses (evaluation spec §13.2, amended after the S1 spike, 2026-09-23;
clauses 1 and 3 amended again in fix round 1 after review findings I1/I2):

1. the candidate's node is not, and is not inside, the node of any selector-map
   entry other than the indexed one — compared by backend_node_id over the
   map's VALUES, never by key or xpath;
2. it is not inside the indexed node;
3. it does not contain the indexed node — a shadow-including containment test
   (walking parentNode/host), never a plain DOM Node.contains, because
   Node.contains stops at a shadow boundary and browser-use indexes nodes
   inside open shadow roots;
4. it is not a <label> whose control is the indexed node;
5. the indexed node, resolved by its backend id, is connected and keeps its tag.

Four guards keep clause 1 honest, because "not in the map" has causes other
than "browser-use judged it not interactive":
- freshness: a selector-map node of the candidate's tag that is no longer
  connected means the page re-rendered after the map was built — a clone of an
  indexed node would otherwise read "not indexed" (S1 verifier, D1);
- visibility: browser-use indexes an interactive node only when it is visible,
  intersects the viewport horizontally, lies within ± 1000 px of it vertically
  (browser_use/dom/service.py:339-344) and is not painted over by another
  element (browser_use/dom/serializer/paint_order.py) — so a hidden, off-screen
  or covered candidate proves nothing. Known hole: a hit-test ignores
  pointer-events:none overlays, which browser-use's paint-order filter does not;
- folded children: browser-use folds a child that fills an <a>/<button> into
  its parent (browser_use/dom/serializer/serializer.py:785-793, 822-877), so
  such a child is absent from the map by design — it can still be inside a
  DIFFERENT indexed node, which clause 1's ancestor walk must catch;
- web-component shapes: browser-use indexes the node INSIDE a shadow root, not
  its host, and a light-DOM node slotted into a shadow control is not a DOM
  descendant of it. A shadow HOST that holds another indexed node passes as
  "not in the map" unless its shadowRoots subtree is inspected directly; a
  light-DOM child slotted into a shadow control is only found via the FLAT
  (composed) tree, walking assignedSlot ahead of parentElement — and that walk
  cannot see slotting into a CLOSED root at all (assignedSlot is null there),
  so a candidate sitting in the light DOM of a closed shadow host reads UNKNOWN
  rather than risk a false accept. A user-agent shadow root (<input>, <select>,
  <textarea>, <video>, <details> all have one) is the browser's own chrome, never
  a place another indexed node could live, and is ignored by both checks.

The ancestor check this buys also fires for a candidate that sits inside any
OTHER always-indexed container — a scrollable grid, a listbox, a menu, any
clickable container browser-use's own serializer role list treats as
interactive — which K then declines exactly like a web-component host; that is
by design, not a hole specific to shadow DOM.

Everything uncertain reads UNKNOWN and keeps the reject — the opposite of
_identity_mismatch's own unknown-means-accept rule, on purpose.

Referenced by: browser_service.agent.actions.find_unique_locator_action
Depends on: browser-use BrowserSession (_dom_watchdog.selector_map,
get_or_create_cdp_session), a Playwright Page
"""

import asyncio
import contextlib
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadIdentityFacts:
    """What one live read established. A non-empty ``unknown`` means: cannot decide."""

    unknown: Tuple[str, ...] = ()
    indexed_ok: bool = False
    is_indexed: bool = False
    inside_other_indexed: bool = False
    hosts_other_indexed: bool = False
    inside: bool = False
    contains: bool = False
    labels_index: bool = False
    candidate_tag: str = ""
    candidate_centre: Optional[Tuple[float, float]] = None


def read_step_verdict(facts: Optional[ReadIdentityFacts]) -> Tuple[bool, str]:
    """(accept, why). Accept only when every clause holds and nothing is unknown."""
    if facts is None:
        return False, "unknown: the live read failed"
    if facts.unknown:
        return False, "unknown: " + "; ".join(facts.unknown)
    if not facts.indexed_ok:
        return False, "clause 5: the indexed node is gone or changed tag"
    if facts.is_indexed:
        return False, "clause 1: the candidate is itself an indexed node"
    if facts.inside_other_indexed:
        return False, "clause 1: the candidate is inside another indexed node"
    if facts.hosts_other_indexed:
        return False, "clause 1: the candidate hosts another indexed node"
    if facts.inside:
        return False, "clause 2: the candidate is inside the indexed node"
    if facts.contains:
        return False, "clause 3: the candidate contains the indexed node"
    if facts.labels_index:
        return False, "clause 4: the candidate is the label of the indexed node"
    return True, "all five clauses hold"


# browser_use.dom.service.DomService's default viewport_threshold (0.13.7,
# service.py:64): nodes up to this many px outside the viewport count as visible.
BROWSER_USE_VIEWPORT_THRESHOLD_PX = 1000

# Measured in the S1 spike: the read is ~5 ms; the freshness guard adds ~10 ms
# at 201 map nodes and ~27 ms at 583. The bound only engages on a hung target —
# timing out is a failed read, which keeps the reject.
READ_IDENTITY_TIMEOUT_S = 2.0
_PARK_TIMEOUT_MS = 1000
_GLOBAL_PREFIX = "__bs_read_identity_"
# The cleanup runs after a timeout too, when the renderer may still be hung, and
# page.evaluate has no timeout of its own — unbounded, it would outlive the read.
_CLEANUP_TIMEOUT_S = 0.5

# Runs with `this` = the candidate. Clauses 2-5 plus the frame, shadow-root and
# visibility unknowns, in one call. The visibility test hit-tests the candidate's
# own centre point and checks browser-use's exact viewport window, but it does not
# reproduce browser-use's paint-order filter node-for-node — a candidate covered by
# its own opaque child (e.g. an icon-only button) passes this hit-test even though
# browser-use's filter might drop the underlying node, so this can only ADD a
# reject relative to browser-use's own judgement, never remove one it would make.
_CLAUSES_FN = r"""
function(indexed, indexedTag, threshold) {
  function nextFlat(n) {
    return n.assignedSlot || n.parentElement || (n.parentNode && n.parentNode.host) || null;
  }
  const cand = this;
  const out = {unknown: [], indexed_ok: false, inside: false, contains: false,
               labels_index: false, centre: null};
  let mainFrame = false;
  try { mainFrame = window.top === window && cand.ownerDocument === document; }
  catch (e) { mainFrame = false; }
  if (!mainFrame) out.unknown.push('the candidate is not in the main frame');
  const root = cand.getRootNode();
  if (!root || root.nodeType === 11) out.unknown.push('the candidate is in a shadow root');
  const rect = cand.getBoundingClientRect();
  out.centre = [rect.left + rect.width / 2, rect.top + rect.height / 2];
  const style = window.getComputedStyle(cand);
  const shown = rect.width > 0 && rect.height > 0
    && style.display !== 'none' && style.visibility !== 'hidden'
    && style.visibility !== 'collapse' && parseFloat(style.opacity || '1') > 0
    && (typeof cand.checkVisibility !== 'function'
        || cand.checkVisibility({opacityProperty: true, visibilityProperty: true}));
  if (!shown) out.unknown.push('the candidate is not visible');
  // browser-use's own window (dom/service.py:339-344): the threshold widens it
  // vertically only; horizontally the node must intersect the viewport itself.
  const vw = document.documentElement.clientWidth;
  const vh = document.documentElement.clientHeight;
  const inWindow = rect.bottom > -threshold && rect.top < vh + threshold
    && rect.right > 0 && rect.left < vw;
  if (!inWindow) out.unknown.push("the candidate is outside browser-use's indexing window");
  // browser-use drops a node another element paints over (paint-order filter), so
  // "not in the map" says nothing about a covered candidate. Hit-test its centre;
  // a centre outside the viewport cannot be hit-tested and reads unknown too.
  const [cx, cy] = out.centre;
  const hit = (cx >= 0 && cy >= 0 && cx < vw && cy < vh) ? document.elementFromPoint(cx, cy) : null;
  if (!hit || !(hit === cand || cand.contains(hit))) {
    out.unknown.push('the candidate is covered or outside the viewport');
  }
  out.indexed_ok = !!(indexed && indexed.isConnected && typeof indexed.tagName === 'string'
    && indexed.tagName.toLowerCase() === indexedTag);
  if (out.indexed_ok) {
    // Flat-tree ancestor walk (composed tree): a slotted light-DOM child's real
    // parentElement is its light-DOM host, not the shadow node it renders inside,
    // so plain Node.contains cannot see clause 2 for a web-component candidate.
    out.inside = false;
    if (indexed !== cand) {
      let n = nextFlat(cand);
      while (n) {
        if (n === indexed) { out.inside = true; break; }
        n = nextFlat(n);
      }
    }
    // Shadow-including containment: Node.contains stops at a shadow boundary, but
    // browser-use indexes nodes inside open shadow roots, so a shadow HOST whose
    // shadow root holds the indexed node must still read "contains".
    out.contains = false;
    if (indexed !== cand) {
      let n = indexed.parentNode || indexed.host;
      while (n) {
        if (n === cand) { out.contains = true; break; }
        n = n.parentNode || n.host;
      }
    }
    out.labels_index = cand.tagName === 'LABEL' && cand.control === indexed;
  }
  return out;
}
"""

_IS_CONNECTED_FN = "function() { return this.isConnected; }"

# Measured: CDP's Runtime.callFunctionOn rejects a call whose ARGUMENT objectId
# belongs to a different JS world than the `this` objectId ("Argument should
# belong to the same JavaScript world as target object") — a protocol-level
# error, so the indexed node's frame must be checked in its OWN world, before
# ever being offered as an argument to a function bound to the candidate's world.
_MAIN_FRAME_FN = "function() { return window.top === window; }"

# browser-use folds a child that fills an <a>/<button> into its parent
# (browser_use/dom/serializer/serializer.py:785-793, 822-877), so such a child is
# absent from the map by design. Clause 1 must also reject when the candidate is
# inside a DIFFERENT indexed node, not just when it IS one. The walk is the FLAT
# (composed) tree, so a light-DOM node slotted into a shadow button's <slot> is
# still found as inside that button; assignedSlot is null for a slot inside a
# CLOSED shadow root, so a closed host is caught separately (see below), never by
# this walk pretending to see through it.
_ANCESTORS_FN = (
    "function() { "
    "const next = (n) => n.assignedSlot || n.parentElement || (n.parentNode && n.parentNode.host) || null; "
    "const arr = []; let n = next(this); "
    "while (n) { arr.push(n); n = next(n); } return arr; }"
)


async def _inside_other_indexed(
    cdp_session, candidate_object: str, other_indexed_ids: Set[int], group: str, candidate_tag: str
) -> Tuple[bool, Optional[str]]:
    """(inside_other_indexed, unknown_reason). True when a flat-tree ancestor of the
    candidate is a same-target selector-map node other than the indexed one. Any
    failure here must propagate (fail closed) — never swallowed, unlike the
    freshness guard's per-node resolve. An empty ancestor list is only legitimate
    for <html> (which has none); for anything else it means the walk broke, not
    that the candidate has no ancestors.

    assignedSlot is null for a slot inside a CLOSED shadow root, so the walk cannot
    see slotting into closed content; if a CLOSED author shadow host sits on the
    chain it actually walked, the walk cannot be trusted here and this reads
    unknown rather than risk a false negative."""
    if not other_indexed_ids:
        return False, None
    ancestors = await _cdp(
        cdp_session,
        "Runtime.callFunctionOn",
        {"functionDeclaration": _ANCESTORS_FN, "objectId": candidate_object, "objectGroup": group},
    )
    array_object_id = (ancestors.get("result") or {}).get("objectId")
    if not array_object_id:
        raise RuntimeError("ancestor walk returned no array object")
    props = await _cdp(
        cdp_session, "Runtime.getProperties", {"objectId": array_object_id, "ownProperties": True}
    )
    ancestor_object_ids = [
        (p.get("value") or {}).get("objectId")
        for p in props.get("result") or []
        if (p.get("value") or {}).get("objectId")
    ]
    if not ancestor_object_ids:
        if candidate_tag == "html":
            return False, None
        raise RuntimeError("ancestor walk found no ancestors for a non-<html> candidate")
    described = await asyncio.gather(
        *(_cdp(cdp_session, "DOM.describeNode", {"objectId": oid}) for oid in ancestor_object_ids)
    )
    ancestor_ids: Set[int] = set()
    closed_shadow_host = False
    for d in described:
        node = d.get("node") or {}
        backend_id = _backend_id(node.get("backendNodeId"))
        if backend_id is None:
            raise RuntimeError("an ancestor node has no backend id")
        ancestor_ids.add(backend_id)
        if any(sr.get("shadowRootType") == "closed" for sr in node.get("shadowRoots") or []):
            closed_shadow_host = True
    if closed_shadow_host:
        return False, "the candidate sits in the light DOM of a closed shadow host"
    return bool(ancestor_ids & other_indexed_ids), None


# Only author roots (open/closed) can hide another indexed node inside the
# candidate; a user-agent root (<input>, <select>, <textarea>, <video>, <details>)
# is browser's own chrome, never a place another indexed node could live.
_AUTHOR_SHADOW_ROOT_TYPES = {"open", "closed"}


def _shadow_backend_ids(node: Dict[str, Any]) -> Set[int]:
    """Every backend id found inside NODE's shadowRoots subtrees (recursively,
    including nested shadow roots) — never the host's own light-DOM children,
    which live under NODE's own "children", not under "shadowRoots"."""
    ids: Set[int] = set()

    def walk(n: Dict[str, Any]) -> None:
        backend_id = _backend_id(n.get("backendNodeId"))
        if backend_id is not None:
            ids.add(backend_id)
        for child in n.get("children") or []:
            walk(child)
        for shadow_root in n.get("shadowRoots") or []:
            walk(shadow_root)

    for shadow_root in node.get("shadowRoots") or []:
        walk(shadow_root)
    return ids


async def _hosts_other_indexed(
    cdp_session,
    candidate_object: str,
    candidate_node: Dict[str, Any],
    other_indexed_ids: Set[int],
) -> bool:
    """True when the candidate is a shadow host and some OTHER indexed node lives
    anywhere inside its shadow tree(s). browser-use indexes the interactive node
    INSIDE a shadow root, not its host, so a host is otherwise invisible to clause
    1's "not in the map" test even though it visually wraps a different indexed
    control (e.g. two custom-element twins, or an alert box whose shadow content
    holds its own indexed Close button)."""
    if not other_indexed_ids:
        return False
    if not any(
        sr.get("shadowRootType") in _AUTHOR_SHADOW_ROOT_TYPES
        for sr in candidate_node.get("shadowRoots") or []
    ):
        return False
    deep = await _cdp(
        cdp_session, "DOM.describeNode", {"objectId": candidate_object, "depth": -1, "pierce": True}
    )
    hosted_ids = _shadow_backend_ids(deep.get("node") or {})
    return bool(hosted_ids & other_indexed_ids)


def _backend_id(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _watchdog_selector_map(browser_session) -> Optional[Dict[Any, Any]]:
    """The map the agent's index came from. Never get_selector_map(): in browser-use
    0.13.7 it returns the cached map first (session.py:2716-2721), so most of the
    time it IS this same map — the real divergence is a scroll clearing the
    watchdog's map (default_action_watchdog.py:522-524), which get_selector_map()
    would silently rebuild/renumber. Reading the watchdog directly means K reads
    that as "no selector map" and fails closed, instead of comparing the candidate
    against a map the index may not have come from (Review Focus 1)."""
    watchdog = getattr(browser_session, "_dom_watchdog", None)
    selector_map = getattr(watchdog, "selector_map", None)
    if isinstance(selector_map, dict) and selector_map:
        return selector_map
    return None


async def _cdp(cdp_session, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fail closed on a page-JS error: a CDP response that carries ``exceptionDetails``
    (e.g. a throwing evaluated function) is NOT a successful result — Runtime domain
    calls return HTTP-success with the thrown Error object standing in for the return
    value, which a caller reading it as an array/object would silently misread as
    empty/false rather than as a broken read."""
    domain, name = method.split(".", 1)
    call = getattr(getattr(cdp_session.cdp_client.send, domain), name)
    result = await call(params=params, session_id=cdp_session.session_id)
    if isinstance(result, dict) and result.get("exceptionDetails"):
        raise RuntimeError(f"{method} raised: {result['exceptionDetails']!r}")
    return result


async def _any_disconnected(cdp_session, backend_ids: Iterable[int], group: str) -> bool:
    """True when any of these nodes is gone or detached. A failed resolve counts as gone."""

    async def connected(backend_id: int) -> bool:
        try:
            resolved = await _cdp(
                cdp_session, "DOM.resolveNode", {"backendNodeId": backend_id, "objectGroup": group}
            )
            answer = await _cdp(
                cdp_session,
                "Runtime.callFunctionOn",
                {
                    "functionDeclaration": _IS_CONNECTED_FN,
                    "objectId": resolved["object"]["objectId"],
                    "returnByValue": True,
                },
            )
            return (answer.get("result") or {}).get("value") is True
        except Exception:
            return False

    results = await asyncio.gather(*(connected(b) for b in backend_ids))
    return not all(results)


async def read_identity_facts(
    page,
    browser_session,
    element_data: Optional[Dict[str, Any]],
    candidate_locator: str,
) -> Optional[ReadIdentityFacts]:
    """One live read of the facts K v1 decides on. None = the read failed or timed out."""
    try:
        return await asyncio.wait_for(
            _read(page, browser_session, element_data, candidate_locator),
            timeout=READ_IDENTITY_TIMEOUT_S,
        )
    except Exception as e:
        logger.warning(
            f"   ⚠️ Read-step identity read failed for '{candidate_locator}': "
            f"{type(e).__name__}: {e}"
        )
        return None


async def _read(page, browser_session, element_data, candidate_locator) -> ReadIdentityFacts:
    indexed_id = _backend_id((element_data or {}).get("backendNodeId"))
    if indexed_id is None:
        return ReadIdentityFacts(unknown=("no backend id for the indexed node",))
    selector_map = _watchdog_selector_map(browser_session)
    if selector_map is None:
        return ReadIdentityFacts(unknown=("no selector map",))
    indexed_entries = [
        n
        for n in selector_map.values()
        if _backend_id(getattr(n, "backend_node_id", None)) == indexed_id
    ]
    if not indexed_entries:
        return ReadIdentityFacts(unknown=("the indexed node is not in the selector map",))
    target_id = getattr(indexed_entries[0], "target_id", None)
    same_target = [
        n for n in selector_map.values() if str(getattr(n, "target_id", None)) == str(target_id)
    ]

    cdp_session = await browser_session.get_or_create_cdp_session(target_id=target_id, focus=False)
    name = _GLOBAL_PREFIX + uuid.uuid4().hex
    group = "bs-read-identity-" + uuid.uuid4().hex[:8]
    try:
        await page.locator(candidate_locator).evaluate(
            "(el, n) => { window[n] = el; }", name, timeout=_PARK_TIMEOUT_MS
        )
        parked = await _cdp(
            cdp_session,
            "Runtime.evaluate",
            {"expression": f"window[{name!r}]", "objectGroup": group},
        )
        candidate_object = (parked.get("result") or {}).get("objectId")
        if not candidate_object:
            return ReadIdentityFacts(
                unknown=("the candidate is not reachable on the indexed node's target",)
            )
        described = await _cdp(cdp_session, "DOM.describeNode", {"objectId": candidate_object})
        node = described.get("node") or {}
        candidate_id = _backend_id(node.get("backendNodeId"))
        candidate_tag = (node.get("nodeName") or "").lower()
        if candidate_id is None or not candidate_tag:
            return ReadIdentityFacts(unknown=("the candidate node could not be described",))
        is_indexed = any(
            _backend_id(getattr(n, "backend_node_id", None)) == candidate_id for n in same_target
        )
        other_indexed_ids: Set[int] = {
            b
            for b in (_backend_id(getattr(n, "backend_node_id", None)) for n in same_target)
            if b is not None and b != indexed_id
        }
        hosts_other_indexed = await _hosts_other_indexed(
            cdp_session, candidate_object, node, other_indexed_ids
        )
        inside_other_indexed, closed_shadow_unknown = await _inside_other_indexed(
            cdp_session, candidate_object, other_indexed_ids, group, candidate_tag
        )

        indexed_arg: Dict[str, Any] = {"value": None}
        indexed_not_main_frame = False
        with contextlib.suppress(Exception):
            resolved = await _cdp(
                cdp_session, "DOM.resolveNode", {"backendNodeId": indexed_id, "objectGroup": group}
            )
            resolved_object_id = resolved["object"]["objectId"]
            frame_check = await _cdp(
                cdp_session,
                "Runtime.callFunctionOn",
                {
                    "functionDeclaration": _MAIN_FRAME_FN,
                    "objectId": resolved_object_id,
                    "returnByValue": True,
                },
            )
            if (frame_check.get("result") or {}).get("value") is True:
                indexed_arg = {"objectId": resolved_object_id}
            else:
                indexed_not_main_frame = True
        indexed_tag = ((element_data or {}).get("tagName") or "").lower()
        answer = await _cdp(
            cdp_session,
            "Runtime.callFunctionOn",
            {
                "functionDeclaration": _CLAUSES_FN,
                "objectId": candidate_object,
                "arguments": [
                    indexed_arg,
                    {"value": indexed_tag},
                    {"value": BROWSER_USE_VIEWPORT_THRESHOLD_PX},
                ],
                "returnByValue": True,
            },
        )
        clauses = (answer.get("result") or {}).get("value") or {}
        unknown = list(clauses.get("unknown") or [])
        if closed_shadow_unknown:
            unknown.append(closed_shadow_unknown)
        if indexed_not_main_frame:
            unknown.append("the indexed node is not in the main frame")

        same_tag_ids = [
            b
            for b in (
                _backend_id(getattr(n, "backend_node_id", None))
                for n in same_target
                if (getattr(n, "node_name", "") or "").lower() == candidate_tag
            )
            if b is not None
        ]
        if same_tag_ids and await _any_disconnected(cdp_session, same_tag_ids, group):
            unknown.append(
                f"a selector-map <{candidate_tag}> is no longer on the page "
                f"(re-rendered after the map was built)"
            )

        centre = clauses.get("centre")
        return ReadIdentityFacts(
            unknown=tuple(unknown),
            indexed_ok=bool(clauses.get("indexed_ok")),
            is_indexed=is_indexed,
            inside_other_indexed=inside_other_indexed,
            hosts_other_indexed=hosts_other_indexed,
            inside=bool(clauses.get("inside")),
            contains=bool(clauses.get("contains")),
            labels_index=bool(clauses.get("labels_index")),
            candidate_tag=candidate_tag,
            candidate_centre=(float(centre[0]), float(centre[1])) if centre else None,
        )
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                page.evaluate("(n) => { delete window[n]; }", name), timeout=_CLEANUP_TIMEOUT_S
            )
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                _cdp(cdp_session, "Runtime.releaseObjectGroup", {"objectGroup": group}),
                timeout=_CLEANUP_TIMEOUT_S,
            )
