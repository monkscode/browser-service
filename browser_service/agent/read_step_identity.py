"""
Read-step identity check for the agent-candidate path (wrong-element fix, K v1).

When the candidate identity guard (actions._identity_mismatch) rejects a unique
agent candidate on a READ step, the index — not the candidate — can be the
wrong evidence. u07: the agent indexed the sidebar "Dashboard" link while its
candidate named the page's <h6> heading; the reject sent the cascade back to
the link, and the test passed while verifying the wrong element.

K decides only on light-DOM, main-frame pages. There, the call site may accept
such a candidate only when one live read proves all five clauses (evaluation
spec §13.2, amended after the S1 spike, 2026-09-23; clause 1 amended in fix
round 1; the scope narrowed to light-DOM pages in working session 3):

1. the candidate's node is not, and is not inside, the node of any selector-map
   entry other than the indexed one — compared by backend_node_id over the
   map's VALUES, never by key or xpath; "inside" walks the candidate's
   light-DOM ancestors (parentElement);
2. it is not inside the indexed node (plain DOM containment);
3. it does not contain the indexed node (plain DOM containment);
4. it is not a <label> whose control is the indexed node;
5. the indexed node, resolved by its backend id, is connected and keeps its tag.

Plain DOM containment is the whole truth only on a light-DOM page: it stops at
a shadow boundary, browser-use indexes nodes INSIDE shadow roots (never their
hosts), and a node slotted into a shadow control is not a DOM descendant of it.
So K does not decide any shadow shape; each reads UNKNOWN:
- B1 the candidate is not in its document tree (it sits in a shadow root);
- B2 the candidate itself hosts an author (open or closed) shadow root;
- B3 a light-DOM ancestor of the candidate hosts an author shadow root — this
  covers a node slotted into a web component, open or closed, because
  DOM.describeNode reports closed roots too;
- B4 the indexed node is in a shadow root (author or user-agent);
- B5 a same-target selector-map node sits inside a user-agent shadow root.
A user-agent root on the CANDIDATE side is the browser's own chrome — Chromium
gives <img alt>, <input>, <textarea>, <select>, <details>, <video>, <audio>,
<meter>, <progress>, <object>, <embed> and <marquee> one — so B2 and B3 ignore
it, or K would switch off for image reads. browser-use 0.13.7 indexes no node
inside a user-agent root (measured across 18 host kinds); B5 keeps K fail-closed
should an upgrade start to. A candidate or indexed node in a child frame reads
UNKNOWN too.

Three guards keep clause 1 honest, because "not in the map" has causes other
than "browser-use judged it not interactive":
- freshness: a selector-map node of the candidate's tag that is no longer
  connected means the page re-rendered after the map was built — a clone of an
  indexed node would otherwise read "not indexed" (S1 verifier, D1);
- visibility: browser-use indexes an interactive node only when it is visible,
  intersects the viewport horizontally, lies within ± 1000 px of it vertically
  (browser_use/dom/service.py:339-344) and is not painted over by another
  element (browser_use/dom/serializer/paint_order.py) — so a hidden, off-screen
  or covered candidate proves nothing. Two known holes, both where the centre
  hit-test and browser-use's paint-order filter disagree: a hit-test ignores
  pointer-events:none overlays, which the filter does not; and the filter drops
  a node covered by its OWN opaque children (paint_order.py:195-205), which the
  hit-test passes because the hit lands inside the candidate. In the second case
  the candidate is visible and the agent could not have indexed it, which is
  K's premise;
- folded children: browser-use folds a child that fills an <a>/<button> into
  its parent (browser_use/dom/serializer/serializer.py:785-793, 822-877), so
  such a child is absent from the map by design — it can still be inside a
  DIFFERENT indexed node, which clause 1's ancestor walk must catch.

The ancestor check this buys also fires for a candidate that sits inside any
OTHER always-indexed container — a scrollable grid, a listbox, a menu, any
clickable container browser-use's own serializer role list treats as
interactive — which K then declines; that is by design.

Every page-JS function reads DOM properties through the prototypes, never the
instance: named elements clobber them (<form><input name="parentElement">
makes form.parentElement that INPUT; <form name="host"> makes document.host that
FORM), and a clobbered walk used to loop forever and hang the page's renderer.
The ancestor walk is also capped at 4096 hops.

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
# reproduce browser-use's paint-order filter node-for-node, and the two disagree
# both ways. Known hole 1: a hit-test ignores a pointer-events:none overlay, which
# the paint-order filter does not. Known hole 2: the filter drops any node whose
# box is covered by later-painted opaque boxes, its own children included
# (browser_use/dom/serializer/paint_order.py:195-205) — a candidate covered by its
# own opaque child passes this hit-test (the hit lands inside it) yet can be
# absent from the map for that reason. The candidate is visible in that case,
# which matches K's premise: the agent could not have indexed it.
#
# Every DOM property is read through its prototype (P/G/M below), never the
# instance, because named elements clobber instance reads: <img
# name="documentElement"> makes document.documentElement that IMG, and <form><input
# name="tagName"> makes form.tagName that INPUT. window.top, window.getComputedStyle
# and document are unforgeable or own properties of the global and stay direct.
_CLAUSES_FN = r"""
function(indexed, indexedTag, threshold) {
  const P = (proto, name) => Object.getOwnPropertyDescriptor(proto, name);
  const G = (proto, name) => P(proto, name).get;
  const M = (proto, name) => P(proto, name).value;
  const ownerDocumentOf = G(Node.prototype, 'ownerDocument');
  const isConnectedOf = G(Node.prototype, 'isConnected');
  const getRootNode = M(Node.prototype, 'getRootNode');
  const contains = M(Node.prototype, 'contains');
  const tagNameOf = G(Element.prototype, 'tagName');
  const getRect = M(Element.prototype, 'getBoundingClientRect');
  const visibilityDescriptor = P(Element.prototype, 'checkVisibility');
  const checkVisibility = visibilityDescriptor ? visibilityDescriptor.value : undefined;
  const clientWidthOf = G(Element.prototype, 'clientWidth');
  const clientHeightOf = G(Element.prototype, 'clientHeight');
  const documentElementOf = G(Document.prototype, 'documentElement');
  const elementFromPoint = M(Document.prototype, 'elementFromPoint');
  const controlOf = G(HTMLLabelElement.prototype, 'control');
  const cand = this;
  const out = {unknown: [], indexed_ok: false, inside: false, contains: false,
               labels_index: false, centre: null, in_document_tree: false};
  let mainFrame = false;
  try { mainFrame = window.top === window && ownerDocumentOf.call(cand) === document; }
  catch (e) { mainFrame = false; }
  if (!mainFrame) out.unknown.push('the candidate is not in the main frame');
  out.in_document_tree = getRootNode.call(cand) === document;
  if (!out.in_document_tree) out.unknown.push('the candidate is in a shadow root');
  const rect = getRect.call(cand);
  out.centre = [rect.left + rect.width / 2, rect.top + rect.height / 2];
  const style = window.getComputedStyle(cand);
  const shown = rect.width > 0 && rect.height > 0
    && style.display !== 'none' && style.visibility !== 'hidden'
    && style.visibility !== 'collapse' && parseFloat(style.opacity || '1') > 0
    && (typeof checkVisibility !== 'function'
        || checkVisibility.call(cand, {opacityProperty: true, visibilityProperty: true}));
  if (!shown) out.unknown.push('the candidate is not visible');
  // browser-use's own window (dom/service.py:339-344): the threshold widens it
  // vertically only; horizontally the node must intersect the viewport itself.
  const docEl = documentElementOf.call(document);
  const vw = clientWidthOf.call(docEl);
  const vh = clientHeightOf.call(docEl);
  const inWindow = rect.bottom > -threshold && rect.top < vh + threshold
    && rect.right > 0 && rect.left < vw;
  if (!inWindow) out.unknown.push("the candidate is outside browser-use's indexing window");
  // browser-use drops a node another element paints over (paint-order filter), so
  // "not in the map" says nothing about a covered candidate. Hit-test its centre;
  // a centre outside the viewport cannot be hit-tested and reads unknown too.
  const [cx, cy] = out.centre;
  const hit = (cx >= 0 && cy >= 0 && cx < vw && cy < vh)
    ? elementFromPoint.call(document, cx, cy) : null;
  if (!hit || !(hit === cand || contains.call(cand, hit))) {
    out.unknown.push('the candidate is covered or outside the viewport');
  }
  out.indexed_ok = !!(indexed && indexed instanceof Element && isConnectedOf.call(indexed)
    && tagNameOf.call(indexed).toLowerCase() === indexedTag);
  if (out.indexed_ok) {
    // Plain DOM containment, both ways: K decides only on light-DOM pages (every
    // shadow shape reads unknown), where it is the whole truth.
    out.inside = indexed !== cand && contains.call(indexed, cand);
    out.contains = indexed !== cand && contains.call(cand, indexed);
    out.labels_index = cand instanceof HTMLLabelElement && controlOf.call(cand) === indexed;
  }
  return out;
}
"""

_IS_CONNECTED_FN = (
    "function() { return Object.getOwnPropertyDescriptor(Node.prototype, 'isConnected')"
    ".get.call(this); }"
)

# Measured: CDP's Runtime.callFunctionOn rejects a call whose ARGUMENT objectId
# belongs to a different JS world than the `this` objectId ("Argument should
# belong to the same JavaScript world as target object") — a protocol-level
# error, so the indexed node's frame, and whether it sits in a shadow root, must
# be checked in its OWN world, before ever being offered as an argument to a
# function bound to the candidate's world. A connected node whose root is not its
# document is in a shadow root (author or user-agent); a detached node is left to
# clause 5.
_MAIN_FRAME_FN = (
    "function() { "
    "const N = (name) => Object.getOwnPropertyDescriptor(Node.prototype, name); "
    "const connected = N('isConnected').get.call(this); "
    "return {main_frame: window.top === window, "
    "in_shadow_root: connected && N('getRootNode').value.call(this) !== document}; }"
)

# browser-use folds a child that fills an <a>/<button> into its parent
# (browser_use/dom/serializer/serializer.py:785-793, 822-877), so such a child is
# absent from the map by design. Clause 1 must also reject when the candidate is
# inside a DIFFERENT indexed node, not just when it IS one. The walk is the plain
# light-DOM parentElement chain (K decides only on light-DOM pages), read through
# Node.prototype — a form control named "parentElement" would otherwise loop the
# walk FORM → INPUT → FORM forever — and capped, so no page shape can hang it.
_ANCESTOR_WALK_CAP = 4096
_ANCESTORS_FN = (
    "function() { "
    "const parentOf = Object.getOwnPropertyDescriptor(Node.prototype, 'parentElement').get; "
    "const arr = []; let n = parentOf.call(this); "
    "while (n) { "
    f"if (arr.length >= {_ANCESTOR_WALK_CAP}) "
    f"throw new Error('the ancestor walk passed {_ANCESTOR_WALK_CAP} hops'); "
    "arr[arr.length] = n; n = parentOf.call(n); } "
    "return arr; }"
)

# Only author roots (open/closed) make a web component; a user-agent root (<img
# alt>, <input>, <select>, <textarea>, <video>, <details> and more) is the
# browser's own chrome, never a place another indexed node could live.
_AUTHOR_SHADOW_ROOT_TYPES = {"open", "closed"}


def _hosts_author_shadow_root(described_node: Dict[str, Any]) -> bool:
    return any(
        root.get("shadowRootType") in _AUTHOR_SHADOW_ROOT_TYPES
        for root in described_node.get("shadowRoots") or []
    )


async def _light_dom_ancestors(
    cdp_session,
    candidate_object: str,
    group: str,
    candidate_tag: str,
    *,
    outside_document_tree: bool,
) -> Tuple[Set[int], bool]:
    """(the backend ids of the candidate's light-DOM ancestors, whether any of them
    hosts an author shadow root). Runs on every read. Any failure here must
    propagate (fail closed) — never swallowed, unlike the freshness guard's
    per-node resolve. An empty ancestor list is legitimate only for <html>, and
    for a candidate outside its document tree (the top child of a shadow root has
    no parentElement; that candidate already reads unknown); for anything else it
    means the walk broke, not that the candidate has no ancestors."""
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
        if candidate_tag == "html" or outside_document_tree:
            return set(), False
        raise RuntimeError("ancestor walk found no ancestors for a non-<html> candidate")
    described = await asyncio.gather(
        *(_cdp(cdp_session, "DOM.describeNode", {"objectId": oid}) for oid in ancestor_object_ids)
    )
    ancestor_ids: Set[int] = set()
    hosts_shadow = False
    for d in described:
        node = d.get("node") or {}
        backend_id = _backend_id(node.get("backendNodeId"))
        if backend_id is None:
            raise RuntimeError("an ancestor node has no backend id")
        ancestor_ids.add(backend_id)
        hosts_shadow = hosts_shadow or _hosts_author_shadow_root(node)
    return ancestor_ids, hosts_shadow


# browser-use links every node to its parent, and a shadow-root node to its host,
# through parent_node; a shadow-root node carries shadow_root_type, the plain
# string "user-agent" | "open" | "closed" (cdp_use/cdp/dom/types.py:85).
_MAP_PARENT_WALK_CAP = 10_000


def _map_node_in_user_agent_root(nodes: Iterable[Any]) -> bool:
    """True when any of these selector-map nodes sits inside a user-agent shadow
    root. A node without parent_node (a test's SimpleNamespace) reads "not inside".
    Raises past the cap, which fails the read closed."""
    for node in nodes:
        current = node
        hops = 0
        while current is not None:
            if getattr(current, "shadow_root_type", None) == "user-agent":
                return True
            hops += 1
            if hops > _MAP_PARENT_WALK_CAP:
                raise RuntimeError(
                    f"a selector-map parent chain passed {_MAP_PARENT_WALK_CAP} hops"
                )
            current = getattr(current, "parent_node", None)
    return False


def _backend_id(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _watchdog_selector_map(browser_session) -> Optional[Dict[Any, Any]]:
    """The map the agent's index came from. Never get_selector_map(): in browser-use
    0.13.7 a scroll's clear_cache() only rebinds the watchdog's selector_map to None
    (browser/watchdogs/dom_watchdog.py:848-850), while the session's
    _cached_selector_map still holds the pre-scroll dict (browser/session.py:2474),
    and get_selector_map() returns that cached dict first (session.py:2716-2721).
    Nothing is rebuilt or renumbered, but after a scroll that dict no longer
    matches the page. K reads the watchdog's map, so after a scroll it sees "no
    selector map" and fails closed (Review Focus 1)."""
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


async def _resolve_indexed(
    cdp_session, indexed_id: int, group: str
) -> Tuple[Dict[str, Any], bool, bool]:
    """(the clauses call's argument for the indexed node, it is not in the main
    frame, it is in a shadow root). Both facts come from one call in the indexed
    node's OWN world (see _MAIN_FRAME_FN). A failed resolve leaves the argument
    null, which clause 5 rejects."""
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
        where = (frame_check.get("result") or {}).get("value")
        if not isinstance(where, dict):
            where = {}
        in_shadow_root = where.get("in_shadow_root") is not False
        if where.get("main_frame") is True:
            return {"objectId": resolved_object_id}, False, in_shadow_root
        return {"value": None}, True, in_shadow_root
    return {"value": None}, False, False


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
    ua_map_node = _map_node_in_user_agent_root(same_target)

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
        indexed_arg, indexed_not_main_frame, indexed_in_shadow_root = await _resolve_indexed(
            cdp_session, indexed_id, group
        )
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
        # The ancestor read runs on every read, even when the map holds no node
        # other than the index: B3 must see a web-component ancestor regardless.
        ancestor_ids, ancestor_hosts_shadow = await _light_dom_ancestors(
            cdp_session,
            candidate_object,
            group,
            candidate_tag,
            outside_document_tree=clauses.get("in_document_tree") is False,
        )
        if _hosts_author_shadow_root(node):
            unknown.append("the candidate hosts a shadow root (web component)")
        if ancestor_hosts_shadow:
            unknown.append("an ancestor of the candidate hosts a shadow root (web component)")
        if indexed_not_main_frame:
            unknown.append("the indexed node is not in the main frame")
        if indexed_in_shadow_root:
            unknown.append("the indexed node is in a shadow root")
        if ua_map_node:
            unknown.append(
                "a selector-map node sits inside a browser-internal (user-agent) shadow root"
            )

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
            inside_other_indexed=bool(ancestor_ids & other_indexed_ids),
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
