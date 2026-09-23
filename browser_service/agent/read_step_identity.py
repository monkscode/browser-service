"""
Read-step identity check for the agent-candidate path (wrong-element fix, K v1).

When the candidate identity guard (actions._identity_mismatch) rejects a unique
agent candidate on a READ step, the index — not the candidate — can be the
wrong evidence. u07: the agent indexed the sidebar "Dashboard" link while its
candidate named the page's <h6> heading; the reject sent the cascade back to
the link, and the test passed while verifying the wrong element.

The call site may accept such a candidate only when one live read proves all
five clauses (evaluation spec §13.2, amended after the S1 spike, 2026-09-23):

1. the candidate's node is not the node of any selector-map entry — compared by
   backend_node_id over the map's VALUES, never by key or xpath;
2. it is not inside the indexed node;
3. it does not contain the indexed node;
4. it is not a <label> whose control is the indexed node;
5. the indexed node, resolved by its backend id, is connected and keeps its tag.

Two guards keep clause 1 honest, because "not in the map" has causes other
than "browser-use judged it not interactive":
- freshness: a selector-map node of the candidate's tag that is no longer
  connected means the page re-rendered after the map was built — a clone of an
  indexed node would otherwise read "not indexed" (S1 verifier, D1);
- visibility: browser-use indexes an interactive node only when it is visible,
  intersects the viewport horizontally, lies within ± 1000 px of it vertically
  (browser_use/dom/service.py:339-344) and is not painted over by another
  element (browser_use/dom/serializer/paint_order.py) — so a hidden, off-screen
  or covered candidate proves nothing. Known hole: a hit-test ignores
  pointer-events:none overlays, which browser-use's paint-order filter does not.

Everything uncertain reads UNKNOWN and keeps the reject — the opposite of
_identity_mismatch's own unknown-means-accept rule, on purpose.

Referenced by: browser_service.agent.actions.find_unique_locator_action
Depends on: browser-use BrowserSession (_dom_watchdog.selector_map,
get_or_create_cdp_session), a Playwright Page
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadIdentityFacts:
    """What one live read established. A non-empty ``unknown`` means: cannot decide."""

    unknown: Tuple[str, ...] = ()
    indexed_ok: bool = False
    is_indexed: bool = False
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
    if facts.inside:
        return False, "clause 2: the candidate is inside the indexed node"
    if facts.contains:
        return False, "clause 3: the candidate contains the indexed node"
    if facts.labels_index:
        return False, "clause 4: the candidate is the label of the indexed node"
    return True, "all five clauses hold"
