"""
K v1's live read against real Chromium (fixture lane — runs in CI).

The production read goes through browser-use's CDP session. Here the same
cdp_use-style call shape is served by Playwright's own CDP session
(page.context.new_cdp_session), and the selector map is built from REAL
backend ids. The S1 verifier measured both sessions return the same
backendNodeId (25 of 25 probes). The browser-use session itself is proven by
tests/test_agent/test_live_read_step_identity.py (live lane) and by the live u07 runs.
"""

import asyncio
import contextlib
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.read_step_identity import (
    READ_IDENTITY_TIMEOUT_S,
    read_identity_facts,
    read_step_verdict,
)

pytestmark = pytest.mark.integration

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "test_locators"
    / "locator_fixtures"
    / "read_step_identity.html"
)

# Every interactive element on the fixture page — what browser-use would index.
# Re-measured 2026-09-23 on a real browser-use 0.13.7 session after the F1 web
# component elements were added (throwaway probe, session scratchpad, never in
# the worktree). browser-use also indexes the <button> inside #xclosed's CLOSED
# shadow root, but Playwright cannot select into closed shadow content (0 matches
# for "#xclosed button"), so that one node is deliberately left out here — this
# tuple is "every node the fixture-lane tests can address", not the full map.
INDEXED = (
    "#nav-admin",
    "a.menu-item",
    "a.shopping_cart_link",
    "#shopping_cart_container",
    "#export",
    "#save",
    "#save-link",
    "#city",
    "#flash button.close",
    "#dropdown",
    "#badge",
    "#alert2 button.close",
    "#xbtn button.inner",
    "#xbtn span.xb-lbl",
    "#xclosed span.xc-lbl",
    "details",
    "summary",
)


class _Send:
    """cdp_use-style ``send.<Domain>.<method>(params=..., session_id=...)`` over Playwright CDP."""

    def __init__(self, pw_cdp):
        self._pw_cdp = pw_cdp

    def __getattr__(self, domain):
        pw_cdp = self._pw_cdp

        class _Domain:
            def __getattr__(self, method):
                async def call(params=None, session_id=None):
                    return await pw_cdp.send(f"{domain}.{method}", params or {})

                return call

        return _Domain()


class _FakeBrowserSession:
    def __init__(self, pw_cdp, target_id, selector_map):
        self._dom_watchdog = SimpleNamespace(selector_map=selector_map)
        self._cdp_session = SimpleNamespace(
            cdp_client=SimpleNamespace(send=_Send(pw_cdp)), session_id=None, target_id=target_id
        )
        self.requested = []

    async def get_or_create_cdp_session(self, target_id=None, focus=True):
        self.requested.append((target_id, focus))
        return self._cdp_session


@contextlib.asynccontextmanager
async def _page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()
        await page.set_content(FIXTURE.read_text(encoding="utf-8"))
        cdp = await page.context.new_cdp_session(page)
        try:
            yield page, cdp
        finally:
            await browser.close()


async def _node(page, cdp, selector):
    name = "__t_" + uuid.uuid4().hex
    await page.locator(selector).evaluate("(el, n) => { window[n] = el; }", name)
    parked = await cdp.send("Runtime.evaluate", {"expression": f"window[{name!r}]"})
    described = await cdp.send("DOM.describeNode", {"objectId": parked["result"]["objectId"]})
    await page.evaluate("(n) => { delete window[n]; }", name)
    return described["node"]["backendNodeId"], described["node"]["nodeName"]


async def _node_in_frame(cdp, frame_selector, selector):
    """Resolve a node inside a same-origin child frame by backend id, through the
    DOM domain only (no Runtime/window parking, which would need the iframe's own
    execution context) — the same shape _node() returns for the main frame."""
    doc = await cdp.send("DOM.getDocument", {})
    root_id = doc["root"]["nodeId"]
    iframe = await cdp.send("DOM.querySelector", {"nodeId": root_id, "selector": frame_selector})
    described = await cdp.send("DOM.describeNode", {"nodeId": iframe["nodeId"], "pierce": True})
    content_doc_id = described["node"]["contentDocument"]["nodeId"]
    inner = await cdp.send("DOM.querySelector", {"nodeId": content_doc_id, "selector": selector})
    inner_described = await cdp.send("DOM.describeNode", {"nodeId": inner["nodeId"]})
    return inner_described["node"]["backendNodeId"], inner_described["node"]["nodeName"]


async def _setup(page, cdp, indexed, map_selectors=INDEXED):
    """A selector map of real backend ids, and element_data for the indexed node."""
    target_id = (await cdp.send("Target.getTargetInfo"))["targetInfo"]["targetId"]
    selector_map = {}
    # Keyed by index (enumerate), like browser-use's own map — never by backend id,
    # so a regression that compares map KEYS instead of values would fail here.
    for i, selector in enumerate(map_selectors, start=1):
        backend_id, node_name = await _node(page, cdp, selector)
        selector_map[i] = SimpleNamespace(
            backend_node_id=backend_id, node_name=node_name, target_id=target_id
        )
    backend_id, node_name = await _node(page, cdp, indexed)
    element_data = {"tagName": node_name.lower(), "backendNodeId": backend_id}
    return _FakeBrowserSession(cdp, target_id, selector_map), element_data, target_id


async def _verdict(page, cdp, *, indexed, candidate, map_selectors=INDEXED):
    session, element_data, _ = await _setup(page, cdp, indexed, map_selectors=map_selectors)
    facts = await read_identity_facts(page, session, element_data, candidate)
    return facts, read_step_verdict(facts)


# --- accepts -------------------------------------------------------------


async def test_u07_shape_accepts():
    async with _page() as (page, cdp):
        facts, (accept, why) = await _verdict(
            page, cdp, indexed="a.menu-item", candidate="h6.crumb"
        )
    assert accept is True, why
    assert facts.candidate_tag == "h6"
    assert facts.candidate_centre is not None


async def test_q07_cart_shape_accepts():
    async with _page() as (page, cdp):
        _, (accept, why) = await _verdict(
            page, cdp, indexed="#shopping_cart_container", candidate="span.title"
        )
    assert accept is True, why


async def test_mirror_is_accepted_owner_trade_off():
    """Owner-accepted trade-off (2026-09-23): the index is right, the candidate is a
    same-text node with no index and no containment or label link. K cannot tell
    this from u07 and accepts it. 0 such calls among the corpus's 81 rejects."""
    async with _page() as (page, cdp):
        _, (accept, why) = await _verdict(
            page, cdp, indexed="#export", candidate="span.export-note"
        )
    assert accept is True, why


# --- clause rejects -------------------------------------------------------


async def test_reverse_indexed_candidate_rejects_clause_1():
    async with _page() as (page, cdp):
        _, (accept, why) = await _verdict(page, cdp, indexed="#save", candidate="#save-link")
    assert accept is False
    assert "clause 1" in why


async def test_option_inside_select_rejects():
    """q08: the option's box is 0x0 (unknown) AND it is inside the index (clause 2)."""
    async with _page() as (page, cdp):
        facts, (accept, _) = await _verdict(
            page, cdp, indexed="#dropdown", candidate="#dropdown option[selected='selected']"
        )
    assert accept is False
    assert facts.inside is True


async def test_candidate_containing_index_rejects_clause_3():
    """u12: div#flash contains the indexed Close button."""
    async with _page() as (page, cdp):
        _, (accept, why) = await _verdict(
            page, cdp, indexed="#flash button.close", candidate="#flash"
        )
    assert accept is False
    assert "clause 3" in why


async def test_candidate_inside_another_indexed_node_rejects_clause_1():
    """browser-use folds a child that fills an <a>/<button> into its parent
    (serializer.py:785-793), so the inner span is absent from the map by design —
    a candidate reading "not in the map" can still be inside a DIFFERENT indexed node."""
    async with _page() as (page, cdp):
        _, (accept, why) = await _verdict(
            page, cdp, indexed="#save", candidate="#save-link span.lbl"
        )
    assert accept is False
    assert "inside another indexed node" in why


async def test_shadow_host_containing_index_rejects_clause_3():
    """A shadow HOST whose open shadow root holds the indexed button: Node.contains
    stops at the shadow boundary, so clause 3 must walk parentNode/host to catch it."""
    async with _page() as (page, cdp):
        facts, (accept, why) = await _verdict(
            page, cdp, indexed="#alert2 button.close", candidate="#alert2"
        )
    assert accept is False
    assert "clause 3" in why
    assert facts.contains is True


async def test_host_of_another_indexed_node_rejects_clause_1():
    """F1(a): the candidate is a shadow HOST whose shadow root holds a DIFFERENT
    indexed node (#alert2's shadow Close button, indexed separately from #flash's).
    browser-use indexes the button inside the shadow root, never its host, so
    "not in the map" has this fourth cause too."""
    async with _page() as (page, cdp):
        _, (accept, why) = await _verdict(
            page, cdp, indexed="#flash button.close", candidate="#alert2"
        )
    assert accept is False
    assert "hosts another indexed node" in why


# Measured 2026-09-23: real browser-use 0.13.7 indexes BOTH #xbtn's shadow button
# AND its slotted span (see the re-measured INDEXED tuple above) — the light-DOM
# fold-into-parent heuristic the reviewer's evidence relied on for a plain
# <a>/<button> does not apply across this shadow boundary on this fixture. That
# makes span.xb-lbl itself an indexed node here, which trips clause 1's
# is_indexed check before the ancestor walk is ever exercised. The clause 1/2
# ancestor-walk logic this covers is real regardless (the module docstring's own
# "folded children" note says the omission can happen for OTHER shapes), so these
# two tests use a reduced map that omits the span — modelling the shape where it
# is not separately indexed, same as the reviewer's own ad hoc probe fixture.
_MAP_WITHOUT_SLOTTED_SPAN = tuple(s for s in INDEXED if s != "#xbtn span.xb-lbl")


async def test_slotted_into_another_indexed_node_rejects_clause_1():
    """F1(b): a span slotted into #xbtn's shadow button, while #save is indexed —
    the flat-tree ancestor walk must find the (different) indexed button via
    assignedSlot, not the host via parentElement."""
    async with _page() as (page, cdp):
        _, (accept, why) = await _verdict(
            page,
            cdp,
            indexed="#save",
            candidate="#xbtn span.xb-lbl",
            map_selectors=_MAP_WITHOUT_SLOTTED_SPAN,
        )
    assert accept is False
    assert "inside another indexed node" in why


async def test_slotted_into_the_indexed_node_rejects_clause_2():
    """F1(b): the same slotted span, but this time the shadow button ITSELF is the
    indexed node — clause 2, not clause 1."""
    async with _page() as (page, cdp):
        _, (accept, why) = await _verdict(
            page,
            cdp,
            indexed="#xbtn button.inner",
            candidate="#xbtn span.xb-lbl",
            map_selectors=_MAP_WITHOUT_SLOTTED_SPAN,
        )
    assert accept is False
    assert "clause 2" in why


async def test_slotted_into_a_closed_host_reads_unknown():
    """F1(c): assignedSlot is null for a slot inside a CLOSED root, so the flat-tree
    walk cannot see past #xclosed's boundary — it must read unknown, not accept."""
    async with _page() as (page, cdp):
        _, (accept, why) = await _verdict(
            page, cdp, indexed="#save", candidate="#xclosed span.xc-lbl"
        )
    assert accept is False
    assert "closed shadow host" in why


async def test_user_agent_shadow_host_does_not_block_accept():
    """<details> reports a shadowRoots entry (measured: shadowRootType "user-agent",
    Chromium's own internal implementation of the disclosure triangle) — the F1(a)
    host check must ignore it, never treat <details> as hosting another indexed
    node just because it has *a* shadow root. Candidate is <details> itself (the
    host); DETAILS/SUMMARY are dropped from the map so is_indexed/ancestor checks
    do not also fire — real browser-use indexes both (measured), which is a
    separate, correct rejection unrelated to this UA-root guard."""
    async with _page() as (page, cdp):
        _, (accept, why) = await _verdict(
            page,
            cdp,
            indexed="a.menu-item",
            candidate="details",
            map_selectors=tuple(s for s in INDEXED if s not in ("details", "summary")),
        )
    assert accept is True, why


async def test_page_js_error_in_ancestor_walk_fails_closed():
    """F2: a page script that throws inside the ancestor walk must fail the read
    closed (None), not silently read "not inside" and accept."""
    async with _page() as (page, cdp):
        session, element_data, _ = await _setup(page, cdp, "#save")
        await page.evaluate(
            "() => { const d = Object.getOwnPropertyDescriptor(Node.prototype, 'parentElement');"
            " Object.defineProperty(Element.prototype, 'parentElement', {configurable: true,"
            " get() { if (this.classList && this.classList.contains('lbl'))"
            " throw new Error('page broke it'); return d.get.call(this); }}); }"
        )
        facts = await read_identity_facts(page, session, element_data, "#save-link span.lbl")
    assert facts is None
    assert read_step_verdict(facts)[0] is False


async def test_indexed_node_in_a_child_frame_reads_unknown():
    """F3: the indexed node resolves fine by backend id, but it lives in a
    same-origin child frame — the candidate (main frame) must read unknown, never
    a false clause-1/2/3 pass or accept."""
    async with _page() as (page, cdp):
        await page.set_content(
            "<html><body><h6 class='crumb'>Dashboard</h6>"
            "<iframe id='frm' srcdoc=\"<button id='inner'>Click</button>\"></iframe>"
            "</body></html>"
        )
        await page.wait_for_selector("iframe#frm")
        backend_id, node_name = await _node_in_frame(cdp, "#frm", "#inner")
        element_data = {"tagName": node_name.lower(), "backendNodeId": backend_id}
        target_id = (await cdp.send("Target.getTargetInfo"))["targetInfo"]["targetId"]
        session = _FakeBrowserSession(
            cdp,
            target_id,
            {
                1: SimpleNamespace(
                    backend_node_id=backend_id, node_name=node_name, target_id=target_id
                )
            },
        )
        facts = await read_identity_facts(page, session, element_data, "h6.crumb")
    accept, why = read_step_verdict(facts)
    assert accept is False
    assert "not in the main frame" in why


async def test_label_for_indexed_input_rejects_clause_4():
    async with _page() as (page, cdp):
        _, (accept, why) = await _verdict(page, cdp, indexed="#city", candidate="label[for=city]")
    assert accept is False
    assert "clause 4" in why


async def test_indexed_node_removed_rejects_clause_5():
    async with _page() as (page, cdp):
        session, element_data, _ = await _setup(page, cdp, "a.menu-item")
        await page.evaluate("document.querySelector('a.menu-item').remove()")
        facts = await read_identity_facts(page, session, element_data, "h6.crumb")
    accept, why = read_step_verdict(facts)
    assert accept is False
    assert "clause 5" in why


async def test_indexed_node_replaced_by_clone_rejects_clause_5():
    async with _page() as (page, cdp):
        session, element_data, _ = await _setup(page, cdp, "a.menu-item")
        await page.evaluate(
            "(() => { const a = document.querySelector('a.menu-item'); a.replaceWith(a.cloneNode(true)); })()"
        )
        facts = await read_identity_facts(page, session, element_data, "h6.crumb")
    accept, why = read_step_verdict(facts)
    assert accept is False
    assert "clause 5" in why


# --- fail-closed unknowns --------------------------------------------------


async def test_same_tag_indexed_node_replaced_reads_unknown():
    """S1 verifier D1: an indexed node swapped for an identical clone gets a new backend
    id and would read "not indexed". The freshness guard must catch it."""
    async with _page() as (page, cdp):
        session, element_data, _ = await _setup(page, cdp, "a.menu-item")
        await page.evaluate(
            "(() => { const s = document.querySelector('#badge'); s.replaceWith(s.cloneNode(true)); })()"
        )
        facts = await read_identity_facts(page, session, element_data, "#badge")
    accept, why = read_step_verdict(facts)
    assert facts.is_indexed is False  # without the guard this would be an ACCEPT
    assert accept is False
    assert "re-rendered" in why


async def test_hidden_candidate_reads_unknown():
    async with _page() as (page, cdp):
        _, (accept, why) = await _verdict(
            page, cdp, indexed="a.menu-item", candidate="span.hidden-dup"
        )
    assert accept is False
    assert "not visible" in why


async def test_far_offscreen_candidate_reads_unknown():
    async with _page() as (page, cdp):
        _, (accept, why) = await _verdict(
            page, cdp, indexed="a.menu-item", candidate="span.far-away"
        )
    assert accept is False
    assert "indexing window" in why


async def test_shadow_root_candidate_reads_unknown():
    async with _page() as (page, cdp):
        _, (accept, why) = await _verdict(
            page, cdp, indexed="a.menu-item", candidate="span.in-shadow"
        )
    assert accept is False
    assert "shadow root" in why


async def test_off_right_candidate_reads_unknown():
    """browser-use widens its window vertically only (dom/service.py:339-344)."""
    async with _page() as (page, cdp):
        _, (accept, why) = await _verdict(
            page, cdp, indexed="a.menu-item", candidate="span.off-right"
        )
    assert accept is False
    assert "indexing window" in why


async def test_covered_candidate_reads_unknown():
    """browser-use's paint-order filter drops a covered node from the map."""
    async with _page() as (page, cdp):
        _, (accept, why) = await _verdict(
            page, cdp, indexed="a.menu-item", candidate="span.covered-twin"
        )
    assert accept is False
    assert "covered" in why


# Busies the page's main thread for 10 s, starting on the next task.
_BUSY_JS = "setTimeout(() => { const t = Date.now(); while (Date.now() - t < 10000) {} }, 0); 1"


async def test_hung_renderer_read_stays_bounded():
    """A renderer that hangs mid-read must not stretch the read past its bound.
    The cleanup runs after the timeout too, and page.evaluate has no timeout."""
    async with _page() as (page, cdp):
        session, element_data, _ = await _setup(page, cdp, "a.menu-item")
        real_send = session._cdp_session.cdp_client.send

        class _HangOnEvaluate:
            def __getattr__(self, domain):
                real = getattr(real_send, domain)

                class _Domain:
                    def __getattr__(self, method):
                        async def call(params=None, session_id=None):
                            if (domain, method) == ("Runtime", "evaluate"):
                                await cdp.send("Runtime.evaluate", {"expression": _BUSY_JS})
                                await asyncio.sleep(0.2)
                            return await getattr(real, method)(params=params, session_id=session_id)

                        return call

                return _Domain()

        session._cdp_session.cdp_client.send = _HangOnEvaluate()
        started = time.monotonic()
        facts = await read_identity_facts(page, session, element_data, "h6.crumb")
        elapsed = time.monotonic() - started
    assert facts is None
    assert elapsed < READ_IDENTITY_TIMEOUT_S + 1.5, f"read took {elapsed:.1f} s"


async def test_no_backend_id_reads_unknown():
    async with _page() as (page, cdp):
        session, element_data, _ = await _setup(page, cdp, "a.menu-item")
        element_data.pop("backendNodeId")
        facts = await read_identity_facts(page, session, element_data, "h6.crumb")
    accept, why = read_step_verdict(facts)
    assert accept is False
    assert "no backend id" in why


async def test_no_watchdog_map_reads_unknown():
    """Review Focus 1: a map from get_selector_map()'s fallback is never trusted."""
    async with _page() as (page, cdp):
        session, element_data, _ = await _setup(page, cdp, "a.menu-item")
        session._dom_watchdog = SimpleNamespace(selector_map={})
        facts = await read_identity_facts(page, session, element_data, "h6.crumb")
    accept, why = read_step_verdict(facts)
    assert accept is False
    assert "no selector map" in why


async def test_candidate_that_matches_nothing_returns_none():
    """Review Focus 2: an unparkable candidate is a failed read, never a raise."""
    async with _page() as (page, cdp):
        session, element_data, _ = await _setup(page, cdp, "a.menu-item")
        facts = await read_identity_facts(page, session, element_data, "#does-not-exist")
    assert facts is None
    assert read_step_verdict(facts)[0] is False


async def test_new_document_after_map_rejects():
    """Review Focus 3: a navigation between map build and read."""
    async with _page() as (page, cdp):
        session, element_data, _ = await _setup(page, cdp, "a.menu-item")
        await page.set_content("<html><body><h6 class='crumb'>Dashboard</h6></body></html>")
        facts = await read_identity_facts(page, session, element_data, "h6.crumb")
    assert read_step_verdict(facts)[0] is False


async def test_candidate_on_another_page_reads_unknown():
    """Review Focus 4: the Playwright page is not the indexed node's target."""
    async with _page() as (page, cdp):
        session, element_data, target_id = await _setup(page, cdp, "a.menu-item")
        other = await page.context.new_page()
        await other.set_content("<html><body><p>other tab</p></body></html>")
        other_cdp = await page.context.new_cdp_session(other)
        wrong_target = _FakeBrowserSession(other_cdp, target_id, session._dom_watchdog.selector_map)
        facts = await read_identity_facts(page, wrong_target, element_data, "h6.crumb")
    accept, why = read_step_verdict(facts)
    assert accept is False
    assert "not reachable" in why


# --- hygiene ---------------------------------------------------------------


async def test_asks_for_the_indexed_target_without_stealing_focus():
    async with _page() as (page, cdp):
        session, element_data, target_id = await _setup(page, cdp, "a.menu-item")
        await read_identity_facts(page, session, element_data, "h6.crumb")
    assert session.requested == [(target_id, False)]


async def test_window_global_is_cleaned_up():
    async with _page() as (page, cdp):
        session, element_data, _ = await _setup(page, cdp, "a.menu-item")
        await read_identity_facts(page, session, element_data, "h6.crumb")
        left = await page.evaluate(
            "Object.keys(window).filter(k => k.startsWith('__bs_read_identity_'))"
        )
    assert left == []
