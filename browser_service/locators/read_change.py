"""Did a read's value change because of the action before it? (F1)

Before bs performs an action it keeps the page's HTML (snapshot_html). When a later
READ is located, read_changed_since parses that HTML inside the live page (DOMParser:
no script runs, nothing loads, no extra tab) and evaluates the read's OWN locator
there. "changed" = the locator matched exactly one element before the action and its
normalized textContent differs from the live element's. NLRF then makes the test wait
for that change. These cases are "unknown" (no mark = today's behaviour): frame hops,
engines other than css / id= / xpath= / nth=, any locator whose evaluation on the LIVE
page does not find exactly Playwright's element, a live element whose shown text
(innerText, what Get Text returns) is not its textContent (hidden text), and a locator
that does not find that same element, once, when the CURRENT page is re-parsed the way
the kept page was (JS-built nesting the HTML parser never produces).

Residuals, by design: (1) nesting the HTML parser never produces that exists only
BEFORE the action (the action removes or re-renders it) and touches the address - the
read target itself, an element on its path, or a block above a positional step - can
still read as "changed"; (2) hidden text that existed only before the action is not seen
(the live element shows none now). Accepted limit (owner): a read target holding an
inline <script> (e.g. JSON-LD) or an SVG <title> is "unknown" even when its visible text
changed (the hidden-text check), so such reads get no mark.

CAME BACK (read_came_back): bs keeps the page before EVERY action it performs, P_0 ... P_k-1
(None = an action was performed and no page was kept: a gap). For each READ it locates, the
read's own locator is evaluated on every kept page (T_i = its text there: one match -> that
text, none -> absent, several -> ambiguous) and compared with its live text L, by the SAME
engine as read_changed_since. came_back = some T_i == L and a LATER T_j != L (absent or
ambiguous counts as different), i.e. the value returned to an earlier one, so a wait that stops
at the first change of the read could stop on the in-between value. Fail-safe, each = True (the
test gets no inserted lines = today's behaviour): a gap in the kept pages; a read that cannot
be checked (empty locator, a frame hop, an engine other than css / id= / xpath= / nth=, a live
element that is not exactly Playwright's, hidden text, a current page that re-parses to
something else, a <noscript> target live or on a kept page); any error or timeout. No kept
page -> False. The caller adds the cases it owns: a read flagged earlier in the workflow, a
read in a frame, a re-ranked locator swap (tasks/workflow.py commit_reranked_winner).

Referenced by: browser_service/agent/registration.py (find_unique_locator).
Depends on: playwright page objects (duck-typed); playwright.async_api.Error (to tell an
expected failure from a bug when logging).
"""

import asyncio
import logging

from playwright.async_api import Error as PlaywrightError

logger = logging.getLogger(__name__)

SNAPSHOT_TIMEOUT_S = 2.0
EVAL_TIMEOUT_S = 2.0

# ONE engine, shared by both evaluators below: `norm` / `textOf` (the text of an element with every
# <noscript> descendant removed), `makeRun(addr)` (the read's locator evaluated on a document) and
# `liveUnknown(run, live)` (every reason the LIVE side cannot be compared). It runs in the page.
_ENGINE_JS = r"""
  const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();
  // A parsed document has scripting off, so <noscript> children parse as elements; on the live page
  // they are one raw-text node. Compare text with every <noscript> descendant removed on both sides.
  const textOf = (el) => {
    const c = el.cloneNode(true);
    c.querySelectorAll('noscript').forEach((n) => n.remove());
    return norm(c.textContent);
  };
  const makeRun = (addr) => {
    const segs = addr.split(/\s>>\s(?!>)/);
    return (doc) => {
      let roots = [doc];
      for (const raw of segs) {
        let s = raw.trim(), m;
        if ((m = s.match(/^nth=(-?\d+)$/))) {
          const k = Number(m[1]); const el = k < 0 ? roots[roots.length + k] : roots[k];
          roots = el ? [el] : []; continue;
        }
        if (/^[a-z][\w-]*=/i.test(s) && !/^(css|id|xpath)=/i.test(s)) throw new Error('engine');
        if (/^css=/i.test(s)) s = s.slice(4);
        if ((m = s.match(/^id=(.+)$/i))) s = '[id="' + m[1].replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"]';
        const out = [];
        if (/^xpath=/i.test(s) || s.startsWith('//') || s.startsWith('(')) {
          const xp = /^xpath=/i.test(s) ? s.slice(6) : s;
          for (const r of roots) {
            // Playwright's XPath engine scopes a leading '/' to a non-document root with a '.'.
            const path = r.nodeType !== 9 && xp.startsWith('/') ? '.' + xp : xp;
            const it = (r.ownerDocument || r).evaluate(path, r, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
            for (let i = 0; i < it.snapshotLength; i++) out.push(it.snapshotItem(i));
          }
        } else {
          for (const r of roots) for (const e of r.querySelectorAll(s)) out.push(e);
        }
        roots = [...new Set(out)];
      }
      return roots;
    };
  };
  const liveUnknown = (run, live) => {
    const liveHits = run(document);
    if (liveHits.length !== 1 || liveHits[0] !== live) return true;
    if (live.tagName === 'NOSCRIPT') return true;
    // Get Text returns innerText (what is shown); the comparison uses textContent. Text that is
    // not shown (display:none, ...) makes the two differ: no mark.
    const squash = (s) => (s || '').replace(/\s+/g, '').toLowerCase();
    if (squash(live.innerText) !== squash(textOf(live))) return true;
    // The kept page went through the HTML parser; the live DOM may hold nesting the parser never
    // produces (JS-built). Parse the CURRENT page the same way: the address must find exactly one
    // element there, with the live element's text, or the two sides are not comparable.
    // Use the page's own doctype, serialized as page.content() does (none = quirks mode).
    const dt = document.doctype ? new XMLSerializer().serializeToString(document.doctype) : '';
    const cur = new DOMParser().parseFromString(dt + document.documentElement.outerHTML, 'text/html');
    const curHits = run(cur);
    if (curHits.length !== 1 || textOf(curHits[0]) !== textOf(live)) return true;
    return false;
  };
"""

# One kept page -> 'changed' | 'same' | 'absent' | 'ambiguous' | 'unknown'.
_EVAL_JS = (
    "([html, addr, live]) => {"
    + _ENGINE_JS
    + r"""
  try {
    const run = makeRun(addr);
    if (liveUnknown(run, live)) return 'unknown';
    const preHits = run(new DOMParser().parseFromString(html, 'text/html'));
    if (preHits.length === 0) return 'absent';
    if (preHits.length > 1) return 'ambiguous';
    if (preHits[0].tagName === 'NOSCRIPT') return 'unknown';
    return textOf(preHits[0]) !== textOf(live) ? 'changed' : 'same';
  } catch (e) { return 'unknown'; }
}"""
)

# Every kept page at once -> one status per page ('same' = the read's text there equals its live
# text; 'changed' | 'absent' | 'ambiguous' = it differs there), or the string 'unknown' when the
# live side is not comparable. A kept page that cannot be compared is 'unknown' too.
_EVAL_ALL_JS = (
    "([htmls, addr, live]) => {"
    + _ENGINE_JS
    + r"""
  try {
    const run = makeRun(addr);
    if (liveUnknown(run, live)) return 'unknown';
    const liveText = textOf(live);
    return htmls.map((html) => {
      try {
        const preHits = run(new DOMParser().parseFromString(html, 'text/html'));
        if (preHits.length === 0) return 'absent';
        if (preHits.length > 1) return 'ambiguous';
        if (preHits[0].tagName === 'NOSCRIPT') return 'unknown';
        return textOf(preHits[0]) !== liveText ? 'changed' : 'same';
      } catch (e) { return 'unknown'; }
    });
  } catch (e) { return 'unknown'; }
}"""
)


def _log_skip(what: str, error: Exception) -> None:
    """INFO for an expected failure (timeout, Playwright error); WARNING for anything else (a bug)."""
    if isinstance(error, (TimeoutError, PlaywrightError)):  # Playwright's TimeoutError is an Error
        logger.info(f"   read-change {what} skipped ({type(error).__name__})")
    else:
        logger.warning(f"   read-change {what} skipped ({type(error).__name__}: {error})")


async def snapshot_html(page) -> str | None:
    """The page's current HTML (main frame), or None when it cannot be read in time."""
    try:
        return await asyncio.wait_for(page.content(), timeout=SNAPSHOT_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001 — a snapshot must never break discovery
        _log_skip("snapshot", e)
        return None


async def read_changed_since(page, locator: str, html: str | None) -> str:
    """'changed' | 'same' | 'absent' | 'ambiguous' | 'unknown' for `locator` against the kept HTML."""
    if not html or not locator or ">>>" in locator:
        return "unknown"
    try:
        handle = await page.locator(locator).first.element_handle(timeout=1000)
        if handle is None:
            return "unknown"
        status = await asyncio.wait_for(
            page.evaluate(_EVAL_JS, [html, locator, handle]), timeout=EVAL_TIMEOUT_S
        )
        return status if status in ("changed", "same", "absent", "ambiguous") else "unknown"
    except Exception as e:  # noqa: BLE001 — never break discovery
        _log_skip("check", e)
        return "unknown"


_PAGE_STATUSES = ("same", "changed", "absent", "ambiguous")


def came_back_from_statuses(statuses: list[str]) -> bool:
    """The came-back formula over one status per kept page, oldest first.

    "same" = the read's text on that page equals its live text; "changed" / "absent" /
    "ambiguous" = it differs there. True when some page is "same" and a LATER page differs.
    A status outside those four (the engine's "unknown": not comparable) is fail-safe True."""
    if any(s not in _PAGE_STATUSES for s in statuses):
        return True
    seen_same = False
    for status in statuses:
        if status == "same":
            seen_same = True
        elif seen_same:
            return True
    return False


async def read_came_back(page, locator: str, htmls: list[str | None]) -> bool:
    """Did the read's value come back to an earlier value? (one `page.evaluate` for all pages)

    `htmls` = the page kept before every action bs performed, in order; None = a gap (an action
    was performed and no page was kept). No kept page -> False. Anything that cannot be checked
    (a gap, an empty or framed locator, a live side the engine cannot compare, any error or a
    timeout) -> True, the fail-safe: NLRF then gives the test no inserted lines."""
    if not htmls:
        return False
    if any(not h for h in htmls) or not locator or ">>>" in locator:
        return True
    try:
        handle = await page.locator(locator).first.element_handle(timeout=1000)
        if handle is None:
            return True
        statuses = await asyncio.wait_for(
            page.evaluate(_EVAL_ALL_JS, [htmls, locator, handle]), timeout=EVAL_TIMEOUT_S
        )
        if not isinstance(statuses, list) or len(statuses) != len(htmls):
            return True
        return came_back_from_statuses(statuses)
    except Exception as e:  # noqa: BLE001 — never break discovery
        _log_skip("came-back check", e)
        return True
