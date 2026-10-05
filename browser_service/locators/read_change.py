"""Did a read's value change because of the action before it? (F1)

Before bs performs an action it keeps the page's HTML (snapshot_html). When a later
READ is located, read_changed_since parses that HTML inside the live page (DOMParser:
no script runs, nothing loads, no extra tab) and evaluates the read's OWN locator
there. "changed" = the locator matched exactly one element before the action and its
normalized textContent differs from the live element's. NLRF then makes the test wait
for that change. Anything this cannot evaluate faithfully is "unknown" (no mark =
today's behaviour): frame hops, engines other than css / id= / xpath= / nth=, and any
locator whose evaluation on the LIVE page does not find exactly Playwright's element.

Referenced by: browser_service/agent/registration.py (find_unique_locator).
Depends on: playwright page objects (duck-typed).
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

SNAPSHOT_TIMEOUT_S = 2.0
EVAL_TIMEOUT_S = 2.0

_EVAL_JS = r"""([html, addr, live]) => {
  const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const segs = addr.split(/\s>>\s(?!>)/);
  const run = (doc) => {
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
          const it = (r.ownerDocument || r).evaluate(xp, r, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
          for (let i = 0; i < it.snapshotLength; i++) out.push(it.snapshotItem(i));
        }
      } else {
        for (const r of roots) for (const e of r.querySelectorAll(s)) out.push(e);
      }
      roots = [...new Set(out)];
    }
    return roots;
  };
  try {
    const liveHits = run(document);
    if (liveHits.length !== 1 || liveHits[0] !== live) return 'unknown';
    const preHits = run(new DOMParser().parseFromString(html, 'text/html'));
    if (preHits.length === 0) return 'absent';
    if (preHits.length > 1) return 'ambiguous';
    return norm(preHits[0].textContent) !== norm(live.textContent) ? 'changed' : 'same';
  } catch (e) { return 'unknown'; }
}"""


async def snapshot_html(page) -> str | None:
    """The page's current HTML (main frame), or None when it cannot be read in time."""
    try:
        return await asyncio.wait_for(page.content(), timeout=SNAPSHOT_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001 — a snapshot must never break discovery
        logger.info(f"   read-change snapshot skipped ({type(e).__name__})")
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
        logger.info(f"   read-change check skipped ({type(e).__name__})")
        return "unknown"
