"""
Read-target policy (E2b, 2026-09-15 locator evaluation).

A read step's locator must not embed the value it reads. amazon u01 shipped
[aria-label="Portronics Toad 23 Wireless Optical Mouse…"] for "the first
product name": unique and right at discovery, dead on the next search,
because the name IS the data. Corpus read locators (1,921 runs): a literal
of >=40 chars passed 0 of 4; literals under 10 chars passed 40 of 42
(text="John" >> nth=0 on a static table).

Applied after the cascade has chosen, and only when both hold:
- the step is a read (get_text / get_attribute) with an expected_text;
- the locator embeds a literal of >= DATA_LITERAL_MIN_CHARS that equals,
  contains or is contained in expected_text.
The locator is then rewritten to id=<id> when the element carries a stable
id, or otherwise to <container> >> nth=<i> >> <descendant> when the target
sits in a REPEATED container (a list item, a table row, a result card — the
structural evidence that the text is per-item data, not a label; a long
static heading has no such ancestor). Either rewrite is validated live to
resolve to the SAME element. When neither validates, the original stands
(demote, never delete).

Not applied to collections (the collection handler owns them), row-anchored
results (anchored on the QA's row datum on purpose), or iframe results. An
element with a stable id is rewritten straight to that id (validated live,
same-element check) instead of the container-ordinal rewrite below — the
agent-candidate path's element_info can carry an id that never lands in
all_locators, so workflow.py's priority check has nothing there to force.

Referenced by: agent/actions.py
Depends on: locators/action_fit.py, locators/stability.py
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from browser_service.locators.action_fit import FIT_READ_TIMEOUT_MS, READ_ACTIONS
from browser_service.locators.stability import STABLE, classify_locator, score_stability

logger = logging.getLogger(__name__)

# How far up the ancestor chain the container search looks. Measured on
# live amazon.in (2026-09-16): the repeated s-search-result card sits 14
# ancestors above the product title, behind intra-card wrappers that each
# have a single same-tag sibling, so a shallower walk gives up inside one
# card and the data-bound locator stands. The walk returns at the FIRST
# qualifying ancestor, so a larger bound can only turn "no rewrite" into a
# rewrite — it can never substitute a different container for one already
# found, and every other guard (>=2 same-shape peers, shared non-per-item
# evidence, STABLE tokens, count==1, same-element validation) is unchanged.
CONTAINER_WALK_MAX_ANCESTORS = 16

# Same band as the nth-child strategy — structural and positional.
READ_TARGET_PRIORITY = 9

# Mirrors smart_locator.py's PRIORITY_ID (native id attribute is priority 1).
READ_TARGET_ID_PRIORITY = 1

_Q = r"(?:\"([^\"]*)\"|'([^']*)')"
_VALUE_PARTS = (
    re.compile(r"(?:^|>>|\s)text\s*=\s*" + _Q),
    re.compile(r":(?:has-text|text|text-is)\(\s*" + _Q + r"\s*\)"),
    re.compile(r"\[\s*(?:title|aria-label|alt|placeholder)\s*[*^$~|]?=\s*" + _Q + r"\s*\]"),
    re.compile(r"role\s*=\s*[\w-]+\s*\[[^\]]*\bname\s*[*^$~|]?=\s*" + _Q),
    re.compile(r"text\(\)\s*=\s*" + _Q),
    re.compile(r"contains\(\s*(?:text\(\)|\.|normalize-space\([^)]*\))\s*,\s*" + _Q + r"\s*\)"),
    re.compile(r"normalize-space\([^)]*\)\s*=\s*" + _Q),
    re.compile(r"@(?:title|aria-label|alt|placeholder)\s*=\s*" + _Q),
)


def normalize_text(text: Any) -> str:
    """Casefold, collapse whitespace (NBSP included), drop a trailing ellipsis."""
    return " ".join(str(text or "").split()).casefold().rstrip(".…").rstrip()


_HAS_TEXT_ANY = re.compile(r":(?:has-text|text|text-is)\(([^)]*)\)")


def displayed_literals(locator: str) -> List[str]:
    """Every displayed-text literal the locator matches on (text=, :has-text, title / aria-label / alt / placeholder, role name, xpath text forms), quoted or unquoted."""
    out: List[str] = []
    for pattern in _VALUE_PARTS:
        for m in pattern.finditer(locator or ""):
            out.append(next((g for g in m.groups() if g is not None), ""))
    # Unquoted forms, parsed without a backtracking regex: `text=Foo bar` as a
    # whole `>>` hop, and `:has-text(Foo)`.
    for part in (locator or "").split(">>"):
        hop = part.strip()
        if hop[:5].lower() == "text=" and hop[5:6] not in ("'", '"'):
            out.append(hop[5:].strip())
    for m in _HAS_TEXT_ANY.finditer(locator or ""):
        inner = m.group(1).strip()
        if inner[:1] not in ("'", '"'):
            out.append(inner)
    return out


def carries_value(locator: str, observed: Optional[str]) -> bool:
    """True when the locator matches on the observed text: equal, or one contains the other when both are >= 3 characters."""
    value = normalize_text(observed)
    if not value:
        return False
    for literal in displayed_literals(locator):
        lit = normalize_text(literal)
        if lit and (
            lit == value or (len(lit) >= 3 and len(value) >= 3 and (lit in value or value in lit))
        ):
            return True
    return False


def own_text_equals(own: Optional[str], observed: Optional[str]) -> bool:
    """True when an element's own text shows the observed text (normalised), a trailing ... allowed on either side."""
    o, v = normalize_text(own), normalize_text(observed)
    if not o or not v:
        return False
    if o == v:
        return True
    if str(observed).rstrip().endswith(("...", "…")) and o.startswith(v):
        return True
    return str(own).rstrip().endswith(("...", "…")) and v.startswith(o)


def own_texts_match(texts: List[Optional[str]], observed: Optional[str]) -> bool:
    """The element shows what the agent saw in ANY of its own texts."""
    return any(own_text_equals(t, observed) for t in texts or [])


# Every text the element itself shows or announces: rendered text, a control's
# value / placeholder, an image's alt, the title tooltip, the aria-label.
OWN_TEXT_JS = """el => [(el.innerText || '').trim(), el.value || '', el.getAttribute('placeholder') || '',
    el.getAttribute('alt') || '', el.getAttribute('title') || '', el.getAttribute('aria-label') || '']"""

STRUCTURAL_CANDIDATES_JS = """el => {
  const esc = s => CSS.escape(s);
  const ROWS = ['TR', 'LI', 'ARTICLE'];
  const POS = ['TR', 'LI', 'ARTICLE', 'TD', 'TH'];
  const classesOf = n => Array.from(n.classList || []);
  const out = [];
  const add = (kind, locator, tokens) => out.push({kind, locator, tokens});
  const tag = el.tagName.toLowerCase();
  if (el.id) add('id', /^[A-Za-z][\\w-]*$/.test(el.id) ? `id=${el.id}` : `[id="${el.id}"]`, [['id', el.id]]);
  for (const a of ['data-testid', 'data-test', 'data-qa', 'data-cy']) {
    const v = el.getAttribute(a);
    if (v && !v.includes('"')) add('test-id', `[${a}="${v}"]`, [[a, v]]);
  }
  const step = n => { const c = classesOf(n)[0]; return n.tagName.toLowerCase() + (c ? '.' + esc(c) : ''); };
  const stepTokens = n => { const c = classesOf(n)[0]; return c ? [['class', c]] : []; };
  // The shortest selector that finds exactly el inside container a: tag.class, tag, then the
  // hop path with R2 pins.
  const innerSel = a => {
    const tries = [];
    const c = classesOf(el)[0];
    if (c) tries.push([`${tag}.${esc(c)}`, [['class', c]]]);
    tries.push([tag, []]);
    const hops = [], hopTokens = [];
    for (let n = el; n !== a; n = n.parentElement) {
      let h = step(n);
      const same = Array.from(n.parentElement.children).filter(s => s.matches(h));
      if (same.length > 1 || POS.includes(n.tagName))
        h += `:nth-of-type(${Array.from(n.parentElement.children).filter(s => s.tagName === n.tagName).indexOf(n) + 1})`;
      hops.unshift(h);
      hopTokens.push(...stepTokens(n));
    }
    tries.push([hops.join(' > '), hopTokens]);
    for (const [s, t] of tries) {
      const m = a.querySelectorAll(s);
      if (m.length === 1 && m[0] === el) return [s, t];
    }
    return null;
  };
  // R1 + same shape: a container is a shared data-* VALUE (data-component-type, …), a shared
  // data-* attribute NAME, a shared class, or a row/list tag; its peers must each hold exactly
  // one element the inner selector finds. Never a bare div.
  const SHARED_ATTRS = ['data-component-type', 'data-testid', 'data-test', 'data-qa', 'data-cy', 'role'];
  const containerBases = a => {
    const t = a.tagName.toLowerCase();
    const out = [];
    for (const at of SHARED_ATTRS) {
      const v = a.getAttribute(at);
      if (v && !v.includes('"')) out.push([`[${at}="${v}"]`, [[at, v]]]);
    }
    for (const at of Array.from(a.attributes))
      if (at.name.startsWith('data-') && !SHARED_ATTRS.includes(at.name)) out.push([`${t}[${at.name}]`, []]);
    for (const c of classesOf(a)) out.push([`${t}.${esc(c)}`, [['class', c]]]);
    if (ROWS.includes(a.tagName)) out.push([t, []]);
    return out;
  };
  const anchorOf = n => {
    for (let a = n.parentElement; a; a = a.parentElement) {
      if (a.id) return [`#${esc(a.id)}`, a, [['id', a.id]]];
      for (const t of ['data-testid', 'data-test']) {
        const v = a.getAttribute(t);
        if (v && !v.includes('"')) return [`[${t}="${v}"]`, a, [[t, v]]];
      }
      const c = classesOf(a)[0];
      if (c && document.querySelectorAll(`${a.tagName.toLowerCase()}.${esc(c)}`).length === 1)
        return [`${a.tagName.toLowerCase()}.${esc(c)}`, a, [['class', c]]];
    }
    return null;
  };
  let cardFound = false;
  for (let a = el.parentElement, depth = 0; a && a.parentElement && depth < __MAX_DEPTH__ && !cardFound;
       a = a.parentElement, depth++) {
    const inner = innerSel(a);
    if (!inner) continue;
    const [innerS, innerTokens] = inner;
    for (const [base, baseTokens] of containerBases(a)) {
      const peers = Array.from(a.parentElement.children).filter(p => p !== a && p.matches(base)
        && p.querySelectorAll(innerS).length === 1);
      if (!peers.length && !ROWS.includes(a.tagName)) continue;   // a lone row still means "row 1"
      const sel = (a.parentElement.tagName === 'TBODY' ? 'tbody > ' : '') + base;
      let scope = '', scopeTokens = [], root = document;
      const anc = anchorOf(a);
      if (anc && document.querySelectorAll(sel).length !== anc[1].querySelectorAll(sel).length)
        [scope, root, scopeTokens] = [anc[0] + ' >> ', anc[1], anc[2]];
      const index = Array.from(root.querySelectorAll(sel)).indexOf(a);
      if (index < 0) continue;
      add('card-path', `${scope}${sel} >> nth=${index} >> ${innerS}`, [...scopeTokens, ...baseTokens, ...innerTokens]);
      cardFound = true;
      break;
    }
  }
  const cls = classesOf(el);
  for (const c of cls) add('class', `${tag}.${esc(c)}`, [['class', c]]);
  if (cls.length > 1) add('class', tag + cls.map(c => '.' + esc(c)).join(''), cls.map(c => ['class', c]));
  // Anchored short path: the nearest uniquely identifiable ancestor (id, test id or a class
  // unique in the document) and the shortest selector inside it — generic, usually position-free.
  const near = anchorOf(el);
  if (near) {
    const inside = innerSel(near[1]);
    if (inside) add('anchored-path', `${near[0]} >> ${inside[0]}`, [...near[2], ...inside[1]]);
  }
  const xpathOf = (node, stopAtId) => {
    const parts = [];
    for (let n = node; n && n.nodeType === 1; n = n.parentElement) {
      if (stopAtId && n !== node && n.id) { parts.unshift(`//*[@id="${n.id}"]`); return [parts.join('/'), n.id]; }
      let i = 1;
      for (let s = n.previousElementSibling; s; s = s.previousElementSibling) if (s.tagName === n.tagName) i++;
      parts.unshift(`${n.tagName.toLowerCase()}[${i}]`);
    }
    return ['/' + parts.join('/'), null];
  };
  const [anchored, anchorId] = xpathOf(el, true);
  if (anchorId) add('anchored-xpath', 'xpath=' + anchored, [['id', anchorId]]);
  add('absolute-xpath', 'xpath=' + xpathOf(el, false)[0], []);
  return out;
}""".replace("__MAX_DEPTH__", str(CONTAINER_WALK_MAX_ANCESTORS))

ELEMENT_FROM_POINT_JS = "([x, y]) => document.elementFromPoint(x, y)"


async def _unique_same(search_context, locator: str, handle) -> bool:
    try:
        loc = search_context.locator(locator)
        if await loc.count() != 1:
            return False
        same = await loc.evaluate("(el, t) => el === t", handle, timeout=FIT_READ_TIMEOUT_MS)
    except Exception:
        return False
    return same is True


_VISIBLE_RETRY_KINDS = frozenset({"id", "test-id", "class", "anchored-path"})


async def structural_address(search_context, handle, observed: str) -> Optional[Dict[str, Any]]:
    """The first structural address of the element that is stable, value-free, unique and the same node — with Playwright's `>> visible=true` filter for a hidden twin — or None."""
    try:
        candidates = await handle.evaluate(STRUCTURAL_CANDIDATES_JS)
    except Exception as e:
        logger.info(f"   ⚠️ Read-address candidate read failed: {e}")
        return None
    for cand in candidates or []:
        locator = cand.get("locator") or ""
        if any(score_stability(kind, value) != STABLE for kind, value in cand.get("tokens") or []):
            continue
        if carries_value(locator, observed):
            continue
        if await _unique_same(search_context, locator, handle):
            return {"locator": locator, "kind": cand.get("kind", "")}
        # A hidden copy (display:none menu, responsive twin) makes an otherwise
        # good address match twice — Playwright's visibility filter keeps the real one.
        if cand.get("kind") in _VISIBLE_RETRY_KINDS and await _unique_same(
            search_context, locator + " >> visible=true", handle
        ):
            return {"locator": locator + " >> visible=true", "kind": cand.get("kind", "")}
    return None


async def _pointed_sources(search_context, element_data, point) -> List[Tuple[str, Any]]:
    sources: List[Tuple[str, Any]] = []
    xpath = ((element_data or {}).get("xpath") or "").strip()
    if xpath:
        try:
            handle = await search_context.locator("xpath=/" + xpath.lstrip("/")).element_handle(
                timeout=FIT_READ_TIMEOUT_MS
            )
            if handle:
                sources.append(("pointed element", handle))
        except Exception as e:
            logger.info(f"   ⚠️ Read-address: indexed element not reachable: {e}")
    if point is not None:
        try:
            js_handle = await search_context.evaluate_handle(
                ELEMENT_FROM_POINT_JS, [point[0], point[1]]
            )
            handle = js_handle.as_element()
            if handle:
                sources.append(("screen point", handle))
        except Exception as e:
            logger.info(f"   ⚠️ Read-address: vision point lookup failed: {e}")
    return sources


def _short(text: Any) -> str:
    return normalize_text(text)[:40]


async def resolve_read_address(
    search_context,
    observed: str,
    element_data: Optional[Dict[str, Any]],
    point: Optional[Tuple[float, float]],
) -> Dict[str, Any]:
    """Confirm what the agent pointed at — the indexed element, then the element under its vision point — by its own texts, and return its structural address, or {'unconfirmed': reason}."""
    reasons: List[str] = []
    for source, handle in await _pointed_sources(search_context, element_data, point):
        try:
            own = await handle.evaluate(OWN_TEXT_JS)
        except Exception:
            continue
        if not own_texts_match(own, observed):
            shown = next((t for t in own if t), "")
            reasons.append(f"the {source} shows {_short(shown)!r}, not {_short(observed)!r}")
            continue
        address = await structural_address(search_context, handle, observed)
        if address:
            return {**address, "source": source}
        reasons.append(
            f"the {source} shows {_short(observed)!r} but has no unique address without that text"
        )
    return {"unconfirmed": "; ".join(reasons) or "the agent pointed at no element"}


# Walk up from the target to the nearest ancestor that is one of >=2
# same-shape siblings (each holding exactly one element like the target),
# and describe it as a container selector + its document-order index + the
# target's selector inside it. Only shared, non-per-item evidence names the
# container: a shared data-*/role value, a shared class, or li/tr/article
# under a parent with an id or class. Returns null when none qualifies.
CONTAINER_ORDINAL_JS = """el => {
    const SHARED_ATTRS = ['data-component-type', 'data-testid', 'data-test', 'data-qa',
                          'data-cy', 'role'];
    const ROW_TAGS = ['li', 'tr', 'article'];
    const tag = el.tagName.toLowerCase();
    const cls = el.classList.length ? el.classList[0] : '';
    let node = el;
    for (let depth = 0; depth < __MAX_DEPTH__ && node.parentElement; depth++) {
        const tokens = [];
        let descendant = '';
        if (node !== el) {
            const withClass = cls ? tag + '.' + CSS.escape(cls) : '';
            if (withClass && node.querySelectorAll(withClass).length === 1) {
                descendant = withClass;
                tokens.push(['class', cls]);
            } else if (node.querySelectorAll(tag).length === 1) {
                descendant = tag;
            } else {
                node = node.parentElement;
                continue;
            }
        }
        const parent = node.parentElement;
        const peers = Array.from(parent.children).filter(c =>
            c.tagName === node.tagName
            && (!descendant || c.querySelectorAll(descendant).length === 1));
        if (peers.length >= 2) {
            const ntag = node.tagName.toLowerCase();
            let sel = '';
            for (const a of SHARED_ATTRS) {
                const v = node.getAttribute(a);
                if (v && peers.filter(p => p.getAttribute(a) === v).length >= 2) {
                    sel = '[' + a + '=' + JSON.stringify(v) + ']';
                    tokens.push([a, v]);
                    break;
                }
            }
            if (!sel) {
                const shared = Array.from(node.classList)
                    .find(c => peers.filter(p => p.classList.contains(c)).length >= 2);
                if (shared) {
                    sel = ntag + '.' + CSS.escape(shared);
                    tokens.push(['class', shared]);
                }
            }
            if (!sel && ROW_TAGS.includes(ntag)) {
                if (parent.id) {
                    sel = '#' + CSS.escape(parent.id) + ' > ' + ntag;
                    tokens.push(['id', parent.id]);
                } else if (parent.classList.length) {
                    sel = parent.tagName.toLowerCase() + '.'
                        + CSS.escape(parent.classList[0]) + ' > ' + ntag;
                    tokens.push(['class', parent.classList[0]]);
                }
            }
            if (sel) {
                const index = Array.from(document.querySelectorAll(sel)).indexOf(node);
                if (index >= 0) {
                    return {container: sel, index: index, descendant: descendant, tokens: tokens};
                }
            }
        }
        node = parent;
    }
    return null;
}""".replace("__MAX_DEPTH__", str(CONTAINER_WALK_MAX_ANCESTORS))


async def _resolves_to_same_element(search_context, locator: str, candidate: str) -> bool:
    """True when ``candidate`` resolves to exactly one element, the SAME
    element ``locator`` resolves to. Every failure — a non-unique count, a
    raised exception, a timeout — is False; never turn an uncertain read
    into an accept."""
    try:
        target = search_context.locator(locator)
        rewritten = search_context.locator(candidate)
        if await rewritten.count() != 1:
            return False
        handle = await target.element_handle(timeout=FIT_READ_TIMEOUT_MS)
        same = await rewritten.evaluate("(el, t) => el === t", handle, timeout=FIT_READ_TIMEOUT_MS)
    except Exception as e:
        logger.info(f"   ⚠️ Read-target validation failed for '{candidate}': {e}")
        return False
    return same is True


async def build_container_ordinal(search_context, locator: str) -> Optional[str]:
    """A container-ordinal locator that resolves to exactly the element
    ``locator`` resolves to, or None. Every failure is None."""
    target = search_context.locator(locator)
    try:
        info = await target.evaluate(CONTAINER_ORDINAL_JS, timeout=FIT_READ_TIMEOUT_MS)
    except Exception as e:
        logger.info(f"   ⚠️ Read-target container read failed for '{locator}': {e}")
        return None
    if not isinstance(info, dict) or not info.get("container"):
        return None
    for kind, token in info.get("tokens") or []:
        if score_stability(kind, token) != STABLE:
            return None
    candidate = f"{info['container']} >> nth={int(info['index'])}"
    if info.get("descendant"):
        candidate += f" >> {info['descendant']}"
    return (
        candidate if await _resolves_to_same_element(search_context, locator, candidate) else None
    )


async def apply_read_target_policy(
    search_context,
    result: Dict[str, Any],
    action: Optional[str],
    expected_text: Optional[str],
    iframe_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Rewrite a found read locator that embeds the value being read."""
    if action not in READ_ACTIONS or not result.get("found") or iframe_context:
        return result
    if result.get("row_anchored") or result.get("element_type") == "collection":
        return result
    locator = result.get("best_locator") or ""
    literal = locator if carries_value(locator, expected_text) else ""
    if not literal:
        return result
    info_id = ((result.get("element_info") or {}).get("id") or "").strip()
    if info_id and score_stability("id", info_id) == STABLE:
        id_locator = f"id={info_id}"
        if await _resolves_to_same_element(search_context, locator, id_locator):
            id_stability = classify_locator(id_locator)
            logger.info(
                f"   📎 READ TARGET REWRITE: '{locator}' embeds the value being read — using "
                f"'{id_locator}' (signal: read-target-id)"
            )
            return {
                **result,
                "best_locator": id_locator,
                "stability": id_stability,
                # The only entry: workflow.py's PHASE-2 re-ranker scores every
                # unique+valid entry and would otherwise re-promote the original.
                "all_locators": [
                    {
                        "type": "id",
                        "locator": id_locator,
                        "priority": READ_TARGET_ID_PRIORITY,
                        "strategy": "ID selector from element_data",
                        "count": 1,
                        "unique": True,
                        "valid": True,
                        "validated": True,
                        "validation_method": "playwright",
                        "stability": id_stability,
                    }
                ],
                "read_target_rewritten_from": locator,
            }
    rewritten = await build_container_ordinal(search_context, locator)
    if not rewritten:
        logger.info(
            f"   📎 READ TARGET: '{locator}' embeds the value being read but no repeated "
            f"container validated — keeping it (signal: read-target-kept)"
        )
        return result
    stability = classify_locator(rewritten)
    logger.info(
        f"   📎 READ TARGET REWRITE: '{locator}' embeds the value being read — using "
        f"'{rewritten}' (signal: read-target-rewrite)"
    )
    return {
        **result,
        "best_locator": rewritten,
        "stability": stability,
        # The only entry: workflow.py's PHASE-2 re-ranker scores every
        # unique+valid entry and would otherwise re-promote the original.
        "all_locators": [
            {
                "type": "read-target-ordinal",
                "locator": rewritten,
                "priority": READ_TARGET_PRIORITY,
                "strategy": "Repeated-container ordinal (read target)",
                "count": 1,
                "unique": True,
                "valid": True,
                "validated": True,
                "validation_method": "playwright",
                "stability": stability,
            }
        ],
        "read_target_rewritten_from": locator,
    }
