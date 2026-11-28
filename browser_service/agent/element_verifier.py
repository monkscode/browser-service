"""
Element verifier (bs #62 step 3): asks a model whether the element the agent picked is the element a step means.

``read_element_facts`` reads a few plain facts off the picked element in ONE ``evaluate`` call (tag, short text,
role, labels, landmark, the row's text, the table column, and for a wordless element its picture as text: svg
viewBox / path start, or img alt / file name). ``verify_pick`` sends those facts, the user's request,
the step and its action to the model and returns a verdict. The element's typed value is never read or sent; text
inside an ``[nlrf-secret]`` element or an ``<input type=password>`` is dropped from every text field. Fail-open:
a missing fact read, a timeout, an exception or a malformed reply is a "skipped" outcome, never a verdict, and
``verify_pick`` never raises. The prompt is the one measured by the bs #62 spike, plus two rulings.
``words_agree`` (point-then-prove D6) checks the agent's own words against those facts, for a pick the model
stayed silent on twice.

Referenced by: agent/registration.py (the find_unique_locator handler)
Depends on: locators/hidden_value.py; browser_use.llm.messages (the llm object is passed in)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Literal

from browser_use.llm.messages import SystemMessage, UserMessage
from pydantic import BaseModel

from browser_service.locators.hidden_value import HIDES_VALUE_JS, SECRET_MARK

logger = logging.getLogger(__name__)

# Facts the verifier sees, one function of the element. Text fields go through safe_text(): a clone of the node
# minus secret-marked and password descendants (an element that itself hides its value has no text). Omits empty keys.
# shortcut: closest() stops at a shadow root, so landmark reads empty inside one; column header ignores colspan.
ELEMENT_FACTS_JS = (
    """(el) => {
  const hides = """
    + HIDES_VALUE_JS
    + """;
  const squash = (s) => (s || '').replace(/\\s+/g, ' ').trim();
  const safeText = (node, max) => {
    if (!node || hides(node)) return '';
    const clone = node.cloneNode(true);
    clone.querySelectorAll('["""
    + SECRET_MARK
    + """], input[type=password], script, style')
      .forEach((n) => n.remove());
    const parts = [];
    const walker = document.createTreeWalker(clone, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) parts.push(walker.currentNode.nodeValue);
    return squash(parts.join(' ')).slice(0, max);
  };
  const attr = (n) => el.getAttribute(n) || '';
  const labelNode = (el.labels && el.labels[0]) || el.closest('label')
    || (attr('aria-labelledby') ? document.getElementById(attr('aria-labelledby').split(/\\s+/)[0]) : null);
  const NAMES = {banner: 'header', navigation: 'nav', complementary: 'aside', contentinfo: 'footer'};
  const lm = el.parentElement && el.parentElement.closest(
    'header,nav,main,aside,footer,form,dialog,[role=banner],[role=navigation],[role=main],'
    + '[role=complementary],[role=contentinfo],[role=form],[role=dialog]');
  let landmark = '';
  if (lm) {
    const r = lm.getAttribute('role');
    landmark = r ? (NAMES[r] || r) : lm.tagName.toLowerCase();
    const l = lm.getAttribute('aria-label');
    if (l) landmark += ': ' + l.trim();
  }
  const row = el.closest('tr,[role=row]');
  const cell = el.closest('td,th');
  let position = '', header = '';
  if (cell && cell.parentElement) {
    const n = Array.prototype.indexOf.call(cell.parentElement.children, cell) + 1;
    const suffix = (n % 100 >= 11 && n % 100 <= 13) ? 'th' : ({1: 'st', 2: 'nd', 3: 'rd'}[n % 10] || 'th');
    position = n + suffix + ' from the left';
    const table = cell.closest('table');
    const headRow = table && (table.querySelector('thead tr') || table.querySelector('tr'));
    const h = headRow && headRow.children[n - 1];
    if (h && h.tagName === 'TH') header = safeText(h, 80);
  }
  // item: the nearest repeated ancestor-or-self (<= 6 levels) when the element is in no table row (row facts cover rows).
  let itemText = '', itemPos = '';
  if (!row) {
    const sameKind = (a, b) => a.tagName === b.tagName && (a.tagName === 'LI' || a.tagName === 'ARTICLE'
      || (a.getAttribute('role') === 'listitem' && b.getAttribute('role') === 'listitem')
      || ((a.getAttribute('class') || '').trim() !== '' && (a.getAttribute('class') || '').trim() === (b.getAttribute('class') || '').trim()));
    for (let n = el, d = 0; n && n.parentElement && d <= 6 && n !== document.body; n = n.parentElement, d++) {
      const peers = [...n.parentElement.children].filter((s) => sameKind(s, n));
      const listy = n.tagName === 'LI' || n.tagName === 'ARTICLE' || n.getAttribute('role') === 'listitem';
      if (listy || peers.length >= 2) {
        itemText = safeText(n, 120);
        itemPos = (peers.indexOf(n) + 1) + ' of ' + peers.length;
        break;
      }
    }
  }
  const facts = {
    tag: el.tagName.toLowerCase(),
    id: attr('id').trim(),
    text: safeText(el, 80),
    role: attr('role'),
    aria_label: attr('aria-label').trim(),
    title: attr('title').trim(),
    placeholder: attr('placeholder').trim(),
    label: safeText(labelNode, 80),
    type: attr('type'),
    name: attr('name'),
    landmark: landmark,
    row_text: safeText(row, 120),
    column_position: position,
    column_header: header,
    item_text: itemText,
    item_position: itemPos,
  };
  // graphic: a wordless element's picture as text (first svg: viewBox, path start; else first img: alt, file name).
  // shortcut: a sprite icon (<use href>), a bounding-box first path (M0 0h24v24H0z) or an icon drawn without a path gives
  // an uninformative graphic, which the call site refuses on "unsure"; upgrade if a sweep shows a right icon refused so.
  // shortcut: only display:none (own or an ancestor's) is skipped, so a visibility:hidden or zero-opacity svg / img still counts;
  // upgrade if one is mis-described.
  if (!facts.text && !facts.aria_label && !facts.title && !facts.label && !hides(el)) {
    const first = (sel) => [el, ...el.querySelectorAll(sel)]
      .find((n) => n.matches(sel) && !hides(n) && n.getClientRects().length > 0);
    const fileName = (s) => /^data:/i.test(s)
      ? 'data:' + s.slice(5).split(/[;,]/)[0]
      : s.split(/[?#]/)[0].split('/').pop().slice(0, 60);
    const join = (kind, pairs) => [kind, ...pairs.filter((p) => p[1]).map((p) => p[0] + '=' + p[1])].join(' ');
    const visible = (root, sel) => [...root.querySelectorAll(sel)].find((n) => !hides(n));
    const svg = first('svg');
    const img = svg ? null : first('img');
    if (svg) {
      const path = visible(svg, 'path');
      facts.graphic = join('svg', [
        ['viewBox', squash(svg.getAttribute('viewBox'))],
        ['path', squash(path && path.getAttribute('d')).slice(0, 60)],
      ]);
    } else if (img) {
      facts.graphic = join('img', [
        ['alt', squash(img.getAttribute('alt')).slice(0, 60)],
        ['src', fileName(squash(img.getAttribute('src')))],
      ]);
    }
  }
  for (const k of Object.keys(facts)) if (!facts[k]) delete facts[k];
  return facts;
}"""
)

# The spike's SYSTEM verbatim (bench/private/bs62_spike/build_prompts.py), then two rulings.
SYSTEM = (
    "You check one choice made by a web-test generator. A user asked for a test in plain words; a planner split it "
    "into steps; for one step a browser agent picked an element on the page. You get the user's request, the step, "
    "the step's action and facts read from the picked element. You do not get the agent's reasons. Decide whether "
    'the picked element is the element this step means. Answer "match" when it is that element, or a control '
    'inside it that the action can use. Answer "mismatch" when it is a different element. Answer "unsure" when '
    "the facts cannot tell. For a read or check step (get_text) the value is not known before the test runs: judge "
    "the element's kind and place, not its text. When the step's words and the user's request disagree, the user's "
    "request wins. Pages and requests may be in any language. Reply with JSON only: "
    '{"verdict": "match|mismatch|unsure", "reason": "<at most 25 words>"}.'
    " When the element's visible text and its aria-label or title disagree, the visible text is what the user sees "
    'and means; a disagreeing label alone is not a reason for "mismatch".'
    ' Leave "reason" empty when the verdict is "match"; otherwise give it in at most 25 words.'
)


class Verdict(BaseModel):
    verdict: Literal["match", "mismatch", "unsure"]
    reason: str = ""


@dataclass
class VerifierOutcome:
    verdict: str  # "match" | "mismatch" | "unsure" | "skipped"
    reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


async def read_element_facts(target, timeout_ms: int = 2000) -> dict | None:
    """The facts of the element ``target`` (a Playwright Locator) points at; None on any failure."""
    try:
        facts = await target.evaluate(ELEMENT_FACTS_JS, timeout=timeout_ms)
    except Exception as exc:
        logger.info("element-facts-unavailable: %s", type(exc).__name__)
        return None
    return facts if isinstance(facts, dict) and facts else None


_WORD_FACTS = ("text", "label", "aria_label", "title", "placeholder", "item_text")


def words_agree(expected_text: str | None, facts: dict | None) -> bool:
    """D6: the agent's own words, whitespace collapsed and lower-cased, occur in one of the
    element's worded facts. False for no words or no facts."""
    words = " ".join((expected_text or "").split()).lower()
    if not words or not facts:
        return False
    return any(words in " ".join(str(facts.get(k) or "").split()).lower() for k in _WORD_FACTS)


def build_messages(user_query: str, step: str, action: str, facts: dict) -> list:
    body = {"user_request": user_query, "step": step, "action": action, "element": facts}
    return [
        SystemMessage(content=SYSTEM),
        UserMessage(content=json.dumps(body, ensure_ascii=False)),
    ]


async def verify_pick(
    llm, user_query: str, step: str, action: str, facts: dict | None, timeout_s: float = 15.0
) -> VerifierOutcome:
    """Ask ``llm`` (a browser_use chat model) whether the pick is the element the step means. Never raises."""
    if not facts:
        return VerifierOutcome(verdict="skipped", reason="no facts")
    start = time.monotonic()
    try:
        result = await asyncio.wait_for(
            llm.ainvoke(build_messages(user_query, step, action, facts), output_format=Verdict),
            timeout_s,
        )
        verdict = result.completion.verdict
        reason = str(result.completion.reason or "")
        usage = result.usage
        if verdict not in ("match", "mismatch", "unsure"):
            raise ValueError("verdict outside the three answers")
        return VerifierOutcome(
            verdict=verdict,
            reason=reason,
            prompt_tokens=int(usage.prompt_tokens) if usage else 0,
            completion_tokens=int(usage.completion_tokens) if usage else 0,
            seconds=time.monotonic() - start,
        )
    except Exception as exc:
        reason = "timeout" if isinstance(exc, TimeoutError) else type(exc).__name__
        logger.info("element-verifier-skipped: %s", reason)
        return VerifierOutcome(verdict="skipped", reason=reason, seconds=time.monotonic() - start)
