"""
Element verifier (bs #62 step 3): asks a model whether the element the agent picked is the element a step means.

``read_element_facts`` reads a few plain facts off the picked element in ONE ``evaluate`` call (tag, short text,
role, labels, landmark, the row's text, the table column). ``verify_pick`` sends those facts, the user's request,
the step and its action to the model and returns a verdict. The element's typed value is never read or sent; text
inside an ``[nlrf-secret]`` element or an ``<input type=password>`` is dropped from every text field. Fail-open:
a missing fact read, a timeout, an exception or a malformed reply is a "skipped" outcome, never a verdict, and
``verify_pick`` never raises. The prompt is the one measured by the bs #62 spike, plus three rulings.

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
  const facts = {
    tag: el.tagName.toLowerCase(),
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
  };
  for (const k of Object.keys(facts)) if (!facts[k]) delete facts[k];
  return facts;
}"""
)

# The spike's SYSTEM verbatim (bench/private/bs62_spike/build_prompts.py), then three rulings.
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
    " For a step that chooses, reads or checks an option of a dropdown, the dropdown that holds that option "
    "(a <select>, or the custom control that opens its list) is the right element: the option is set and read "
    "through it."
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
