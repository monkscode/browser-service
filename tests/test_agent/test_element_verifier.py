"""Unit tests for the element verifier (bs #62 step 3): a fake llm, no network."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from browser_service.agent.element_verifier import (
    SYSTEM,
    Verdict,
    build_messages,
    verify_pick,
)

TWO_LINES = [
    "When the element's visible text and its aria-label or title disagree, the visible text is what the user sees "
    'and means; a disagreeing label alone is not a reason for "mismatch".',
    'Leave "reason" empty when the verdict is "match"; otherwise give it in at most 25 words.',
]
DROPPED_DROPDOWN_SENTENCE = (
    "For a step that chooses, reads or checks an option of a dropdown, the dropdown that holds that option "
    "(a <select>, or the custom control that opens its list) is the right element: the option is set and read "
    "through it."
)
FACTS = {"tag": "button", "text": "Search", "landmark": "main"}


class FakeLLM:
    def __init__(self, completion=None, usage=None, delay=0.0, exc=None):
        self.completion, self.usage, self.delay, self.exc = completion, usage, delay, exc
        self.calls = []

    async def ainvoke(self, messages, output_format=None):
        self.calls.append((messages, output_format))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return SimpleNamespace(completion=self.completion, usage=self.usage)


def _usage(p=120, c=15):
    return SimpleNamespace(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


@pytest.mark.parametrize("verdict", ["match", "mismatch", "unsure"])
async def test_match_mismatch_unsure_pass_through(verdict):
    llm = FakeLLM(Verdict(verdict=verdict, reason="because"), _usage())
    out = await verify_pick(llm, "search shoes", "click search", "click", FACTS)
    assert (out.verdict, out.reason) == (verdict, "because")
    assert (out.prompt_tokens, out.completion_tokens) == (120, 15)
    assert out.seconds >= 0
    assert llm.calls[0][1] is Verdict


async def test_timeout_is_skipped_not_mismatch():
    llm = FakeLLM(Verdict(verdict="mismatch"), _usage(), delay=0.5)
    out = await verify_pick(llm, "q", "s", "click", FACTS, timeout_s=0.05)
    assert out.verdict == "skipped"
    assert out.seconds < 0.4


async def test_exception_is_skipped():
    llm = FakeLLM(exc=RuntimeError("provider down"))
    out = await verify_pick(llm, "q", "s", "click", FACTS)
    assert out.verdict == "skipped"


async def test_bad_reply_is_skipped():
    llm = FakeLLM(completion=None, usage=_usage())
    assert (await verify_pick(llm, "q", "s", "click", FACTS)).verdict == "skipped"


async def test_no_facts_makes_no_call():
    llm = FakeLLM(Verdict(verdict="match"), _usage())
    out = await verify_pick(llm, "q", "s", "click", None)
    assert out.verdict == "skipped"
    assert llm.calls == []


def test_messages_hold_no_value_key_and_the_two_lines():
    system, user = build_messages("find shoes", "click the search box", "click", FACTS)
    for line in TWO_LINES:
        assert line in SYSTEM
    assert DROPPED_DROPDOWN_SENTENCE not in SYSTEM
    assert "dropdown" not in SYSTEM
    assert system.content == SYSTEM
    body = json.loads(user.content)
    assert set(body) == {"user_request", "step", "action", "element"}
    assert body["element"] == FACTS
    assert "value" not in body["element"]


async def test_reason_is_optional():
    assert Verdict(verdict="match").reason == ""
    llm = FakeLLM(Verdict(verdict="match"), usage=None)
    out = await verify_pick(llm, "q", "s", "click", FACTS)
    assert (out.verdict, out.reason) == ("match", "")
    assert (out.prompt_tokens, out.completion_tokens) == (0, 0)
