"""The verifier's model client: a clone of the agent's own, timed and token-tracked (bs #62 step 3).

Referenced by: browser_service.tasks.workflow
Depends on: browser_use.llm.google, browser_use.tokens.service
"""

import dataclasses
import logging

import pytest

from browser_service.tasks.workflow import instrument_llm_timing
from tests.test_tasks.test_workflow_execution import (  # noqa: F401
    FakeAgent,
    arm,
    run_workflow,
    workflow_harness,
)


def _vertex():
    from browser_use.llm.google import ChatGoogle

    return ChatGoogle(
        model="gemini-2.5-flash",
        vertexai=True,
        credentials=None,
        project="p",
        location="global",
        temperature=0.1,
        thinking_budget=0,
    )


def _api_key():
    from browser_use.llm.google import ChatGoogle

    return ChatGoogle(model="gemini-2.5-flash", api_key="k", temperature=0.1, thinking_budget=0)


@pytest.mark.parametrize("make", [_vertex, _api_key])
def test_clone_keeps_provider_fields_and_sets_max_retries_1(make):
    original = make()
    clone = dataclasses.replace(original, max_retries=1)

    assert clone is not original
    assert clone.max_retries == 1
    for f in dataclasses.fields(original):
        if f.name != "max_retries" and f.init:
            assert getattr(clone, f.name) == getattr(original, f.name), f.name


def test_instrument_llm_timing_appends_to_a_given_list():
    import asyncio

    class Fake:
        async def ainvoke(self, messages, output_format=None, **kwargs):
            return "ok"

    shared: list[float] = []
    a, b = Fake(), Fake()
    assert instrument_llm_timing(a, shared) is shared
    assert instrument_llm_timing(b, shared) is shared
    asyncio.run(a.ainvoke([]))
    asyncio.run(b.ainvoke([]))
    assert len(shared) == 2
    # no list given: a fresh one, as before
    assert instrument_llm_timing(Fake()) == []


def test_second_registered_instance_sums_into_usage_summary():
    import asyncio

    from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
    from browser_use.tokens.service import TokenCost

    def usage(p, c):
        return ChatInvokeUsage(
            prompt_tokens=p,
            prompt_cached_tokens=None,
            prompt_cache_creation_tokens=None,
            prompt_image_tokens=None,
            completion_tokens=c,
            total_tokens=p + c,
        )

    class Fake:
        provider = "google"
        model = "gemini-2.5-flash"

        def __init__(self, u):
            self._u = u

        async def ainvoke(self, messages, output_format=None, **kwargs):
            return ChatInvokeCompletion(completion="x", usage=self._u)

    tc = TokenCost(include_cost=False)
    first, second = Fake(usage(100, 10)), Fake(usage(7, 3))
    tc.register_llm(first)
    tc.register_llm(second)

    async def go():
        await first.ainvoke([])
        await second.ainvoke([])
        return await tc.get_usage_summary()

    summary = asyncio.run(go())
    assert summary.total_prompt_tokens == 107
    assert summary.total_completion_tokens == 13


@dataclasses.dataclass
class _DataclassLlm:
    model: str = "m"
    max_retries: int = 9

    async def ainvoke(self, messages, output_format=None, **kwargs):  # pragma: no cover
        return None


class _Tracker:
    def __init__(self):
        self.registered = []

    def register_llm(self, llm):
        self.registered.append(llm)


def _run(harness, flag, monkeypatch, llm=None, tracker=True):
    from unittest.mock import patch

    from browser_service.tasks import workflow as wf

    harness.config.locator.enable_element_verifier = flag
    arm(harness, [])
    if llm is not None:
        harness.chat_google.return_value = llm
    trk = _Tracker()
    with patch.object(wf, "register_custom_actions", return_value=True) as register:
        if tracker:
            with patch.object(FakeAgent, "token_cost_service", trk, create=True):
                run_workflow(harness)
        else:
            run_workflow(harness)
    return register, trk


def test_flag_off_passes_no_verifier(workflow_harness, monkeypatch):
    register, trk = _run(workflow_harness, False, monkeypatch, llm=_DataclassLlm())
    assert register.call_args.kwargs["verifier_llm"] is None
    assert register.call_args.kwargs["user_query"] == "find the search box"
    assert trk.registered == []


def test_flag_on_with_a_non_dataclass_client_passes_none_and_warns_once(
    workflow_harness, monkeypatch, caplog
):
    with caplog.at_level(logging.WARNING):
        register, _ = _run(workflow_harness, True, monkeypatch)  # the harness's MagicMock client
    assert register.call_args.kwargs["verifier_llm"] is None
    assert len([r for r in caplog.records if "verifier" in r.getMessage().lower()]) == 1


def test_flag_on_hands_the_clone_and_the_query_to_registration(workflow_harness, monkeypatch):
    original = _DataclassLlm()
    register, trk = _run(workflow_harness, True, monkeypatch, llm=original)
    clone = register.call_args.kwargs["verifier_llm"]
    assert isinstance(clone, _DataclassLlm) and clone is not original
    assert clone.max_retries == 1 and original.max_retries == 9
    assert register.call_args.kwargs["user_query"] == "find the search box"
    assert trk.registered == [clone]


def test_a_failed_token_registration_drops_the_verifier(workflow_harness, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        register, _ = _run(
            workflow_harness, True, monkeypatch, llm=_DataclassLlm(), tracker=False
        )  # FakeAgent has no token_cost_service -> AttributeError
    assert register.call_args.kwargs["verifier_llm"] is None
    assert any("verifier" in r.getMessage().lower() for r in caplog.records)
