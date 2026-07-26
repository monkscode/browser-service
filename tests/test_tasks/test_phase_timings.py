"""Phase-timing and agent-diagnostic extraction for the efficiency check.

Referenced by: browser_service.tasks.workflow
Depends on: browser_use.agent.views
"""
import asyncio

import pytest

from browser_service.tasks.workflow import (
    extract_agent_diagnostics,
    instrument_llm_timing,
    summarize_dom_samples,
)


class _Meta:
    def __init__(self, duration):
        self.duration_seconds = duration


class _Result:
    def __init__(self, error=None):
        self.error = error


class _Item:
    def __init__(self, duration, error=None):
        self.metadata = _Meta(duration)
        self.result = [_Result(error)]


class _History:
    def __init__(self, items, steps):
        self.history = items
        self._steps = steps

    def number_of_steps(self):
        return self._steps


def test_extract_agent_diagnostics_counts_429_steps_and_their_cost():
    history = _History(
        [
            _Item(2.0),
            _Item(1.5, error="ModelProviderError: 429 RESOURCE_EXHAUSTED"),
            _Item(1.9, error="RESOURCE_EXHAUSTED. quota"),
            _Item(3.0, error="element not found"),
        ],
        steps=4,
    )
    diag = extract_agent_diagnostics(history, dom_samples=[10, 20, 30])
    assert diag["agent_steps"] == 4
    assert diag["llm_429_count"] == 2
    assert diag["retry_lost_s"] == pytest.approx(3.4)
    assert diag["dom_elements_max"] == 30
    assert diag["dom_elements_median"] == 20


def test_extract_agent_diagnostics_handles_clean_run():
    history = _History([_Item(2.0), _Item(1.0)], steps=2)
    diag = extract_agent_diagnostics(history, dom_samples=[5])
    assert diag["llm_429_count"] == 0
    assert diag["retry_lost_s"] == 0.0


def test_extract_agent_diagnostics_never_raises_on_malformed_history():
    """Instrumentation must never break the pipeline it measures."""
    diag = extract_agent_diagnostics(None, dom_samples=[])
    assert diag["agent_steps"] == 0
    assert diag["dom_elements_max"] == 0
    assert diag["retry_lost_s"] == 0.0


def test_extract_agent_diagnostics_survives_missing_metadata():
    """A step with no metadata must not crash the sum."""
    item = _Item(0.0, error="429 RESOURCE_EXHAUSTED")
    item.metadata = None
    diag = extract_agent_diagnostics(_History([item], steps=1), dom_samples=[])
    assert diag["llm_429_count"] == 1
    assert diag["retry_lost_s"] == 0.0


def test_summarize_dom_samples_reports_max_and_median():
    assert summarize_dom_samples([100, 2143, 1876]) == (2143, 1876)
    assert summarize_dom_samples([]) == (0, 0)


class _FakeLLM:
    """Minimal stand-in for ChatGoogle: one awaitable ainvoke we can steer."""

    def __init__(self, delay=0.0, error=None, result="ok"):
        self.delay = delay
        self.error = error
        self.result = result
        self.seen = []

    async def ainvoke(self, messages, output_format=None, **kwargs):
        self.seen.append((messages, output_format, kwargs))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


async def test_timer_records_one_duration_per_call_and_reports_the_max():
    llm = _FakeLLM()
    timer = instrument_llm_timing(llm)

    await llm.ainvoke(["a"])
    await llm.ainvoke(["b"])
    await llm.ainvoke(["c"])

    assert timer.calls == 3
    assert timer.total_s > 0.0
    assert timer.max_s > 0.0
    assert timer.max_s <= timer.total_s


async def test_timer_max_tracks_the_slowest_call():
    llm = _FakeLLM()
    timer = instrument_llm_timing(llm)

    await llm.ainvoke(["fast"])
    llm.delay = 0.05
    await llm.ainvoke(["slow"])

    assert timer.max_s >= 0.05
    assert timer.calls == 2


async def test_timer_passes_the_return_value_through_untouched():
    llm = _FakeLLM(result="the completion")
    instrument_llm_timing(llm)

    assert await llm.ainvoke(["a"]) == "the completion"


async def test_timer_records_a_failed_call_and_re_raises():
    """A failed call still burned wall clock; it must be counted, not swallowed."""
    llm = _FakeLLM(delay=0.02, error=RuntimeError("429 RESOURCE_EXHAUSTED"))
    timer = instrument_llm_timing(llm)

    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        await llm.ainvoke(["a"])

    assert timer.calls == 1
    assert timer.total_s >= 0.02


async def test_timer_lets_cancellation_propagate():
    """asyncio.wait_for(llm_timeout) at agent/service.py:1173 cancels the inner
    coroutine. Catching that would change agent behaviour."""
    llm = _FakeLLM(delay=0.02, error=asyncio.CancelledError())
    timer = instrument_llm_timing(llm)

    with pytest.raises(asyncio.CancelledError):
        await llm.ainvoke(["a"])

    assert timer.calls == 1


async def test_timer_forwards_output_format_positionally_and_by_keyword():
    """browser-use's own wrapper calls original(messages, output_format, **kwargs)
    positionally (tokens/service.py:352); the step loop passes it as a keyword
    (agent/service.py:1937-1940). Both must reach the real method."""
    llm = _FakeLLM()
    instrument_llm_timing(llm)

    await llm.ainvoke(["a"], dict, session_id="s1")
    await llm.ainvoke(["b"], output_format=list, session_id="s2")

    assert llm.seen[0] == (["a"], dict, {"session_id": "s1"})
    assert llm.seen[1] == (["b"], list, {"session_id": "s2"})


async def test_two_llm_instances_keep_separate_accumulators():
    """Up to 10 workflows run concurrently in a ThreadPoolExecutor, each with its
    own llm_instance. Module-level state would cross-contaminate them."""
    a, b = _FakeLLM(delay=0.03), _FakeLLM()
    timer_a, timer_b = instrument_llm_timing(a), instrument_llm_timing(b)

    await asyncio.gather(
        a.ainvoke(["a1"]), a.ainvoke(["a2"]), b.ainvoke(["b1"])
    )

    assert timer_a.calls == 2
    assert timer_b.calls == 1
    assert timer_a.total_s > timer_b.total_s
