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


class _Usage:
    """Stands in for browser-use's UsageSummary (tokens/views.py:107)."""

    def __init__(self, entry_count):
        self.entry_count = entry_count


class _History:
    def __init__(self, items, steps, usage=None):
        self.history = items
        self._steps = steps
        self.usage = usage

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
    assert diag["llm_calls_actual"] == 0
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


class _StubTimer:
    def __init__(self, total_s=0.0, max_s=0.0, calls=0):
        self.total_s = total_s
        self.max_s = max_s
        self.calls = calls


def test_diagnostics_no_longer_reports_agent_steps():
    """agent_steps duplicated browser_use_llm_calls exactly on 30/30 bench rows.
    llm_calls_actual replaces it with a number that differs whenever a step
    retries (agent/service.py:1655 calls the model up to twice per step)."""
    diag = extract_agent_diagnostics(_History([_Item(1.0)], steps=1), dom_samples=[])
    assert "agent_steps" not in diag


def test_diagnostics_reports_exactly_the_nine_measured_fields():
    diag = extract_agent_diagnostics(_History([_Item(1.0)], steps=1), dom_samples=[])
    assert set(diag) == {
        "dom_elements_max", "dom_elements_median", "llm_429_count", "retry_lost_s",
        "llm_total_s", "llm_max_s", "llm_calls_actual", "steps_total_s",
        "llm_coverage_gap",
    }


def test_steps_total_sums_step_metadata_durations():
    history = _History([_Item(2.0), _Item(1.5), _Item(0.25)], steps=3)
    diag = extract_agent_diagnostics(history, dom_samples=[])
    assert diag["steps_total_s"] == pytest.approx(3.75)


def test_steps_total_is_zero_for_empty_or_missing_history():
    assert extract_agent_diagnostics(None, dom_samples=[])["steps_total_s"] == 0.0
    assert extract_agent_diagnostics(
        _History([], steps=0), dom_samples=[]
    )["steps_total_s"] == 0.0


def test_steps_total_survives_a_step_with_no_metadata():
    item = _Item(0.0)
    item.metadata = None
    history = _History([_Item(2.0), item], steps=2)
    assert extract_agent_diagnostics(history, dom_samples=[])["steps_total_s"] == 2.0


def test_llm_fields_come_from_the_timer():
    history = _History([_Item(2.0)], steps=1)
    timer = _StubTimer(total_s=12.3456, max_s=7.8912, calls=3)
    diag = extract_agent_diagnostics(history, dom_samples=[], llm_timer=timer)
    assert diag["llm_total_s"] == 12.346
    assert diag["llm_max_s"] == 7.891
    assert diag["llm_calls_actual"] == 3


def test_llm_fields_are_zero_without_a_timer():
    diag = extract_agent_diagnostics(_History([_Item(1.0)], steps=1), dom_samples=[])
    assert diag["llm_total_s"] == 0.0
    assert diag["llm_max_s"] == 0.0
    assert diag["llm_calls_actual"] == 0


def test_coverage_gap_is_zero_when_every_call_was_ours():
    history = _History([_Item(1.0)], steps=1, usage=_Usage(entry_count=3))
    diag = extract_agent_diagnostics(
        history, dom_samples=[], llm_timer=_StubTimer(calls=3)
    )
    assert diag["llm_coverage_gap"] == 0


def test_coverage_gap_is_positive_when_calls_escaped_the_wrapper():
    """The fallback swap at agent/service.py:2003 reassigns self.llm and
    re-registers it for token tracking, orphaning our wrapper. entry_count then
    exceeds our count, and the run's numbers are void."""
    history = _History([_Item(1.0)], steps=1, usage=_Usage(entry_count=5))
    diag = extract_agent_diagnostics(
        history, dom_samples=[], llm_timer=_StubTimer(calls=3)
    )
    assert diag["llm_coverage_gap"] == 2


def test_coverage_gap_is_negative_when_a_call_failed():
    """Failed calls return no usage (tokens/service.py:356 guards on
    `if result.usage:`), so we count them and entry_count does not. Benign."""
    history = _History([_Item(1.0)], steps=1, usage=_Usage(entry_count=2))
    diag = extract_agent_diagnostics(
        history, dom_samples=[], llm_timer=_StubTimer(calls=3)
    )
    assert diag["llm_coverage_gap"] == -1


def test_coverage_gap_is_none_when_usage_is_unavailable():
    """Not 0: "could not check" must never read as "coverage was perfect". None
    becomes an empty CSV cell, which trips the populated-on-all-rows gate."""
    history = _History([_Item(1.0)], steps=1, usage=None)
    diag = extract_agent_diagnostics(
        history, dom_samples=[], llm_timer=_StubTimer(calls=3)
    )
    assert diag["llm_coverage_gap"] is None


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
