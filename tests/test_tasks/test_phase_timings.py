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
    """Only .history and .usage are read by extract_agent_diagnostics."""

    def __init__(self, items, usage=None):
        self.history = items
        self.usage = usage


def test_extract_agent_diagnostics_counts_429_steps_and_their_cost():
    history = _History(
        [
            _Item(2.0),
            _Item(1.5, error="ModelProviderError: 429 RESOURCE_EXHAUSTED"),
            _Item(1.9, error="RESOURCE_EXHAUSTED. quota"),
            _Item(3.0, error="element not found"),
        ],
    )
    diag = extract_agent_diagnostics(history, dom_samples=[10, 20, 30])
    assert diag["llm_429_count"] == 2
    assert diag["retry_lost_s"] == pytest.approx(3.4)
    assert diag["dom_elements_max"] == 30
    assert diag["dom_elements_median"] == 20


@pytest.mark.parametrize(
    "message",
    [
        "ModelProviderError: 429 RESOURCE_EXHAUSTED",
        "Resource exhausted, please retry",
        "You exceeded your current quota exceeded",
        "429 Too Many Requests",
        "Rate limit reached for model",
    ],
)
def test_every_indicator_browser_use_uses_is_counted(message):
    """browser-use classifies a provider error as 429 on any of five
    lowercased indicators (llm/google/chat.py:512-516) precisely because
    providers phrase quota errors differently. Matching only the literal
    RESOURCE_EXHAUSTED token missed the rest, and a miss reads as
    llm_429_count = 0 — "no rate limiting" — which is the false-clean this
    metric exists to prevent. The bench is pinned to Vertex, not the Gemini
    Developer API, so the wording is not ours to assume.

    The five here are the phrase indicators. browser-use's list also carries a
    bare "429", which ours deliberately drops — see
    test_an_element_index_that_happens_to_be_429_is_not_a_rate_limit."""
    diag = extract_agent_diagnostics(_History([_Item(1.5, error=message)]), dom_samples=[])
    assert diag["llm_429_count"] == 1
    assert diag["retry_lost_s"] == pytest.approx(1.5)


def test_an_ordinary_error_is_not_counted_as_a_429():
    diag = extract_agent_diagnostics(
        _History([_Item(1.5, error="element not found")]), dom_samples=[]
    )
    assert diag["llm_429_count"] == 0
    assert diag["retry_lost_s"] == 0.0


@pytest.mark.parametrize(
    "message",
    [
        # tools/service.py:623 and :1548 — a tool failure, not a provider error.
        "Element index 429 not available - page may have changed.",
        "Element with index 429 does not exist.",
        "Element index 4290 not found in browser state",
        # A duration that happens to contain the digits.
        "Action timed out after 4290ms",
    ],
)
def test_an_element_index_that_happens_to_be_429_is_not_a_rate_limit(message):
    """`_is_rate_limit_error` runs against every ActionResult.error, not just
    provider errors — tool failures flow through it too, and they quote the
    element index. On a page indexing 2000+ elements (the case this
    instrumentation exists to price) index 429 is ordinary, so a bare "429"
    substring would charge that step's whole duration to retry_lost_s.

    Dropping the bare token costs no recall on the messages this project
    actually sees: both real Vertex 429s in logs/ carry a phrase —
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Resource
    exhausted...'status': 'RESOURCE_EXHAUSTED'}}" and "429 Too Many
    Requests' for url 'https://aiplatform.googleapis.com/...".
    """
    diag = extract_agent_diagnostics(_History([_Item(1.5, error=message)]), dom_samples=[])
    assert diag["llm_429_count"] == 0
    assert diag["retry_lost_s"] == 0.0


@pytest.mark.parametrize(
    "message",
    [
        # Verbatim shapes taken from logs/ — the two forms Vertex actually returns.
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Resource exhausted. "
        "Please try again later.', 'status': 'RESOURCE_EXHAUSTED'}}",
        "429 Too Many Requests' for url "
        "'https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/"
        "google/models/gemini-3.5-flash:generateContent'",
    ],
)
def test_the_real_vertex_429_texts_are_still_counted_without_the_bare_token(message):
    """Guards the recall side of the test above: these are the messages the
    metric exists to catch, and each is matched on a phrase, not on "429"."""
    diag = extract_agent_diagnostics(_History([_Item(1.5, error=message)]), dom_samples=[])
    assert diag["llm_429_count"] == 1
    assert diag["retry_lost_s"] == pytest.approx(1.5)


def test_extract_agent_diagnostics_handles_clean_run():
    history = _History([_Item(2.0), _Item(1.0)])
    diag = extract_agent_diagnostics(history, dom_samples=[5])
    assert diag["llm_429_count"] == 0
    assert diag["retry_lost_s"] == 0.0


def test_extract_agent_diagnostics_never_raises_on_malformed_history():
    """Instrumentation must never break the pipeline it measures."""
    diag = extract_agent_diagnostics(None, dom_samples=[])
    assert diag["llm_calls_actual"] == 0
    assert diag["dom_elements_max"] == 0
    assert diag["retry_lost_s"] == 0.0


def _item_with_unusable_result():
    """`.result` present but not iterable — `for r in 42` raises TypeError."""
    item = _Item(1.0)
    item.result = 42
    return item


def _item_with_unusable_duration():
    """`duration_seconds` present but not numeric — `float(...)` raises ValueError."""
    item = _Item(1.0)
    item.metadata = _Meta("not a number")
    return item


@pytest.mark.parametrize("make_item", [_item_with_unusable_result, _item_with_unusable_duration])
def test_a_history_item_that_explodes_mid_scan_is_suppressed(make_item, caplog):
    """The docstring promises "Never raises", and this is the branch that keeps
    the promise — the only one the rest of the file left unexercised.

    Whatever raised, the caller still gets a well-formed dict: the three
    history-derived fields are assigned only after the loop, so they keep their
    "not measured" zeros rather than a half-summed total. The DOM and LLM
    fields are computed before the loop and must survive intact.
    """
    diag = extract_agent_diagnostics(
        _History([make_item()]), dom_samples=[10, 30], llm_durations=[2.5]
    )

    assert diag["llm_429_count"] == 0
    assert diag["retry_lost_s"] == 0.0
    assert diag["steps_total_s"] == 0.0
    # Measured before the loop, so unaffected by the failure.
    assert diag["dom_elements_max"] == 30
    assert diag["llm_total_s"] == 2.5
    assert diag["llm_calls_actual"] == 1
    assert "Agent diagnostics extraction failed" in caplog.text


def test_extract_agent_diagnostics_survives_missing_metadata():
    """A step with no metadata must not crash the sum."""
    item = _Item(0.0, error="429 RESOURCE_EXHAUSTED")
    item.metadata = None
    diag = extract_agent_diagnostics(_History([item]), dom_samples=[])
    assert diag["llm_429_count"] == 1
    assert diag["retry_lost_s"] == 0.0


def test_summarize_dom_samples_reports_max_and_median():
    assert summarize_dom_samples([100, 2143, 1876]) == (2143, 1876)
    assert summarize_dom_samples([]) == (0, 0)


def test_diagnostics_reports_exactly_the_nine_measured_fields():
    """Exactly these nine — which also pins that agent_steps is gone.

    agent_steps duplicated browser_use_llm_calls exactly on 30/30 bench rows.
    llm_calls_actual replaces it with a number that differs whenever a step
    retries (agent/service.py:1655 calls the model up to twice per step)."""
    diag = extract_agent_diagnostics(_History([_Item(1.0)]), dom_samples=[])
    assert set(diag) == {
        "dom_elements_max",
        "dom_elements_median",
        "llm_429_count",
        "retry_lost_s",
        "llm_total_s",
        "llm_max_s",
        "llm_calls_actual",
        "steps_total_s",
        "llm_coverage_gap",
    }


def test_steps_total_sums_step_metadata_durations():
    history = _History([_Item(2.0), _Item(1.5), _Item(0.25)])
    diag = extract_agent_diagnostics(history, dom_samples=[])
    assert diag["steps_total_s"] == pytest.approx(3.75)


def test_steps_total_is_zero_for_empty_or_missing_history():
    assert extract_agent_diagnostics(None, dom_samples=[])["steps_total_s"] == 0.0
    assert extract_agent_diagnostics(_History([]), dom_samples=[])["steps_total_s"] == 0.0


def test_steps_total_survives_a_step_with_no_metadata():
    item = _Item(0.0)
    item.metadata = None
    history = _History([_Item(2.0), item])
    assert extract_agent_diagnostics(history, dom_samples=[])["steps_total_s"] == 2.0


def test_llm_fields_are_derived_from_the_recorded_durations():
    history = _History([_Item(2.0)])
    diag = extract_agent_diagnostics(history, dom_samples=[], llm_durations=[7.8912, 3.0, 1.4544])
    assert diag["llm_total_s"] == 12.346
    assert diag["llm_max_s"] == 7.891
    assert diag["llm_calls_actual"] == 3


def test_llm_fields_are_zero_without_a_timer():
    diag = extract_agent_diagnostics(_History([_Item(1.0)]), dom_samples=[])
    assert diag["llm_total_s"] == 0.0
    assert diag["llm_max_s"] == 0.0
    assert diag["llm_calls_actual"] == 0


def test_an_empty_duration_list_is_a_measurement_not_a_missing_one():
    """Wrapped but never called must not read the same as never wrapped: an
    empty list is falsy, so anything testing truthiness here would silently
    report "not measured" and skip the coverage check."""
    history = _History([_Item(1.0)], usage=_Usage(entry_count=0))
    diag = extract_agent_diagnostics(history, dom_samples=[], llm_durations=[])
    assert diag["llm_total_s"] == 0.0
    assert diag["llm_calls_actual"] == 0
    assert diag["llm_coverage_gap"] == 0  # checked, not skipped


def test_coverage_gap_is_zero_when_every_call_was_ours():
    history = _History([_Item(1.0)], usage=_Usage(entry_count=3))
    diag = extract_agent_diagnostics(history, dom_samples=[], llm_durations=[1.0, 2.0, 3.0])
    assert diag["llm_coverage_gap"] == 0


def test_a_positive_coverage_gap_would_mean_a_second_registered_llm():
    """A positive gap is UNREACHABLE in the current configuration — this pins
    the arithmetic, not a state the service can reach.

    Our wrapper is installed before register_llm, so tracked_ainvoke nests
    strictly outside it and entry_count can never exceed our count for the
    instance we wrapped. The only way to go positive is a SECOND registered
    LLM whose calls we never saw, and there is none: page_extraction_llm and
    judge_llm default to the same object (agent/service.py:249-252),
    register_llm early-returns on a duplicate instance id
    (tokens/service.py:337-339), and compaction_llm/fallback_llm are None.

    Kept so that if a future Agent(...) kwarg introduces one, the number moves
    and this test documents what it would mean."""
    history = _History([_Item(1.0)], usage=_Usage(entry_count=5))
    diag = extract_agent_diagnostics(history, dom_samples=[], llm_durations=[1.0, 2.0, 3.0])
    assert diag["llm_coverage_gap"] == 2


def test_coverage_gap_is_negative_when_a_call_failed():
    """Failed calls return no usage (tokens/service.py:356 guards on
    `if result.usage:`), so we count them and entry_count does not. Benign."""
    history = _History([_Item(1.0)], usage=_Usage(entry_count=2))
    diag = extract_agent_diagnostics(history, dom_samples=[], llm_durations=[1.0, 2.0, 3.0])
    assert diag["llm_coverage_gap"] == -1


def test_coverage_gap_is_none_when_usage_is_unavailable():
    """Not 0: "could not check" must never read as "coverage was perfect". None
    becomes an empty CSV cell, which trips the populated-on-all-rows gate."""
    history = _History([_Item(1.0)], usage=None)
    diag = extract_agent_diagnostics(history, dom_samples=[], llm_durations=[1.0, 2.0, 3.0])
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
    durations = instrument_llm_timing(llm)

    await llm.ainvoke(["a"])
    await llm.ainvoke(["b"])
    await llm.ainvoke(["c"])

    assert len(durations) == 3
    assert all(d > 0.0 for d in durations)
    assert max(durations) <= sum(durations)


async def test_timer_max_tracks_the_slowest_call():
    llm = _FakeLLM()
    durations = instrument_llm_timing(llm)

    await llm.ainvoke(["fast"])
    llm.delay = 0.05
    await llm.ainvoke(["slow"])

    assert max(durations) >= 0.05
    assert len(durations) == 2


async def test_timer_passes_the_return_value_through_untouched():
    llm = _FakeLLM(result="the completion")
    instrument_llm_timing(llm)

    assert await llm.ainvoke(["a"]) == "the completion"


async def test_timer_records_a_failed_call_and_re_raises():
    """A failed call still burned wall clock; it must be counted, not swallowed."""
    llm = _FakeLLM(delay=0.02, error=RuntimeError("429 RESOURCE_EXHAUSTED"))
    durations = instrument_llm_timing(llm)

    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        await llm.ainvoke(["a"])

    assert len(durations) == 1
    assert durations[0] >= 0.02


async def test_timer_lets_cancellation_propagate():
    """asyncio.wait_for(llm_timeout) at agent/service.py:1173 cancels the inner
    coroutine. Catching that would change agent behaviour."""
    llm = _FakeLLM(delay=0.02, error=asyncio.CancelledError())
    durations = instrument_llm_timing(llm)

    with pytest.raises(asyncio.CancelledError):
        await llm.ainvoke(["a"])

    assert len(durations) == 1


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
    durations_a, durations_b = instrument_llm_timing(a), instrument_llm_timing(b)

    await asyncio.gather(a.ainvoke(["a1"]), a.ainvoke(["a2"]), b.ainvoke(["b1"]))

    assert len(durations_a) == 2
    assert len(durations_b) == 1
    assert sum(durations_a) > sum(durations_b)
