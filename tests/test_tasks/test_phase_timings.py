"""Phase-timing and agent-diagnostic extraction for the efficiency check.

Referenced by: browser_service.tasks.workflow
Depends on: browser_use.agent.views
"""
import pytest

from browser_service.tasks.workflow import (
    extract_agent_diagnostics,
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
