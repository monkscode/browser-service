"""
Decision table for K v1's read-step verdict (pure function, fast lane).

The verdict ACCEPTS only when every clause holds and nothing is unknown.
Anything uncertain keeps the reject — the opposite of _identity_mismatch's
unknown-means-accept rule, on purpose (evaluation spec §13.2, amended 2026-09-23).
"""

import pytest

from browser_service.agent.read_step_identity import ReadIdentityFacts, read_step_verdict

ALL_CLEAR = ReadIdentityFacts(indexed_ok=True, candidate_tag="h6", candidate_centre=(310.0, 30.0))


def test_all_clauses_hold_accepts():
    accept, why = read_step_verdict(ALL_CLEAR)
    assert accept is True
    assert "all five clauses hold" in why


def test_no_facts_keeps_reject():
    accept, why = read_step_verdict(None)
    assert accept is False
    assert why.startswith("unknown")


@pytest.mark.parametrize(
    "reason",
    [
        "no backend id for the indexed node",
        "no selector map",
        "the candidate is in a shadow root",
        "the candidate is not visible",
    ],
)
def test_any_unknown_keeps_reject(reason):
    facts = ReadIdentityFacts(unknown=(reason,), indexed_ok=True)
    accept, why = read_step_verdict(facts)
    assert accept is False
    assert reason in why


@pytest.mark.parametrize(
    "field, clause",
    [
        ("is_indexed", "clause 1"),
        ("inside_other_indexed", "clause 1"),
        ("inside", "clause 2"),
        ("contains", "clause 3"),
        ("labels_index", "clause 4"),
    ],
)
def test_each_clause_rejects(field, clause):
    facts = ReadIdentityFacts(indexed_ok=True, **{field: True})
    accept, why = read_step_verdict(facts)
    assert accept is False
    assert clause in why


def test_indexed_node_gone_rejects_clause_5():
    accept, why = read_step_verdict(ReadIdentityFacts(indexed_ok=False))
    assert accept is False
    assert "clause 5" in why


def test_unknown_wins_over_an_otherwise_clean_read():
    facts = ReadIdentityFacts(unknown=("the candidate is not in the main frame",), indexed_ok=True)
    assert read_step_verdict(facts)[0] is False
