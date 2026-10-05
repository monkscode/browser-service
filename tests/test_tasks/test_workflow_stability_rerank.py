"""
Task 10 (E1) — the PHASE-2 re-ranker must respect stability tiers.

score_locator gives any id-type locator 100 on TYPE alone; before Task 10
the re-ranker sorted on quality_score only and would have flipped a
stability-demoted volatile id straight back to best_locator, silently
undoing the locator engine's ordering. rerank_sort_key is the shared
verdict both sides read.
"""

from browser_service.locators.stability import STABLE
from browser_service.tasks.workflow import (
    commit_reranked_winner,
    find_forceable_id_locator,
    rerank_sort_key,
)


def test_stable_name_beats_volatile_id_despite_lower_score():
    volatile_id = {
        "type": "id",
        "locator": "id=ext-gen1042",
        "quality_score": 100,
        "stability": "volatile",
    }
    stable_name = {
        "type": "name",
        "locator": '[name="username"]',
        "quality_score": 96,
        "stability": "stable",
    }
    ordered = sorted([volatile_id, stable_name], key=rerank_sort_key)
    assert ordered[0] is stable_name


def test_positional_sorts_below_volatile():
    positional = {
        "locator": 'text="Home" >> nth=0',
        "quality_score": 65,
        "stability": "positional",
    }
    volatile = {
        "locator": "id=ember472",
        "quality_score": 100,
        "stability": "volatile",
    }
    ordered = sorted([positional, volatile], key=rerank_sort_key)
    assert ordered[0] is volatile


def test_missing_stability_defaults_to_stable():
    """Entries from older payloads (no stability field) keep score order."""
    unmarked_id = {"locator": "id=save_button", "quality_score": 100}
    marked_name = {
        "locator": '[name="save"]',
        "quality_score": 96,
        "stability": "stable",
    }
    ordered = sorted([marked_name, unmarked_id], key=rerank_sort_key)
    assert ordered[0] is unmarked_id


def test_within_tier_higher_score_wins():
    a = {"locator": "id=save_button", "quality_score": 100, "stability": "stable"}
    b = {"locator": '[name="save"]', "quality_score": 96, "stability": "stable"}
    assert sorted([b, a], key=rerank_sort_key)[0] is a


# ---------------------------------------------------------------------------
# commit_reranked_winner — best_locator, stability and all_locators describe
# the SAME chosen locator and must move together. Updating best_locator while
# leaving a prior winner's stability behind mislabels the emitted locator.
# ---------------------------------------------------------------------------


def test_commit_syncs_stability_when_winner_downgrades():
    # The old best was a 'stable' id that got filtered out as non-unique; the
    # only survivor is a volatile id. Top-level stability must follow the new
    # winner, not keep the filtered-out locator's tier.
    result = {"best_locator": "id=login", "stability": "stable", "all_locators": []}
    scored = [{"locator": "id=ext-gen42", "stability": "volatile", "quality_score": 100}]
    commit_reranked_winner(result, scored)
    assert result["best_locator"] == "id=ext-gen42"
    assert result["stability"] == "volatile"
    assert result["all_locators"] is scored


def test_commit_defaults_missing_stability_to_stable():
    result = {"best_locator": "id=old", "stability": "volatile"}
    scored = [{"locator": '[name="x"]', "quality_score": 90}]
    commit_reranked_winner(result, scored)
    assert result["stability"] == STABLE


def test_commit_syncs_stability_for_forced_id_correction():
    # The priority-violation guard forces a stable id to the front of
    # all_locators, displacing a positional best. Top-level stability must
    # become the id's tier, not keep the displaced locator's.
    result = {"best_locator": "//div[3]//button", "stability": "positional", "all_locators": []}
    reordered = [
        {"locator": "id=login", "stability": "stable", "quality_score": 100},
        {"locator": "//div[3]//button", "stability": "positional", "quality_score": 60},
    ]
    commit_reranked_winner(result, reordered)
    assert result["best_locator"] == "id=login"
    assert result["stability"] == "stable"


# ---------------------------------------------------------------------------
# find_forceable_id_locator — the id-priority forcer must not re-promote an
# entry the action-fitness check (E2a) rejected. E2a marks a rejected entry
# with BOTH valid=False and action_misfit=<reason>; the forcer must key off
# action_misfit specifically, not valid, because valid=False also has
# pre-existing writers unrelated to action-fitness (a raised validation
# error, "no validation data") that must stay forceable with the flag off.
# ---------------------------------------------------------------------------


def test_first_id_shaped_entry_returned_when_nothing_marked():
    locators = [
        {"locator": 'text="Home"'},
        {"locator": "id=save_button"},
        {"locator": "#other_id"},
    ]
    index, entry = find_forceable_id_locator(locators)
    assert index == 1
    assert entry is locators[1]


def test_action_misfit_entry_skipped_for_later_clean_id():
    locators = [
        {"locator": "id=readonly_field", "valid": False, "action_misfit": "<input> is read-only"},
        {"locator": "#save_button"},
    ]
    index, entry = find_forceable_id_locator(locators)
    assert index == 1
    assert entry is locators[1]


def test_all_id_shaped_entries_rejected_returns_none():
    locators = [
        {"locator": "id=readonly_field", "valid": False, "action_misfit": "<input> is read-only"},
        {"locator": "#disabled_field", "valid": False, "action_misfit": "<input> is disabled"},
        {"locator": 'text="Home"'},
    ]
    index, entry = find_forceable_id_locator(locators)
    assert index is None
    assert entry is None


def test_valid_false_without_action_misfit_still_forceable():
    # Pins the flag-off invariant: valid=False alone (no action_misfit key)
    # must NOT block the forcer. This is the test that fails if the guard
    # is simplified to a `valid` check instead of `action_misfit`.
    locators = [
        {"locator": "id=save_button", "valid": False},
    ]
    index, entry = find_forceable_id_locator(locators)
    assert index == 0
    assert entry is locators[0]


def test_no_id_shaped_entry_returns_none():
    locators = [
        {"locator": 'text="Home"'},
        {"locator": '[name="save"]'},
    ]
    index, entry = find_forceable_id_locator(locators)
    assert index is None
    assert entry is None


def test_commit_drops_the_read_change_mark_when_the_locator_changes():
    # changed_by_action was proven for the OLD address only (F1).
    result = {"best_locator": "css=#old", "changed_by_action": "elem_1", "all_locators": []}
    scored = [{"locator": "css=#new", "quality_score": 90, "stability": "stable"}]
    commit_reranked_winner(result, scored)
    assert result["best_locator"] == "css=#new"
    assert "changed_by_action" not in result


def test_commit_keeps_the_read_change_mark_when_the_locator_is_the_same():
    result = {"best_locator": "css=#same", "changed_by_action": "elem_1", "all_locators": []}
    scored = [{"locator": "css=#same", "quality_score": 90, "stability": "stable"}]
    commit_reranked_winner(result, scored)
    assert result["changed_by_action"] == "elem_1"
