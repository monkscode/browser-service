"""
Task 10 (E1) — the PHASE-2 re-ranker must respect stability tiers.

score_locator gives any id-type locator 100 on TYPE alone; before Task 10
the re-ranker sorted on quality_score only and would have flipped a
stability-demoted volatile id straight back to best_locator, silently
undoing the locator engine's ordering. rerank_sort_key is the shared
verdict both sides read.
"""

from browser_service.tasks.workflow import rerank_sort_key


def test_stable_name_beats_volatile_id_despite_lower_score():
    volatile_id = {
        "type": "id", "locator": "id=ext-gen1042",
        "quality_score": 100, "stability": "volatile",
    }
    stable_name = {
        "type": "name", "locator": '[name="username"]',
        "quality_score": 96, "stability": "stable",
    }
    ordered = sorted([volatile_id, stable_name], key=rerank_sort_key)
    assert ordered[0] is stable_name


def test_positional_sorts_below_volatile():
    positional = {
        "locator": 'text="Home" >> nth=0',
        "quality_score": 65, "stability": "positional",
    }
    volatile = {
        "locator": "id=ember472",
        "quality_score": 100, "stability": "volatile",
    }
    ordered = sorted([positional, volatile], key=rerank_sort_key)
    assert ordered[0] is volatile


def test_missing_stability_defaults_to_stable():
    """Entries from older payloads (no stability field) keep score order."""
    unmarked_id = {"locator": "id=save_button", "quality_score": 100}
    marked_name = {
        "locator": '[name="save"]', "quality_score": 96, "stability": "stable",
    }
    ordered = sorted([marked_name, unmarked_id], key=rerank_sort_key)
    assert ordered[0] is unmarked_id


def test_within_tier_higher_score_wins():
    a = {"locator": "id=save_button", "quality_score": 100, "stability": "stable"}
    b = {"locator": '[name="save"]', "quality_score": 96, "stability": "stable"}
    assert sorted([b, a], key=rerank_sort_key)[0] is a
