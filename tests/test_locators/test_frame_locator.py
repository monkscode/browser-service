"""
The one rule for a locator inside an iframe (browser-service #35).

``frame >>> locator`` used to be written four times as a nested function in
smart_locator.py, applied to a result by a fifth, and stripped by hand in
agent/actions.py. These tests pin the single module that now holds it:

- add_frame / strip_frame: the string rule and its inverse.
- apply_frame_to_result: what the cascade's exits always did to a result.
- ensure_frame_on_result: the gate at the exit of find_unique_locator_action.
  It may only ever ADD a missing frame — a result that already carries it, a
  not-found result and a call with no frame come back as the same, unchanged
  object. That is what makes "no change on working paths" checkable.
"""

import copy

import pytest

from browser_service.locators.frame_locator import (
    add_frame,
    apply_frame_to_result,
    ensure_frame_on_result,
    strip_frame,
)

FRAME = 'iframe[id="editor-frame"]'
ORDINAL_FRAME = "iframe >> nth=1"


# ---------------------------------------------------------------------------
# add_frame / strip_frame
# ---------------------------------------------------------------------------


def test_add_frame_prefixes_the_locator():
    assert add_frame("p[contenteditable='true']", FRAME) == f"{FRAME} >>> p[contenteditable='true']"


@pytest.mark.parametrize("no_frame", [None, ""])
def test_add_frame_without_a_frame_returns_the_locator(no_frame):
    assert add_frame("#save-btn", no_frame) == "#save-btn"


def test_add_frame_is_idempotent():
    once = add_frame("#save-btn", FRAME)
    assert add_frame(once, FRAME) == once


def test_strip_frame_removes_the_frame_hop():
    assert strip_frame(f"{FRAME} >>> #save-btn", FRAME) == "#save-btn"


@pytest.mark.parametrize("no_frame", [None, ""])
def test_strip_frame_without_a_frame_returns_the_locator(no_frame):
    assert strip_frame("#save-btn", no_frame) == "#save-btn"


def test_strip_frame_leaves_a_bare_locator_alone():
    assert strip_frame("#save-btn", FRAME) == "#save-btn"


def test_strip_frame_leaves_another_frames_locator_alone():
    other = 'iframe[id="other"] >>> #save-btn'
    assert strip_frame(other, FRAME) == other


def test_strip_frame_undoes_add_frame():
    locator = 'tr:has-text("64625") >> [aria-label="Edit"] >> visible=true >> nth=0'
    assert strip_frame(add_frame(locator, FRAME), FRAME) == locator


# ---------------------------------------------------------------------------
# apply_frame_to_result
# ---------------------------------------------------------------------------


def _bare_result(**extra):
    return {
        "found": True,
        "best_locator": "#save-btn",
        "stability": "stable",
        "all_locators": [
            {"type": "id", "locator": "#save-btn", "stability": "stable"},
            {"type": "name", "locator": '[name="save"]', "stability": "stable"},
        ],
        **extra,
    }


def test_apply_prefixes_the_best_locator_and_every_alternative():
    result = apply_frame_to_result(_bare_result(), FRAME)
    assert result["best_locator"] == f"{FRAME} >>> #save-btn"
    assert [entry["locator"] for entry in result["all_locators"]] == [
        f"{FRAME} >>> #save-btn",
        f'{FRAME} >>> [name="save"]',
    ]
    assert result["iframe_context"] == FRAME
    assert result["stability"] == "stable"


def test_apply_without_a_frame_returns_the_result_untouched():
    result = _bare_result()
    before = copy.deepcopy(result)
    assert apply_frame_to_result(result, None) is result
    assert result == before


def test_apply_ordinal_frame_makes_the_result_positional():
    """`iframe >> nth=1 >>> x` re-resolves the Nth iframe: DOM order."""
    result = apply_frame_to_result(_bare_result(), ORDINAL_FRAME)
    assert result["stability"] == "positional"
    assert [entry["stability"] for entry in result["all_locators"]] == ["positional"] * 2


def test_apply_row_anchored_result_is_judged_on_the_frame_hop_only():
    """A row-anchored locator ends in `>> nth=0`, the containment collapse —
    structural, not positional. Only an ordinal FRAME makes it positional."""
    collapsed = 'tr:has-text("64625") >> [aria-label="Edit"] >> visible=true >> nth=0'

    def anchored():
        return {
            "found": True,
            "best_locator": collapsed,
            "stability": "stable",
            "row_anchored": True,
            "all_locators": [{"locator": collapsed, "stability": "stable"}],
        }

    named = apply_frame_to_result(anchored(), FRAME)
    assert named["best_locator"] == f"{FRAME} >>> {collapsed}"
    assert named["stability"] == "stable"

    ordinal = apply_frame_to_result(anchored(), ORDINAL_FRAME)
    assert ordinal["stability"] == "positional"


# ---------------------------------------------------------------------------
# ensure_frame_on_result — the gate
# ---------------------------------------------------------------------------


def test_gate_adds_the_frame_to_a_bare_result():
    result = _bare_result()
    out = ensure_frame_on_result(result, FRAME)
    assert out is result
    assert out["best_locator"] == f"{FRAME} >>> #save-btn"
    assert [e for e in out["all_locators"] if not e["locator"].startswith(f"{FRAME} >>> ")] == []
    assert out["iframe_context"] == FRAME


def test_gate_prefixes_a_bare_alternative_of_a_prefixed_best():
    """The re-ranker can promote ANY entry, so one bare alternative is enough
    to ship a locator without its frame."""
    result = _bare_result(best_locator=f"{FRAME} >>> #save-btn")
    result["all_locators"][0]["locator"] = f"{FRAME} >>> #save-btn"
    out = ensure_frame_on_result(result, FRAME)
    assert [entry["locator"] for entry in out["all_locators"]] == [
        f"{FRAME} >>> #save-btn",
        f'{FRAME} >>> [name="save"]',
    ]


def test_gate_ordinal_frame_makes_a_bare_result_positional():
    out = ensure_frame_on_result(_bare_result(), ORDINAL_FRAME)
    assert out["best_locator"] == f"{ORDINAL_FRAME} >>> #save-btn"
    assert out["stability"] == "positional"


def _prefixed(inner: str, **extra):
    locator = f"{FRAME} >>> {inner}"
    return {
        "found": True,
        "best_locator": locator,
        "stability": "stable",
        "all_locators": [{"locator": locator, "stability": "stable"}],
        **extra,
    }


# One result per exit that already carried the frame, in the shape that exit
# returns it. Only two of the five set the iframe_context key — the gate must
# not "complete" the other three.
ALREADY_PREFIXED = {
    "element_data": _prefixed("#save-btn", iframe_context=FRAME),
    "text_first": _prefixed('text="Save draft"', element_type=None),
    "collection": _prefixed("#chapters > li.chapter-item", is_collection=True, unique=False),
    "semantic": _prefixed('role=button[name="Save draft"]'),
    "accessibility": _prefixed('role=textbox[name="Work email"]', iframe_context=FRAME),
}


@pytest.mark.parametrize("exit_name", sorted(ALREADY_PREFIXED))
def test_gate_returns_an_already_prefixed_result_untouched(exit_name):
    result = copy.deepcopy(ALREADY_PREFIXED[exit_name])
    before = copy.deepcopy(result)
    assert ensure_frame_on_result(result, FRAME) is result
    assert result == before


def test_gate_does_not_remark_a_prefixed_result_under_an_ordinal_frame():
    """Nothing bare means nothing to do — even where the positional rule
    would have something to say. The exit that built the result owns that."""
    locator = f"{ORDINAL_FRAME} >>> #save-btn"
    result = {
        "found": True,
        "best_locator": locator,
        "stability": "stable",
        "all_locators": [{"locator": locator, "stability": "stable"}],
    }
    before = copy.deepcopy(result)
    assert ensure_frame_on_result(result, ORDINAL_FRAME) is result
    assert result == before


@pytest.mark.parametrize("no_frame", [None, ""])
def test_gate_without_a_frame_returns_the_result_untouched(no_frame):
    result = _bare_result()
    before = copy.deepcopy(result)
    assert ensure_frame_on_result(result, no_frame) is result
    assert result == before


def test_gate_leaves_a_not_found_result_untouched():
    """The coordinate fallback's not-found result still lists the locators it
    tried. They are diagnostics, not answers — nothing is added to them."""
    result = {
        "found": False,
        "best_locator": None,
        "error": "no unique locator",
        "all_locators": [{"locator": "li.chapter-item", "count": 3, "unique": False}],
        "candidate_locators": ["li.chapter-item"],
    }
    before = copy.deepcopy(result)
    assert ensure_frame_on_result(result, FRAME) is result
    assert result == before


@pytest.mark.parametrize(
    "result",
    [
        pytest.param({"found": True, "best_locator": None}, id="none-best-no-alternatives"),
        pytest.param(
            {"found": True, "best_locator": None, "all_locators": None}, id="none-alternatives"
        ),
        pytest.param(
            {"found": True, "best_locator": None, "all_locators": [{"type": "id"}, None, "x"]},
            id="alternatives-without-a-locator",
        ),
        pytest.param({"found": True}, id="no-locator-keys"),
    ],
)
def test_gate_does_not_raise_on_a_result_with_no_locator(result):
    before = copy.deepcopy(result)
    assert ensure_frame_on_result(result, FRAME) is result
    assert result == before


def test_gate_prefixes_a_bare_best_next_to_malformed_alternatives():
    result = {"found": True, "best_locator": "#save-btn", "all_locators": [{"type": "id"}, None]}
    out = ensure_frame_on_result(result, FRAME)
    assert out["best_locator"] == f"{FRAME} >>> #save-btn"
    assert out["all_locators"] == [{"type": "id"}, None]


@pytest.mark.parametrize("not_a_result", [None, "id=x", ["id=x"]])
def test_gate_returns_a_non_dict_as_is(not_a_result):
    assert ensure_frame_on_result(not_a_result, FRAME) is not_a_result
