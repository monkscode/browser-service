"""Point-then-prove D3 + D4 wiring, no browser: the identity gate, the proven candidate, the retry.

- The gate (``_through_identity_gate``) sits between the inner-control re-aim and the frame gate.
  It judges a found result of a performed step (IDENTITY_ACTIONS) by the call's own point and
  index, and refuses with ``identity_unproven`` when the locator does not land on the target.
  Select steps, reads, locate-only steps, action None and page None pass untouched.
- A proven agent candidate skips the row-anchor reject, the identity reject and the word check;
  the action fit and the volatile demotion still run.
- registration: the first refusal asks again with a screenshot; the second is NOT FOUND with no
  ``error`` (browser-use ends a run after 5 error results).

The DOM reads (``pointed_target_or_raise``, ``pointed_target``, ``proven``) are mocked here; they
run on real Chromium in test_identity_gate_fixtures.py.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_service.agent import actions
from browser_service.agent.actions import (
    _through_frame_gate,
    _through_identity_gate,
    _through_inner_control,
    find_unique_locator_action,
)
from browser_service.agent.registration import register_custom_actions
from tests.test_agent.test_indexed_read_screen_point import FakeNode
from tests.test_agent.test_point_box_honesty import _call as _registration_call
from tests.test_agent.test_point_box_honesty import _frame
from tests.test_agent.test_vision_escalation import (
    FakeAgentWithSettings,
    FakeBrowserSession,
    _fake_playwright,
)

TARGET = object()
FRAME = 'iframe[id="f"]'
FOUND = {"element_id": "elem_1", "found": True, "best_locator": "id=toggle", "unique": True}
WHY = (
    "finds <section id=root> at (640, 300); the element you pointed at is "
    "<input type=checkbox> at (412, 230)"
)
PROVEN_LINE = "(signal: pointed-identity-proven)"
REFUSED_LINE = "(signal: pointed-identity-refused)"


def _body(result):
    """A stand-in for the action's body with the parameters the gates read."""

    async def body(
        x,
        y,
        element_id,
        element_description,
        expected_text=None,
        candidate_locator=None,
        element_data=None,
        page=None,
        iframe_context=None,
        is_collection=None,
        row_anchor_text=None,
        action=None,
        vision_point=None,
        pointed_by_index=False,
    ):
        return dict(result)

    return body


def _page(count=1):
    page = MagicMock()
    for root in (page, page.frame_locator.return_value):
        root.locator.return_value.count = AsyncMock(return_value=count)
    return page


def _reads(target=TARGET, verdict=(True, "same")):
    resolve = (
        AsyncMock(side_effect=target)
        if isinstance(target, Exception)
        else AsyncMock(return_value=target)
    )
    return resolve, AsyncMock(return_value=verdict)


async def _gated(result=FOUND, *, target=TARGET, verdict=(True, "same"), **call):
    resolve, judge = _reads(target, verdict)
    kwargs = {"x": 10, "y": 20, "element_id": "elem_1", "element_description": "the toggle"}
    kwargs.update({"page": _page(), "action": "click", **call})
    with (
        patch.object(actions, "pointed_target_or_raise", new=resolve),
        patch.object(actions, "proven", new=judge),
    ):
        out = await _through_identity_gate(_body(result))(**kwargs)
    return out, resolve, judge


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_decorator_order():
    """inner-control re-aim → identity gate → frame gate → the body."""
    first = find_unique_locator_action
    assert first.__code__.co_name == "aimed"
    assert first.__wrapped__.__code__.co_name == "identity_gated"
    assert first.__wrapped__.__wrapped__.__code__.co_name == "gated"


async def test_gate_sees_the_framed_string():
    """The frame gate adds the hop first; ``proven`` gets the framed string and the frame."""
    resolve, judge = _reads()
    stack = _through_identity_gate(_through_frame_gate(_body({**FOUND, "best_locator": "id=in"})))
    with (
        patch.object(actions, "pointed_target_or_raise", new=resolve),
        patch.object(actions, "proven", new=judge),
    ):
        out = await stack(10, 20, "elem_1", "d", page=_page(), iframe_context=FRAME, action="click")

    assert out["best_locator"] == f"{FRAME} >>> id=in"
    assert judge.await_args.args[1:] == (FRAME, f"{FRAME} >>> id=in", TARGET, False)


async def test_proven_result_passes_with_its_signal(caplog):
    data = {"xpath": "html/body/input", "tagName": "input"}
    with caplog.at_level(logging.INFO, logger="browser_service.agent.actions"):
        out, resolve, judge = await _gated(element_data=data, pointed_by_index=True)

    assert out == FOUND
    assert resolve.await_args.args[1:] == (None, data, True, (10, 20))
    assert judge.await_args.args[1:] == (None, "id=toggle", TARGET, False)
    assert [m for m in caplog.messages if m.endswith(PROVEN_LINE)] == [
        f"   ✅ POINTED IDENTITY PROVEN: 'id=toggle' lands on the pointed element (same) "
        f"{PROVEN_LINE}"
    ]


async def test_refusal_shape(caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.actions"):
        out, _, _ = await _gated(verdict=(False, WHY))

    assert out == {
        **FOUND,
        "found": False,
        "best_locator": None,
        "error": f"Your locator {WHY}.",
        "error_type": "NotPointedElement",
        "identity_unproven": WHY,
    }
    assert [m for m in caplog.messages if m.endswith(REFUSED_LINE)] == [
        f"   ⛔ POINTED IDENTITY REFUSED: 'id=toggle' — {WHY} {REFUSED_LINE}"
    ]


async def test_no_target_is_refused_without_a_relation_read():
    out, _, judge = await _gated(target=None)

    judge.assert_not_awaited()
    assert out["found"] is False
    assert out["identity_unproven"] == "no element at the point"
    assert out["error"] == "Your point is on no element."


async def test_unreadable_target_is_refused():
    out, _, judge = await _gated(target=RuntimeError("frame detached"))

    judge.assert_not_awaited()
    assert out["found"] is False
    assert out["identity_unproven"] == "unreadable"
    assert out["error"] == (
        "The element you pointed at could not be read; the page may have changed."
    )


async def test_collection_is_judged_as_one():
    _, _, judge = await _gated({**FOUND, "unique": False}, is_collection=True)

    assert judge.await_args.args[4] is True


@pytest.mark.parametrize("is_collection", [None, False])
async def test_non_unique_result_is_judged_as_a_collection(is_collection):
    """A result that says it is not unique is a collection even when the call did not say so;
    judged as one element it would be refused."""
    _, _, judge = await _gated({**FOUND, "unique": False}, is_collection=is_collection)

    assert judge.await_args.args[4] is True


@pytest.mark.parametrize(
    ("result", "call"),
    [
        pytest.param(FOUND, {"page": None}, id="page-none"),
        pytest.param(FOUND, {"action": "select"}, id="select-step"),
        pytest.param(FOUND, {"action": "get_text"}, id="read-step"),
        pytest.param(FOUND, {"action": None}, id="action-none"),
        pytest.param({**FOUND, "found": False, "best_locator": None}, {}, id="not-found"),
    ],
)
async def test_out_of_scope_passes_untouched(result, call):
    out, resolve, judge = await _gated(result, verdict=(False, WHY), **call)

    assert out == result
    resolve.assert_not_awaited()
    judge.assert_not_awaited()


async def test_reaimed_inner_control_proven():
    """The re-aimed call is judged at the control's centre with its data, never at vision_point."""
    inner = SimpleNamespace(
        x=300.0,
        y=200.0,
        element_data={"xpath": "html/body/li/a", "tagName": "a", "textContent": "Edit"},
        is_answer=AsyncMock(return_value=True),
    )
    resolve, judge = _reads(verdict=(True, "same"))
    stack = _through_inner_control(_through_identity_gate(_body(FOUND)))
    with (
        patch.object(actions, "find_inner_control", new=AsyncMock(return_value=inner)),
        patch.object(actions, "pointed_target_or_raise", new=resolve),
        patch.object(actions, "proven", new=judge),
    ):
        out = await stack(
            x=100,
            y=100,
            element_id="elem_1",
            element_description="Edit",
            element_data={"xpath": "html/body/li", "tagName": "li", "textContent": "Edit"},
            page=_page(),
            action="click",
            vision_point=(100, 100),
            pointed_by_index=True,
        )

    assert out["inner_control"] is True
    assert resolve.await_count == 1
    assert resolve.await_args.args[1:] == (None, inner.element_data, True, (300.0, 200.0))


# ---------------------------------------------------------------------------
# The candidate: proof replaces the word and identity vetoes
# ---------------------------------------------------------------------------


class FakeLocator:
    """One element: every evaluate answers with its surface (the resolved read, the word check)."""

    def __init__(self, surface):
        self._surface = surface

    async def count(self):
        return 1

    def nth(self, i):
        return self

    async def bounding_box(self):
        return {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}

    async def evaluate(self, js, *args, **kwargs):
        text = self._surface.get("textContent", "")
        return {
            "textContent": text,
            "textContentLength": len(text),
            "innerText": text,
            "placeholder": "",
            "ariaLabel": "",
            "value": self._surface.get("value", ""),
            "labelText": "",
            "labelledbyText": "",
            "tagName": self._surface.get("tagName", "input"),
            "id": self._surface.get("id", ""),
            "isContentEditable": False,
        }


class FakePage:
    url = "https://todomvc.test/"

    def __init__(self, surface):
        self._surface = surface

    def locator(self, selector):
        return FakeLocator(self._surface)

    def frame_locator(self, selector):
        return self

    async def evaluate(self, *args, **kwargs):
        return None


CHECKBOX = {"tagName": "input", "value": "on", "textContent": ""}
# Each veto today sends the right checkbox to the cascade.
VETOES = {
    "identity": dict(element_data={"tagName": "section", "id": "root", "textContent": "todos"}),
    "word": dict(expected_text="Buy milk"),
    "row-anchor": dict(
        candidate_locator='text="Buy milk"',
        row_anchor_text="Buy milk",
        element_data={"tagName": "input", "textContent": "toggle"},
    ),
}
NOT_FOUND = {"found": False, "error": "cascade found nothing"}


async def _candidate(
    proof,
    action="click",
    surface=CHECKBOX,
    fit="",
    cascade_result=NOT_FOUND,
    pointed=TARGET,
    **overrides,
):
    """The real action on the fake page; ``proof`` is ``proven``'s verdict or a stand-in for it;
    ``pointed`` is what ``pointed_target`` returns."""
    kwargs = dict(
        x=100,
        y=100,
        element_id="elem_1",
        element_description="the toggle of Buy milk",
        candidate_locator="li:first-child .toggle",
        page=FakePage(surface),
        is_collection=False,
        action=action,
    )
    kwargs.update(overrides)
    cascade = AsyncMock(return_value=dict(cascade_result))
    target = AsyncMock(return_value=pointed)
    judge = AsyncMock(side_effect=proof) if callable(proof) else AsyncMock(return_value=proof)
    with (
        patch("browser_service.locators.find_unique_locator_at_coordinates", new=cascade),
        patch.object(actions, "pointed_target", new=target),
        patch.object(actions, "pointed_target_or_raise", new=AsyncMock(return_value=TARGET)),
        patch.object(actions, "proven", new=judge),
        patch.object(actions, "check_action_fit", new=AsyncMock(return_value=fit)),
    ):
        out = await find_unique_locator_action(**kwargs)
    return out, cascade, target, judge


@pytest.mark.parametrize("veto", sorted(VETOES))
async def test_proven_candidate_skips_the_veto(caplog, veto):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.actions"):
        out, cascade, target, judge = await _candidate((True, "same"), **VETOES[veto])

    assert out["found"] is True
    assert out["all_locators"][0]["type"] == "candidate"
    cascade.assert_not_awaited()
    assert target.await_args_list[0].args[1:] == (
        None,
        VETOES[veto].get("element_data"),
        False,
        (100, 100),
    )
    assert judge.await_args_list[0].args[1:] == (None, out["best_locator"], TARGET, False)
    assert len([m for m in caplog.messages if m.endswith("(signal: candidate-proven)")]) == 1


CASCADE_FOUND = {"found": True, "best_locator": "id=toggle-1", "unique": True, "count": 1}
UNPROVEN_LINE = (
    f"   ⛔ CANDIDATE NOT PROVEN: 'li:first-child .toggle' — {WHY}; continuing with smart "
    f"locator finder (signal: candidate-unproven)"
)


def _only_proven(locator):
    """A stand-in for ``proven``: only ``locator`` lands on the pointed element."""

    async def judge(page, iframe_context, candidate, target, is_collection):
        return (True, "same") if candidate == locator else (False, WHY)

    return judge


@pytest.mark.parametrize("row_anchor_text", [None, "  "], ids=["no-row", "blank-row"])
async def test_unproven_candidate_goes_to_the_cascade_whose_proven_answer_ships(
    caplog, row_anchor_text
):
    """D4: no veto would stop this candidate today; unproven, it is set aside for the cascade."""
    with caplog.at_level(logging.INFO, logger="browser_service.agent.actions"):
        out, cascade, _, _ = await _candidate(
            _only_proven("id=toggle-1"),
            cascade_result=CASCADE_FOUND,
            row_anchor_text=row_anchor_text,
        )

    cascade.assert_awaited()
    assert out["found"] is True
    assert out["best_locator"] == "id=toggle-1"
    assert [m for m in caplog.messages if m.endswith("(signal: candidate-unproven)")] == [
        UNPROVEN_LINE
    ]


async def test_unproven_candidate_and_an_unproven_cascade_answer_are_refused():
    out, cascade, _, _ = await _candidate(
        _only_proven(None), cascade_result={**CASCADE_FOUND, "best_locator": "id=root"}
    )

    cascade.assert_awaited()
    assert out["found"] is False
    assert out["best_locator"] is None
    assert out["identity_unproven"] == WHY


ROW_DATA = {"tagName": "input", "xpath": "html/body/ul/li[2]/input"}


@pytest.mark.parametrize(
    ("pointed", "why", "error"),
    [
        pytest.param(TARGET, WHY, f"Your locator {WHY}.", id="another-element"),
        pytest.param(
            None, "no element at the point", "Your point is on no element.", id="no-element"
        ),
    ],
)
async def test_unproven_candidate_on_a_row_step_is_refused_without_the_cascade(
    caplog, pointed, why, error
):
    """Design §6: on a step that names its row, a candidate off the pointed element is refused at
    once, in the exit gate's shape; registration's identity branch gives the one retry."""
    with caplog.at_level(logging.INFO, logger="browser_service.agent.actions"):
        out, cascade, _, _ = await _candidate(
            _only_proven("id=toggle-1"),
            cascade_result=CASCADE_FOUND,
            pointed=pointed,
            element_data=ROW_DATA,
            row_anchor_text="Buy milk",
        )

    cascade.assert_not_awaited()
    assert out["found"] is False
    assert out["best_locator"] is None
    assert out["error"] == error
    assert out["error_type"] == "NotPointedElement"
    assert out["identity_unproven"] == why
    assert [m for m in caplog.messages if m.endswith("(signal: candidate-unproven)")] == [
        f"   ⛔ CANDIDATE NOT PROVEN: 'li:first-child .toggle' — {why}; refused: the step "
        f"names a row (signal: candidate-unproven)"
    ]


@pytest.mark.parametrize("action", ["select", "get_text", None])
async def test_out_of_scope_candidate_on_a_row_step_untouched(action):
    """A select, a read or flag off reads no proof, so a row anchor refuses nothing there."""
    out, _, target, judge = await _candidate(
        (False, WHY), action=action, element_data=ROW_DATA, row_anchor_text="Buy milk"
    )

    target.assert_not_awaited()
    judge.assert_not_awaited()
    assert "identity_unproven" not in out
    assert out["found"] is True
    assert out["all_locators"][0]["type"] == "candidate"


async def test_candidate_is_judged_at_the_calls_own_point():
    """Never at vision_point: a re-aimed call keeps that on the dead container."""
    _, _, target, _ = await _candidate(
        (True, "same"), element_data=ROW_DATA, pointed_by_index=True, vision_point=(640, 300)
    )

    assert target.await_args_list[0].args[1:] == (None, ROW_DATA, True, (100, 100))


async def test_proven_candidate_still_takes_the_action_fit(caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.actions"):
        _, cascade, _, _ = await _candidate((True, "same"), fit="<div> cannot be clicked")

    cascade.assert_awaited()
    assert any(m.endswith("(signal: candidate-demoted)") for m in caplog.messages)


async def test_proven_candidate_still_takes_the_volatile_demotion(caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.actions"):
        _, cascade, _, _ = await _candidate(
            (True, "same"), candidate_locator="id=react-select-3-option-2"
        )

    cascade.assert_awaited()
    assert any(m.endswith("(signal: candidate-demoted)") for m in caplog.messages)


async def test_select_step_untouched():
    """A select step keeps today's candidate path: no proof read, the word check still vetoes."""
    out, cascade, target, judge = await _candidate(
        (True, "same"), action="select", **VETOES["word"]
    )

    target.assert_not_awaited()
    judge.assert_not_awaited()
    cascade.assert_awaited()
    assert out["found"] is False


async def test_flag_off_candidate_untouched():
    out, cascade, target, judge = await _candidate((True, "same"), action=None, **VETOES["word"])

    target.assert_not_awaited()
    judge.assert_not_awaited()
    cascade.assert_awaited()


# ---------------------------------------------------------------------------
# registration: one retry with a screenshot, then NOT FOUND with no error
# ---------------------------------------------------------------------------

REFUSED = {
    "element_id": "elem_1",
    "found": False,
    "best_locator": None,
    "error": f"Your locator {WHY}.",
    "error_type": "NotPointedElement",
    "identity_unproven": WHY,
}
RETRY_TEXT = (
    "Your locator finds <section id=root> at (640, 300); the element you pointed at is "
    "<input type=checkbox> at (412, 230). "
    "Point at the element itself and call find_unique_locator again with its element_index or "
    "the coordinates of its centre. A screenshot is attached to your next message."
)


async def _register_and_run(calls, agent=None):
    agent = agent or FakeAgentWithSettings("auto")
    elements = [{"id": "elem_1", "action": "click"}, {"id": "elem_2", "action": "click"}]
    with (
        patch(
            "browser_service.agent.actions.find_unique_locator_action",
            new=AsyncMock(side_effect=lambda **kw: {**REFUSED, "element_id": kw["element_id"]}),
        ),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
    ):
        assert register_custom_actions(agent, elements=elements) is True
        return [
            await agent.tools.registry.execute_action(
                "find_unique_locator",
                {"x": 500, "y": 300, "element_description": "the toggle", **params},
                browser_session=FakeBrowserSession(),
            )
            for params in calls
        ], agent


@pytest.mark.parametrize("index", [{"element_index": 7}, {}], ids=["indexed", "index-less"])
async def test_first_refusal_asks_again_with_a_screenshot(caplog, index):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        (first,), agent = await _register_and_run([{"element_id": "elem_1", **index}])

    assert first.error == RETRY_TEXT
    assert first.extracted_content == RETRY_TEXT
    assert first.metadata["include_screenshot"] is True
    assert first.is_done is not True
    assert agent.settings.use_vision is True
    assert any(m.endswith("(signal: identity-retry)") for m in caplog.messages)


async def test_second_refusal_is_not_found_with_no_error(caplog):
    with caplog.at_level(logging.INFO, logger="browser_service.agent.registration"):
        (_, second), _ = await _register_and_run(
            [{"element_id": "elem_1"}, {"element_id": "elem_1"}]
        )

    assert second.error is None
    assert "NOT FOUND" in second.extracted_content
    assert "do not retry" in second.extracted_content.lower()
    assert not second.metadata.get("include_screenshot")
    assert second.metadata["found"] is False
    assert any(m.endswith("(signal: identity-not-found)") for m in caplog.messages)


async def test_identity_counter_is_per_element():
    (_, other), _ = await _register_and_run([{"element_id": "elem_1"}, {"element_id": "elem_2"}])

    assert other.error.startswith("Your locator finds")


async def test_frame_lookup_swap_makes_the_call_not_by_index():
    """An index found a node in a frame, then the frame lookup swapped in a smaller box at its
    centre: element_data is no longer the indexed node, so the target must come from the point."""
    frame = _frame(900, 1000)  # x 900..1900
    big = _box_node("DIV", 1000, 100, 400, 400)  # centre (1200, 300)
    small = _box_node("SPAN", 1150, 250, 100, 100)  # holds (1200, 300)
    nodes = {3: small, 4: big, 9: frame}

    swapped, _ = await _registration_call("click", {"element_index": 4}, nodes=nodes)
    kept, _ = await _registration_call("click", {"element_index": 3}, nodes=nodes)

    assert swapped["element_data"]["tagName"] == "span"
    assert swapped["pointed_by_index"] is False
    assert kept["pointed_by_index"] is True


def _box_node(tag, x, y, width, height):
    node = FakeNode(x=x, y=y, width=width, height=height)
    node.node_name = tag
    return node
