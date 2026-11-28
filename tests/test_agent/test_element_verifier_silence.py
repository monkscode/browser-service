"""Point-then-prove D6 (the silence rule) and the D4 amendment (point-only picks), stub verifier.

A ``skipped`` first answer is asked once more on a picture-only pick (bs #97, unchanged), on a
performed step (``IDENTITY_ACTIONS``) and on a point-only pick, when facts were read. After two
silences a picture-only pick is refused; a performed step ships only when the agent's own words
occur in its facts AND it holds no other control (``count_inner_controls``). A point-only pick (a
locate-only step the point alone found: no element data, no words) ships only on a "match";
anything else, the verifier off included, is refused like a mismatch. Reads, selects, worded
locate-only steps and the flag off keep 615c9e7's rules (R11).

Harness: test_element_verifier_wiring.py's ``_run`` (the real registered handler, the engine and
Playwright faked); ``count_inner_controls`` is patched at registration's seam. The real-DOM count
is pinned in tests/test_locators/test_pointed_identity_fixtures.py.
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from browser_service.agent import element_verifier
from browser_service.config import config
from browser_service.locators import pointed_identity
from tests.test_agent.test_element_verifier_wiring import (
    CLICK,
    FACTS,
    FOUND,
    GRAPHIC_FACTS,
    LLM,
    LOGGER,
    QUERY,
    REASON,
    _run,
    _shipped,
)
from tests.test_agent.test_no_evidence_gate import _call, cascade, page

COUNT = "browser_service.agent.registration.count_inner_controls"
SILENT = ("skipped", "timeout")
WORDS = {"expected_text": "  sign   IN "}  # FACTS' text, other case and spacing
HOVER = {
    "id": "elem_2",
    "action": "get_text",
    "value": "",
    "description": "the avatar to hover",
    "locate_only": True,
}
RETRY = "find_unique_locator again"


def _refused(result):
    assert RETRY in result.error
    assert result.metadata["found"] is False
    assert result.metadata["best_locator"] is None
    assert result.metadata["include_screenshot"] is True


async def _silent(count, verdicts=(SILENT, SILENT), calls=("elem_1",), **kw):
    with patch(COUNT, new=count):
        return await _run(list(calls), verdicts, **kw)


async def _hover(verdicts=(), calls=("elem_2",), **kw):
    """A point-only pick: a locate-only step, no index data (empty selector map), no words."""
    with patch.object(config.locator, "enable_action_fit", True):
        return await _run(list(calls), verdicts, elements=(HOVER,), **kw)


# ---- D6: the re-ask and the silence rule ------------------------------------------------------


async def test_worded_pick_skipped_then_match_ships(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (result,), m = await _run(["elem_1"], [SILENT, "match"])
    _shipped(result)
    m.interaction.assert_awaited_once()
    assert m.verify.await_count == 2
    assert m.verify.await_args_list[0] == m.verify.await_args_list[1]
    assert "(signal: element-verifier-silent-reask)" in caplog.text
    assert "(signal: element-verifier-graphic-reask)" not in caplog.text


async def test_silent_twice_words_agree_no_inner_control_ships(caplog):
    count = AsyncMock(return_value=0)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (result,), m = await _silent(count, extra=WORDS)
    _shipped(result)
    m.interaction.assert_awaited_once()
    assert m.verify.await_count == 2
    # counted on the same Locator the facts were read from
    count.assert_awaited_once_with("locator(css=#pick)")
    m.read.assert_awaited_once_with("locator(css=#pick)")
    assert "(signal: element-verifier-silent-words-agree)" in caplog.text
    assert "silent-words-refused" not in caplog.text
    assert "Sign in" not in caplog.text  # no fact logged


async def test_silent_twice_root_container_refused(caplog):
    count = AsyncMock(return_value=3)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (first, second), m = await _silent(
            count, (SILENT,) * 4, calls=("elem_1", "elem_1"), extra=WORDS
        )
    _refused(first)
    assert "3 other controls" in first.error
    assert "timeout" not in first.error  # a skip reason is no judgement
    assert "NOT FOUND" in second.extracted_content and second.error is None
    m.interaction.assert_not_awaited()
    assert "(signal: element-verifier-silent-words-refused)" in caplog.text
    assert "(signal: element-verifier-retry)" in caplog.text
    assert "(signal: element-verifier-not-found)" in caplog.text
    assert "silent-words-agree" not in caplog.text


@pytest.mark.parametrize(
    "extra",
    [{"expected_text": "Checkout"}, None, {"expected_text": "  "}],
    ids=["other", "none", "blank"],
)
async def test_silent_twice_words_disagree_refused(extra, caplog):
    count = AsyncMock(return_value=0)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (first,), m = await _silent(count, extra=extra)
    _refused(first)
    count.assert_not_awaited()  # words first: no page read when they disagree
    m.interaction.assert_not_awaited()
    assert "(signal: element-verifier-silent-words-refused)" in caplog.text


async def test_graphic_silent_twice_refused(caplog):
    """bs #97 unchanged, and it wins over the silence rule even where the words would agree."""
    facts = {**GRAPHIC_FACTS, "item_text": "Sign in"}
    count = AsyncMock(return_value=0)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (first,), m = await _silent(count, extra=WORDS, read_facts=AsyncMock(return_value=facts))
    _refused(first)
    assert "could not confirm it is the element the step means" in first.error
    count.assert_not_awaited()
    assert "(signal: element-verifier-graphic-reask)" in caplog.text
    assert "(signal: element-verifier-skipped-graphic-refused)" in caplog.text
    assert "silent-reask" not in caplog.text and "silent-words" not in caplog.text


def test_words_agree_unit():
    agree = element_verifier.words_agree
    assert agree("Buy  MILK", {"item_text": "buy milk Walk the dog"})
    assert agree(" sign\nin ", {"tag": "a", "text": "Sign   in"})
    for key in ("text", "label", "aria_label", "title", "placeholder", "item_text"):
        assert agree("Search", {key: "search"}), key
    assert not agree("Search", {"row_text": "Search", "id": "search", "column_header": "Search"})
    assert not agree(None, FACTS)
    assert not agree("   ", FACTS)
    assert not agree("Sign in", None)
    assert not agree("Sign in", {})


async def test_verifier_off_path_unchanged(caplog):
    count = AsyncMock(return_value=3)
    worded_hover = {**HOVER, "id": "elem_1"}
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (click,), c = await _silent(count, (), verifier_llm=None)
        with patch.object(config.locator, "enable_action_fit", True):
            (hover,), h = await _silent(
                count,
                (),
                verifier_llm=None,
                elements=(worded_hover,),
                extra={"expected_text": "Me"},
            )
    _shipped(click)
    _shipped(hover)
    for m in (c, h):
        m.read.assert_not_awaited()
        m.verify.assert_not_awaited()
        m.interaction.assert_awaited_once()
    count.assert_not_awaited()
    assert "element-verifier" not in caplog.text


async def test_facts_unavailable_ships_as_today(caplog):
    count = AsyncMock(return_value=3)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (result,), m = await _silent(
            count, [("skipped", "no facts")], read_facts=AsyncMock(return_value=None)
        )
    _shipped(result)
    m.verify.assert_awaited_once()  # no re-ask: verify_pick makes no model call without facts
    count.assert_not_awaited()
    assert "silent-reask" not in caplog.text and "silent-words" not in caplog.text


async def test_silent_twice_label_wrapping_its_checkbox_ships(caplog):
    check = {"id": "elem_1", "action": "check", "value": "", "description": "the I agree box"}
    facts = {"tag": "label", "text": "I agree to the terms"}
    count = AsyncMock(return_value=0)  # the real count of a label around its own checkbox: fixtures
    engine = {**FOUND, "best_locator": "css=label.agree"}
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (result,), m = await _silent(
            count,
            elements=(check,),
            engines=[engine],
            read_facts=AsyncMock(return_value=facts),
            extra={"expected_text": "I agree"},
        )
    _shipped(result, "css=label.agree")
    count.assert_awaited_once_with("locator(css=label.agree)")
    assert "(signal: element-verifier-silent-words-agree)" in caplog.text


# ---- ruling 7: a count that cannot be read is not 0 -------------------------------------------


@pytest.mark.parametrize(
    "count",
    [
        AsyncMock(side_effect=RuntimeError("detached")),
        AsyncMock(return_value=None),
        AsyncMock(return_value="0"),
    ],
    ids=["raises", "none", "string"],
)
async def test_unreadable_count_is_refused_not_shipped(count, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (first,), m = await _silent(count, extra=WORDS)
    _refused(first)
    m.interaction.assert_not_awaited()
    assert "(signal: element-verifier-silent-words-refused)" in caplog.text
    assert "the pick ships as today" not in caplog.text


# ---- R11: the non-graphic re-ask and the silence rule are for performed steps only ------------

READ_1 = {"id": "elem_1", "action": "get_text", "value": "", "description": "the product name"}
SELECT = {"id": "elem_1", "action": "select", "value": "Blue", "description": "the colour list"}
WORDED_HOVER = {**HOVER, "id": "elem_1"}


@pytest.mark.parametrize(
    ("elements", "extra", "fit"),
    [
        ((READ_1,), None, True),
        ((SELECT,), WORDS, True),
        ((WORDED_HOVER,), {"expected_text": "Account"}, True),
        ((CLICK,), WORDS, False),
    ],
    ids=["read-no-words", "select", "worded-hover", "flag-off"],
)
async def test_out_of_scope_silence_ships_with_one_ask(elements, extra, fit, caplog):
    """Reads, selects, worded locate-only steps and the flag off keep 615c9e7's rule: only a
    picture is asked again, and a silent answer ships (a count of 3 would refuse a click)."""
    count = AsyncMock(return_value=3)
    with (
        caplog.at_level(logging.INFO, logger=LOGGER),
        patch.object(config.locator, "enable_action_fit", fit),
    ):
        (result,), m = await _silent(count, elements=elements, extra=extra)
    _shipped(result)
    m.verify.assert_awaited_once()
    count.assert_not_awaited()
    assert "silent-reask" not in caplog.text and "silent-words" not in caplog.text


@pytest.mark.parametrize("second", ["match", SILENT], ids=["then-match", "silent-twice"])
async def test_point_only_silence_is_asked_once_more_and_needs_a_match(second, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (first,), m = await _hover([SILENT, second])
    assert m.verify.await_count == 2
    assert "(signal: element-verifier-silent-reask)" in caplog.text
    assert "silent-words" not in caplog.text
    if second == "match":
        _shipped(first)
    else:
        _refused(first)
        assert "(signal: element-verifier-point-only-refused)" in caplog.text


# ---- the D4 amendment: a point-only pick ships only on "match" -------------------------------


async def test_point_only_hover_match_ships(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (result,), m = await _hover(["match"])
    _shipped(result)
    m.verify.assert_awaited_once_with(LLM, QUERY, HOVER["description"], "locate", FACTS)
    assert "point-only-refused" not in caplog.text


async def test_point_only_hover_unsure_refused(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (first, second), m = await _hover(
            [("unsure", "an avatar, but which one")] * 2, calls=("elem_2", "elem_2")
        )
    _refused(first)
    assert "an avatar, but which one" in first.error  # the model's own judgement is kept
    assert "NOT FOUND" in second.extracted_content and second.error is None
    assert second.metadata["found"] is False
    assert "(signal: element-verifier-point-only-refused)" in caplog.text
    assert "(signal: element-verifier-not-found)" in caplog.text


async def test_point_only_hover_facts_unavailable_refused(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (first,), m = await _hover(
            [("skipped", "no facts")], read_facts=AsyncMock(return_value=None)
        )
    _refused(first)
    assert "no facts" not in first.error
    m.verify.assert_awaited_once()
    assert "(signal: element-verifier-point-only-refused)" in caplog.text


async def test_point_only_upload_not_asked_refused(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (first,), m = await _hover(engines=[{**FOUND, "element_type": "file-upload"}])
    _refused(first)
    m.read.assert_not_awaited()
    m.verify.assert_not_awaited()
    assert "(signal: element-verifier-not-asked)" in caplog.text
    assert "(signal: element-verifier-point-only-refused)" in caplog.text


async def test_point_only_with_verifier_off_refused(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (first, second), m = await _hover(calls=("elem_2", "elem_2"), verifier_llm=None)
    _refused(first)
    assert "NOT FOUND" in second.extracted_content and second.error is None
    m.read.assert_not_awaited()
    m.verify.assert_not_awaited()
    assert "(signal: element-verifier-point-only-refused)" in caplog.text
    assert "(signal: element-verifier-not-found)" in caplog.text


async def test_worded_hover_unsure_still_ships(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (result,), m = await _hover(["unsure"], extra={"expected_text": "Account"})
    _shipped(result)
    assert "point-only-refused" not in caplog.text


@pytest.mark.parametrize("where", ["verify", "read", "count"])
async def test_point_only_exception_returns_the_retry(where, caplog):
    """Ruling 6: an error anywhere in the verifier step refuses a point-only pick, as a returned
    result; the outer fail-open never ships it."""
    boom = AsyncMock(side_effect=RuntimeError("boom"))
    kw = {"read_facts": boom} if where == "read" else {}
    # "verify": no verdict queued, so the verify_pick stub raises when it is called
    verdicts = () if where == "verify" else (SILENT, SILENT)
    with caplog.at_level(logging.INFO, logger=LOGGER), patch(COUNT, new=boom):
        (first,), m = await _hover(verdicts, **kw)
    if where == "verify":
        m.verify.assert_awaited_once()
    _refused(first)
    m.interaction.assert_not_awaited()
    assert "(signal: element-verifier-point-only-refused)" in caplog.text
    assert "the pick ships as today" not in caplog.text


async def test_point_only_requery_with_the_accepted_locator_ships(caplog):
    """Ruling 4: that locator was already accepted."""
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (first, second), m = await _hover(["match"], calls=("elem_2", "elem_2"))
    _shipped(first)
    assert second.error is None
    m.verify.assert_awaited_once()
    assert (
        "re-query returned the accepted locator (signal: element-verifier-not-asked)" in caplog.text
    )
    assert "point-only-refused" not in caplog.text


async def test_point_only_requery_refusal_keeps_the_validated_locator(caplog):
    """Ruling 5: a point-only refusal on a re-query takes the requery-kept return."""
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (first, second), m = await _hover(
            ["match", "unsure"],
            calls=("elem_2", "elem_2"),
            engines=[FOUND, {**FOUND, "best_locator": "css=#other"}],
        )
    _shipped(first)
    assert second.error is None and "Locator: css=#pick" in second.extracted_content
    assert "(signal: element-verifier-requery-kept)" in caplog.text
    assert "(signal: element-verifier-point-only-refused)" in caplog.text


CONFIRMED = ("could not be confirmed", "that could not be confirmed")
JUDGED = ("judged the new pick a different element", "the verifier judged different")


@pytest.mark.parametrize(
    ("spec", "second", "extra", "wording"),
    [
        (HOVER, ["unsure"], None, CONFIRMED),
        (CLICK, [SILENT, SILENT], {"expected_text": "Checkout"}, CONFIRMED),
        (CLICK, [("mismatch", REASON)], None, JUDGED),
    ],
    ids=["point-only-unsure", "silence-refused", "mismatch"],
)
async def test_requery_refusal_says_why_the_validated_locator_stands(
    spec, second, extra, wording, caplog
):
    """R12: only a "mismatch" says the verifier judged the new pick different."""
    eid = spec["id"]
    other = CONFIRMED if wording is JUDGED else JUDGED
    with (
        caplog.at_level(logging.INFO, logger=LOGGER),
        patch.object(config.locator, "enable_action_fit", True),
        patch(COUNT, new=AsyncMock(return_value=0)),
    ):
        (first, second_result), m = await _run(
            [eid, eid],
            ["match", *second],
            elements=(spec,),
            engines=[FOUND, {**FOUND, "best_locator": "css=#other"}],
            extra=extra,
        )
    _shipped(first)
    assert second_result.error is None
    assert "Locator: css=#pick" in second_result.extracted_content
    assert wording[0] in second_result.extracted_content
    assert other[0] not in second_result.extracted_content
    line = next(msg for msg in caplog.messages if "signal: element-verifier-requery-kept" in msg)
    assert wording[1] in line and other[1] not in line


# ---- ruling 1: one definition, used by the pass-through and by registration ------------------


def test_point_only_predicate():
    po = pointed_identity.point_only
    assert po(None, None, True, "get_text", None)
    assert po({}, "  ", True, "get_text", False)
    assert not po({"tagName": "a"}, None, True, "get_text", None)  # element data
    assert not po(None, "Me", True, "get_text", None)  # words
    assert not po(None, None, False, "get_text", None)  # not locate-only
    assert not po(None, None, True, None, None)  # flag off
    assert not po(None, None, True, "get_text", True)  # a collection
    assert not po(None, None, True, "click", None)  # the identity gate proves it


async def test_the_no_evidence_pass_through_asks_the_predicate(page, cascade):
    target = AsyncMock(return_value=object())
    with (
        patch("browser_service.agent.actions.pointed_target", new=target),
        patch("browser_service.agent.actions.point_only", return_value=False) as po,
    ):
        result = await _call(page, action="get_text", locate_only=True)
    po.assert_called_once_with(None, None, True, "get_text", None)
    assert result["error_type"] == "NoElementEvidence"
    cascade.assert_not_called()


async def test_registration_asks_the_predicate_with_the_engines_inputs(caplog):
    with (
        caplog.at_level(logging.INFO, logger=LOGGER),
        patch("browser_service.agent.registration.point_only", return_value=True) as po,
        patch.object(config.locator, "enable_action_fit", True),
    ):
        (first,), m = await _run(["elem_1"], ["unsure"], extra={"expected_text": "Sign in"})
    po.assert_called_once_with(None, "Sign in", False, "click", None)
    _refused(first)
    assert "(signal: element-verifier-point-only-refused)" in caplog.text
