"""bs #62 step 3: the verifier's answer decides whether a pick ships, is asked again once, or is NOT FOUND.

Harness: the real registered handler through browser-use's Tools registry, the engine and Playwright faked,
``_do_interaction`` faked at its seam, ``element_verifier.read_element_facts`` / ``verify_pick`` patched
(the test_read_address_retry.py pattern). The extraction test feeds the handler's own metadata through
workflow.py's result extraction.
"""

import asyncio
import logging
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_service.agent.element_verifier import VerifierOutcome
from browser_service.agent.registration import register_custom_actions
from tests.test_agent.test_vision_escalation import (
    FakeAgent,
    FakeAgentWithSettings,
    FakeBrowserSession,
    _fake_playwright,
)
from tests.test_tasks.test_workflow_execution import (  # workflow_harness is a fixture
    FakeActionResult,
    FakeStep,
    arm,
    run_workflow,
    workflow_harness,
)

LOGGER = "browser_service.agent.registration"
VERIFIER = "browser_service.agent.element_verifier"
QUERY = "Open the shop, sign in, then read the first product name"
MARKER = "Zq62ValueMarker"
CLICK = {
    "id": "elem_1",
    "action": "click",
    "value": "",
    "description": "the Sign In button in the header",
}
READ = {"id": "elem_2", "action": "get_text", "value": "", "description": "the first product name"}
TYPE = {"id": "elem_1", "action": "input", "value": MARKER, "description": "the search box"}
FOUND = {"found": True, "best_locator": "css=#pick", "validated": True}
FACTS = {"tag": "a", "text": "Sign in", "landmark": "footer"}
REASON = "it is the footer link, not the header button"
PARAMS = {"x": 500, "y": 300, "element_description": "the Sign In button", "element_index": 7}
LLM = object()


def _outcome(verdict, reason=""):
    return VerifierOutcome(verdict, reason, prompt_tokens=310, completion_tokens=12, seconds=0.84)


async def _run(
    calls,
    verdicts=(),
    *,
    engines=None,
    elements=(CLICK, READ),
    agent=None,
    verifier_llm=LLM,
    read_facts=None,
    extra=None,
):
    """Register once, then call find_unique_locator once per element id in ``calls``.

    ``verdicts`` (a verdict, or (verdict, reason)) feed verify_pick in order; ``engines`` feed the engine
    (FOUND for every call by default). Returns the ActionResults and the mocks."""
    agent = agent or FakeAgent()
    engines = engines or [FOUND] * len(calls)
    engine = AsyncMock(
        side_effect=[{**e, "element_id": eid} for e, eid in zip(engines, calls, strict=True)]
    )
    pw = _fake_playwright()
    page = pw.return_value.start.return_value.chromium.connect_over_cdp.return_value.contexts[
        0
    ].pages[0]
    page.locator = MagicMock(side_effect=lambda sel: f"locator({sel})")
    mocks = SimpleNamespace(
        verify=AsyncMock(
            side_effect=[_outcome(*v) if isinstance(v, tuple) else _outcome(v) for v in verdicts]
        ),
        read=read_facts or AsyncMock(return_value=FACTS),
        interaction=AsyncMock(return_value=("", "not_applicable", "click", "")),
    )
    with (
        patch("browser_service.agent.actions.find_unique_locator_action", new=engine),
        patch("browser_service.agent.registration._do_interaction", new=mocks.interaction),
        patch(f"{VERIFIER}.verify_pick", new=mocks.verify),
        patch(f"{VERIFIER}.read_element_facts", new=mocks.read),
        patch("playwright.async_api.async_playwright", new=pw),
    ):
        assert (
            register_custom_actions(
                agent, elements=list(elements), user_query=QUERY, verifier_llm=verifier_llm
            )
            is True
        )
        results = [
            await agent.tools.registry.execute_action(
                "find_unique_locator",
                {**PARAMS, **(extra or {}), "element_id": eid},
                browser_session=FakeBrowserSession(),
            )
            for eid in calls
        ]
    return results, mocks


def _shipped(result, locator="css=#pick"):
    assert result.error is None
    assert result.metadata["found"] is True
    assert result.metadata["best_locator"] == locator


async def test_first_mismatch_reasks_with_reason_screenshot_and_found_false():
    agent = FakeAgentWithSettings("auto")
    (first,), m = await _run(["elem_1"], [("mismatch", REASON)], agent=agent)
    assert REASON in first.error
    assert first.extracted_content == first.error
    assert "find_unique_locator again" in first.error
    assert "screenshot" in first.error.lower()
    assert first.is_done is not True
    assert first.long_term_memory is None
    assert first.metadata["include_screenshot"] is True
    assert first.metadata["found"] is False
    assert first.metadata["best_locator"] is None
    assert first.metadata["error"] == REASON
    assert agent.settings.use_vision is True
    m.interaction.assert_not_awaited()


async def test_second_mismatch_is_not_found_and_says_do_not_retry():
    (_, second), m = await _run(["elem_2", "elem_2"], [("mismatch", REASON)] * 2)
    assert "NOT FOUND" in second.extracted_content
    assert "do not retry" in second.extracted_content.lower()
    # no error: browser-use counts an error result as an agent failure (5 in a row end the run)
    assert second.error is None
    assert second.is_done is not True
    assert second.long_term_memory is None
    assert not second.metadata.get("include_screenshot")
    assert second.metadata["found"] is False
    assert second.metadata["best_locator"] is None
    assert second.metadata["error"] == REASON
    m.interaction.assert_not_awaited()


async def test_action_step_not_found_says_do_not_perform():
    (_, click_nf), _ = await _run(["elem_1", "elem_1"], [("mismatch", REASON)] * 2)
    (_, read_nf), _ = await _run(["elem_2", "elem_2"], [("mismatch", REASON)] * 2)
    assert "do not perform this step's action yourself" in click_nf.extracted_content.lower()
    assert "perform" not in read_nf.extracted_content.lower()
    assert click_nf.error is None and read_nf.error is None


async def test_verifier_answers_never_chain_into_agent_failures():
    """Two elements rejected twice, then a third rejected once: no two results in a row carry error."""
    third = {"id": "elem_3", "action": "click", "value": "", "description": "the cart icon"}
    calls = ["elem_1", "elem_1", "elem_2", "elem_2", "elem_3"]
    results, _ = await _run(calls, [("mismatch", REASON)] * 5, elements=(CLICK, READ, third))
    errors = [r.error is not None for r in results]
    assert errors == [True, False, True, False, True]
    assert not any(a and b for a, b in zip(errors, errors[1:]))


async def test_match_proceeds_to_the_interaction(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (result,), m = await _run(["elem_1"], ["match"], elements=(TYPE,))
    _shipped(result)
    m.interaction.assert_awaited_once()
    # the inputs: the request, NLRF's step words and its action — never the step's value
    m.verify.assert_awaited_once_with(LLM, QUERY, TYPE["description"], "input", FACTS)
    m.read.assert_awaited_once_with("locator(css=#pick)")
    line = next(
        r.getMessage() for r in caplog.records if "signal: element-verifier-match" in r.getMessage()
    )
    assert "elem_1" in line and "310" in line and "12" in line and "0.8s" in line
    assert MARKER not in caplog.text
    assert "Sign in" not in caplog.text  # the facts are never logged


@pytest.mark.parametrize("verdict", ["unsure", "skipped"])
async def test_unsure_and_skipped_proceed(verdict, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (result,), m = await _run(["elem_1"], [(verdict, "timeout")])
    _shipped(result)
    m.interaction.assert_awaited_once()
    assert f"(signal: element-verifier-{verdict})" in caplog.text


async def test_counter_is_per_element():
    (_, other), _ = await _run(["elem_1", "elem_2"], [("mismatch", REASON)] * 2)
    assert "NOT FOUND" not in other.error
    assert "find_unique_locator again" in other.error
    assert other.metadata["include_screenshot"] is True


async def test_requery_of_accepted_element_keeps_it_with_no_error(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (first, second), m = await _run(
            ["elem_1", "elem_1"],
            ["match", ("mismatch", REASON)],
            engines=[FOUND, {**FOUND, "best_locator": "css=#other"}],
        )
    _shipped(first)
    assert second.error is None
    assert second.metadata is None
    assert "Locator: css=#pick" in second.extracted_content
    assert "the earlier validated locator stands" in second.extracted_content
    assert "css=#pick" in second.long_term_memory
    m.interaction.assert_awaited_once()
    assert "(signal: element-verifier-requery-kept)" in caplog.text


@pytest.mark.parametrize(
    ("engine", "extra", "element_id"),
    [
        ({**FOUND, "element_type": "collection"}, None, "elem_1"),
        ({**FOUND, "element_type": "file-upload"}, None, "elem_1"),
        (FOUND, {"is_collection": True}, "elem_1"),
        (FOUND, None, "elem_9"),  # no NLRF step for this id
    ],
    ids=["collection", "file-upload", "is_collection", "no-step"],
)
async def test_collection_upload_and_is_collection_make_no_call(engine, extra, element_id, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (result,), m = await _run([element_id], engines=[engine], extra=extra)
    _shipped(result)
    m.read.assert_not_awaited()
    m.verify.assert_not_awaited()
    m.interaction.assert_awaited_once()
    assert "(signal: element-verifier-not-asked)" in caplog.text


async def test_no_verifier_llm_makes_no_call(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (result,), m = await _run(["elem_1"], verifier_llm=None)
    _shipped(result)
    m.read.assert_not_awaited()
    m.verify.assert_not_awaited()
    m.interaction.assert_awaited_once()
    assert "element-verifier" not in caplog.text


@pytest.mark.parametrize(
    ("info", "target"),
    [
        ({"hidden_input": True, "input_locator": "input#agree"}, "locator(input#agree)"),
        ({"hidden_input": True, "input_locator": ""}, "locator(css=label.proxy)"),
        ({"hidden_input": False, "input_locator": "input#agree"}, "locator(css=label.proxy)"),
    ],
    ids=["proxy", "no-input-locator", "not-hidden"],
)
async def test_hidden_input_proxy_reads_facts_from_input_locator(info, target):
    engine = {**FOUND, "best_locator": "css=label.proxy", "element_info": info}
    (result,), m = await _run(["elem_1"], ["match"], engines=[engine])
    _shipped(result, "css=label.proxy")
    m.read.assert_awaited_once_with(target)


def _verdict_line(caplog):
    return next(r.getMessage() for r in caplog.records if "Element verifier for " in r.getMessage())


async def test_verdict_line_names_the_judged_locator_and_keeps_both_parsers(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        await _run(["elem_1"], ["match"])
    line = _verdict_line(caplog)
    assert "tokens, 0.8s) on 'css=#pick'" in line
    # the two log parsers that read this line
    assert re.search(r"Element verifier for [^ :]+: [a-z]+", line)
    m = re.search(
        r"Element verifier for (elem_\d+): (\w+) \(prompt (\d+) / completion (\d+) tokens, ([\d.]+)s\)",
        line,
    )
    assert m and m.groups() == ("elem_1", "match", "310", "12", "0.8")


async def test_mismatch_verdict_line_names_the_rejected_picks_locator(caplog):
    engine = {**FOUND, "best_locator": "css=footer a.signin"}
    with caplog.at_level(logging.INFO, logger=LOGGER):
        await _run(["elem_1"], [("mismatch", REASON)], engines=[engine])
    line = _verdict_line(caplog)
    assert "on 'css=footer a.signin'" in line
    assert ": mismatch" in line


async def test_verdict_line_names_the_input_locator_not_the_proxy(caplog):
    info = {"hidden_input": True, "input_locator": "input#agree"}
    engine = {**FOUND, "best_locator": "css=label.proxy", "element_info": info}
    with caplog.at_level(logging.INFO, logger=LOGGER):
        await _run(["elem_1"], ["match"], engines=[engine])
    line = _verdict_line(caplog)
    assert "on 'input#agree'" in line
    assert "label.proxy" not in line


async def test_requery_with_the_accepted_locator_makes_no_call(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (_, second), m = await _run(["elem_1", "elem_1"], ["match"])
    (_, today), t = await _run(["elem_1", "elem_1"], verifier_llm=None)
    m.verify.assert_awaited_once()
    assert m.interaction.await_count == t.interaction.await_count == 2
    assert (second.extracted_content, second.long_term_memory, second.error) == (
        today.extracted_content,
        today.long_term_memory,
        today.error,
    )
    assert second.metadata == today.metadata
    assert "(signal: element-verifier-not-asked)" in caplog.text


async def test_wiring_error_ships_the_pick(caplog):
    broken = AsyncMock(side_effect=RuntimeError("boom"))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        (result,), m = await _run(["elem_1"], ["mismatch"], read_facts=broken)
    _shipped(result)
    m.verify.assert_not_awaited()
    m.interaction.assert_awaited_once()
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "RuntimeError" in r.getMessage()
    ]
    assert len(warnings) == 1


@pytest.mark.parametrize("rejections", [1, 2], ids=["re-asked", "not-found"])
def test_rejected_pick_is_not_extracted(rejections, workflow_harness):
    results, _ = asyncio.run(_run(["elem_1"] * rejections, [("mismatch", REASON)] * rejections))
    arm(workflow_harness, [FakeStep([FakeActionResult(r.metadata)]) for r in results])
    run_workflow(workflow_harness, elements=[{"id": "elem_1", "description": CLICK["description"]}])
    entry = workflow_harness.updates[-1][1]["results"]["results"][0]
    assert entry["element_id"] == "elem_1"
    assert entry["found"] is False
    assert not entry.get("best_locator")
    assert entry["error"] == REASON
