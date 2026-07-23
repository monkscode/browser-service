"""
Interaction outcome in the surviving long_term_memory line
(2026-07-23 failure root-cause design, Stage 2).

On success the handler returns BOTH extracted_content (the "✅ AUTO-ACTION
COMPLETE" confirmation) and long_term_memory (a locator-only sticky note).
browser-use 0.12.6 agent/message_manager/service.py:327-331 is an if/elif:
when long_term_memory is present, extracted_content is DISCARDED. The line
that survives says a locator was found and nothing about the interaction.

For an element whose action the agent cannot see on screen — a password
field shows dots — there is then no evidence in either channel, so it
re-queries find_unique_locator. Median 19 repeat calls, until
dynamic_max_steps kills the run. Loop rate: password fields 27.2%, ordinary
text inputs 2.4%.

Fix: record the interaction OUTCOME in the surviving line. Status-accurate by
construction — performed_actions is added to only on confirmed success
(registration.py:419, :494), so auto_failed can never read as performed.

This reverses the April cf99834 "keep long_term_memory minimal" rule, with
owner approval: that rule guards against forward-looking GUIDANCE biasing
later steps. An interaction outcome is a fact about what already happened.

Harness conventions from test_reregistration_downgrade_guard.py: the real
registered handler through the browser-use Tools registry, engine +
Playwright mocked. _do_interaction is faked at its own seam, reproducing its
real contract (registration.py:869 short-circuit, :874 non-interactive
actions, performed_actions.add only on confirmed success).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_service.agent.registration import register_custom_actions


class FakeBrowserSession:
    cdp_url = "ws://127.0.0.1:9222/devtools/browser/00000000-0000-0000-0000-000000000000"
    cdp_client = None
    is_cdp_connected = True
    _original_viewport_size = (1920, 1080)
    agent_focus_target_id = None

    async def get_selector_map(self):
        return {}

    async def get_current_page_url(self):
        return "https://example.org/login"


class FakeAgent:
    tools = None


def _fake_playwright():
    fake_page = AsyncMock()
    fake_page.url = "https://example.org/login"
    ctx = MagicMock()
    ctx.pages = [fake_page]
    fake_browser = MagicMock()
    fake_browser.contexts = [ctx]
    fake_instance = MagicMock()
    fake_instance.chromium.connect_over_cdp = AsyncMock(return_value=fake_browser)
    starter = MagicMock()
    starter.start = AsyncMock(return_value=fake_instance)
    return MagicMock(return_value=starter)


PASSWORD = {"id": "elem_2", "action": "input", "value": "admin"}
READ_ONLY = {"id": "elem_3", "action": "get_text", "value": ""}

STRONG = {
    "found": True,
    "best_locator": "input[type='password']",
    "validated": True,
    "count": 1,
    "unique": True,
    "valid": True,
    "validation_method": "playwright",
}
STRONG_ALT = {**STRONG, "best_locator": "id=password", "semantic_match": True}


def _params(element_id: str) -> dict:
    return {
        "x": 640,
        "y": 400,
        "element_id": element_id,
        "element_description": "password field in the login form",
        "element_index": 12,
    }


def _fake_interaction(statuses: list[str]):
    """Stand-in for _do_interaction that honours its real contract.

    `statuses` supplies the outcome for each interaction actually attempted.
    Returns the fake plus the list of element_ids it was called with."""
    pending = list(statuses)
    calls: list[str] = []

    async def _fake(
        browser_session,
        active_page,
        locator_str,
        element_id,
        element_index,
        element_specs,
        performed_actions,
        **kwargs,
    ):
        calls.append(element_id)
        # registration.py:869 — already-performed elements short-circuit.
        if element_id in performed_actions:
            return "", "not_applicable", "", ""
        spec = element_specs.get(element_id, {})
        action = spec.get("action", "get_text")
        value = spec.get("value", "")
        # registration.py:874 — actions that perform nothing.
        if action not in ("input", "type", "click", "submit", "select", "check", "uncheck"):
            return "", "not_applicable", action, value
        status = pending.pop(0)
        if status == "auto_ok":
            # registration.py:419/:494 — added only on confirmed success.
            performed_actions.add(element_id)
            return f"\n✅ AUTO-ACTION COMPLETE: Typed '{value}'", "auto_ok", action, value
        return f"\n⚠️ AUTO-ACTION FAILED ({action})", "auto_failed", action, value

    return _fake, calls


async def _run(elements, engine_results, statuses, element_id="elem_2"):
    """Register once (shared closure state), then execute the action once per
    engine result."""
    agent = FakeAgent()
    session = FakeBrowserSession()
    engine = AsyncMock(side_effect=[dict(r) for r in engine_results])
    fake_interaction, calls = _fake_interaction(statuses)
    with (
        patch("browser_service.agent.actions.find_unique_locator_action", new=engine),
        patch("browser_service.agent.registration._do_interaction", new=fake_interaction),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
    ):
        assert register_custom_actions(agent, elements=elements) is True
        results = []
        for _ in engine_results:
            results.append(
                await agent.tools.registry.execute_action(
                    "find_unique_locator",
                    _params(element_id),
                    browser_session=session,
                )
            )
        return results, calls


class TestOutcomeIsRecorded:
    @pytest.mark.asyncio
    async def test_auto_ok_says_performed(self):
        (result,), _ = await _run([PASSWORD, READ_ONLY], [STRONG], ["auto_ok"])
        assert result.long_term_memory == (
            "elem_2 ✅ input performed · validated = input[type='password']"
        )

    @pytest.mark.asyncio
    async def test_auto_failed_never_reads_as_performed(self):
        """The agent must keep seeing the failure — it is the trigger for the
        native-action recovery. Claiming success would suppress a repair the
        run genuinely needs."""
        (result,), _ = await _run([PASSWORD, READ_ONLY], [STRONG], ["auto_failed"])
        assert result.long_term_memory == (
            "elem_2 ⚠️ input NOT performed · validated = input[type='password']"
        )
        assert "✅" not in result.long_term_memory

    @pytest.mark.asyncio
    async def test_read_only_element_claims_no_interaction(self):
        """get_text performs nothing. Saying otherwise would be a lie, so the
        wording stays exactly as it is today."""
        (result,), _ = await _run([PASSWORD, READ_ONLY], [STRONG], [], element_id="elem_3")
        assert result.long_term_memory == "elem_3 validated = input[type='password']"


class TestRepeatOfCompletedElement:
    @pytest.mark.asyncio
    async def test_repeat_after_success_says_already_performed(self):
        results, calls = await _run(
            [PASSWORD, READ_ONLY], [STRONG, STRONG_ALT], ["auto_ok"]
        )
        assert results[1].long_term_memory == (
            "elem_2 ✅ already performed · validated = input[type='password']"
        )
        # Guardrail: the repeat short-circuits before re-running the interaction.
        assert calls == ["elem_2"]

    @pytest.mark.asyncio
    async def test_repeat_keeps_the_first_validated_locator(self):
        """The first validation ran at the step-order-correct page state (D3)."""
        results, _ = await _run(
            [PASSWORD, READ_ONLY], [STRONG, STRONG_ALT], ["auto_ok"]
        )
        assert "input[type='password']" in results[1].long_term_memory
        assert "id=password" not in results[1].long_term_memory

    @pytest.mark.asyncio
    async def test_repeat_after_failure_retries_and_does_not_claim_success(self):
        """A failed interaction is NOT 'already performed'. The element is in
        _completed_elements (its locator was found) but not in
        _performed_actions, so the re-query must re-attempt rather than
        short-circuit — otherwise the guardrail would kill a real recovery."""
        results, calls = await _run(
            [PASSWORD, READ_ONLY], [STRONG, STRONG_ALT], ["auto_failed", "auto_ok"]
        )
        assert "already performed" not in results[1].long_term_memory
        assert calls == ["elem_2", "elem_2"]
        assert results[1].long_term_memory == "elem_2 ✅ input performed · validated = id=password"


class TestDowngradeBlockCarriesOutcome:
    @pytest.mark.asyncio
    async def test_blocked_downgrade_reports_the_interaction(self):
        """The downgrade early return is the other construction site for the
        sticky note; it must not regress to a locator-only line."""
        weak = {**STRONG, "best_locator": "role=dialog", "semantic_match": False}
        results, _ = await _run([PASSWORD, READ_ONLY], [STRONG, weak], ["auto_ok"])
        assert results[1].long_term_memory == (
            "elem_2 ✅ already performed · validated = input[type='password']"
        )
