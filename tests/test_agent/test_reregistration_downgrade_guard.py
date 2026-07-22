"""
Dialog-clobber guard D3 — re-registration must never downgrade
(discovered 2026-07-16, ASTPP gate r2: q02 r3 wf 99b7c4b8, q04 r1
wf 975435dd).

The chain's last defect: the agent re-queried the already-validated Sign
In button at the WRONG page state (announcement modal up, post-login);
the second result — a semantically-mismatched dialog locator — replaced
the correct locator validated 43s earlier, both in the handler's
completion dict (last-write-wins by design comment) and in workflow.py's
metadata extraction ("replace with latest"). For the generated test the
FIRST, step-order-correct validation is the right one: the re-query ran
against a page state the step never sees at runtime.

Rule: a fully-validated result (validated=True and semantic_match not
False — degraded paths mark themselves False; strong paths set True or
omit the key) is only replaced by another fully-validated result. A
downgrade is discarded at the source: the blocked ActionResult carries
NO metadata, so workflow.py's extraction never sees the degraded result.
Upgrades and equal-strength re-validations keep today's behavior.
Signal: re-registration-downgrade-blocked

Harness conventions from test_vision_escalation.py: the real registered
handler through the browser-use Tools registry, engine + Playwright
mocked.
"""

import logging
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
        return "https://sujal.astppbilling.org/"


class FakeAgent:
    tools = None


def _fake_playwright():
    fake_page = AsyncMock()
    fake_page.url = "https://sujal.astppbilling.org/"
    ctx = MagicMock()
    ctx.pages = [fake_page]
    fake_browser = MagicMock()
    fake_browser.contexts = [ctx]
    fake_instance = MagicMock()
    fake_instance.chromium.connect_over_cdp = AsyncMock(return_value=fake_browser)
    starter = MagicMock()
    starter.start = AsyncMock(return_value=fake_instance)
    return MagicMock(return_value=starter)


ELEMENTS = [{"id": "elem_3", "action": "get_text", "value": ""}]
PARAMS = {
    "x": 640,
    "y": 400,
    "element_id": "elem_3",
    "element_description": "Sign In button in the login form",
    "element_index": 12,
}

STRONG_A = {
    "found": True,
    "best_locator": "id=save_button",
    "validated": True,
    "count": 1,
    "unique": True,
    "valid": True,
    "validation_method": "playwright",
}
STRONG_B = {**STRONG_A, "best_locator": 'text="Sign In"', "semantic_match": True}
WEAK_DIALOG = {
    **STRONG_A,
    "best_locator": 'role=dialog[name="Smarter Support Starts Here"]',
    "semantic_match": False,
}


async def _run_sequence(first: dict, second: dict):
    """Register once (shared closure state), execute the action twice with
    the engine returning `first` then `second`.

    registration.py imports find_unique_locator_action at REGISTRATION
    time (function-level import), so the engine patch must wrap
    register_custom_actions, not just the execute calls."""
    agent = FakeAgent()
    session = FakeBrowserSession()
    engine = AsyncMock(side_effect=[dict(first), dict(second)])
    with (
        patch("browser_service.agent.actions.find_unique_locator_action", new=engine),
        patch("playwright.async_api.async_playwright", new=_fake_playwright()),
    ):
        assert register_custom_actions(agent, elements=ELEMENTS) is True
        results = []
        for _ in range(2):
            results.append(
                await agent.tools.registry.execute_action(
                    "find_unique_locator",
                    dict(PARAMS),
                    browser_session=session,
                )
            )
        return results


class TestDowngradeBlocked:
    @pytest.mark.asyncio
    async def test_weak_requery_does_not_replace_validated(self, caplog):
        """The clobber replay: strong accept, then a semantic_match=False
        re-query — the degraded result is discarded at the source."""
        with caplog.at_level(logging.INFO):
            first, second = await _run_sequence(STRONG_A, WEAK_DIALOG)
        assert first.metadata and first.metadata.get("best_locator") == "id=save_button"
        assert "re-registration-downgrade-blocked" in caplog.text
        # The blocked ActionResult must not feed workflow extraction.
        assert not (
            second.metadata
            and second.metadata.get("found")
            and second.metadata.get("best_locator") == WEAK_DIALOG["best_locator"]
        )
        # The agent is told the element is already validated with the KEPT locator.
        assert "id=save_button" in (second.extracted_content or "")

    @pytest.mark.asyncio
    async def test_strong_requery_still_replaces(self):
        """Equal-strength re-validation keeps today's behavior — the
        latest fully-validated result wins (legit retry/update)."""
        first, second = await _run_sequence(STRONG_A, STRONG_B)
        assert second.metadata is not None
        assert second.metadata.get("best_locator") == STRONG_B["best_locator"]

    @pytest.mark.asyncio
    async def test_upgrade_replaces_weak_first_result(self):
        """A weak first result (degraded path) is replaced by a later
        fully-validated one — the guard blocks downgrades only."""
        first, second = await _run_sequence(WEAK_DIALOG, STRONG_A)
        assert second.metadata is not None
        assert second.metadata.get("best_locator") == "id=save_button"
