"""
Unit and integration tests for interaction reliability helpers.

Tests the module-level helpers added to browser_service.agent.registration:
  _do_interaction_playwright  — layered Playwright retry chain
  _do_interaction             — event-path orchestrator with Playwright fallback

All unit tests mock at the boundary (active_page, browser_session, locator methods).
No live CDP or live browser required for unit tests.

Integration tests (marked with @pytest.mark.integration) require a real browser.

Coverage:
  #3  Malformed locator → auto_failed, no uncaught exception
  #4  Contenteditable: Control+a before press_sequentially in Tier 2 input
  #5  Tom Select: dispatchEvent reaches wrapper-bound change listener  [integration]
  #7  CDP stall on get_element_by_index → 10s timeout → Playwright fallback
  #8  Idempotency: second _do_interaction call for same element_id is no-op
  #9  performed_actions isolation: two independent sets do not interfere
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _import_helpers():
    from browser_service.agent.registration import (
        _do_interaction,
        _do_interaction_playwright,
    )
    return _do_interaction, _do_interaction_playwright


# ─────────────────────────────────────────────────────────────────────────────
# #3 — Malformed locator: _do_interaction_playwright must not raise
# ─────────────────────────────────────────────────────────────────────────────

class TestMalformedLocator:
    """_do_interaction_playwright catches invalid selectors and returns auto_failed."""

    async def test_malformed_locator_returns_auto_failed(self, performed_actions):
        _, _do_interaction_playwright = _import_helpers()

        mock_page = MagicMock()
        mock_page.locator.side_effect = Exception("Invalid selector: ###invalid")

        note, status = await _do_interaction_playwright(
            active_page=mock_page,
            locator_str="###invalid",
            action="click",
            value="",
            element_id="elem_1",
            performed_actions=performed_actions,
        )

        assert status == "auto_failed"
        assert "AUTO-ACTION FAILED" in note
        # Must not raise — outer except catches and surfaces as auto_failed
        assert "elem_1" not in performed_actions  # guard not set on failure


# ─────────────────────────────────────────────────────────────────────────────
# #4 — Contenteditable: Control+a clears all before press_sequentially
# ─────────────────────────────────────────────────────────────────────────────

class TestTier2ControlA:
    """Tier 2 input must call press('Control+a') before press_sequentially."""

    async def test_ctrl_a_before_sequential(self, performed_actions):
        _, _do_interaction_playwright = _import_helpers()

        # page.locator() is sync in Playwright — use MagicMock so the call returns
        # mock_loc directly without wrapping it in a coroutine (which AsyncMock does).
        mock_page = MagicMock()
        mock_loc = MagicMock()
        mock_page.locator.return_value = mock_loc

        # Tier 1: fill() raises — forces Tier 2
        mock_loc.fill = AsyncMock(side_effect=Exception("fill not supported on contenteditable"))
        # Tier 2: triple_click, press, press_sequentially all succeed
        mock_loc.triple_click = AsyncMock()
        mock_loc.press = AsyncMock()
        mock_loc.press_sequentially = AsyncMock()
        # input_value() raises (contenteditable) — auto_partial path
        mock_loc.input_value = AsyncMock(side_effect=Exception("input_value not supported"))

        note, status = await _do_interaction_playwright(
            active_page=mock_page,
            locator_str="div[contenteditable]",
            action="input",
            value="hello world",
            element_id="elem_1",
            performed_actions=performed_actions,
        )

        # Control+a must be called
        press_calls = [call.args[0] for call in mock_loc.press.call_args_list]
        assert "Control+a" in press_calls, "press('Control+a') must be called in Tier 2"

        # Control+a must precede press_sequentially
        ctrl_a_call_index = press_calls.index("Control+a")
        assert ctrl_a_call_index == 0, "Control+a must be the first press call"
        mock_loc.press_sequentially.assert_awaited_once()

        assert status in ("auto_ok", "auto_partial")
        assert "key events" in note
        assert "elem_1" in performed_actions  # guard set on success


# ─────────────────────────────────────────────────────────────────────────────
# #5 — Tom Select: dispatchEvent fires on wrapper-bound listener  [integration]
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skip(
    reason="Needs pytest-playwright's live 'page' fixture; pytest-playwright's sync "
    "event loop conflicts with this suite's asyncio_mode=auto. Run under a dedicated "
    "async-Playwright setup."
)
@pytest.mark.integration
class TestTomSelectWrapperEvent:
    """
    Verify that the Tom Select JS tier fires a change event on the wrapper element,
    not just on the underlying <select>. Requires a real browser (Playwright fixture).
    """

    async def test_tom_select_dispatches_change_on_wrapper(self, page):
        """change event must bubble to wrapper-bound listeners after setValue."""
        await page.set_content("""
            <div class="ts-wrapper">
                <div class="ts-control">pick one</div>
                <select id="s">
                    <option value="opt1">Option One</option>
                    <option value="opt2">Option Two</option>
                </select>
            </div>
            <script>
                const select = document.getElementById('s');
                select.tomselect = {
                    setValue(val, silent) { select.value = val; }
                };
                window._wrapperChangeCount = 0;
                document.querySelector('.ts-wrapper').addEventListener('change', () => {
                    window._wrapperChangeCount++;
                });
            </script>
        """)

        from browser_service.agent.registration import _do_interaction_playwright
        note, status = await _do_interaction_playwright(
            active_page=page,
            locator_str=".ts-control",
            action="select",
            value="Option One",
            element_id="elem_1",
            performed_actions=set(),
            dropdown_framework="tom-select",
            select_id=None,
        )

        assert status == "auto_ok", f"Expected auto_ok, got {status!r}: {note}"
        wrapper_count = await page.evaluate("window._wrapperChangeCount")
        assert wrapper_count >= 1, "change event must reach wrapper-bound listeners via bubbling"


# ─────────────────────────────────────────────────────────────────────────────
# #7 — CDP stall: 10s ceiling fires, falls through to Playwright
# ─────────────────────────────────────────────────────────────────────────────

class TestCdpStallFallthrough:
    """A stalled get_element_by_index must not hang _do_interaction — Playwright handles it."""

    async def test_stalled_get_element_by_index_falls_through(self, performed_actions):
        _do_interaction, _ = _import_helpers()

        mock_bs = AsyncMock()

        async def stalled_index_lookup(index):
            await asyncio.sleep(9999)

        mock_bs.get_element_by_index.side_effect = stalled_index_lookup

        # page.locator() is sync — use MagicMock so it returns mock_loc directly.
        # wait_for_load_state is async (called by _wait_for_page_stability after click).
        mock_page = MagicMock()
        mock_page.wait_for_load_state = AsyncMock()
        mock_loc = MagicMock()
        mock_page.locator.return_value = mock_loc
        mock_loc.click = AsyncMock()  # Playwright Tier 1 succeeds

        element_specs = {"elem_1": {"action": "click", "value": ""}}

        with patch(
            "browser_service.agent.registration.asyncio.wait_for",
            side_effect=asyncio.TimeoutError,
        ):
            note, status, action, value = await _do_interaction(
                browser_session=mock_bs,
                active_page=mock_page,
                locator_str="button#submit",
                element_id="elem_1",
                element_index=42,
                element_specs=element_specs,
                performed_actions=performed_actions,
            )

        assert status == "auto_ok", f"Expected auto_ok from Playwright fallback, got: {status}"
        mock_loc.click.assert_awaited_once()
        assert "elem_1" in performed_actions


# ─────────────────────────────────────────────────────────────────────────────
# #8 — Idempotency: second call for same element_id is no-op
# ─────────────────────────────────────────────────────────────────────────────

class TestIdempotency:
    """_do_interaction must not re-perform an action already in performed_actions."""

    async def test_second_call_returns_not_applicable(self, performed_actions):
        _do_interaction, _ = _import_helpers()

        mock_page = MagicMock()
        mock_page.wait_for_load_state = AsyncMock()  # called by _wait_for_page_stability
        mock_loc = MagicMock()
        mock_page.locator.return_value = mock_loc
        mock_loc.click = AsyncMock()

        element_specs = {"elem_1": {"action": "click", "value": ""}}

        # First call — should succeed
        note, status, action, value = await _do_interaction(
            browser_session=None,
            active_page=mock_page,
            locator_str="button#login",
            element_id="elem_1",
            element_index=None,
            element_specs=element_specs,
            performed_actions=performed_actions,
        )
        assert status == "auto_ok"
        assert "elem_1" in performed_actions
        assert mock_loc.click.await_count == 1

        # Second call — must be a no-op (idempotency guard)
        note2, status2, _, _ = await _do_interaction(
            browser_session=None,
            active_page=mock_page,
            locator_str="button#login",
            element_id="elem_1",
            element_index=None,
            element_specs=element_specs,
            performed_actions=performed_actions,
        )
        assert status2 == "not_applicable"
        assert note2 == ""
        assert mock_loc.click.await_count == 1  # no additional click


# ─────────────────────────────────────────────────────────────────────────────
# #9 — performed_actions isolation: two independent sets do not interfere
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformedActionsIsolation:
    """Two workflows with separate performed_actions sets must not interfere."""

    async def test_two_workflows_independent(self):
        _do_interaction, _ = _import_helpers()

        mock_page = MagicMock()
        mock_page.wait_for_load_state = AsyncMock()  # called by _wait_for_page_stability
        mock_loc = MagicMock()
        mock_page.locator.return_value = mock_loc
        mock_loc.click = AsyncMock()

        element_specs = {"elem_1": {"action": "click", "value": ""}}

        set_workflow_a = set()
        set_workflow_b = set()

        # Workflow A performs the action
        _, status_a, _, _ = await _do_interaction(
            browser_session=None,
            active_page=mock_page,
            locator_str="button#submit",
            element_id="elem_1",
            element_index=None,
            element_specs=element_specs,
            performed_actions=set_workflow_a,
        )
        assert status_a == "auto_ok"
        assert "elem_1" in set_workflow_a

        # Workflow B must still be able to perform the same action independently
        _, status_b, _, _ = await _do_interaction(
            browser_session=None,
            active_page=mock_page,
            locator_str="button#submit",
            element_id="elem_1",
            element_index=None,
            element_specs=element_specs,
            performed_actions=set_workflow_b,
        )
        assert status_b == "auto_ok"
        assert "elem_1" in set_workflow_b
        assert mock_loc.click.await_count == 2  # both workflows clicked


# ─────────────────────────────────────────────────────────────────────────────
# #10 — Tom Select Tier 0: select_id path uses getElementById JS, skips locator
# ─────────────────────────────────────────────────────────────────────────────

class TestTomSelectTier0SelectId:
    """When select_id is provided, Tier 0 (getElementById) must succeed without touching loc.evaluate."""

    async def test_tier0_select_id_succeeds(self, performed_actions):
        _, _do_interaction_playwright = _import_helpers()

        mock_page = MagicMock()
        mock_loc = MagicMock()
        mock_page.locator.return_value = mock_loc

        # page.evaluate is async — JS returns "ok" on success (string, not bool)
        mock_page.evaluate = AsyncMock(return_value="ok")
        # loc.evaluate must NOT be called when select_id succeeds
        mock_loc.evaluate = AsyncMock(return_value="ok")

        note, status = await _do_interaction_playwright(
            active_page=mock_page,
            locator_str=".ts-control",
            action="select",
            value="Option One",
            element_id="elem_1",
            performed_actions=performed_actions,
            dropdown_framework="tom-select",
            select_id="role",
        )

        assert status == "auto_ok"
        assert "AUTO-SELECT" in note
        # page.evaluate called for getElementById tier; loc.evaluate must not be called
        mock_page.evaluate.assert_awaited_once()
        mock_loc.evaluate.assert_not_awaited()
        assert "elem_1" in performed_actions

    async def test_tier0_falls_through_to_tier0b_on_false(self, performed_actions):
        """If getElementById JS returns False (option not found), fall through to Tier 0b."""
        _, _do_interaction_playwright = _import_helpers()

        mock_page = MagicMock()
        mock_loc = MagicMock()
        mock_page.locator.return_value = mock_loc

        # Tier 0 diagnostic: option not found in ts.options
        mock_page.evaluate = AsyncMock(return_value="no_opt")
        # Tier 0b succeeds via locator-based traversal — JS returns "ok"
        mock_loc.evaluate = AsyncMock(return_value="ok")

        note, status = await _do_interaction_playwright(
            active_page=mock_page,
            locator_str=".ts-control",
            action="select",
            value="Option One",
            element_id="elem_1",
            performed_actions=performed_actions,
            dropdown_framework="tom-select",
            select_id="role",
        )

        assert status == "auto_ok"
        mock_page.evaluate.assert_awaited_once()
        mock_loc.evaluate.assert_awaited_once()
        assert "elem_1" in performed_actions


# ─────────────────────────────────────────────────────────────────────────────
# #11 — Tom Select Tier 0b: sibling DOM traversal  [integration]
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skip(
    reason="Needs pytest-playwright's live 'page' fixture; pytest-playwright's sync "
    "event loop conflicts with this suite's asyncio_mode=auto. Run under a dedicated "
    "async-Playwright setup."
)
@pytest.mark.integration
class TestTomSelectSiblingDomTraversal:
    """
    Verify Tier 0b works when <select> is the *previous sibling* of .ts-wrapper
    (standard Tom Select DOM) — not a child. Requires a real browser.
    """

    async def test_select_previous_sibling_of_wrapper(self, page):
        """JS traversal must find <select> via previousElementSibling, not querySelector('select')."""
        await page.set_content("""
            <div id="container">
                <select id="role">
                    <option value="admin">Administrator</option>
                    <option value="viewer">Viewer</option>
                </select>
                <div class="ts-wrapper">
                    <div class="ts-control" id="ts-ctrl">pick role</div>
                </div>
            </div>
            <script>
                const select = document.getElementById('role');
                select.tomselect = {
                    setValue(val, silent) { select.value = val; }
                };
            </script>
        """)

        from browser_service.agent.registration import _do_interaction_playwright
        note, status = await _do_interaction_playwright(
            active_page=page,
            locator_str="#ts-ctrl",
            action="select",
            value="Administrator",
            element_id="elem_1",
            performed_actions=set(),
            dropdown_framework="tom-select",
            select_id=None,
        )

        assert status == "auto_ok", f"Expected auto_ok, got {status!r}: {note}"
        value = await page.locator("#role").evaluate("el => el.value")
        assert value == "admin", f"Expected 'admin', got '{value}'"


# ─────────────────────────────────────────────────────────────────────────────
# #12 — _strip_rf_select_prefix: all branch coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestStripRfSelectPrefix:
    """_strip_rf_select_prefix must only strip known RF strategy prefixes."""

    def _strip(self, value: str) -> str:
        from browser_service.agent.registration import _strip_rf_select_prefix
        return _strip_rf_select_prefix(value)

    def test_empty_string_unchanged(self):
        assert self._strip("") == ""

    def test_single_token_unchanged(self):
        assert self._strip("India") == "India"

    def test_label_prefix_stripped(self):
        assert self._strip("label India") == "India"

    def test_value_prefix_stripped(self):
        assert self._strip("value some_val") == "some_val"

    def test_text_prefix_stripped(self):
        assert self._strip("text My Option") == "My Option"

    def test_index_prefix_stripped(self):
        assert self._strip("index 0") == "0"

    def test_prefix_case_insensitive(self):
        assert self._strip("Label India") == "India"

    def test_multi_word_option_not_stripped(self):
        # "United" is not an RF prefix — the full string must be preserved
        assert self._strip("United States") == "United States"

    def test_multi_word_value_preserved_after_strip(self):
        assert self._strip("text Hello World") == "Hello World"


# ─────────────────────────────────────────────────────────────────────────────
# #13 — Tom Select action remap: input/type → select when dropdown_framework=tom-select
# ─────────────────────────────────────────────────────────────────────────────

class TestTomSelectActionRemap:
    """When dropdown_framework='tom-select', action 'input' or 'type' must be remapped to 'select'."""

    async def test_input_remapped_to_select(self, performed_actions):
        _do_interaction, _ = _import_helpers()

        mock_page = MagicMock()
        mock_loc = MagicMock()
        mock_page.locator.return_value = mock_loc
        mock_page.wait_for_load_state = AsyncMock()

        # Tier 0b (locator traversal) succeeds — JS returns "ok"
        mock_loc.evaluate = AsyncMock(return_value="ok")

        element_specs = {"elem_1": {"action": "input", "value": "India"}}

        note, status, action, value = await _do_interaction(
            browser_session=None,
            active_page=mock_page,
            locator_str=".ts-control",
            element_id="elem_1",
            element_index=None,
            element_specs=element_specs,
            performed_actions=performed_actions,
            dropdown_framework="tom-select",
        )

        # Action must have been remapped to "select" — JS evaluate is the select path
        assert status == "auto_ok", f"Expected auto_ok after remap, got: {status}"
        assert action == "select", f"Expected action='select' after remap, got: {action!r}"
        mock_loc.evaluate.assert_awaited_once()
        assert "elem_1" in performed_actions

    async def test_type_remapped_to_select(self, performed_actions):
        _do_interaction, _ = _import_helpers()

        mock_page = MagicMock()
        mock_loc = MagicMock()
        mock_page.locator.return_value = mock_loc
        mock_page.wait_for_load_state = AsyncMock()
        mock_loc.evaluate = AsyncMock(return_value="ok")

        element_specs = {"elem_1": {"action": "type", "value": "Male"}}

        note, status, action, value = await _do_interaction(
            browser_session=None,
            active_page=mock_page,
            locator_str=".ts-control",
            element_id="elem_1",
            element_index=None,
            element_specs=element_specs,
            performed_actions=performed_actions,
            dropdown_framework="tom-select",
        )

        assert status == "auto_ok"
        assert action == "select"
        assert "elem_1" in performed_actions


# ─────────────────────────────────────────────────────────────────────────────
# Task D (G4) — flatpickr Tier 0 in the input chain
# ─────────────────────────────────────────────────────────────────────────────

class TestFlatpickrInputTier:
    """datepicker_framework='flatpickr' routes input actions through the
    widget's setDate API BEFORE any fill/type tier. Readonly flatpickr
    inputs never become editable, so fill() waits its full timeout and
    triple_click leaves the calendar overlay open — the JS tier is the
    only deterministic path (verified live on ASTPP 2026-07-08)."""

    async def test_setdate_js_runs_before_fill(self, performed_actions):
        _, _do_interaction_playwright = _import_helpers()

        mock_page = MagicMock()
        mock_loc = MagicMock()
        mock_page.locator.return_value = mock_loc
        mock_loc.evaluate = AsyncMock(return_value="ok")
        mock_loc.fill = AsyncMock()

        note, status = await _do_interaction_playwright(
            active_page=mock_page,
            locator_str="id=customer_cdr_from_date",
            action="input",
            value="2026-07-01",
            element_id="elem_1",
            performed_actions=performed_actions,
            datepicker_framework="flatpickr",
        )

        assert status == "auto_ok", f"Expected auto_ok, got {status!r}: {note}"
        assert "flatpickr" in note.lower()
        mock_loc.evaluate.assert_awaited_once()
        mock_loc.fill.assert_not_awaited()
        assert "elem_1" in performed_actions

    async def test_setdate_diag_failure_falls_through_to_fill(self, performed_actions):
        """Fail-open contract: a 'no_fp' diag (element lost its instance)
        must fall through to the generic input tiers, same as Tom Select."""
        _, _do_interaction_playwright = _import_helpers()

        mock_page = MagicMock()
        mock_loc = MagicMock()
        mock_page.locator.return_value = mock_loc
        mock_loc.evaluate = AsyncMock(return_value="no_fp")
        mock_loc.fill = AsyncMock()
        mock_loc.input_value = AsyncMock(return_value="2026-07-01")

        note, status = await _do_interaction_playwright(
            active_page=mock_page,
            locator_str="id=customer_cdr_from_date",
            action="input",
            value="2026-07-01",
            element_id="elem_1",
            performed_actions=performed_actions,
            datepicker_framework="flatpickr",
        )

        assert status == "auto_ok"
        mock_loc.fill.assert_awaited_once()
        assert "elem_1" in performed_actions

    async def test_no_datepicker_framework_skips_js_tier(self, performed_actions):
        """Plain inputs must be untouched — no widget JS ever runs."""
        _, _do_interaction_playwright = _import_helpers()

        mock_page = MagicMock()
        mock_loc = MagicMock()
        mock_page.locator.return_value = mock_loc
        mock_loc.evaluate = AsyncMock()
        mock_loc.fill = AsyncMock()
        mock_loc.input_value = AsyncMock(return_value="hello")

        note, status = await _do_interaction_playwright(
            active_page=mock_page,
            locator_str="#search_box",
            action="input",
            value="hello",
            element_id="elem_1",
            performed_actions=performed_actions,
        )

        assert status == "auto_ok"
        mock_loc.evaluate.assert_not_awaited()
        mock_loc.fill.assert_awaited_once()

    async def test_event_path_skipped_for_flatpickr(self, performed_actions):
        """browser-use's TypeTextEvent 'succeeds' uselessly on a readonly
        input — _do_interaction must skip the event path entirely (same
        contract as Tom Select) so the JS tier runs."""
        _do_interaction, _ = _import_helpers()

        mock_bs = AsyncMock()

        mock_page = MagicMock()
        mock_page.wait_for_load_state = AsyncMock()
        mock_loc = MagicMock()
        mock_page.locator.return_value = mock_loc
        mock_loc.evaluate = AsyncMock(return_value="ok")

        element_specs = {"elem_1": {"action": "input", "value": "2026-07-01"}}

        note, status, action, value = await _do_interaction(
            browser_session=mock_bs,
            active_page=mock_page,
            locator_str="id=customer_cdr_from_date",
            element_id="elem_1",
            element_index=7,
            element_specs=element_specs,
            performed_actions=performed_actions,
            datepicker_framework="flatpickr",
        )

        assert status == "auto_ok"
        mock_bs.get_element_by_index.assert_not_awaited()
        mock_loc.evaluate.assert_awaited_once()
        assert "elem_1" in performed_actions
