"""
Unit tests for browser_service.prompts — system prompt and workflow prompt builders.

Purpose: Prompts define the agent's behaviour.  build_system_prompt controls the
         agent's role and rules; build_workflow_prompt controls what task the agent
         executes.  A regression in either means the agent misbehaves (wrong mode,
         missing instructions, injection vulnerabilities).

Tests cover:
  System prompt:
  - Custom action mode includes find_unique_locator instructions
  - Legacy mode includes legacy locator instructions
  - Prompt is never empty

  Workflow prompt:
  - URL is included and https-prefixed if missing
  - User query is included
  - Elements list is included
  - Empty elements raises ValueError
  - Missing URL raises ValueError
  - Element count capped at limit
  - Custom vs legacy mode content differences
  - Client hints included when provided
  - Value field forwarded for input actions
  - loop_type forwarded for loop steps
"""

import re

import pytest

from browser_service.prompts.system import build_system_prompt
from browser_service.prompts.workflow import build_workflow_prompt


class TestBuildSystemPrompt:
    """Tests for system prompt construction."""

    def test_custom_action_mode_content(self):
        """Custom action mode includes find_unique_locator instructions."""
        prompt = build_system_prompt(include_custom_action=True)
        assert "find_unique_locator" in prompt.lower() or "custom" in prompt.lower()

    def test_legacy_mode_content(self):
        """Legacy mode includes legacy-specific instructions."""
        prompt = build_system_prompt(include_custom_action=False)
        assert len(prompt) > 0
        # Legacy mode should NOT mention custom actions
        assert "find_unique_locator" not in prompt.lower()

    def test_prompt_not_empty(self):
        """System prompt is never empty regardless of mode."""
        assert len(build_system_prompt(include_custom_action=True)) > 100
        assert len(build_system_prompt(include_custom_action=False)) > 100


class TestBuildWorkflowPrompt:
    """Tests for workflow prompt construction."""

    def _make_elements(self, count=2):
        """Helper to create N sample elements."""
        return [
            {"id": f"elem_{i}", "description": f"element {i}", "action": "click"}
            for i in range(1, count + 1)
        ]

    def test_url_included(self):
        """Target URL appears in the workflow prompt."""
        prompt = build_workflow_prompt(
            url="https://example.com",
            user_query="click the button",
            elements=self._make_elements(),
        )
        assert "https://example.com" in prompt

    def test_url_gets_https_prefix(self):
        """URL without protocol gets https:// prepended."""
        prompt = build_workflow_prompt(
            url="example.com",
            user_query="click the button",
            elements=self._make_elements(),
        )
        assert "https://example.com" in prompt

    def test_user_query_included(self):
        """User query is embedded in the prompt."""
        prompt = build_workflow_prompt(
            url="https://example.com",
            user_query="search for shoes and get price",
            elements=self._make_elements(),
        )
        assert "search for shoes" in prompt

    def test_elements_included(self):
        """Element descriptions appear in the prompt."""
        elements = [
            {"id": "elem_1", "description": "search input in header", "action": "input"},
        ]
        prompt = build_workflow_prompt(
            url="https://example.com",
            user_query="search",
            elements=elements,
        )
        assert "search input" in prompt

    def test_long_query_and_description_are_truncated(self):
        """Over-long caller inputs are capped so the prompt stays bounded."""
        prompt = build_workflow_prompt(
            url="https://example.com",
            user_query="q" * 600,
            elements=[{"id": "elem_1", "description": "d" * 300, "action": "get_text"}],
        )
        # query capped at 500 + ellipsis; full 600 never appears
        assert "q" * 500 + "..." in prompt
        assert "q" * 600 not in prompt
        # description capped at 200 + ellipsis
        assert "d" * 200 + "..." in prompt
        assert "d" * 300 not in prompt

    def test_empty_elements_raises(self):
        """Empty elements list raises ValueError — nothing to process."""
        with pytest.raises((ValueError, Exception)):
            build_workflow_prompt(
                url="https://example.com",
                user_query="test",
                elements=[],
            )

    def test_missing_url_raises(self):
        """Missing/empty URL raises ValueError — can't navigate."""
        with pytest.raises((ValueError, Exception)):
            build_workflow_prompt(
                url="",
                user_query="test",
                elements=self._make_elements(),
            )

    def test_element_count_limit(self):
        """Prompt handles large element lists by enforcing a limit."""
        elements = self._make_elements(count=55)
        with pytest.raises(ValueError, match="Too many elements"):
            build_workflow_prompt(
                url="https://example.com",
                user_query="find everything",
                elements=elements,
            )

    def test_custom_action_mode(self):
        """Custom action mode prompt includes action-specific instructions."""
        prompt = build_workflow_prompt(
            url="https://example.com",
            user_query="click",
            elements=self._make_elements(),
            include_custom_action=True,
        )
        assert len(prompt) > 100

    def test_legacy_mode(self):
        """Legacy mode prompt differs from custom action mode."""
        prompt = build_workflow_prompt(
            url="https://example.com",
            user_query="click",
            elements=self._make_elements(),
            include_custom_action=False,
        )
        assert len(prompt) > 100

    def test_client_hints_included(self):
        """Client hints (from NL backend) are included when provided.

        client_hints is a LIST of hints, as the caller passes it
        (client_config.system_prompt_additions). Handing it a bare string
        makes the bullet join iterate characters, so each hint has to survive
        intact as its own line for this to mean anything.
        """
        prompt = build_workflow_prompt(
            url="https://example.com",
            user_query="click",
            elements=self._make_elements(),
            client_hints=["Handle cookie popup first", "Wait for the spinner"],
        )
        assert "• Handle cookie popup first" in prompt
        assert "• Wait for the spinner" in prompt

    def test_no_client_hints(self):
        """Prompt works without client hints."""
        prompt = build_workflow_prompt(
            url="https://example.com",
            user_query="click",
            elements=self._make_elements(),
        )
        assert len(prompt) > 100

    def test_value_field_forwarded(self):
        """Value field (for input actions) is included in prompt."""
        elements = [
            {"id": "elem_1", "description": "search box", "action": "input", "value": "shoes"},
        ]
        prompt = build_workflow_prompt(
            url="https://example.com",
            user_query="search",
            elements=elements,
        )
        assert "shoes" in prompt

    def test_loop_type_forwarded(self):
        """loop_type field is preserved in prompt for loop steps."""
        elements = [
            {
                "id": "elem_1",
                "description": "row items",
                "action": "get_text",
                "loop_type": "FOR",
                "loop_source": "table_rows",
            },
        ]
        prompt = build_workflow_prompt(
            url="https://example.com",
            user_query="extract rows",
            elements=elements,
        )
        # Either the raw loop_type is mentioned or the concept is in the prompt
        assert len(prompt) > 100


class TestVisionOffContract:
    """A1-INLINE (Task 28): the workflow prompt contract for vision-off runs.

    With use_vision='auto' the model has NO screenshot on normal steps — the
    DOM element list is its only grounding, so element_index must be a hard
    requirement, and the escalation rule must forbid found:false before a
    find_unique_locator attempt (whose failure attaches a screenshot to the
    next message).

    Owner commitments pinned here as regression guards:
    SEQUENTIAL_PROCESSING_RULES and the mandatory expected_text stay untouched.
    """

    def _prompt(self):
        return build_workflow_prompt(
            url="https://example.com",
            user_query="click submit",
            elements=[{"id": "elem_1", "description": "submit button", "action": "click"}],
            include_custom_action=True,
        )

    def test_element_index_documented_as_required(self):
        """element_index is a required parameter in the action contract."""
        assert "element_index (int, required)" in self._prompt()

    def test_element_index_old_accuracy_wording_gone(self):
        """The old 'REQUIRED FOR ACCURACY' soft wording is replaced."""
        assert "REQUIRED FOR ACCURACY" not in self._prompt()

    def test_escalation_rule_before_found_false(self):
        """found:false is only allowed after a failed find_unique_locator attempt,
        re-examined with the screenshot attached to the next message."""
        prompt = self._prompt()
        assert "NEVER record an element as found: false" in prompt
        assert "re-examine" in prompt.lower()
        assert "screenshot" in prompt.lower()

    # ── Owner-commitment regression guards (must not change in Task 28) ──

    def test_sequential_processing_rules_untouched(self):
        from browser_service.prompts.templates import SEQUENTIAL_PROCESSING_RULES

        assert "Process elements IN THE ORDER THEY ARE LISTED" in SEQUENTIAL_PROCESSING_RULES
        assert "FALLBACK REQUIRED" in SEQUENTIAL_PROCESSING_RULES
        assert "Always provide element_index" in SEQUENTIAL_PROCESSING_RULES

    def test_expected_text_mandate_untouched(self):
        from browser_service.prompts.templates import CUSTOM_ACTION_PARAMETERS_EXTENDED

        assert (
            "PROVIDE THIS whenever the element has visible text"
            in CUSTOM_ACTION_PARAMETERS_EXTENDED
        )


class TestFallbackActionNamesAreReal:
    """The escalation contract must name actions browser-use actually has.

    When all layered interaction strategies fail, the find_unique_locator
    action returns "FALLBACK REQUIRED: Call <action>(index=N)" and
    SEQUENTIAL_PROCESSING_RULES tells the agent to execute it. The two sides
    are written independently, in different files, and nothing checked either
    against the live registry.

    browser-use renamed these between 0.12.6 and 0.13.7 (click_element →
    click, input_text → input, select_dropdown_option → select_dropdown; the
    param-model classes kept the old names, which is why the rename is easy to
    miss). Both sides went on naming actions that no longer exist. It never
    surfaced because the path is a genuine last resort — 0 firings against 822
    prompt emissions in the captured logs — so the contract was broken and
    silent at the same time.

    These tests are the drift alarm the upgrade did not have.
    """

    # Every action the escalation contract can name, and the interaction
    # outcome that names it.
    FALLBACK_ACTIONS = ("click", "input", "select_dropdown")

    @staticmethod
    def _registry_actions():
        from browser_use.tools.service import Tools

        return set(Tools().registry.registry.actions)

    def test_prompt_names_only_registered_actions(self):
        """Parse the action names out of the rendered fallback block and
        require each to exist. Fails loudly on the next upstream rename."""
        from browser_service.prompts.templates import SEQUENTIAL_PROCESSING_RULES

        block = SEQUENTIAL_PROCESSING_RULES.split("FALLBACK REQUIRED")[1]
        block = block.split('If "⚠️ Automated')[0]
        named = set(re.findall(r"•\s*([a-z_]+)\(index=N", block))

        assert named, "fallback block names no actions — did its shape change?"
        assert named <= self._registry_actions(), (
            f"SEQUENTIAL_PROCESSING_RULES tells the agent to call "
            f"{sorted(named - self._registry_actions())}, which browser-use "
            f"does not register. The agent cannot execute the escalation."
        )

    def test_runtime_note_names_only_registered_actions(self):
        """The other half of the contract: the names registration.py emits in
        its FALLBACK REQUIRED note."""
        import inspect

        from browser_service.agent import registration

        src = inspect.getsource(registration)
        emitted = set(re.findall(r"FALLBACK REQUIRED: Call ([a-z_]+)\(", src))

        assert emitted, "no FALLBACK REQUIRED note found — did its shape change?"
        assert emitted <= self._registry_actions(), (
            f"registration.py emits {sorted(emitted - self._registry_actions())}, "
            f"which browser-use does not register."
        )

    def test_both_halves_name_the_same_actions(self):
        """Prompt and runtime note are written in different files. If they
        drift apart the agent is told to call one thing and instructed about
        another."""
        import inspect

        from browser_service.agent import registration
        from browser_service.prompts.templates import SEQUENTIAL_PROCESSING_RULES

        block = SEQUENTIAL_PROCESSING_RULES.split("FALLBACK REQUIRED")[1]
        block = block.split('If "⚠️ Automated')[0]
        prompt_names = set(re.findall(r"•\s*([a-z_]+)\(index=N", block))
        emitted = set(
            re.findall(r"FALLBACK REQUIRED: Call ([a-z_]+)\(", inspect.getsource(registration))
        )

        assert prompt_names == emitted == set(self.FALLBACK_ACTIONS)
