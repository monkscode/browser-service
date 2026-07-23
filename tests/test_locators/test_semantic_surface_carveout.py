"""
q05 guard (i) — empty-surface interactive carve-out for the agent-candidate
semantic check (ASTPP gate q05, root-caused 2026-07-16).

The recorded failure (1 of 3 corrupt-expected_text q05 runs): the vision
agent attached expected_text="Email *" (the LABEL's text) to the Email
INPUT. The candidate ``input[name='email']`` was unique and valid, but
``validate_semantic_match``'s locator path vetoed it: the input's entire
semantic surface is empty (no placeholder, no aria, no value, and the
label association is BROKEN — ASTPP's <label for="Email"> points at an id
the input does not have, live-verified 2026-07-17). An element with no
text surface cannot contradict any expected_text; vetoing it re-enacts
the return-something-over-fail family this engine keeps removing.

Fix: opt-in ``accept_empty_interactive`` — mirrors the probe-18 node-path
carve-out (interactive tag + empty surface -> accept on uniqueness),
passed ONLY by the agent-candidate check in actions.py. All other
validate_semantic_match call sites keep today's behavior.

Same conventions as test_form_control_semantics.py: real headless
Chromium against a static fixture, marked integration.
"""

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.smart_locator import validate_semantic_match

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent / "locator_fixtures"


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page_obj = await ctx.new_page()
        await page_obj.goto((FIXTURES_DIR / "semantic_surface_carveout.html").resolve().as_uri())
        try:
            yield page_obj
        finally:
            await browser.close()


async def test_default_still_vetoes_empty_surface_input(page):
    """Regression pin: without the opt-in, the q05 veto stands (the five
    other validate_semantic_match call sites keep today's behavior)."""
    ok, _ = await validate_semantic_match(None, "Email *", page=page, locator="input[name='email']")
    assert ok is False


async def test_carveout_accepts_empty_surface_interactive(page):
    """The q05 case: unique input, empty surface, label-text
    expected_text -> accept on uniqueness when opted in."""
    ok, _ = await validate_semantic_match(
        None,
        "Email *",
        page=page,
        locator="input[name='email']",
        accept_empty_interactive=True,
    )
    assert ok is True


async def test_carveout_does_not_bypass_real_mismatch(page):
    """A surface-bearing input that genuinely disagrees is still vetoed
    with the flag on — the carve-out is for surface-LESS elements only."""
    ok, _ = await validate_semantic_match(
        None,
        "Email *",
        page=page,
        locator="input[name='username']",
        accept_empty_interactive=True,
    )
    assert ok is False


async def test_carveout_ignores_non_interactive_nodes(page):
    """A surface-less div has no interactive affordance — never carved
    out (mirrors the node-path rule exactly)."""
    ok, _ = await validate_semantic_match(
        None,
        "Email *",
        page=page,
        locator="#empty-div",
        accept_empty_interactive=True,
    )
    assert ok is False
