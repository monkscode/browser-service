"""
Workflow task processing module.

This module handles the execution of workflow tasks, including:
- Browser session management
- Agent initialization and execution
- Locator extraction and validation
- Result processing and metrics tracking
"""

import asyncio
import json
import logging
import re
import statistics
import time

import structlog

logger = logging.getLogger(__name__)
from typing import Any, Dict, List, Optional

from browser_service.locators.stability import (
    STABLE,
    score_stability,
    stability_rank,
)


def rerank_sort_key(loc: dict) -> tuple:
    """
    PHASE-2 re-rank order (E1): stability tier first, quality score second.

    A volatile id scores 100 on TYPE alone, so score-only sorting would
    re-promote it above the stable name= the locator engine deliberately
    chose — the re-ranker must read the same stability verdict as the
    engine's candidate ordering or one silently undoes the other.
    """
    return (
        stability_rank(loc.get("stability", STABLE)),
        -loc.get("quality_score", 0),
    )


def commit_reranked_winner(result: dict, scored_locators: list[dict]) -> None:
    """Point the top-level result fields at the reranked winner.

    ``best_locator``, ``stability`` and ``all_locators`` describe the SAME
    chosen locator and must move together. Updating ``best_locator`` while
    leaving a prior winner's ``stability`` in place mislabels a
    volatile/positional locator with the old tier (E1) — the emitted payload
    then disagrees with ``all_locators[0]``. Assigning all three here makes
    that drift unrepresentable.
    """
    winner = scored_locators[0]
    result["best_locator"] = winner["locator"]
    result["stability"] = winner.get("stability", STABLE)
    result["all_locators"] = scored_locators


def bind_request_context(workflow_id=None, org_id=None, user_id=None) -> None:
    """Bind the FastAPI-supplied correlation id + org/user into structlog so every
    browser_use.log line for this job carries the same id as the FastAPI side.

    Clears first: workers run on a reused ThreadPoolExecutor thread, and Python does
    NOT reset contextvars between submitted tasks, so without this a task missing
    org_id/user_id would inherit the previous task's identity (cross-request leak).
    Mirrors bind_workflow_context() on the FastAPI side."""
    structlog.contextvars.clear_contextvars()
    ctx = {}
    if workflow_id:
        ctx["workflow_id"] = workflow_id
    if org_id:
        ctx["org_id"] = org_id
    if user_id:
        ctx["user_id"] = user_id
    if ctx:
        structlog.contextvars.bind_contextvars(**ctx)


def _resolve_use_vision(vision_mode: str, custom_actions_enabled: bool):
    """Map config.agent_vision_mode to the browser-use Agent use_vision argument.

    'on' → full vision on every step (escape hatch); 'auto' → vision-off with
    failure-triggered escalation (registration.py attaches the screenshot to
    the next LLM call via ActionResult metadata when find_unique_locator fails
    validation). The escalation hook lives in the custom action, so the legacy
    JS workflow (custom actions disabled) has no failure trigger — 'auto' there
    would mean permanently blind; legacy always gets full vision."""
    if not custom_actions_enabled:
        return True
    return True if vision_mode == "on" else "auto"


def _apply_vision_mode(agent, use_vision) -> None:
    """In 'auto' mode, drop the model-facing screenshot action from the schema.

    browser-use auto-excludes it only when use_vision is not 'auto'
    (agent/service.py:314-320), so 'auto' would otherwise pay the action's
    schema tokens every call. The failure-triggered attachment is
    registry-independent — message_manager reads ActionResult.metadata only —
    so escalation keeps working with the action excluded (verified live,
    forced-failure replay 2026-07-17)."""
    if use_vision == "auto":
        agent.tools.exclude_action("screenshot")


# Import browser-use components
from browser_use import Agent

from browser_service.agent import register_custom_actions
from browser_service.browser import capture_session_pid, cleanup_browser_resources

# Import local modules
from browser_service.config import config
from browser_service.prompts import build_system_prompt, build_workflow_prompt
from browser_service.utils.json_parser import extract_json_for_element
from browser_service.utils.metrics import record_workflow_metrics

try:
    from src.backend.core.config import settings as _nl_settings
except ImportError:
    _nl_settings = None
try:
    from clients import get_client_config
except ImportError:
    # Standalone browser-service mode
    class _MockClientConfig:
        name = "Default"
        minimum_wait_page_load_time = 0.5
        wait_for_network_idle_page_load_time = 1.0
        wait_between_actions = 0.5
        system_prompt_additions = []

    def get_client_config(url: str):
        return _MockClientConfig()
# JSON EXTRACTION HELPERS (Module Level)
# ========================================
# These functions are used to extract element JSON data from agent results.
# Defined at module level for testability and to avoid redefinition on each call.


def _extract_from_result_lines(text: str) -> List[str]:
    """
    Extract JSON from 'Result:' lines printed by browser_use.

    This is the MOST RELIABLE method because:
    1. Always printed by browser_use after JS execution
    2. Contains complete, valid JSON
    3. Has best_locator already selected (first unique locator)
    4. No double-escaping issues

    Args:
        text: Full text output from the agent

    Returns:
        List of JSON strings extracted from Result: lines
    """
    results = []
    # Find all Result: lines
    lines = text.split("\n")
    for line in lines:
        if "Result:" in line and "element_id" in line:
            # Extract everything after "Result:"
            result_start = line.find("Result:")
            if result_start != -1:
                json_part = line[result_start + 7 :].strip()

                # Find complete JSON using brace matching
                if json_part.startswith("{"):
                    brace_count = 0
                    in_string = False
                    escape_next = False

                    for i, char in enumerate(json_part):
                        if escape_next:
                            escape_next = False
                            continue
                        if char == "\\":
                            escape_next = True
                            continue
                        if char == '"':
                            in_string = not in_string
                            continue
                        if in_string:
                            continue

                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                json_str = json_part[: i + 1]
                                results.append(json_str)
                                break

    return results


def summarize_dom_samples(samples: list[int]) -> tuple[int, int]:
    """(max, median) indexed-element counts across agent steps.

    Prices cross_origin_iframes=True, which indexes 2000+ elements on some pages
    and whose token cost has never been measured.
    """
    if not samples:
        return (0, 0)
    return (max(samples), int(statistics.median(samples)))


def instrument_llm_timing(llm) -> list[float]:
    """Wrap llm.ainvoke to record each model call's wall clock.

    Returns the list it appends to — one entry per call, retries included.
    That list is per-instance and closure-local by construction: the service
    runs up to config.max_concurrent_tasks workflows in a ThreadPoolExecutor,
    each with its own llm_instance and Agent. Module-level state would
    cross-contaminate them.

    Call this BEFORE Agent(...) is constructed. Agent.__init__ hands the instance
    to register_llm (browser_use/agent/service.py:419), which captures whatever
    ainvoke is at that moment — so browser-use's token bookkeeping nests OUTSIDE
    this timer and is excluded from what we measure.

    Mirrors browser-use's own wrapping pattern, signature included
    (browser_use/tokens/service.py:345-371), so positional and keyword callers
    both keep working.

    Records in a finally and catches nothing: a timed-out or cancelled call still
    burned wall clock, and swallowing the exception would change agent behaviour.
    """
    durations: list[float] = []
    original = llm.ainvoke

    async def timed_ainvoke(messages, output_format=None, **kwargs):
        started = time.perf_counter()
        try:
            return await original(messages, output_format, **kwargs)
        finally:
            durations.append(time.perf_counter() - started)

    setattr(llm, "ainvoke", timed_ainvoke)
    return durations


def extract_agent_diagnostics(
    agent_result, dom_samples: list[int], llm_durations: list[float] | None = None
) -> dict:
    """DOM size, 429 cost, and the LLM / non-LLM split of agent_run_s.

    Never raises: instrumentation must not break the pipeline it measures. A
    malformed history yields zeros, which read as "not measured" downstream.

    429s are counted here rather than scraped from logs/browser_use.log, which
    accumulates across days and makes a raw count meaningless.

    Raw measurements only. step_non_llm_s (= steps_total_s - llm_total_s) and
    agent_overhead_s (= agent_run_s - steps_total_s) are derived at analysis
    time, so a wrong relationship shows up instead of being baked in.

    llm_coverage_gap is the instrument checking itself: entry_count counts
    tracked calls across every registered LLM instance, ours counts only the one
    we wrapped. Positive means calls happened somewhere we cannot see.
    """
    dom_max, dom_median = summarize_dom_samples(dom_samples)
    # `is not None`, never truthiness: an empty list means "wrapped, zero calls",
    # which is a measurement, not a missing one.
    calls = llm_durations if llm_durations is not None else []
    diagnostics = {
        "dom_elements_max": dom_max,
        "dom_elements_median": dom_median,
        "llm_429_count": 0,
        "retry_lost_s": 0.0,
        "llm_total_s": round(sum(calls), 3),
        "llm_max_s": round(max(calls, default=0.0), 3),
        "llm_calls_actual": len(calls),
        "steps_total_s": 0.0,
        # None, not 0: "we could not check coverage" must never read as
        # "coverage was perfect". An empty CSV cell is the alarm.
        "llm_coverage_gap": None,
    }
    if agent_result is None:
        return diagnostics
    try:
        count = 0
        lost = 0.0
        steps_total = 0.0
        for item in getattr(agent_result, "history", []) or []:
            meta = getattr(item, "metadata", None)
            duration = float(getattr(meta, "duration_seconds", 0.0) or 0.0)
            steps_total += duration
            errors = [
                r.error for r in (getattr(item, "result", None) or [])
                if getattr(r, "error", None)
            ]
            if any("RESOURCE_EXHAUSTED" in str(e) for e in errors):
                count += 1
                lost += duration
        diagnostics["llm_429_count"] = count
        diagnostics["retry_lost_s"] = round(lost, 3)
        diagnostics["steps_total_s"] = round(steps_total, 3)
        entry_count = getattr(getattr(agent_result, "usage", None), "entry_count", None)
        if entry_count is not None and llm_durations is not None:
            diagnostics["llm_coverage_gap"] = int(entry_count) - len(calls)
    except Exception as e:  # never break the pipeline for a metric
        logger.warning(f"Agent diagnostics extraction failed (suppressed): {e}")
    return diagnostics


def _extract_all_element_jsons(text: str) -> List[str]:
    """
    Extract all JSON objects containing element_id from text.

    This is a FALLBACK method when Result: lines are not available.

    Args:
        text: Full text to search for JSON objects

    Returns:
        List of unique JSON strings containing element_id
    """
    found_jsons = []
    # Look for {"element_id": patterns
    for pattern in ['"element_id":', "'element_id':"]:
        pos = 0
        while True:
            pos = text.find(pattern, pos)
            if pos == -1:
                break

            # Find the opening brace
            brace_pos = text.rfind("{", max(0, pos - 50), pos + 20)
            if brace_pos == -1:
                pos += 1
                continue

            # Match braces to find complete JSON
            brace_count = 0
            in_string = False
            escape_next = False

            for i in range(brace_pos, min(len(text), brace_pos + 10000)):
                char = text[i]

                if escape_next:
                    escape_next = False
                    continue
                if char == "\\":
                    escape_next = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue

                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = text[brace_pos : i + 1]
                        if json_str not in found_jsons:
                            found_jsons.append(json_str)
                        break

            pos += 1

    return found_jsons


def process_workflow_task(
    task_id: str,
    elements: List[Dict[str, Any]],
    url: str,
    user_query: str,
    session_config: Dict[str, Any],
    enable_custom_actions: Optional[bool] = None,
    task_processor=None,  # TaskProcessor — used to update task status atomically
    parent_workflow_id: Optional[str] = None,
    org_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    """
    Process elements as a UNIFIED WORKFLOW in a single browser session.

    This is the primary processing function for ALL tasks. Instead of creating separate
    Agent instances for each element, this creates ONE Agent that performs the entire
    workflow: navigate → act → extract all locators in sequence.

    Benefits:
    - Single Agent session (optimal cost)
    - Context preserved across all actions
    - No "empty page" or navigation issues
    - Agent understands the complete workflow
    - Matches user intent naturally

    Args:
        task_id: Unique task identifier
        elements: List of element specs [{"id": "elem_1", "description": "...", "action": "..."}]
        url: Target URL
        user_query: Full user query for context (e.g., "search for shoes and get product name")
        session_config: Browser configuration
        enable_custom_actions: Optional flag to enable/disable custom actions (defaults to config value)
    """
    bind_request_context(workflow_id=parent_workflow_id or task_id, org_id=org_id, user_id=user_id)

    if task_processor is None:
        raise ValueError("task_processor parameter is required for task tracking")

    task_processor.update_task(
        task_id,
        {
            "status": "running",
            "started_at": time.time(),
            "message": f"Processing {len(elements)} elements as unified workflow",
        },
    )

    # Time this task spent queued before a worker picked it up. update_task
    # merges, so created_at (processor.py:88/:131) survives the started_at write.
    _task_row = task_processor.get_task_status(task_id) or {}
    queue_s = max(0.0, time.time() - float(_task_row.get("created_at", time.time())))

    logger.info(f"🚀 Starting WORKFLOW MODE for task {task_id}")
    logger.info(f"   Elements: {len(elements)}")
    logger.info(f"   URL: {url}")
    logger.info(f"   Query: {user_query[:100]}...")

    # Capture enable_custom_actions parameter for use in async function
    # Default to config value if not provided
    if enable_custom_actions is None:
        if _nl_settings is not None and hasattr(_nl_settings, "ENABLE_CUSTOM_ACTIONS"):
            enable_custom_actions_flag = _nl_settings.ENABLE_CUSTOM_ACTIONS
        else:
            enable_custom_actions_flag = config.enable_custom_actions
        logger.info(f"🔧 Using ENABLE_CUSTOM_ACTIONS from config: {enable_custom_actions_flag}")
    else:
        enable_custom_actions_flag = enable_custom_actions
        logger.info(
            f"🔧 Using ENABLE_CUSTOM_ACTIONS from API parameter: {enable_custom_actions_flag}"
        )

    # Browser handles for post-completion cleanup (Task 19 / TIER-0 0.2).
    # run_unified_workflow() deposits whatever resources it created here; the
    # sync wrapper below runs cleanup_browser_resources() with them AFTER the
    # task is marked completed, so the polling client never waits through
    # cleanup (median 11.7s/generation on the 2026-07-06 bench baseline).
    cleanup_handles: Dict[str, Any] = {}

    async def run_unified_workflow():
        """Execute the entire workflow in ONE Agent session."""
        from browser_use.browser.session import BrowserSession
        from browser_use.llm.google import ChatGoogle

        session = None
        connected_browser = None
        playwright_instance = None
        browser_pid = None  # Captured after session.start() for orphan cleanup

        try:
            # Initialize browser session ONCE
            _t_session = time.perf_counter()
            logger.info("🌐 Initializing browser session...")

            # CRITICAL: Set explicit viewport for consistent coordinates
            # browser-use in headful mode (headless=False) defaults to no_viewport=True
            # which makes content fit to window, causing coordinate misalignment.
            # Setting explicit viewport ensures:
            # 1. Vision AI sees page at this exact resolution
            # 2. Coordinates from vision AI match our Playwright validation
            # 3. document.elementFromPoint(x, y) returns the same element vision AI identified
            VIEWPORT_WIDTH = 1920
            VIEWPORT_HEIGHT = 1080

            # ========================================
            # CLIENT-SPECIFIC CONFIGURATION
            # ========================================
            # Get client-specific timing and prompt hints based on URL
            client_config = get_client_config(url)
            logger.info(f"📋 Client config: {client_config.name}")
            if client_config.name != "Default":
                logger.info(
                    f"   Timing: wait_page_load={client_config.minimum_wait_page_load_time}s, "
                    f"network_idle={client_config.wait_for_network_idle_page_load_time}s, "
                    f"wait_between_actions={client_config.wait_between_actions}s"
                )
                if client_config.system_prompt_additions:
                    logger.info(
                        f"   Prompt hints: {len(client_config.system_prompt_additions)} application-specific hints"
                    )

            session = BrowserSession(
                headless=session_config.get("headless", config.headless),
                viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
                no_viewport=False,  # CRITICAL: Force browser-use to respect viewport (default is True when headless=False)
                minimum_wait_page_load_time=client_config.minimum_wait_page_load_time,
                wait_for_network_idle_page_load_time=client_config.wait_for_network_idle_page_load_time,
                wait_between_actions=client_config.wait_between_actions,
                # CRITICAL: Enable iframe content crawling for proper locator extraction
                # Without this, get_browser_state_summary() only returns main page elements (~86)
                # With this, iframe content is indexed in selector_map (2000+ elements)
                cross_origin_iframes=True,
                # Prevent agent from navigating to ad/analytics/tracker domains during element detection
                prohibited_domains=[
                    "ads.google.com",
                    "analytics.google.com",
                    "googletagmanager.com",
                    "doubleclick.net",
                    "facebook.net",
                    "connect.facebook.net",
                ],
            )
            logger.info(
                f"📐 Viewport set to {VIEWPORT_WIDTH}x{VIEWPORT_HEIGHT} for coordinate consistency (no_viewport=False)"
            )

            # browser-use requires explicit start() call
            logger.info("🚀 Starting browser session...")
            await session.start()
            logger.info("✅ Browser session started successfully")
            session_setup_s = time.perf_counter() - _t_session
            _t_agent_setup = time.perf_counter()

            # Capture browser PID immediately — Chrome is guaranteed alive here.
            # Must happen BEFORE agent.run() because browser-use fires
            # on_BrowserStopEvent (killing the parent Chrome) before agent.run()
            # returns, making session.cdp_url unavailable at cleanup time.
            browser_pid = capture_session_pid(session)
            if browser_pid:
                logger.info(f"📍 Captured browser PID {browser_pid} for orphan cleanup")
            else:
                logger.warning("⚠️ Could not capture browser PID — orphan cleanup will be skipped")

            # Calculate dynamic max_steps based on workflow complexity
            # Formula: navigate(1) + process_elements(elements * 3 for find+action+wait) + done(1) + buffer(8)
            # Buffer increased from 5 to 8 for browser-use 0.12.0 agent planning overhead (2-3 steps)
            dynamic_max_steps = 1 + (len(elements) * 3) + 1 + 8
            logger.info(f"📊 Dynamic max_steps: {dynamic_max_steps} (for {len(elements)} elements)")

            # NOTE: Element sequencing is handled by the AI agent based on:
            # 1. The order elements are received (from Step Planner)
            # 2. The 'action' field on each element (input/click/submit vs get_text/get_attribute)
            # 3. Prompt instructions that guide the AI to perform actions before reading results
            # No manual categorization needed - the AI understands workflow semantics

            # ========================================
            # NEW APPROACH: Use Playwright's Built-in Methods
            # ========================================
            # Instead of embedding 2000+ lines of JavaScript in the prompt (causing LLM timeout),
            # we use a simplified prompt where the agent only finds elements and returns coordinates.
            # Then Python uses Playwright's built-in methods for locator extraction and F12-style validation.

            logger.info("🔧 Using Browser Library with Playwright validation")

            # ========================================
            # FEATURE FLAG: ENABLE_CUSTOM_ACTIONS
            # ========================================
            if enable_custom_actions_flag:
                logger.info("🔧 Custom actions ENABLED - Smart locator strategy")
            else:
                logger.info("🔧 Custom actions DISABLED - Legacy JavaScript validation")

            # Build workflow prompt based on feature flag
            unified_objective = build_workflow_prompt(
                user_query=user_query,
                url=url,
                elements=elements,
                include_custom_action=enable_custom_actions_flag,
                client_hints=client_config.system_prompt_additions,
            )
            logger.info(f"📝 Built workflow prompt for {len(elements)} elements")

            # Create Agent with prompts based on feature flag
            # NOTE: browser-use 0.12.0 moved max_steps to agent.run() and renamed
            # system_prompt to override_system_message
            #
            # NOTE: We do NOT manually set session._original_viewport_size here.

            # Build LLM instance based on provider.
            # Credentials for Vertex AI are pre-loaded in BrowserServiceConfig.__init__()
            # and cached on config.llm.vertexai_credentials — no file I/O here.
            if config.llm.model_provider == "vertex":
                llm_instance = ChatGoogle(
                    model=config.llm.google_model,
                    vertexai=True,
                    credentials=config.llm.vertexai_credentials,
                    project=config.llm.vertexai_project,
                    location=config.llm.vertexai_location,
                    temperature=0.1,
                    thinking_budget=0,
                )
                logger.info(
                    f"🔑 Using Vertex AI: project={config.llm.vertexai_project}, "
                    f"location={config.llm.vertexai_location}, model={config.llm.google_model}"
                )
            elif config.llm.model_provider == "local":
                # browser-service does not support local/Ollama — validate() already reports
                # this as an error at startup. Raise explicitly if execution somehow reaches here.
                raise RuntimeError(
                    "MODEL_PROVIDER=local is not supported by browser-service. "
                    "Browser-service requires a Google vision model (gemini or vertex)."
                )
            else:
                # "gemini" — Gemini Developer API
                llm_instance = ChatGoogle(
                    model=config.llm.google_model,
                    api_key=config.llm.google_api_key,
                    temperature=0.1,
                    thinking_budget=0,
                )
                logger.info(f"🔑 Using Gemini API: model={config.llm.google_model}")

            # Wrap BEFORE Agent(...): Agent.__init__ hands this instance to
            # register_llm (agent/service.py:419), which captures whatever
            # ainvoke is then — so token accounting nests outside this timer.
            # One wrap covers every path: page_extraction_llm defaults to the
            # same object (service.py:249-250), compaction_llm falls back to it
            # (service.py:1156), and judge_llm is never invoked (use_judge=False).
            llm_durations = instrument_llm_timing(llm_instance)

            use_vision = _resolve_use_vision(config.agent_vision_mode, enable_custom_actions_flag)
            agent = Agent(
                task=unified_objective,
                browser_session=session,
                llm=llm_instance,
                use_vision=use_vision,
                override_system_message=build_system_prompt(
                    include_custom_action=enable_custom_actions_flag
                ),
                use_thinking=False,
                calculate_cost=True,
                use_judge=False,
            )
            _apply_vision_mode(agent, use_vision)
            logger.info(
                f"👁️ Agent vision mode: {config.agent_vision_mode} (use_vision={use_vision!r})"
            )

            session.llm_screenshot_size = (VIEWPORT_WIDTH, VIEWPORT_HEIGHT)
            logger.info(
                f"LLM screenshot size: {VIEWPORT_WIDTH}x{VIEWPORT_HEIGHT} (matches viewport)"
            )

            # ========================================
            # REGISTER CUSTOM ACTIONS (if enabled)
            # ========================================
            # Register custom actions with the agent after creation.
            # The custom action will get the page object from browser_session during execution,
            # ensuring we use the SAME browser that's already open (no new browser instance needed).
            # This is the key strategy: validate locators using the existing browser_use browser.
            custom_actions_enabled = False

            if enable_custom_actions_flag:
                logger.info("🔧 Attempting to register custom actions...")
                # Pass None for page since the custom action will get it from browser_session during execution.
                # Pass elements so the action handler can set is_done=True automatically when all
                # expected element IDs have been processed, terminating the agent loop without
                # relying on the LLM to call done() with the correct JSON format.
                custom_actions_enabled = register_custom_actions(
                    agent, page=None, elements=elements
                )

                if custom_actions_enabled:
                    logger.info("✅ Custom actions registered successfully")
                    logger.info("   Agent can now call find_unique_locator action")
                    logger.info(
                        "   Custom action will use the existing browser_use browser for validation"
                    )
                    logger.info("   Using smart locator strategy (custom action mode)")
                else:
                    logger.warning("⚠️ Custom action registration failed")
                    logger.warning("   Falling back to legacy workflow (JavaScript validation)")

                    # Fall back to legacy mode
                    unified_objective = build_workflow_prompt(
                        user_query=user_query,
                        url=url,
                        elements=elements,
                        include_custom_action=False,  # Fallback to legacy mode
                        client_hints=client_config.system_prompt_additions,
                    )
                    agent.task = unified_objective
                    agent.override_system_message = build_system_prompt(include_custom_action=False)
                    # The escalation hook died with the failed registration —
                    # 'auto' would leave the legacy agent permanently blind.
                    # Mutating settings post-construction is browser-use's own
                    # pattern (agent/service.py DeepSeek/XAI handling).
                    agent.settings.use_vision = True
                    logger.info(
                        "✅ Agent prompts updated with legacy workflow instructions (full vision restored)"
                    )
            else:
                logger.info("⏭️ Skipping custom action registration (disabled via config)")
                logger.info("   Using legacy workflow mode")

            # Run the unified workflow
            logger.info("🤖 Starting unified Agent...")

            # Log available actions for debugging
            if hasattr(agent, "tools") and agent.tools:
                if hasattr(agent.tools, "registry") and hasattr(agent.tools.registry, "registry"):
                    action_registry = agent.tools.registry.registry
                    if hasattr(action_registry, "actions"):
                        available_actions = list(action_registry.actions.keys())
                        logger.info(f"📋 Available custom actions: {available_actions}")
                    else:
                        logger.info("📋 Tools registry structure unknown")
                else:
                    logger.info("📋 Tools registry structure unknown")
            else:
                logger.info("⚠️ Agent has no tools registered")

            agent_setup_s = time.perf_counter() - _t_agent_setup

            dom_samples: list[int] = []

            async def _on_step_end(agent_obj):
                # get_selector_map() is a cache read (session.py:2569-2583) —
                # no CDP call, so this cannot inflate agent_run_s.
                try:
                    dom_samples.append(len(await session.get_selector_map()))
                except Exception:
                    pass  # a metric must never fail a step

            start_time = time.time()
            try:
                agent_result = await agent.run(
                    max_steps=dynamic_max_steps, on_step_end=_on_step_end
                )
            finally:
                execution_time = time.time() - start_time
                teardown = getattr(agent, "_pw_teardown", None)
                if teardown:
                    try:
                        await teardown()
                    except Exception as e:
                        logger.warning(f"Playwright teardown error (suppressed): {e}")

            _t_postprocess = time.perf_counter()

            logger.info(f"✅ Agent completed in {execution_time:.1f}s")

            # ========================================
            # TOKEN USAGE EXTRACTION from browser-use
            # ========================================
            # Extract actual token usage from AgentHistoryList.usage (UsageSummary)
            token_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
                "actual_cost": 0.0,
            }

            # Debug: Log agent_result structure
            logger.info(f"🔍 DEBUG: agent_result type = {type(agent_result)}")
            logger.info(
                f"🔍 DEBUG: hasattr(agent_result, 'usage') = {hasattr(agent_result, 'usage')}"
            )

            if hasattr(agent_result, "usage") and agent_result.usage:
                usage = agent_result.usage
                logger.info(f"🔍 DEBUG: usage type = {type(usage)}")

                # Try to dump the usage object for full visibility
                try:
                    if hasattr(usage, "model_dump"):
                        logger.info(f"🔍 DEBUG: usage.model_dump() = {usage.model_dump()}")
                    elif hasattr(usage, "__dict__"):
                        logger.info(f"🔍 DEBUG: usage.__dict__ = {usage.__dict__}")
                except Exception as e:
                    logger.warning(f"🔍 DEBUG: Could not dump usage object: {e}")

                # Extract token counts from UsageSummary
                token_usage = {
                    "input_tokens": getattr(usage, "total_prompt_tokens", 0) or 0,
                    "output_tokens": getattr(usage, "total_completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                    "cached_tokens": getattr(usage, "total_prompt_cached_tokens", 0) or 0,
                    "actual_cost": getattr(usage, "total_cost", 0.0) or 0.0,
                }

                logger.info("📊 ACTUAL TOKEN USAGE from browser-use:")
                logger.info(f"   Input tokens (prompt): {token_usage['input_tokens']}")
                logger.info(f"   Output tokens (completion): {token_usage['output_tokens']}")
                logger.info(f"   Total tokens: {token_usage['total_tokens']}")
                logger.info(f"   Cached tokens: {token_usage['cached_tokens']}")
                logger.info(f"   Actual cost: ${token_usage['actual_cost']:.6f}")
            else:
                logger.warning("⚠️ No token usage data available from agent_result.usage")

                # Fallback: Try agent.token_cost_service if available
                if hasattr(agent, "token_cost_service"):
                    logger.info("🔍 DEBUG: Trying fallback via agent.token_cost_service...")
                    try:
                        usage_summary = await agent.token_cost_service.get_usage_summary()
                        if usage_summary:
                            logger.info(f"🔍 DEBUG: Fallback usage_summary = {usage_summary}")
                            token_usage = {
                                "input_tokens": getattr(usage_summary, "total_prompt_tokens", 0)
                                or 0,
                                "output_tokens": getattr(
                                    usage_summary, "total_completion_tokens", 0
                                )
                                or 0,
                                "total_tokens": getattr(usage_summary, "total_tokens", 0) or 0,
                                "cached_tokens": getattr(
                                    usage_summary, "total_prompt_cached_tokens", 0
                                )
                                or 0,
                                "actual_cost": getattr(usage_summary, "total_cost", 0.0) or 0.0,
                            }
                            logger.info("📊 TOKEN USAGE from fallback (token_cost_service):")
                            logger.info(f"   Total tokens: {token_usage['total_tokens']}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not get usage from token_cost_service: {e}")

            # Check if agent actually used the custom action
            if custom_actions_enabled and hasattr(agent_result, "history"):
                custom_action_calls = 0
                execute_js_calls = 0
                for step in agent_result.history:
                    # find_unique_locator is a registered custom action — it does NOT appear in
                    # model_fields_set, which only covers native browser-use Pydantic actions.
                    # Count it from ActionResult.metadata, which the action handler populates
                    # on every successful call (failed calls have no metadata).
                    if hasattr(step, "result") and step.result:
                        for action_result in step.result:
                            if (
                                hasattr(action_result, "metadata")
                                and isinstance(action_result.metadata, dict)
                                and action_result.metadata.get("element_id")
                                and action_result.metadata.get("found")
                                and action_result.metadata.get("best_locator")
                            ):
                                custom_action_calls += 1
                    # execute_js IS a native browser-use action with a Pydantic model, so
                    # model_fields_set correctly reflects its usage.
                    if step.model_output and step.model_output.action:
                        for action_model in step.model_output.action:
                            if "execute_js" in action_model.model_fields_set:
                                execute_js_calls += 1

                logger.info(
                    f"📊 Action usage: find_unique_locator={custom_action_calls}, execute_js={execute_js_calls}"
                )

                if custom_action_calls == 0 and execute_js_calls > 0:
                    logger.warning(
                        "⚠️ Agent used execute_js instead of find_unique_locator custom action!"
                    )
                    logger.warning(
                        "   This may indicate the custom action wasn't properly registered or visible to the agent"
                    )
                elif custom_action_calls > 0:
                    logger.info(
                        f"✅ Agent successfully used find_unique_locator custom action {custom_action_calls} times"
                    )

            # ========================================
            # METRICS LOGGING: LLM Call Count
            # ========================================
            # Track LLM calls from agent history for cost tracking
            # Phase: Error Handling and Logging | Requirements: 6.1, 6.2, 6.3, 9.5
            llm_call_count = 0
            if hasattr(agent_result, "history") and agent_result.history:
                # Count steps that involved LLM calls (agent actions)
                llm_call_count = len(agent_result.history)
                logger.info(f"📊 METRIC: Total LLM calls in workflow: {llm_call_count}")

            # Log custom action usage
            logger.info(f"📊 METRIC: Custom actions enabled: {custom_actions_enabled}")
            logger.info(f"📊 METRIC: Workflow execution time: {execution_time:.2f}s")

            # ========================================
            # EXTRACT RESULTS FROM CUSTOM ACTION METADATA (IF ENABLED)
            # ========================================
            # When custom actions are enabled, results are stored directly in ActionResult.metadata
            # This is the FASTEST path - no coordinate parsing, no Playwright extraction needed
            results_list = []
            # element_id -> the payload that was rejected below. Kept out of
            # results_list so a later successful call for the same element still
            # wins; read by the backfill to report WHY a locator is missing.
            rejected_payloads: dict = {}

            if custom_actions_enabled and hasattr(agent_result, "history") and agent_result.history:
                logger.info("🎯 Extracting results from custom action metadata (primary path)...")

                for idx, step in enumerate(agent_result.history):
                    if hasattr(step, "result") and step.result:
                        if isinstance(step.result, list):
                            for action_result in step.result:
                                if hasattr(action_result, "metadata") and isinstance(
                                    action_result.metadata, dict
                                ):
                                    metadata = action_result.metadata
                                    elem_id = metadata.get("element_id")

                                    # Check if this is a custom action result with locator data
                                    if (
                                        elem_id
                                        and metadata.get("found")
                                        and metadata.get("best_locator")
                                    ):
                                        logger.info(
                                            f"   ✅ Found custom action result for {elem_id}: {metadata.get('best_locator')}"
                                        )

                                        # Check if we already have this element
                                        existing_idx = next(
                                            (
                                                i
                                                for i, r in enumerate(results_list)
                                                if r.get("element_id") == elem_id
                                            ),
                                            None,
                                        )

                                        if existing_idx is None:
                                            # First occurrence, add it
                                            metadata["metrics"] = {"custom_action_used": True}
                                            results_list.append(metadata)
                                        else:
                                            # Already have it (agent retry/update), replace with latest
                                            logger.info(
                                                f"   🔄 Updating {elem_id} with latest result"
                                            )
                                            metadata["metrics"] = {"custom_action_used": True}
                                            results_list[existing_idx] = metadata
                                    elif elem_id:
                                        # Not found, or found without a locator.
                                        # Either way it carries the engine's
                                        # reason — the only copy of it.
                                        rejected_payloads[elem_id] = metadata

                if results_list:
                    logger.info(
                        f"   🎉 Extracted {len(results_list)}/{len(elements)} elements via custom action metadata"
                    )
                    logger.info(
                        "   ⏭️  Skipping coordinate parsing and Playwright extraction (not needed)"
                    )
                else:
                    logger.warning(
                        "   ⚠️  No custom action results found in metadata - will try fallback methods"
                    )

            # results_list already initialized above after custom action extraction
            workflow_completed = False

            # OPTIMIZATION: When custom actions are enabled and we already have results, skip extraction
            if custom_actions_enabled and results_list:
                logger.info(
                    f"✅ Already have {len(results_list)} results from custom actions - skipping coordinate-based extraction"
                )
                workflow_completed = len(results_list) == len(elements)
            # Continue with existing result processing if needed
            logger.info(
                f"📝 Workflow completed: {workflow_completed}, Results: {len(results_list)}"
            )

            # Skip old JavaScript-based result parsing - we've already extracted locators using Playwright
            # The old logic below is kept for backward compatibility but won't be executed
            # since results_list is already populated

            # OLD PARSING LOGIC (SKIPPED) - removed dead code that referenced undefined variables

            # If no structured results, try to extract individual element results from history
            # NOTE: When custom actions are enabled, we should already have results from the metadata extraction above
            if not results_list:
                if custom_actions_enabled:
                    logger.warning(
                        "⚠️ Custom actions enabled but no results extracted from metadata - trying fallback history parsing..."
                    )
                    logger.warning(
                        "   This shouldn't normally happen - check custom action implementation"
                    )
                else:
                    logger.warning(
                        "No structured workflow results, attempting to extract from history..."
                    )

                # APPROACH 1: Extract actual result content from agent history
                logger.info("   Approach 1: Extracting from agent history steps...")

                # Build a list of all result strings from history
                # CRITICAL: Try multiple ways to access the content to avoid double-escaping
                result_strings = []
                direct_results = []  # Store parsed results directly from tool execution

                # DEBUG: Check what attributes agent_result has
                logger.info(f"   🔍 agent_result type: {type(agent_result)}")
                logger.info(
                    f"   🔍 agent_result has all_results: {hasattr(agent_result, 'all_results')}"
                )
                logger.info(f"   🔍 agent_result has history: {hasattr(agent_result, 'history')}")

                # Strategy 1: Try all_results attribute
                if hasattr(agent_result, "all_results") and agent_result.all_results:
                    for action_result in agent_result.all_results:
                        # PRIORITY 1: Check for metadata attribute (custom actions)
                        if hasattr(action_result, "metadata") and isinstance(
                            action_result.metadata, dict
                        ):
                            if "element_id" in action_result.metadata:
                                direct_results.append(action_result.metadata)
                                continue

                        # PRIORITY 2: Try result attribute
                        if hasattr(action_result, "result") and action_result.result:
                            if isinstance(action_result.result, dict):
                                direct_results.append(action_result.result)
                                continue
                            content = str(action_result.result)
                            if content not in result_strings:
                                result_strings.append(content)

                # Strategy 2: Try history attribute (MOST IMPORTANT for execute_js results)
                if hasattr(agent_result, "history") and agent_result.history:
                    logger.info(f"   ✅ Found history with {len(agent_result.history)} items")

                    for idx, step in enumerate(agent_result.history):
                        logger.info(f"   📋 history[{idx}] type: {type(step)}")
                        logger.info(f"   📋 history[{idx}] has result: {hasattr(step, 'result')}")

                        # PRIORITY 0: Check if step.result contains ActionResult objects with metadata
                        if hasattr(step, "result") and step.result:
                            logger.info(f"   🔍 history[{idx}].result type: {type(step.result)}")

                            # step.result is a list of ActionResult objects
                            if isinstance(step.result, list):
                                logger.info(
                                    f"   🔍 history[{idx}].result is a list with {len(step.result)} items"
                                )
                                for result_idx, action_result in enumerate(step.result):
                                    logger.info(
                                        f"   🔍 history[{idx}].result[{result_idx}] type: {type(action_result)}"
                                    )
                                    logger.info(
                                        f"   🔍 history[{idx}].result[{result_idx}] has metadata: {hasattr(action_result, 'metadata')}"
                                    )

                                    if hasattr(action_result, "metadata") and isinstance(
                                        action_result.metadata, dict
                                    ):
                                        logger.info(
                                            f"   🎯 history[{idx}].result[{result_idx}].metadata is a dict! Direct access possible"
                                        )
                                        logger.info(
                                            f"   🔍 metadata keys: {list(action_result.metadata.keys())}"
                                        )
                                        if "element_id" in action_result.metadata:
                                            direct_results.append(action_result.metadata)
                                            logger.info(
                                                f"   ✅ Found element_id in history[{idx}].result[{result_idx}].metadata dict!"
                                            )

                            # If result is not a list, check if it has metadata directly
                            elif hasattr(step.result, "metadata") and isinstance(
                                step.result.metadata, dict
                            ):
                                logger.info(
                                    f"   🎯 history[{idx}].result.metadata is a dict! Direct access possible"
                                )
                                if step.result.metadata and "element_id" in step.result.metadata:
                                    logger.info(
                                        f"   🔍 metadata keys: {list(step.result.metadata.keys())}"
                                    )
                                    direct_results.append(step.result.metadata)
                                    logger.info(
                                        f"   ✅ Found element_id in history[{idx}].result.metadata dict!"
                                    )

                        # Check for tool_results in state
                        if hasattr(step, "state") and hasattr(step.state, "tool_results"):
                            for tool_result in step.state.tool_results:
                                # Check for metadata or dict
                                if hasattr(tool_result, "metadata") and isinstance(
                                    tool_result.metadata, dict
                                ):
                                    if "element_id" in tool_result.metadata:
                                        direct_results.append(tool_result.metadata)
                                        continue
                                elif isinstance(tool_result, dict) and "element_id" in tool_result:
                                    direct_results.append(tool_result)
                                    continue
                                elif hasattr(tool_result, "result") and isinstance(
                                    tool_result.result, dict
                                ):
                                    if "element_id" in tool_result.result:
                                        direct_results.append(tool_result.result)
                                        continue

                # Strategy 3: If still nothing, try converting entire agent_result to string as last resort
                if not result_strings and not direct_results:
                    result_strings.append(str(agent_result))

                # PRIORITY: If we found direct dict results, use them immediately!
                if direct_results:
                    logger.info(
                        f"   🎉 Found {len(direct_results)} direct dict results (NO PARSING NEEDED)!"
                    )
                    for direct_result in direct_results:
                        elem_id = direct_result.get("element_id")
                        # Same gate as the primary path: a found claim without a
                        # locator is not a result. Accepting one here would let
                        # it be counted as successful with best_locator None.
                        if (
                            elem_id
                            and direct_result.get("found")
                            and direct_result.get("best_locator")
                        ):
                            existing_idx = next(
                                (
                                    i
                                    for i, r in enumerate(results_list)
                                    if r.get("element_id") == elem_id
                                ),
                                None,
                            )
                            if existing_idx is not None:
                                # Replace existing result (agent retry/correction)
                                old_locator = results_list[existing_idx].get("best_locator")
                                new_locator = direct_result.get("best_locator")
                                logger.info(
                                    f"   🔄 Replacing {elem_id}: '{old_locator}' → '{new_locator}' (agent retry/correction)"
                                )
                                results_list[existing_idx] = direct_result
                            else:
                                # First occurrence, add it
                                results_list.append(direct_result)
                                logger.info(
                                    f"   ✅ Direct access: {elem_id} (best_locator: {direct_result.get('best_locator')})"
                                )
                        elif elem_id:
                            rejected_payloads[elem_id] = direct_result

                    # If we got all elements via direct access, we're completely done!
                    if len(results_list) == len(elements):
                        logger.info(
                            f"   🏆 All {len(elements)} elements extracted via DIRECT ACCESS (fastest path)!"
                        )
                        # Early exit - skip all string parsing and jump to re-ranking
                        # No need to process result_strings, extract JSON, or check history
                        logger.info(
                            "   ⏭️  Skipping all fallback extraction methods (100% success via direct access)"
                        )
                    else:
                        # Only do string parsing if we're missing elements
                        logger.info(
                            f"   ⚠️  Missing {len(elements) - len(results_list)} elements, will try string-based extraction..."
                        )

                # Combine all result strings (needed for extraction functions)
                full_result_str = "\n".join(result_strings)

                # Only process strings if we don't have all elements yet
                if len(results_list) < len(elements):
                    logger.info(
                        f"   Collected {len(result_strings)} result strings, total length: {len(full_result_str)} characters"
                    )

                    # ROBUST EXTRACTION: Leverage "Result:" pattern from browser_use library
                    # The browser_use library ALWAYS prints "Result: {json}" after JavaScript execution
                    # This is the most reliable source of locator data
                    logger.info("   🎯 Strategy: Extract from 'Result:' lines (most reliable)")

                    # STRATEGY 1: Extract from "Result:" lines (MOST RELIABLE)
                    # STRATEGY 2: Extract any JSON with element_id (FALLBACK)
                    # (Functions defined at module level: _extract_from_result_lines, _extract_all_element_jsons)

                    # Try Strategy 1 first (Result: lines)
                    result_line_jsons = _extract_from_result_lines(full_result_str)
                    if result_line_jsons:
                        logger.info(
                            f"   ✅ Extracted {len(result_line_jsons)} JSON blocks from 'Result:' lines"
                        )
                        extracted_jsons = result_line_jsons
                    else:
                        # Try Strategy 2 (any JSON with element_id)
                        extracted_jsons = _extract_all_element_jsons(full_result_str)
                        if extracted_jsons:
                            logger.info(
                                f"   ✅ Extracted {len(extracted_jsons)} JSON blocks (fallback method)"
                            )
                        else:
                            extracted_jsons = []

                    # Try to parse extracted JSONs directly
                    if extracted_jsons:
                        logger.info("   🚀 Attempting direct JSON parsing...")
                        for json_str in extracted_jsons:
                            if not json_str:
                                continue

                            try:
                                parsed = json.loads(json_str)
                                if not isinstance(parsed, dict):
                                    continue

                                elem_id = parsed.get("element_id")
                                if elem_id and parsed.get("found"):
                                    # CRITICAL: Use validated locators from agent if available
                                    # The agent now validates locators during execution (while browser is open)
                                    # This is more reliable than validating after browser closes

                                    # Initialize variables at the top level to avoid UnboundLocalError
                                    dom_attrs = parsed.get("dom_attributes", {})
                                    dom_id = parsed.get("dom_id") or dom_attrs.get("id")
                                    generated_locators = []

                                    # Check if agent already validated locators
                                    if "locators" in parsed and parsed["locators"]:
                                        # Agent provided locators - verify they were actually validated
                                        generated_locators = parsed["locators"]
                                        logger.info(
                                            f"   📋 Received {len(generated_locators)} locators from agent"
                                        )

                                        # Add priority field if missing and verify validation status
                                        priority_map = {
                                            "id": 1,
                                            "data-testid": 2,
                                            "name": 3,
                                            "css-class": 7,
                                        }

                                        actually_validated_count = 0
                                        for loc in generated_locators:
                                            if "priority" not in loc:
                                                loc["priority"] = priority_map.get(
                                                    loc.get("type"), 10
                                                )

                                            # CRITICAL: Only mark as valid if it's unique (count=1)
                                            # For test automation, only unique locators are usable
                                            if loc.get("validated") and "count" in loc:
                                                # Has validation data from JavaScript
                                                count = loc.get("count", 0)
                                                loc["unique"] = count == 1
                                                # ONLY unique locators are valid for testing
                                                loc["valid"] = count == 1
                                                loc["validated"] = True
                                                actually_validated_count += 1

                                                if count == 1:
                                                    status = "✅ VALID & UNIQUE"
                                                elif count == 0:
                                                    status = "❌ NOT FOUND"
                                                else:
                                                    status = (
                                                        f"❌ INVALID - {count} matches (not unique)"
                                                    )

                                                logger.info(
                                                    f"      {loc['type']}: {loc['locator']} → {status} (agent-validated)"
                                                )
                                            else:
                                                # No validation data - mark as unvalidated
                                                loc["validated"] = False
                                                loc["unique"] = False
                                                loc["valid"] = False
                                                logger.warning(
                                                    f"      {loc['type']}: {loc['locator']} → ⚠️ No validation data"
                                                )

                                        logger.info(
                                            f"   ✅ {actually_validated_count}/{len(generated_locators)} locators have validation data"
                                        )
                                    else:
                                        # Fallback: Generate locators from DOM attributes
                                        logger.info(
                                            "   ⚠️ No pre-validated locators, generating from DOM attributes..."
                                        )

                                        # Priority 1: ID
                                        if dom_id:
                                            generated_locators.append(
                                                {
                                                    "type": "id",
                                                    "locator": f"id={dom_id}",
                                                    "priority": 1,
                                                    # Assume unique/valid but not validated yet
                                                    "unique": None,  # Unknown until validated
                                                    "valid": None,  # Unknown until validated
                                                    "validated": False,  # Not yet validated
                                                }
                                            )

                                        # Priority 2: data-testid
                                        if dom_attrs.get("data-testid"):
                                            generated_locators.append(
                                                {
                                                    "type": "data-testid",
                                                    "locator": f"data-testid={dom_attrs['data-testid']}",
                                                    "priority": 2,
                                                    "unique": None,
                                                    "valid": None,
                                                    "validated": False,
                                                }
                                            )

                                        # Priority 3: name
                                        if dom_attrs.get("name"):
                                            generated_locators.append(
                                                {
                                                    "type": "name",
                                                    "locator": f"name={dom_attrs['name']}",
                                                    "priority": 3,
                                                    "unique": None,
                                                    "valid": None,
                                                    "validated": False,
                                                }
                                            )

                                        # Priority 4: CSS class (if available)
                                        if dom_attrs.get("class"):
                                            first_class = (
                                                dom_attrs["class"].split()[0]
                                                if dom_attrs["class"]
                                                else None
                                            )
                                            if first_class:
                                                element_type = parsed.get("element_type", "div")
                                                generated_locators.append(
                                                    {
                                                        "type": "css-class",
                                                        "locator": f"{element_type}.{first_class}",
                                                        "priority": 7,
                                                        "unique": None,
                                                        "valid": None,
                                                        "validated": False,
                                                    }
                                                )

                                    # No Playwright page here to re-validate against:
                                    # CDP page access went with the locator
                                    # pipelines (5867f33). The locators arrive
                                    # already validated by the browser-service
                                    # engine, which owns that step now.
                                    logger.info(
                                        f"   ⚠️ Page not available, skipping validation for {elem_id} (trusting browser_use)"
                                    )

                                # Select best locator - ONLY use validated, unique, valid locators
                                # valid=True means count=1 (unique and usable for testing)
                                validated_unique = [
                                    loc
                                    for loc in generated_locators
                                    if loc.get("validated")
                                    and loc.get("unique")
                                    and loc.get("valid")
                                ]

                                if validated_unique:
                                    # Found valid unique locators - select best by priority
                                    best_locator = min(
                                        validated_unique, key=lambda x: x["priority"]
                                    )["locator"]
                                    logger.info(
                                        f"   ✅ Selected VALID unique locator: {best_locator}"
                                    )
                                else:
                                    # No valid unique locators found — report why.
                                    best_locator = None

                                    # Log why we couldn't find a valid locator
                                    if generated_locators:
                                        non_unique = [
                                            loc
                                            for loc in generated_locators
                                            if loc.get("validated") and loc.get("count", 0) > 1
                                        ]
                                        not_found = [
                                            loc
                                            for loc in generated_locators
                                            if loc.get("validated") and loc.get("count", 0) == 0
                                        ]
                                        not_validated = [
                                            loc
                                            for loc in generated_locators
                                            if not loc.get("validated")
                                        ]

                                        if non_unique:
                                            logger.error(
                                                f"   ❌ No valid locator: {len(non_unique)} locators are not unique"
                                            )
                                        if not_found:
                                            logger.error(
                                                f"   ❌ No valid locator: {len(not_found)} locators not found on page"
                                            )
                                        if not_validated:
                                            logger.warning(
                                                f"   ⚠️ {len(not_validated)} locators were not validated"
                                            )
                                    else:
                                        logger.error(f"   ❌ No locators generated for {elem_id}")

                                # Find element description
                                elem_desc = next(
                                    (
                                        e.get("description")
                                        for e in elements
                                        if e.get("id") == elem_id
                                    ),
                                    "Unknown element",
                                )

                                # Build result with locators
                                result = {
                                    "element_id": elem_id,
                                    "description": elem_desc,
                                    "found": True,
                                    "best_locator": best_locator,
                                    "all_locators": generated_locators,
                                    "element_info": {
                                        "id": dom_id,
                                        "tagName": parsed.get("element_type", ""),
                                        "text": parsed.get("visible_text", ""),
                                        "className": dom_attrs.get("class", ""),
                                        "name": dom_attrs.get("name", ""),
                                        "testId": dom_attrs.get("data-testid", ""),
                                    },
                                    "coordinates": parsed.get("coordinates", {}),
                                    "validation_summary": {
                                        "total_generated": len(generated_locators),
                                        "valid": sum(
                                            1 for loc in generated_locators if loc.get("valid")
                                        ),
                                        "unique": sum(
                                            1 for loc in generated_locators if loc.get("unique")
                                        ),
                                        "validated": sum(
                                            1 for loc in generated_locators if loc.get("validated")
                                        ),
                                        "best_type": generated_locators[0]["type"]
                                        if generated_locators
                                        else None,
                                    },
                                }

                                # Check if we already have this element
                                existing_idx = next(
                                    (
                                        i
                                        for i, r in enumerate(results_list)
                                        if r.get("element_id") == elem_id
                                    ),
                                    None,
                                )
                                if existing_idx is not None:
                                    # Replace existing result (agent retry/correction)
                                    old_locator = results_list[existing_idx].get("best_locator")
                                    logger.info(
                                        f"   🔄 Replacing {elem_id}: '{old_locator}' → '{best_locator}' (agent retry/correction)"
                                    )
                                    results_list[existing_idx] = result
                                else:
                                    # First occurrence, add it
                                    results_list.append(result)
                                    logger.info(
                                        f"   ✅ Directly parsed and added {elem_id} (best_locator: {best_locator})"
                                    )
                            except json.JSONDecodeError as e:
                                logger.debug(f"   Failed to parse JSON directly: {e}")
                                # Will fall back to pattern matching below

                    # If we got all elements via direct parsing, we're done!
                    if len(results_list) == len(elements):
                        logger.info(
                            f"   🎉 All {len(elements)} elements extracted via direct JSON parsing!"
                        )
                        # Skip pattern matching - we have everything we need
                        # Jump directly to re-ranking section
                        logger.info(
                            "   ⏭️  Skipping string-based extraction (all elements found via direct access)"
                        )

                # ONLY do string-based extraction if we don't have all elements yet
                if len(results_list) < len(elements):
                    logger.info(
                        f"   🔍 Missing {len(elements) - len(results_list)} elements, trying string-based extraction..."
                    )
                else:
                    logger.info(
                        "   ✅ All elements found via direct access, skipping string parsing"
                    )
                    # Skip to re-ranking by using a flag or just not entering the loop
                    # We'll just make the loop conditional

                # Only run pattern matching if we're missing elements
                if len(results_list) < len(elements):
                    # Check which elements we're still missing
                    missing_ids = [
                        e.get("id")
                        for e in elements
                        if not any(r.get("element_id") == e.get("id") for r in results_list)
                    ]
                    logger.info(
                        f"   🔍 Looking for {len(missing_ids)} missing elements: {missing_ids}"
                    )

                    for elem in elements:
                        elem_id = elem.get("id")

                        # Skip if we already have this element
                        if any(r.get("element_id") == elem_id for r in results_list):
                            continue

                        # Check multiple patterns (with and without space after colon)
                        patterns_to_check = [
                            f'"element_id":"{elem_id}"',
                            f'"element_id": "{elem_id}"',
                            f"'element_id':'{elem_id}'",
                            f"'element_id': '{elem_id}'",
                        ]

                        found = any(pattern in full_result_str for pattern in patterns_to_check)

                        if not found:
                            logger.warning(f"   ⚠️  '{elem_id}' not found in result string")
                            continue

                        # Element found in result string - extract JSON data
                        try:
                            elem_data = extract_json_for_element(full_result_str, elem_id)
                            # Same gate as the other two paths.
                            if (
                                elem_data
                                and elem_data.get("found")
                                and elem_data.get("best_locator")
                            ):
                                existing_idx = next(
                                    (
                                        i
                                        for i, r in enumerate(results_list)
                                        if r.get("element_id") == elem_id
                                    ),
                                    None,
                                )
                                if existing_idx is None:
                                    # First occurrence, add it
                                    results_list.append(elem_data)
                                    logger.info(f"   ✅ Extracted {elem_id} from result string")
                            elif elem_data:
                                rejected_payloads[elem_id] = elem_data
                        except Exception as e:
                            logger.exception(f"   ❌ Exception extracting {elem_id}: {e}")
                else:
                    logger.info(
                        "   ⏭️  String-based extraction skipped (all elements already found)"
                    )

            # Elements still missing an entry — none extracted, or only some — are
            # backfilled once, below, after the re-rank and validation passes (both
            # skip not-found entries, so there is nothing for them to do here).
            if not results_list:
                logger.error("Could not extract any element results from workflow")

            # ========================================
            # PHASE 2: POST-PROCESS LOCATOR RE-RANKING
            # ========================================
            # Re-rank locators by quality to ensure best_locator is actually the best
            def score_locator(locator_obj):
                """
                Score locator based on robustness and stability.
                Higher score = better locator.

                STRICT SIX-TIER PRIORITY SYSTEM:
                ================================
                Tier 1 (90-100): Native Attributes - Most stable, browser-native lookups
                    - ID: 100 (best possible - unique, fast, stable)
                    - data-testid: 98 (designed specifically for testing)
                    - name: 96 (semantic, stable for forms)

                Tier 2 (70-89): Semantic Attributes - Accessibility-focused, stable
                    - aria-label: 88 (accessibility attribute, semantic)
                    - title: 85 (semantic attribute, descriptive)

                Tier 3 (50-69): Content-Based - Can change with content updates
                    - text: 65 (content-based, can change)
                    - role: 60 (Playwright-specific, semantic)

                Tier 4 (40-55): Fallback Strategies - Advanced strategies when basic attributes unavailable
                    - parent-id-xpath: 55 (anchored to parent ID, stable)
                    - nth-child: 50 (position-based, moderately stable)
                    - text-xpath: 48 (exact text match, more specific)
                    - attribute-combo: 45 (multiple attributes for uniqueness)

                Tier 5 (30-39): CSS Selectors - Styling-based, can change
                    - CSS with ID: 45 (should use id= instead)
                    - CSS with attribute: 40 (better than class)
                    - Regular CSS class: 35 (styling can change)
                    - Auto-generated class: 32 (very fragile)

                Tier 6 (0-29): XPath - LAST RESORT, fragile, breaks with DOM changes
                    - XPath with ID: 28 (should use id= instead!)
                    - XPath with data-testid: 26 (should use data-testid= instead!)
                    - XPath with semantic attrs: 24 (should use direct attribute)
                    - Text-based XPath: 20 (content can change)
                    - Structural XPath: 10-18 (very fragile, breaks easily)

                Clear score gaps between tiers prevent ties and ensure strict priority.
                """
                locator = locator_obj.get("locator", "")
                locator_type = locator_obj.get("type", "")

                # ========================================
                # TIER 1: NATIVE ATTRIBUTES (90-100)
                # ========================================
                # These are the most stable locators - browser-native lookups
                # that are fast, unique, and rarely change

                if locator_type == "id" or locator.startswith("id="):
                    return 100  # Best possible - unique, fast, stable

                if locator_type == "data-testid" or "data-testid=" in locator:
                    return 98  # Designed specifically for testing

                if locator_type == "name" or locator.startswith("name="):
                    return 96  # Semantic, stable for form elements

                # ========================================
                # TIER 2: SEMANTIC ATTRIBUTES (70-89)
                # ========================================
                # Accessibility-focused attributes that are semantic and stable

                if locator_type == "aria-label" or "aria-label=" in locator:
                    return 88  # Accessibility-focused, semantic

                if "@title=" in locator or locator_type == "title":
                    return 85  # Semantic attribute, descriptive

                # ========================================
                # TIER 3: CONTENT-BASED (50-69)
                # ========================================
                # Locators based on visible content - can change with content updates

                if locator_type == "text" or "text=" in locator:
                    return 65  # Content-based, can change with text updates

                if locator_type == "role" or "role=" in locator:
                    return 60  # Playwright-specific, semantic but content-dependent

                # ========================================
                # TIER 4: CSS SELECTORS (30-49)
                # ========================================
                # Styling-based selectors - can change when CSS is refactored

                if locator_type == "css" or locator.startswith("css="):
                    css_selector = locator.replace("css=", "")

                    if "#" in css_selector:
                        return 45  # CSS with ID (should use id= instead!)

                    if "[" in css_selector:
                        return 40  # CSS with attribute selector

                    # Check for auto-generated classes (very fragile)
                    if re.search(r"[_][0-9a-zA-Z]{5,}", css_selector):
                        return 32  # Auto-generated class (very fragile)

                    return 35  # Regular CSS class (styling can change)

                # ========================================
                # TIER 4.5: FALLBACK STRATEGIES (40-55)
                # ========================================
                # Advanced fallback strategies when basic attributes don't exist
                # These are better than generic CSS/XPath but not as good as native attributes

                # Parent ID + Relative XPath - anchored to stable ID
                if locator_type == "parent-id-xpath":
                    return 55  # Anchored to parent ID (stable), but uses XPath

                # Nth-child selector - position-based, moderately stable
                if locator_type == "nth-child":
                    return 50  # Position-based, can break if siblings change

                # Text-based XPath with exact match - better than generic XPath
                if locator_type == "text-xpath":
                    return 48  # Text-based but exact match (more specific)

                # Attribute combination - multiple attributes for uniqueness
                if locator_type == "attribute-combo":
                    # Multiple attributes (more stable than single class)
                    return 45

                # ========================================
                # TIER 5: XPATH - LAST RESORT (0-29)
                # ========================================
                # XPath locators are fragile and break when DOM structure changes
                # They should ONLY be used when no better option exists
                # Even "good" XPath gets low scores to enforce this priority

                if (
                    "xpath" in locator_type
                    or locator.startswith("//")
                    or locator.startswith("xpath=")
                ):
                    # XPath with ID - should use id= instead!
                    if "@id=" in locator:
                        return 28  # Should use id= locator instead

                    # XPath with data-testid - should use data-testid= instead!
                    if "@data-testid=" in locator or "@data-test=" in locator:
                        return 26  # Should use data-testid= locator instead

                    # XPath with semantic attributes - should use direct attribute
                    if "@aria-label=" in locator or "@title=" in locator:
                        return 24  # Should use direct attribute locator

                    # Text-based XPath - content can change
                    if "text()=" in locator or "contains(text()" in locator:
                        return 20  # Content-based, can change

                    # Structural XPath (worst) - lots of [1], [2], etc.
                    # These break easily when DOM structure changes
                    index_count = locator.count("[1]") + locator.count("[2]") + locator.count("[3]")
                    if index_count >= 3:
                        return 10  # Very structural (extremely fragile)
                    elif index_count >= 2:
                        return 15  # Somewhat structural (fragile)

                    return 18  # Default XPath (still fragile)

                # Unknown/default - below Tier 4
                return 25

            logger.info("🔄 Re-ranking locators by quality score...")
            re_ranked_count = 0

            for result in results_list:
                if not result.get("found", False):
                    continue

                all_locators = result.get("all_locators", [])
                if not all_locators:
                    continue

                # Score each locator - ONLY score unique and valid locators
                scored_locators = []
                skipped_count = 0

                # Check if this is a collection element (collections have unique=False by design)
                element_type = result.get("element_type", "single")
                is_collection = element_type == "collection"

                for loc in all_locators:
                    # CRITICAL FIX: Filter out non-unique or invalid locators before scoring
                    # EXCEPTION: Collections are allowed to have unique=False (they match multiple elements)
                    if not is_collection and not (loc.get("unique") and loc.get("valid")):
                        skipped_count += 1
                        continue  # Skip non-unique or invalid locators (but not collections)

                    try:
                        # Collections already have quality_score from smart_locator_finder, preserve it
                        if is_collection and "quality_score" in loc:
                            scored_locators.append(loc)
                        else:
                            score = score_locator(loc)
                            scored_locators.append({**loc, "quality_score": score})
                    except Exception as e:
                        logger.warning(f"⚠️ Error scoring locator: {e}, skipping")

                # Log filtering results
                element_id = result.get("element_id", "unknown")
                if is_collection:
                    logger.info(
                        f"🔍 {element_id}: COLLECTION element - keeping {len(scored_locators)} locator(s) (unique=False expected)"
                    )
                elif skipped_count > 0:
                    logger.info(
                        f"🔍 {element_id}: Filtered out {skipped_count} non-unique/invalid locators (keeping {len(scored_locators)} unique locators)"
                    )

                if not scored_locators:
                    logger.warning(f"⚠️ {element_id}: No locators available after filtering!")
                    continue

                # Stability tier first, quality score second (E1) — see
                # rerank_sort_key for why score-only sorting would undo the
                # locator engine's demotion.
                scored_locators.sort(key=rerank_sort_key)

                # Log top 3 locators with their scores for debugging
                locator_type_label = "COLLECTION" if is_collection else "UNIQUE"
                logger.info(
                    f"📊 Locator Scores for {element_id} (showing {locator_type_label} locators only):"
                )
                for i, loc in enumerate(scored_locators[:3]):  # Show top 3
                    locator_str = loc.get("locator", "")[:50]  # Truncate long locators
                    quality_score = loc.get("quality_score", 0)
                    loc_type = loc.get("type", "unknown")
                    unique = loc.get("unique", False)
                    valid = loc.get("valid", False)

                    if i == 0:
                        # First locator is the selected best
                        logger.info(
                            f"   {quality_score:3d} - {loc_type:15s} - {locator_str} ⭐ SELECTED AS BEST (unique={unique}, valid={valid})"
                        )
                        # Log warning if XPath is selected as best
                        if (
                            loc_type == "xpath"
                            or locator_str.startswith("xpath=")
                            or locator_str.startswith("//")
                        ):
                            logger.warning(
                                "   ⚠️  XPath used as fallback - no ID, data-testid, name, or aria-label available"
                            )
                    else:
                        logger.info(
                            f"   {quality_score:3d} - {loc_type:15s} - {locator_str} (unique={unique}, valid={valid})"
                        )

                # Update result with re-ranked locators
                old_best = result.get("best_locator", "")
                new_best = scored_locators[0]["locator"]
                new_score = scored_locators[0].get("quality_score", 0)

                if old_best != new_best:
                    # Calculate old score for comparison (helpful for debugging)
                    old_score = score_locator({"locator": old_best}) if old_best else 0
                    logger.info(f"   ✨ {element_id}: Upgraded locator")
                    logger.info(f"      OLD: {old_best} (score: {old_score})")
                    logger.info(f"      NEW: {new_best} (score: {new_score})")
                    re_ranked_count += 1

                commit_reranked_winner(result, scored_locators)

            logger.info(
                f"✅ Re-ranking complete: {re_ranked_count}/{len(results_list)} elements upgraded"
            )

            # ========================================
            # RESULTS VALIDATION - Verify quality_score is present
            # ========================================
            logger.info("🔍 Validating results before return...")
            for result in results_list:
                elem_id = result.get("element_id", "unknown")
                found = result.get("found", False)
                best_locator = result.get("best_locator", "N/A")
                all_locators = result.get("all_locators", [])

                if found:
                    has_scores = all(loc.get("quality_score") is not None for loc in all_locators)
                    logger.info(
                        f"   ✅ {elem_id}: {best_locator} ({len(all_locators)} locators, scored={has_scores})"
                    )
                    if not has_scores and all_locators:
                        logger.warning(f"   ⚠️ {elem_id}: Some locators missing quality_score!")
                else:
                    error = result.get("error", "Unknown")
                    logger.error(f"   ❌ {elem_id}: {error}")
            # ========================================

            # ========================================
            # LOCATOR PRIORITY VALIDATION CHECK
            # ========================================
            # Verify that elements with ID attributes use ID locators
            # This catches cases where the scoring system may have failed
            # or where XPath/other locators were incorrectly prioritized
            logger.info("🔍 Running locator priority validation check...")
            validation_violations = 0

            for result in results_list:
                if not result.get("found", False):
                    continue

                element_info = result.get("element_info", {})
                element_id_attr = element_info.get("id", "").strip()
                best_locator = result.get("best_locator", "")
                all_locators = result.get("all_locators", [])
                elem_id = result.get("element_id", "unknown")

                # Check if element has ID attribute but best_locator is not ID type
                if element_id_attr and element_id_attr != "":
                    # A volatile id (ext-gen1042, tomselect-3, ...) was
                    # deliberately demoted by the stability scorer (E1) —
                    # forcing it back here would undo the demotion.
                    if score_stability("id", element_id_attr) != STABLE:
                        logger.info(
                            f"   ⏭️ {elem_id}: id '{element_id_attr}' is volatile — "
                            f"keeping stability-demoted best_locator {best_locator}"
                        )
                        continue

                    # Determine if best_locator is an ID locator.
                    # Accepts both Playwright explicit format (id=value) and
                    # CSS hash format (#value) — both target the ID attribute.
                    is_id_locator = best_locator.startswith("id=") or best_locator.startswith("#")

                    if not is_id_locator:
                        # PRIORITY VIOLATION DETECTED
                        logger.error(f"❌ PRIORITY VIOLATION: {elem_id}")
                        logger.error(f"   Element has ID attribute: '{element_id_attr}'")
                        logger.error(f"   But best_locator is: {best_locator}")
                        validation_violations += 1

                        # Search for ID locator in all_locators list
                        id_locator = None
                        id_locator_index = None
                        for idx, loc in enumerate(all_locators):
                            loc_str = loc.get("locator", "")
                            if loc_str.startswith("id=") or loc_str.startswith("#"):
                                id_locator = loc
                                id_locator_index = idx
                                break

                        if id_locator:
                            # Automatically correct by forcing ID locator to be best_locator
                            logger.info(f"   🔧 Forcing ID locator: {id_locator['locator']}")

                            # Move ID locator to first position
                            all_locators.pop(id_locator_index)
                            all_locators.insert(0, id_locator)

                            # Sync best_locator, stability and all_locators to
                            # the forced ID winner together — leaving the
                            # displaced non-id best's stability behind would
                            # mislabel the now-stable ID locator (same drift
                            # commit_reranked_winner prevents in the re-ranker).
                            commit_reranked_winner(result, all_locators)

                            logger.info(f"   ✅ Corrected: {elem_id} now uses ID locator")
                        else:
                            # ID locator not found in list - this is a critical issue
                            logger.error(
                                "   ⚠️  CRITICAL: ID locator not found in all_locators list!"
                            )
                            logger.error(f"   Element ID attribute: '{element_id_attr}'")
                            logger.error(
                                f"   Available locators: {[loc.get('type') for loc in all_locators]}"
                            )
                            logger.error("   This indicates a problem with locator generation")

            if validation_violations > 0:
                logger.warning(
                    f"⚠️  Validation found {validation_violations} priority violations (corrected)"
                )
            else:
                logger.info("✅ Validation passed: All elements with ID use ID locators")
            # ========================================

            # Backfill the elements that produced no entry at all: one the agent
            # never called find_unique_locator for, and one it reported as
            # found=False (the extractors above only record found payloads).
            # Without this they are indistinguishable from elements that were
            # never requested, and `failed` below can only ever be 0.
            reported_ids = {r.get("element_id") for r in results_list}
            for elem in elements:
                elem_id = elem.get("id")
                if elem_id in reported_ids:
                    continue
                record = {
                    "element_id": elem_id,
                    "description": elem.get("description"),
                    "found": False,
                    "error": "No locator reported by the agent",
                    "validated": False,
                    "count": 0,
                    "unique": False,
                    "valid": False,
                    "validation_method": "playwright",
                    "metrics": {
                        "execution_time": 0,
                        "estimated_llm_calls": 0,
                        "estimated_cost": 0,
                        # False, not the custom_actions_enabled FLAG:
                        # record_workflow_metrics and the NL backend's tool both
                        # tally this key to report how many elements the custom
                        # action resolved. An element with no result resolved
                        # nothing, whether or not the action was available.
                        "custom_action_used": False,
                    },
                }
                rejected = rejected_payloads.get(elem_id)
                if rejected:
                    # Keep the engine's own diagnosis — error, error_type,
                    # semantic_match, whatever it recorded — but never its found
                    # or identity claims: a payload rejected for claiming found
                    # without a locator must not re-enter as found, and the
                    # requested description is the canonical one.
                    #
                    # metrics is the same exception as approach_metrics below:
                    # this element resolved nothing, so custom_action_used must
                    # stay False. A rejected payload carrying its own metrics
                    # would otherwise ride the overlay in and re-inflate the
                    # custom-action tally record_workflow_metrics computes.
                    own_metrics = record["metrics"]
                    record.update(rejected)
                    record["metrics"] = own_metrics
                    record["element_id"] = elem_id
                    record["description"] = elem.get("description")
                    record["found"] = False
                    # A payload may carry error=None explicitly; .get(default)
                    # downstream would not replace that, so coalesce here.
                    record["error"] = record["error"] or "No locator reported by the agent"
                    # approach_metrics is the exception: element_approach_metrics
                    # feeds a strategy-distribution chart that buckets elements by
                    # fallback_depth, and every row in it today is an element that
                    # WAS located. Failures stamp depth 7, so passing them through
                    # would inflate that bucket with elements the tier did not
                    # actually resolve. Failure-side strategy data needs its own
                    # decision (consistent stamping + a chart that splits the two),
                    # not a side effect of this backfill.
                    record.pop("approach_metrics", None)
                logger.warning(f"   ❌ {elem_id}: {record['error']}")
                results_list.append(record)

            # Calculate metrics
            successful = sum(1 for r in results_list if r.get("found", False))
            failed = len(results_list) - successful

            # ========================================
            # METRICS LOGGING: Cost Calculation
            # ========================================
            # Actual cost is calculated by browser-use's TokenCostService
            # Phase: Error Handling and Logging | Requirements: 6.1, 6.2, 6.3, 9.5

            # Only log cost metrics if TRACK_LLM_COSTS is enabled
            if _nl_settings and _nl_settings.TRACK_LLM_COSTS:
                # Calculate average metrics
                avg_llm_calls_per_element = (
                    llm_call_count / len(elements) if len(elements) > 0 else 0
                )

                logger.info("=" * 80)
                logger.info("📊 WORKFLOW COST METRICS")
                logger.info("=" * 80)
                logger.info(f"Total LLM calls: {llm_call_count}")
                logger.info(f"Average LLM calls per element: {avg_llm_calls_per_element:.1f}")
                logger.info(f"Actual cost (from browser-use): ${token_usage['actual_cost']:.6f}")
                logger.info(
                    f"Cost per element: ${token_usage['actual_cost'] / len(elements):.6f}"
                    if len(elements) > 0
                    else "N/A"
                )
                logger.info(f"Custom actions enabled: {custom_actions_enabled}")
                logger.info(f"Total execution time: {execution_time:.2f}s")
                logger.info(
                    f"Average time per element: {execution_time / len(elements):.2f}s"
                    if len(elements) > 0
                    else "N/A"
                )
                logger.info("--- TOKEN METRICS ---")
                logger.info(f"Total tokens: {token_usage['total_tokens']}")
                logger.info(f"Input tokens (prompt): {token_usage['input_tokens']}")
                logger.info(f"Output tokens (completion): {token_usage['output_tokens']}")
                logger.info(f"Cached tokens: {token_usage['cached_tokens']}")
                logger.info("=" * 80)

            # ========================================
            # VALIDATION VERIFICATION BEFORE WORKFLOW COMPLETION
            # ========================================
            # Verify all elements have proper validation data
            logger.info("🔍 Verifying validation data for all elements...")

            validation_issues = []
            elements_without_validation = []
            elements_not_unique = []
            elements_not_valid = []

            for result in results_list:
                elem_id = result.get("element_id", "unknown")

                # Check if element has validated=True
                if not result.get("validated", False):
                    validation_issues.append(f"{elem_id}: missing validated=True")
                    elements_without_validation.append(elem_id)

                # Check if element has count=1 and unique=True (only for found elements)
                if result.get("found", False):
                    count = result.get("count", 0)
                    unique = result.get("unique", False)
                    valid = result.get("valid", False)
                    element_type = result.get("element_type", "single")

                    # Collections are EXPECTED to have count > 1 and unique=False
                    if element_type == "collection":
                        # For collections, check if count > 1 and valid=True
                        if count > 1 and valid:
                            logger.debug(
                                f"   ✅ {elem_id}: Collection with {count} elements (expected)"
                            )
                        else:
                            validation_issues.append(
                                f"{elem_id}: Invalid collection (count={count}, valid={valid})"
                            )
                            if not valid:
                                elements_not_valid.append(elem_id)
                            if count <= 1:
                                elements_not_unique.append(elem_id)
                    else:
                        # For single elements, require count=1 and unique=True
                        if count != 1 or not unique:
                            validation_issues.append(
                                f"{elem_id}: count={count}, unique={unique} (expected count=1, unique=True)"
                            )
                            elements_not_unique.append(elem_id)

                        if not valid:
                            validation_issues.append(
                                f"{elem_id}: valid={valid} (expected valid=True)"
                            )
                            elements_not_valid.append(elem_id)

            # Log validation summary
            if validation_issues:
                logger.warning(
                    f"⚠️ Validation issues found for {len(validation_issues)} element(s):"
                )
                for issue in validation_issues:
                    logger.warning(f"   - {issue}")
            else:
                logger.info("✅ All elements have complete validation data")

            # Create validation summary for results
            validation_summary = {
                "total_elements": len(results_list),
                "elements_with_validation": len(results_list) - len(elements_without_validation),
                "elements_without_validation": len(elements_without_validation),
                "elements_unique": len(
                    [r for r in results_list if r.get("unique", False) and r.get("found", False)]
                ),
                "elements_not_unique": len(elements_not_unique),
                "elements_valid": len(
                    [r for r in results_list if r.get("valid", False) and r.get("found", False)]
                ),
                "elements_not_valid": len(elements_not_valid),
                "validation_issues": validation_issues,
                "elements_without_validation_list": elements_without_validation,
                "elements_not_unique_list": elements_not_unique,
                "elements_not_valid_list": elements_not_valid,
            }

            logger.info("📊 Validation Summary:")
            logger.info(f"   Total elements: {validation_summary['total_elements']}")
            logger.info(
                f"   Elements with validation: {validation_summary['elements_with_validation']}/{validation_summary['total_elements']}"
            )
            logger.info(
                f"   Elements with unique locators: {validation_summary['elements_unique']}/{successful}"
            )
            logger.info(
                f"   Elements with valid locators: {validation_summary['elements_valid']}/{successful}"
            )

            if validation_issues:
                logger.warning(f"   ⚠️ {len(validation_issues)} validation issue(s) detected")
            # ========================================

            # CRITICAL: Only consider workflow successful if ALL elements have unique locators
            # This ensures we don't proceed with placeholder locators or non-unique locators.
            # Measured per requested element id, not by count: an entry for an id
            # nobody asked for must not stand in for a requested one that is missing.
            found_ids = {r.get("element_id") for r in results_list if r.get("found", False)}
            all_found = bool(elements) and all(e.get("id") in found_ids for e in elements)

            # ========================================
            # COLLECT ELEMENT APPROACH METRICS
            # ========================================
            # Extract approach_metrics from each element result for pattern analysis
            # This enables tracking which locator approach worked best for different element types
            from urllib.parse import urlparse

            url_domain = urlparse(url).netloc if url else ""

            element_approach_metrics = []
            for result in results_list:
                approach_data = result.get("approach_metrics")
                if approach_data:
                    # Create new dict to avoid mutating original result
                    approach_entry = {
                        **approach_data,
                        "element_id": result.get("element_id", ""),
                        "url_domain": url_domain,
                    }
                    element_approach_metrics.append(approach_entry)

            postprocess_s = time.perf_counter() - _t_postprocess

            return {
                "success": all_found,  # Changed from 'successful > 0' to require ALL elements found
                "workflow_mode": True,
                "workflow_completed": workflow_completed,
                "results": results_list,
                "summary": {
                    "total_elements": len(elements),
                    "successful": successful,
                    "failed": failed,
                    "success_rate": successful / len(elements) if len(elements) > 0 else 0,
                    # Cost tracking metrics
                    "total_llm_calls": llm_call_count,
                    "avg_llm_calls_per_element": llm_call_count / len(elements)
                    if len(elements) > 0
                    else 0,
                    "custom_actions_enabled": custom_actions_enabled,
                    # Actual token usage from browser-use
                    "total_tokens": token_usage["total_tokens"],
                    "input_tokens": token_usage["input_tokens"],
                    "output_tokens": token_usage["output_tokens"],
                    "cached_tokens": token_usage["cached_tokens"],
                    "actual_cost": token_usage["actual_cost"],
                    # Per-element approach metrics for pattern analysis
                    "element_approach_metrics": element_approach_metrics,
                    # identify_s phase breakdown (2026-07-26 efficiency check).
                    # The backend adds submit_s and poll_wait_s on its side.
                    "phase_timings": {
                        "queue_s": round(queue_s, 3),
                        "session_setup_s": round(session_setup_s, 3),
                        "agent_setup_s": round(agent_setup_s, 3),
                        "agent_run_s": round(execution_time, 3),
                        "postprocess_s": round(postprocess_s, 3),
                    },
                    "agent_diagnostics": extract_agent_diagnostics(
                        agent_result, dom_samples, llm_durations
                    ),
                },
                "validation_summary": validation_summary,  # Add validation summary to results
                "execution_time": execution_time,
                "session_id": str(id(session)),
            }

        except Exception as e:
            logger.error(f"❌ Workflow task error: {e}", exc_info=True)
            return {
                "success": False,
                "workflow_mode": True,
                "error": str(e),
                "results": [],
                "summary": {
                    "total_elements": len(elements),
                    "successful": 0,
                    "failed": len(elements),
                    "success_rate": 0,
                },
            }
        finally:
            # Playwright cache cleanup removed — no module-level cache exists.
            # Each custom action creates and destroys its own Playwright instance.

            # Task 19: do NOT clean up here. Deposit the handles for the sync
            # wrapper, which runs cleanup AFTER publishing the completed status.
            # This finally runs on success and on exception, so a session whose
            # start() failed is still captured for cleanup.
            cleanup_handles["session"] = session
            cleanup_handles["connected_browser"] = connected_browser
            cleanup_handles["playwright_instance"] = playwright_instance
            cleanup_handles["browser_pid"] = browser_pid

    # Run the async workflow
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)  # unset in finally before loop.close()
    try:
        results = loop.run_until_complete(run_unified_workflow())

        # Update task status
        task_processor.update_task(
            task_id,
            {
                "status": "completed",
                "completed_at": time.time(),
                "message": f"Workflow completed: {results['summary']['successful']}/{results['summary']['total_elements']} elements found",
                "results": results,
            },
        )

        logger.info(f"🎉 Workflow task {task_id} completed successfully")
        if "summary" in results and "success_rate" in results["summary"]:
            logger.info(f"   Success rate: {results['summary']['success_rate'] * 100:.1f}%")

        # ========================================
        # RECORD WORKFLOW METRICS
        # ========================================
        # Send metrics to the workflow metrics API endpoint for persistence
        # Skip if parent_workflow_id is provided (main workflow will handle unified metrics)
        if (
            _nl_settings
            and _nl_settings.TRACK_LLM_COSTS
            and "summary" in results
            and not parent_workflow_id
        ):
            try:
                record_workflow_metrics(
                    workflow_id=task_id,
                    url=url,
                    results=results,
                    session_id=results.get("session_id"),
                    backend_port=_nl_settings.APP_PORT if _nl_settings else 5000,
                )
                logger.info(f"📊 Browser-use metrics recorded for task {task_id}")
            except Exception as metrics_error:
                # Don't fail the workflow if metrics recording fails
                logger.warning(f"⚠️ Failed to record workflow metrics: {metrics_error}")
        elif parent_workflow_id:
            logger.info(
                f"⏭️  Skipping browser-use metrics recording (parent workflow {parent_workflow_id} will handle unified metrics)"
            )

    except Exception as e:
        logger.exception(f"❌ Failed to execute workflow task {task_id}: {e}")
        task_processor.update_task(
            task_id,
            {
                "status": "completed",
                "completed_at": time.time(),
                "message": f"Workflow failed: {str(e)}",
                "results": {
                    "success": False,
                    "workflow_mode": True,
                    "error": str(e),
                    "results": [],
                    "summary": {
                        "total_elements": len(elements),
                        "successful": 0,
                        "failed": len(elements),
                    },
                },
            },
        )
    finally:
        # Task 19 (TIER-0 0.2): browser cleanup runs HERE, after the task was
        # marked completed above — the polling client already has its results
        # and stops waiting. The worker thread (and thus the concurrency slot's
        # real capacity — executor max_workers == max_concurrent_tasks) is
        # still held until Chrome is confirmed dead, so cleanup cannot leak:
        # same guarantees as before, minus the user-visible wait.
        try:
            loop.run_until_complete(
                cleanup_browser_resources(
                    session=cleanup_handles.get("session"),
                    connected_browser=cleanup_handles.get("connected_browser"),
                    playwright_instance=cleanup_handles.get("playwright_instance"),
                    browser_pid=cleanup_handles.get("browser_pid"),
                )
            )
        except Exception as cleanup_error:
            # Results are already published; a cleanup failure must never
            # clobber them. The orphan-detection logging inside
            # cleanup_browser_resources() and the PID-scoped cleanup_worker
            # remain the safety net for leaked Chrome processes.
            logger.error(
                f"⚠️ Post-completion browser cleanup failed: {cleanup_error}", exc_info=True
            )

        # Unset the thread-local loop BEFORE closing it.
        # ThreadPoolExecutor reuses worker threads; without this, the next task
        # on the same thread would inherit the closed loop from set_event_loop()
        # above and fail immediately. asyncio.run() / asyncio.Runner follow the
        # same pattern: set_event_loop(None) then loop.close().
        asyncio.set_event_loop(None)
        loop.close()
