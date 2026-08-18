"""Basic usage example for the browser-service package.

This calls process_workflow_task directly, which is what the Flask route does
after validating a request. In normal operation you would POST to /workflow
rather than importing this function — see the README.

Requires a configured model provider (MODEL_PROVIDER, plus either
GEMINI_API_KEY or the VERTEXAI_* variables) and a browser available to
browser-use. It drives a real browser against the target URL.
"""

from browser_service.config import config
from browser_service.tasks import process_workflow_task


def main() -> None:
    # Config is nested — there is no flat config.GEMINI_API_KEY. Provider and
    # credentials come from the environment; see the README's variable table.
    print(f"provider={config.llm.model_provider} headless={config.headless}")

    # Each element is a spec dict, not a "step". Actions the dispatcher
    # understands: input/type, click/submit, select, check/uncheck.
    elements = [
        {"id": "elem_1", "description": "Login button", "action": "click"},
        {"id": "elem_2", "description": "Username field", "action": "input"},
    ]

    # process_workflow_task is synchronous — do not await it.
    result = process_workflow_task(
        task_id="example-task-1",
        elements=elements,
        url="https://example.com",
        user_query="Log in with a username",
        session_config={},
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
