"""Phase 3: the worker binds the correlation id + org/user into structlog contextvars."""

import structlog

from browser_service.tasks.workflow import bind_request_context


def test_bind_request_context_sets_identity():
    structlog.contextvars.clear_contextvars()
    bind_request_context(workflow_id="wf-1", org_id="org-A", user_id="user-1")
    ctx = structlog.contextvars.get_contextvars()
    assert ctx["workflow_id"] == "wf-1"
    assert ctx["org_id"] == "org-A"
    assert ctx["user_id"] == "user-1"
    structlog.contextvars.clear_contextvars()
