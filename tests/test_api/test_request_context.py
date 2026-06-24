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


def test_bind_request_context_does_not_leak_prior_identity():
    # Workers run on a reused ThreadPoolExecutor thread whose contextvars persist
    # across tasks. A later task missing org_id/user_id must NOT inherit the prior
    # task's identity — bind_request_context clears before binding.
    structlog.contextvars.clear_contextvars()
    bind_request_context(workflow_id="wf-A", org_id="org-A", user_id="user-A")
    bind_request_context(workflow_id="wf-B")  # next task on the same thread, no org/user
    ctx = structlog.contextvars.get_contextvars()
    assert ctx["workflow_id"] == "wf-B"
    assert "org_id" not in ctx
    assert "user_id" not in ctx
    structlog.contextvars.clear_contextvars()
