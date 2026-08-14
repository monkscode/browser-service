"""
Tests that a custom-action result is logged as ONE record, not a line at a time.

_log_failure_result and _log_success_result rendered a terminal banner by
calling the logger once per line — separator rules, blank lines and each
indented detail field were separate log records at error/info level.

Measured 2026-08-14 in Loki, over the five days it held: the aggregate error
panel showed 445 lines, of which 284 (63.8%) were the decoration and body of
these banners rather than distinct errors. 24 real `CUSTOM ACTION FAILED`
events produced 264 of them. Every line carries the workflow_id, so the
per-run log panel showed the same 11-line block when debugging one run.

Nothing parses these lines — checked across bench/, src/, tests/ and tools/
in the consumer repo and across this one — so collapsing them breaks no
reader. The `=` rules and blank lines are dropped: they existed to delimit
consecutive single-line records, which is a job that disappears once the
whole result is one record.
"""

import logging

import pytest

from browser_service.agent.actions import (
    _log_failure_result,
    _log_success_result,
)


FAILURE_ARGS = ("elem_1", "login button on the page (action: click)", 432.0, 248.0)


class TestFailureLogging:

    def test_emits_exactly_one_record(self, caplog):
        with caplog.at_level(logging.ERROR):
            _log_failure_result(
                *FAILURE_ARGS,
                {"error": "Semantic mismatch: Expected 'Login'",
                 "validation_summary": {"total_generated": 4, "valid": 0,
                                        "not_found": 0, "not_unique": 0,
                                        "errors": 0}},
            )

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1, (
            f"one failure produced {len(errors)} error records; each one is a "
            f"separate line in Loki and a separate row in the error panel")

    def test_keeps_every_detail_field(self, caplog):
        with caplog.at_level(logging.ERROR):
            _log_failure_result(
                *FAILURE_ARGS,
                {"error": "Semantic mismatch: Expected 'Login'",
                 "validation_summary": {"total_generated": 4, "valid": 0,
                                        "not_found": 1, "not_unique": 2,
                                        "errors": 3}},
            )

        message = caplog.records[0].getMessage()
        for fragment in (
            "CUSTOM ACTION FAILED for elem_1",
            "Semantic mismatch: Expected 'Login'",
            "Element ID: elem_1",
            "login button on the page (action: click)",
            "(432.0, 248.0)",
            "total_strategies: 4",
            "not_found: 1",
            "not_unique: 2",
            "errors: 3",
        ):
            assert fragment in message, (
                f"collapsing the banner lost {fragment!r}; the detail has to "
                f"survive, only the line count changes")

    def test_no_blank_or_rule_lines(self, caplog):
        with caplog.at_level(logging.ERROR):
            _log_failure_result(*FAILURE_ARGS, {"error": "boom"})

        lines = caplog.records[0].getMessage().split("\n")
        assert all(line.strip() for line in lines), (
            "the collapsed record still contains a blank line")
        assert not any(set(line.strip()) == {"="} for line in lines), (
            "the collapsed record still contains a separator rule")

    def test_validation_summary_stays_optional(self, caplog):
        with caplog.at_level(logging.ERROR):
            _log_failure_result(*FAILURE_ARGS, {"error": "boom"})

        message = caplog.records[0].getMessage()
        assert len(caplog.records) == 1
        assert "Validation Summary" not in message, (
            "a result with no validation_summary must not render an empty one")
        assert "boom" in message


class TestSuccessLogging:
    """Same shape at INFO — 6,332 of the indented lines were this one."""

    def test_emits_exactly_one_record(self, caplog):
        with caplog.at_level(logging.INFO):
            _log_success_result(
                "elem_1",
                {"best_locator": "css=#login",
                 "validated": True, "count": 1, "unique": True, "valid": True,
                 "validation_method": "playwright",
                 "validation_summary": {"best_type": "css", "best_strategy": "id",
                                        "total_generated": 4, "valid": 1,
                                        "unique": 1, "not_found": 0,
                                        "not_unique": 0, "errors": 0}},
            )

        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(infos) == 1, (
            f"one success produced {len(infos)} info records")
        message = infos[0].getMessage()
        assert "CUSTOM ACTION SUCCEEDED for elem_1" in message
        assert "css=#login" in message
        assert "validation_method: playwright" in message
