# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-06

### Added
- Initial release of browser-service package
- AI-powered element identification using browser-use
- Multiple locator strategies (ID, data-testid, name, aria-label, CSS, XPath)
- Playwright-based locator validation
- Unified workflow mode for browser sessions
- Custom actions for smart locator finding
- Comprehensive configuration management
- Browser session lifecycle management
- Locator generation and validation utilities
- Task processing and workflow execution
- API route registration and handlers
- Metrics and logging utilities
- Support for Robot Framework Browser and Selenium libraries

### Changed
- Extracted from monolithic browser_use_service.py
- Modularized into separate components for better maintainability

### Fixed
- Windows UTF-8 compatibility issues
- Resource cleanup on workflow completion

## [Unreleased]

### Fixed

- Workflow cost metrics are no longer always zero. `record_workflow_metrics`
  read `estimated_total_cost`/`estimated_cost_per_element`, keys no producer
  emits — the summary carries `actual_cost` — so every run persisted
  `total_cost=0.0`. Per-element cost is derived with the backend's own formula.
- `custom_action_usage_count` no longer counts elements that produced no
  result. Backfilled failures stamped `metrics.custom_action_used` from the
  custom-actions FLAG rather than from what happened, so a 2-element run
  resolving 1 reported 2 custom-action uses.
- Caller-controlled values in the workflow-submission log lines
  (`parent_workflow_id`, `url`, `user_query`) are flattened before logging. A
  newline in the JSON body reached the log record verbatim and could forge a
  log entry (CWE-117).
- Workflow `success` is now false when a requested element has no locator.
  It was measured against the elements the agent reported rather than the
  elements that were requested, so a run that found 1 of 2 reported success.
- Elements the agent never reported, or reported as not found, are now
  returned as `found: false` results and counted in `summary.failed`,
  carrying the locator engine's own reason (`error`, `error_type`,
  `semantic_match`, …) instead of a generic message.
- A payload claiming `found` with no `best_locator` is now rejected by every
  extraction path. The history-scan fallback accepted it, so it could be
  counted as a successful element whose locator was `None`.
- CDP URL lookup no longer aborts when a session property raises: the reads
  were guarded by `hasattr()`, which only swallows `AttributeError`, so a
  reset session's `cdp_client` skipped the remaining fallback strategies.

### Planned
- Additional locator strategies
- Enhanced error recovery
- Performance optimizations
- Extended documentation
- More comprehensive test coverage
