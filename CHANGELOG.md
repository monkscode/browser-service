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

### Planned
- Additional locator strategies
- Enhanced error recovery
- Performance optimizations
- Extended documentation
- More comprehensive test coverage
