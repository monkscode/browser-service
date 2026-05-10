"""
Fixtures for test_interaction_reliability.py.

performed_actions and element_specs are plain objects constructed per-test.
No monkeypatching needed — helpers accept these as explicit parameters.
"""

import pytest


@pytest.fixture
def performed_actions():
    """Fresh performed_actions set for each test — mirrors the per-workflow closure."""
    return set()
