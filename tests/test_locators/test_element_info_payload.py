"""
Guards the element_info payload contract consumed by nlrf's identify agent.

nlrf's Task G state-verification assembler reads two element_info keys:
  - className   — the class list observed on the element at locate time
                  (captured AFTER the preceding workflow steps ran, so for
                  "click Save, verify the field shows an error" it is the
                  field in its error state; the error-family marker is
                  picked from this evidence)
  - ariaInvalid — the aria-invalid attribute value, for sites that mark
                  invalid fields via ARIA instead of a CSS class

element_info is assembled inline in smart_locator's result build (a
function too entangled with live Playwright objects to unit-test
directly), so these source-level guards keep the contract keys from
being silently dropped in a refactor.
"""

from pathlib import Path

import browser_service.locators.smart_locator as sl_mod

SRC = Path(sl_mod.__file__).read_text(encoding="utf-8")


class TestElementInfoPayloadContract:

    def test_class_name_in_payload(self):
        assert "'className': element_data['className']" in SRC

    def test_aria_invalid_in_payload(self):
        assert "'ariaInvalid': element_data.get('ariaInvalid', '')" in SRC
