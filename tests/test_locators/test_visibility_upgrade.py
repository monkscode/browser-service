"""
Unit tests for visibility-aware uniqueness (G2 / Task A).

Sites keep hidden template DOM in the page permanently (closed Bootstrap
modals with duplicated ids, mobile/desktop dual navs). The raw
count()==1 uniqueness check counted those hidden twins, killed the best
candidates (a duplicated id dies at Priority 1), and the cascade decayed
to parent-CSS/positional locators. ASTPP audit: up to 19 duplicated ids
per page, `#export` x2, `#global-popup-delete-button` x2.

The fix: when a candidate matches >1 elements but exactly one is
visible, upgrade it to `{locator} >> visible=true` instead of
discarding. Covered seams:
  - _upgrade_to_visible_only(): the shared helper
  - _validate_strategy_candidates(): STEP-3 strategy validation
  - _generate_locators_from_element_data(): STEP-0 candidate loop
  - _find_element_by_description(): description fallback
  - _find_element_by_expected_text(): text-first without coordinates
"""

from browser_service.locators.smart_locator import (
    _find_element_by_description,
    _find_element_by_expected_text,
    _generate_locators_from_element_data,
    _upgrade_to_visible_only,
    _validate_strategy_candidates,
)


class FakeLocator:
    """Playwright Locator stand-in: count() plus the element-info
    evaluate() that validate_semantic_match's fallback path runs."""

    def __init__(self, count: int, text: str = ""):
        self._count = count
        self._text = text

    async def count(self) -> int:
        return self._count

    async def evaluate(self, js: str) -> dict:
        return {
            "textContent": self._text,
            "textContentLength": len(self._text),
            "innerText": self._text,
            "placeholder": "",
            "ariaLabel": "",
            "value": "",
            "labelText": "",
            "labelledbyText": "",
        }


class FakeSearchContext:
    """Maps selector string -> FakeLocator; unknown selectors raise
    KeyError so unexpected lookups fail the test loudly."""

    def __init__(self, locators: dict):
        self._locators = locators

    def locator(self, selector: str) -> FakeLocator:
        return self._locators[selector]


class LenientFakeContext(FakeSearchContext):
    """Unknown selectors count 0 — for paths that probe many selector
    shapes (description fallback, text-first)."""

    def locator(self, selector: str) -> FakeLocator:
        return self._locators.get(selector, FakeLocator(0))


class TestUpgradeToVisibleOnly:
    """The shared helper: upgrade applies only on exactly-one-visible."""

    async def test_one_visible_returns_composite(self):
        ctx = FakeSearchContext({"#export >> visible=true": FakeLocator(1)})
        assert await _upgrade_to_visible_only(ctx, "#export") == "#export >> visible=true"

    async def test_two_visible_returns_none(self):
        ctx = FakeSearchContext({"#export >> visible=true": FakeLocator(2)})
        assert await _upgrade_to_visible_only(ctx, "#export") is None

    async def test_zero_visible_returns_none(self):
        """Target itself hidden (e.g. modal closed at validation time) —
        no safe upgrade; behavior stays as before the fix."""
        ctx = FakeSearchContext({"#export >> visible=true": FakeLocator(0)})
        assert await _upgrade_to_visible_only(ctx, "#export") is None

    async def test_probe_error_returns_none(self):
        ctx = FakeSearchContext({})  # KeyError inside -> swallowed
        assert await _upgrade_to_visible_only(ctx, "#export") is None

    async def test_engine_agnostic_chaining(self):
        """text= and xpath= candidates take the same composite form —
        the genericity requirement (works beyond ASTPP/CSS ids)."""
        ctx = FakeSearchContext(
            {
                'text="Save" >> visible=true': FakeLocator(1),
                'xpath=//a[@title="Edit"] >> visible=true': FakeLocator(1),
            }
        )
        assert await _upgrade_to_visible_only(ctx, 'text="Save"') == 'text="Save" >> visible=true'
        assert (
            await _upgrade_to_visible_only(ctx, 'xpath=//a[@title="Edit"]')
            == 'xpath=//a[@title="Edit"] >> visible=true'
        )


class TestStrategyValidatorUpgrade:
    """STEP-3: a duplicated-id strategy must validate as unique via the
    visible-only composite instead of being recorded not-unique."""

    @staticmethod
    def make_strategies() -> list:
        return [
            {"type": "id", "locator": "#export", "priority": 1, "strategy": "Native ID attribute"},
            {
                "type": "text",
                "locator": 'text="Export"',
                "priority": 6,
                "strategy": "Visible text content",
            },
        ]

    async def test_hidden_duplicate_id_upgrades_and_early_exits(self):
        ctx = FakeSearchContext(
            {
                "#export": FakeLocator(2),
                "#export >> visible=true": FakeLocator(1, "Export"),
                'text="Export"': FakeLocator(2),
            }
        )
        result = await _validate_strategy_candidates(
            ctx, self.make_strategies(), expected_text="Export"
        )
        # Upgraded priority-1 id is unique + semantically right -> early exit
        assert len(result) == 1
        entry = result[0]
        assert entry["locator"] == "#export >> visible=true"
        assert entry["unique"] and entry["valid"]
        assert entry["count"] == 1
        assert entry["visibility_filtered"] is True

    async def test_multiple_visible_stays_not_unique(self):
        """Two VISIBLE matches: genuinely ambiguous — no upgrade."""
        ctx = FakeSearchContext(
            {
                "#export": FakeLocator(2),
                "#export >> visible=true": FakeLocator(2),
                'text="Export"': FakeLocator(0),
            }
        )
        result = await _validate_strategy_candidates(ctx, self.make_strategies())
        id_entry = [r for r in result if r["type"] == "id"][0]
        assert id_entry["locator"] == "#export"
        assert not id_entry["unique"]
        assert id_entry["count"] == 2
        assert "visibility_filtered" not in id_entry

    async def test_naturally_unique_candidate_untouched(self):
        ctx = FakeSearchContext(
            {
                "#export": FakeLocator(1, "Export"),
                'text="Export"': FakeLocator(1, "Export"),
            }
        )
        result = await _validate_strategy_candidates(
            ctx, self.make_strategies(), expected_text="Export"
        )
        assert result[0]["locator"] == "#export"
        assert "visibility_filtered" not in result[0]


class TestElementDataLoopUpgrade:
    """STEP-0: the duplicated-id candidate from element_data must come
    back as a visible-only composite with the flag in the payload."""

    @staticmethod
    def make_element_data(**overrides) -> dict:
        data = {
            "tagName": "button",
            "id": "export",
            "name": "",
            "className": "",
            "textContent": "Export",
            "ariaLabel": "",
            "placeholder": "",
            "title": "",
            "role": "",
            "dataTestId": "",
            "type": "",
            "xpath": "",
            "coordinates": {"x": 100, "y": 100},
            "parentId": "",
            "parentClass": "",
        }
        data.update(overrides)
        return data

    async def test_duplicated_id_returns_visible_composite(self):
        ctx = FakeSearchContext(
            {
                "#export": FakeLocator(2),
                "#export >> visible=true": FakeLocator(1, "Export"),
            }
        )
        result = await _generate_locators_from_element_data(
            ctx,
            self.make_element_data(),
            "elem_1",
            "Export button",
            expected_text="Export",
        )
        assert result is not None and result["found"]
        assert result["best_locator"] == "#export >> visible=true"
        assert result["visibility_filtered"] is True
        assert result["stability"] == "stable"  # base id stability kept
        assert result["all_locators"][0]["visibility_filtered"] is True

    async def test_all_copies_hidden_falls_through(self):
        ctx = FakeSearchContext(
            {
                "#export": FakeLocator(2),
                "#export >> visible=true": FakeLocator(0),
            }
        )
        result = await _generate_locators_from_element_data(
            ctx,
            self.make_element_data(),
            "elem_1",
            "Export button",
            expected_text="Export",
        )
        assert result is None  # unchanged pre-fix behavior


class TestDescriptionFallbackUpgrade:
    """Description-derived selectors get the same rescue."""

    async def test_text_selector_upgrades(self):
        ctx = LenientFakeContext(
            {
                'text="Save"': FakeLocator(2),
                'text="Save" >> visible=true': FakeLocator(1, "Save"),
            }
        )
        result = await _find_element_by_description(ctx, "Save button")
        assert result == {"locator": 'text="Save" >> visible=true'}


class TestTextFirstNoCoordsUpgrade:
    """Text-first without coordinates: with coords the existing
    disambiguator already applies a visible filter; without coords the
    upgrade is the only rescue."""

    async def test_upgrade_without_coordinates(self):
        ctx = LenientFakeContext(
            {
                'text="Save"': FakeLocator(2),
                'text="Save" >> visible=true': FakeLocator(1, "Save"),
            }
        )
        result = await _find_element_by_expected_text(
            ctx,
            "Save",
            "Save button",
            x=None,
            y=None,
        )
        assert result is not None
        assert result["locator"] == 'text="Save" >> visible=true'
