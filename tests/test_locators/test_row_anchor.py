"""
Unit tests for row-anchored action locators (G1/G8 / Task B).

Data grids repeat identical action controls on every row (ASTPP audit:
`a[title="Edit"]` x12, `a.login_link` x12, copy icons with no semantics).
The only thing distinguishing "Edit customer 64625" from the other 11
Edit links is the ROW'S DATA. Today those candidates fail count()==1
and the cascade decays to `>> nth=N` — positional, silently clicking a
different customer when sort order or data changes.

The fix: the vision agent passes the row-identifying datum from the QA
step as `row_anchor_text`; when a candidate is non-unique, the engine
tries the row-scoped composite `tr:has-text("{anchor}") >> {candidate}`
(then `li:` for repeated list layouts) and emits it when exactly one
element matches. Anchored-on-QA-data is scored stable. Ambiguous
anchors (two rows match) fall through to today's behavior, flagged
`row_anchor_ambiguous` per the demote-never-delete policy.

Covered seams:
  - _upgrade_to_row_anchor(): the shared helper
  - _generate_locators_from_element_data(): STEP-0 candidate loop
  - _find_element_by_expected_text(): text-first (before nth fallback)
  - _find_element_by_description(): description fallback
  - _validate_strategy_candidates(): STEP-3 strategy validation
  - plumbing tripwires: params exist end-to-end (registration/actions)
"""

import inspect
from pathlib import Path

from browser_service.locators.smart_locator import (
    _derive_row_anchor_text_from_description,
    _find_element_by_description,
    _find_element_by_expected_text,
    _generate_locators_from_element_data,
    _should_treat_as_collection,
    _upgrade_to_row_anchor,
    _validate_strategy_candidates,
    candidate_targets_row_anchor,
    correct_expected_text_for_row_anchor,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeLocator:
    """Playwright Locator stand-in: count(), bounding_box() for the
    coordinate identity check, evaluate() for semantic validation,
    nth() for the disambiguator fallback, and evaluate_all() for the
    containment-chain probe (nested_chain models whether every match
    contains or is contained by the others)."""

    def __init__(
        self,
        count: int,
        text: str = "",
        box: dict = None,
        nth_boxes: list = None,
        nested_chain: bool = False,
    ):
        self._count = count
        self._text = text
        self._box = box
        self._nth_boxes = nth_boxes or []
        self._nested_chain = nested_chain

    async def count(self) -> int:
        return self._count

    async def bounding_box(self):
        return self._box

    async def evaluate_all(self, js: str):
        # Models the containment-chain probe: one nesting chain needs
        # >1 matches where every pair is in an ancestor relationship.
        return self._count > 1 and self._nested_chain

    def nth(self, i: int) -> "FakeLocator":
        box = self._nth_boxes[i] if i < len(self._nth_boxes) else None
        return FakeLocator(1, box=box)

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
        self.probed = []

    def locator(self, selector: str) -> FakeLocator:
        self.probed.append(selector)
        return self._locators[selector]


class LenientFakeContext(FakeSearchContext):
    """Unknown selectors count 0 — for paths that probe many selector
    shapes (description fallback, text-first)."""

    def locator(self, selector: str) -> FakeLocator:
        self.probed.append(selector)
        return self._locators.get(selector, FakeLocator(0))


ANCHOR = "64625"
EDIT = '[title="Edit"]'
TR_COMPOSITE = f'tr:has-text("{ANCHOR}") >> {EDIT}'
LI_COMPOSITE = f'li:has-text("{ANCHOR}") >> {EDIT}'
TR_VISIBLE = f"{TR_COMPOSITE} >> visible=true"
TR_COLLAPSED = f"{TR_VISIBLE} >> nth=0"


class TestUpgradeToRowAnchor:
    """The shared helper: emit the row-scoped composite only when it
    resolves to exactly one element (that vision saw, if coords given)."""

    async def test_tr_anchor_unique_returns_composite(self):
        ctx = FakeSearchContext({TR_COMPOSITE: FakeLocator(1)})
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR)
        assert result == {"locator": TR_COMPOSITE}

    async def test_ambiguous_two_rows_flagged(self):
        """Anchor matches two rows -> ambiguous, not a silent guess
        (Option 1: caller falls through to today's behavior, flagged)."""
        ctx = FakeSearchContext(
            {
                TR_COMPOSITE: FakeLocator(2),
                f"{TR_COMPOSITE} >> visible=true": FakeLocator(2),
            }
        )
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR)
        assert result == {"ambiguous": True}

    async def test_hidden_duplicate_row_rescued_by_visible(self):
        """A hidden template row (modal grid copy) also matching the
        anchor must not kill the upgrade — visible filter rescues."""
        ctx = FakeSearchContext(
            {
                TR_COMPOSITE: FakeLocator(2),
                f"{TR_COMPOSITE} >> visible=true": FakeLocator(1),
            }
        )
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR)
        assert result == {"locator": f"{TR_COMPOSITE} >> visible=true"}

    async def test_li_fallback_when_no_tr(self):
        """Repeated card/list layouts get the same fix via li."""
        ctx = FakeSearchContext(
            {
                TR_COMPOSITE: FakeLocator(0),
                LI_COMPOSITE: FakeLocator(1),
            }
        )
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR)
        assert result == {"locator": LI_COMPOSITE}

    async def test_xpath_candidate_skipped(self):
        """Chained-xpath scoping (relative vs document root) is
        engine-version-dependent, and the generated test runs under RF
        Browser's own Playwright — the upgrade must not even probe it."""
        ctx = FakeSearchContext({})  # any lookup would KeyError
        for xp in ('xpath=//a[@title="Edit"]', '//a[@title="Edit"]', '(//a[@title="Edit"])'):
            assert await _upgrade_to_row_anchor(ctx, xp, ANCHOR) is None
        assert ctx.probed == []

    async def test_empty_anchor_returns_none(self):
        ctx = FakeSearchContext({})
        assert await _upgrade_to_row_anchor(ctx, EDIT, "") is None
        assert await _upgrade_to_row_anchor(ctx, EDIT, "   ") is None
        assert await _upgrade_to_row_anchor(ctx, EDIT, None) is None
        assert ctx.probed == []

    async def test_quote_in_anchor_escaped(self):
        """Anchor data can carry double quotes (John "JJ" Smith)."""
        escaped = 'tr:has-text("John \\"JJ\\" Smith") >> [title="Edit"]'
        ctx = FakeSearchContext({escaped: FakeLocator(1)})
        result = await _upgrade_to_row_anchor(ctx, EDIT, 'John "JJ" Smith')
        assert result == {"locator": escaped}

    async def test_identity_check_rejects_wrong_row(self):
        """Unique match that is NOT the element vision clicked (anchor
        text present in a different row) must not be emitted."""
        far_box = {"x": 900.0, "y": 900.0, "width": 20.0, "height": 20.0}
        ctx = FakeSearchContext({TR_COMPOSITE: FakeLocator(1, box=far_box)})
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR, x=100, y=100)
        assert result is None

    async def test_identity_check_accepts_right_row(self):
        box = {"x": 90.0, "y": 90.0, "width": 40.0, "height": 40.0}
        ctx = FakeSearchContext({TR_COMPOSITE: FakeLocator(1, box=box)})
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR, x=100, y=100)
        assert result == {"locator": TR_COMPOSITE}

    async def test_probe_error_returns_none(self):
        ctx = FakeSearchContext({})  # KeyError inside -> swallowed
        assert await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR) is None


class TestContainmentCollapse:
    """One control wearing several name tags (ASTPP q02, live-verified
    2026-07-17): the site puts title="Edit" on the row's <a> AND on the
    icon <span> inside it, so the row-scoped chain counts 2 and the
    rescue declared AMBIGUOUS — every ASTPP grid action then shipped as
    the warning-flagged positional fallback ([title="Edit"] >> nth=18,
    which drifted from nth=16 overnight as rows were added). When ALL
    visible matches nest into one containment chain they are one
    control: collapse to `>> visible=true >> nth=0` — document order
    puts the parent first, so nth=0 is the outermost by structure, not
    by page order. Anything short of a full chain (second control in
    the row, second matching row) is genuine ambiguity and keeps the
    Option-1 flagged fallback untouched."""

    async def test_nested_pair_collapses_to_outermost(self):
        ctx = FakeSearchContext(
            {
                TR_COMPOSITE: FakeLocator(2),
                TR_VISIBLE: FakeLocator(2, nested_chain=True),
            }
        )
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR)
        assert result == {"locator": TR_COLLAPSED}

    async def test_triple_nesting_is_still_one_control(self):
        """td[title] > a[title] > span[title]: three name tags, one
        control — the chain check is pairwise, not pair-only."""
        ctx = FakeSearchContext(
            {
                TR_COMPOSITE: FakeLocator(3),
                TR_VISIBLE: FakeLocator(3, nested_chain=True),
            }
        )
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR)
        assert result == {"locator": TR_COLLAPSED}

    async def test_nested_pair_plus_separate_control_stays_ambiguous(self):
        """A second non-nested Edit control in the same row is genuine
        ambiguity — the collapse must not paper over it."""
        ctx = FakeSearchContext(
            {
                TR_COMPOSITE: FakeLocator(3),
                TR_VISIBLE: FakeLocator(3, nested_chain=False),
            }
        )
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR)
        assert result == {"ambiguous": True}

    async def test_two_rows_of_nested_pairs_stay_ambiguous(self):
        """Anchor matching two rows stays ambiguous even when each row
        carries the nested-title pattern (cross-row matches never nest)."""
        ctx = FakeSearchContext(
            {
                TR_COMPOSITE: FakeLocator(4),
                TR_VISIBLE: FakeLocator(4, nested_chain=False),
            }
        )
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR)
        assert result == {"ambiguous": True}

    async def test_hidden_template_duplicate_still_collapses(self):
        """Both ASTPP patterns at once: a hidden template row duplicates
        the anchor row AND the control is nested (raw count 4). The
        visible filter reduces to the one row's chain — collapse fires."""
        ctx = FakeSearchContext(
            {
                TR_COMPOSITE: FakeLocator(4),
                TR_VISIBLE: FakeLocator(2, nested_chain=True),
            }
        )
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR)
        assert result == {"locator": TR_COLLAPSED}

    async def test_all_matches_hidden_stays_ambiguous(self):
        """A chain with zero visible members must not be emitted — a
        locator resolving to hidden DOM fails every RF action."""
        ctx = FakeSearchContext(
            {
                TR_COMPOSITE: FakeLocator(2),
                TR_VISIBLE: FakeLocator(0, nested_chain=True),
            }
        )
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR)
        assert result == {"ambiguous": True}

    async def test_collapse_probe_error_stays_ambiguous(self):
        """DOM probe failure is not evidence of nesting — keep today's
        flagged fallback."""

        class ExplodingEvalLocator(FakeLocator):
            async def evaluate_all(self, js: str):
                raise RuntimeError("Execution context was destroyed")

        ctx = FakeSearchContext(
            {
                TR_COMPOSITE: FakeLocator(2),
                TR_VISIBLE: ExplodingEvalLocator(2),
            }
        )
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR)
        assert result == {"ambiguous": True}

    async def test_collapsed_composite_identity_rejects_wrong_element(self):
        """The A5 identity check still guards the collapsed composite:
        a chain that is not what vision saw must not be emitted."""
        far_box = {"x": 900.0, "y": 900.0, "width": 20.0, "height": 20.0}
        ctx = FakeSearchContext(
            {
                TR_COMPOSITE: FakeLocator(2),
                TR_VISIBLE: FakeLocator(2, nested_chain=True),
                TR_COLLAPSED: FakeLocator(1, box=far_box),
            }
        )
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR, x=100, y=100)
        assert result is None

    async def test_collapsed_composite_identity_accepts_right_element(self):
        box = {"x": 90.0, "y": 90.0, "width": 40.0, "height": 40.0}
        ctx = FakeSearchContext(
            {
                TR_COMPOSITE: FakeLocator(2),
                TR_VISIBLE: FakeLocator(2, nested_chain=True),
                TR_COLLAPSED: FakeLocator(1, box=box),
            }
        )
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR, x=100, y=100)
        assert result == {"locator": TR_COLLAPSED}

    async def test_li_nested_pair_collapses(self):
        """Card/list layouts get the same collapse via the li container."""
        li_visible = f"{LI_COMPOSITE} >> visible=true"
        ctx = FakeSearchContext(
            {
                TR_COMPOSITE: FakeLocator(0),
                LI_COMPOSITE: FakeLocator(2),
                li_visible: FakeLocator(2, nested_chain=True),
            }
        )
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR)
        assert result == {"locator": f"{li_visible} >> nth=0"}


class TestElementDataRowAnchor:
    """STEP-0: a row-repeated candidate (aria-label on every row's Edit
    button) must come back row-anchored with the flag in the payload."""

    @staticmethod
    def make_element_data(**overrides) -> dict:
        data = {
            "tagName": "a",
            "id": "",
            "name": "",
            "className": "",
            "textContent": "",
            "ariaLabel": "Edit",
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

    async def test_repeated_candidate_returns_row_composite(self):
        aria = '[aria-label="Edit"]'
        composite = f'tr:has-text("{ANCHOR}") >> {aria}'
        ctx = LenientFakeContext(
            {
                aria: FakeLocator(12),
                composite: FakeLocator(1),
            }
        )
        result = await _generate_locators_from_element_data(
            ctx,
            self.make_element_data(),
            "elem_1",
            "Edit link for customer 64625",
            row_anchor_text=ANCHOR,
        )
        assert result is not None and result["found"]
        assert result["best_locator"] == composite
        assert result["row_anchored"] is True
        assert result["stability"] == "stable"  # QA-specified data
        assert result["all_locators"][0]["row_anchored"] is True

    async def test_no_anchor_param_keeps_old_behavior(self):
        aria = '[aria-label="Edit"]'
        ctx = LenientFakeContext(
            {
                aria: FakeLocator(12),
                f"{aria} >> visible=true": FakeLocator(12),
            }
        )
        result = await _generate_locators_from_element_data(
            ctx,
            self.make_element_data(),
            "elem_1",
            "Edit link for customer 64625",
        )
        assert result is None  # unchanged pre-fix behavior

    async def test_collection_classification_suppressed_by_row_anchor(self):
        """ASTPP gate q02 r1 (2026-07-16): the CLASSIFIER voted 'collection'
        from description keywords ('... in the customer data table') and
        routed to the collection handler INSIDE STEP 0 — bypassing the
        STEP 0.5 trust-order gate entirely — even though the agent's call
        carried row_anchor_text AND is_collection=False. The payload was a
        bare 'tbody > tr' claimed found:true. A named row anchor must
        suppress the collection route here too (same trust order)."""
        from unittest.mock import patch

        anchor = "4727985745"
        aria = '[aria-label="Edit"]'
        composite = f'tr:has-text("{anchor}") >> {aria}'
        ctx = LenientFakeContext(
            {
                aria: FakeLocator(12),
                composite: FakeLocator(1),
            }
        )
        desc = "Edit action icon in the row containing 4727985745 in the customer data table"

        async def _collection_must_not_run(*args, **kwargs):
            raise AssertionError("collection handler must not run for a row-anchored request")

        with patch(
            "browser_service.locators.smart_locator._collection_handler.find_locator",
            side_effect=_collection_must_not_run,
        ):
            result = await _generate_locators_from_element_data(
                ctx,
                self.make_element_data(),
                "elem_4",
                desc,
                row_anchor_text=anchor,
            )
        assert result is not None and result["found"]
        assert result["best_locator"] == composite
        assert result["row_anchored"] is True

    async def test_nested_title_candidate_collapses_and_keeps_stable(self):
        """q02 shape through STEP 0: the repeated candidate's row chain
        double-counts one nested control — collapse must rescue it and
        the payload must keep the BASE candidate's stability (the
        structural nth=0 is not DOM-order; no cry-wolf warning)."""
        aria = '[aria-label="Edit"]'
        composite = f'tr:has-text("{ANCHOR}") >> {aria}'
        collapsed = f"{composite} >> visible=true >> nth=0"
        ctx = LenientFakeContext(
            {
                aria: FakeLocator(12),
                composite: FakeLocator(2),
                f"{composite} >> visible=true": FakeLocator(2, nested_chain=True),
            }
        )
        result = await _generate_locators_from_element_data(
            ctx,
            self.make_element_data(),
            "elem_1",
            "Edit link for customer 64625",
            row_anchor_text=ANCHOR,
        )
        assert result is not None and result["found"]
        assert result["best_locator"] == collapsed
        assert result["row_anchored"] is True
        assert result["stability"] == "stable"

    async def test_ambiguous_anchor_flag_rides_on_fallback_result(self):
        """Anchor matches two rows; the anchored candidate falls through
        but a later unique candidate carries row_anchor_ambiguous so
        nlrf can warn (Option 1)."""
        aria = '[aria-label="Edit"]'
        composite = f'tr:has-text("{ANCHOR}") >> {aria}'
        parent_css = "#actions-col a"
        ctx = LenientFakeContext(
            {
                aria: FakeLocator(12),
                composite: FakeLocator(2),
                f"{composite} >> visible=true": FakeLocator(2),
                f"{aria} >> visible=true": FakeLocator(12),
                parent_css: FakeLocator(1),
            }
        )
        result = await _generate_locators_from_element_data(
            ctx,
            self.make_element_data(parentId="actions-col"),
            "elem_1",
            "Edit link for customer 64625",
            row_anchor_text=ANCHOR,
        )
        assert result is not None and result["found"]
        assert result["best_locator"] == parent_css
        assert result.get("row_anchored") is None
        assert result["row_anchor_ambiguous"] is True


class TestTextFirstRowAnchor:
    """Text-first: row anchor is tried BEFORE the nth= coordinate
    disambiguation — the anchored composite must win when unique."""

    async def test_row_anchor_beats_nth_fallback(self):
        text_sel = 'text="Edit"'
        composite = f'tr:has-text("{ANCHOR}") >> {text_sel}'
        box = {"x": 90.0, "y": 90.0, "width": 40.0, "height": 40.0}
        ctx = LenientFakeContext(
            {
                text_sel: FakeLocator(12),
                composite: FakeLocator(1, box=box),
            }
        )
        result = await _find_element_by_expected_text(
            ctx,
            "Edit",
            "Edit link",
            x=100,
            y=100,
            row_anchor_text=ANCHOR,
        )
        assert result is not None
        assert result["locator"] == composite
        assert result["row_anchored"] is True

    async def test_ambiguous_anchor_falls_to_nth_with_flag(self):
        """Two rows match the anchor -> today's nth fallback, but the
        result is flagged so nlrf can warn (Option 1)."""
        text_sel = 'text="Edit"'
        composite = f'tr:has-text("{ANCHOR}") >> {text_sel}'
        boxes = [
            {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0},
            {"x": 95.0, "y": 95.0, "width": 10.0, "height": 10.0},
        ]
        ctx = LenientFakeContext(
            {
                text_sel: FakeLocator(2, nth_boxes=boxes),
                composite: FakeLocator(2),
                f"{composite} >> visible=true": FakeLocator(2),
                f"{text_sel} >> visible=true": FakeLocator(2, nth_boxes=boxes),
            }
        )
        result = await _find_element_by_expected_text(
            ctx,
            "Edit",
            "Edit link",
            x=100,
            y=100,
            row_anchor_text=ANCHOR,
        )
        assert result is not None
        assert "nth=1" in result["locator"]
        assert result["row_anchor_ambiguous"] is True

    async def test_nested_title_collapse_beats_nth_fallback(self):
        """q02's exact selector shape ([title="Edit"] from the text-first
        list): the nested pair must collapse instead of decaying to the
        coordinate nth fallback, and the base selector must ride along
        for honest stability classification."""
        title_sel = '[title="Edit"]'
        composite = f'tr:has-text("{ANCHOR}") >> {title_sel}'
        collapsed = f"{composite} >> visible=true >> nth=0"
        box = {"x": 90.0, "y": 90.0, "width": 40.0, "height": 40.0}
        ctx = LenientFakeContext(
            {
                title_sel: FakeLocator(12),
                composite: FakeLocator(2),
                f"{composite} >> visible=true": FakeLocator(2, nested_chain=True),
                collapsed: FakeLocator(1, box=box),
            }
        )
        result = await _find_element_by_expected_text(
            ctx,
            "Edit",
            "Edit icon",
            x=100,
            y=100,
            row_anchor_text=ANCHOR,
        )
        assert result is not None
        assert result["locator"] == collapsed
        assert result["row_anchored"] is True
        assert result["row_anchor_base"] == title_sel

    async def test_without_anchor_behavior_unchanged(self):
        text_sel = 'text="Edit"'
        boxes = [
            {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0},
            {"x": 95.0, "y": 95.0, "width": 10.0, "height": 10.0},
        ]
        ctx = LenientFakeContext(
            {
                text_sel: FakeLocator(2, nth_boxes=boxes),
                f"{text_sel} >> visible=true": FakeLocator(2, nth_boxes=boxes),
            }
        )
        result = await _find_element_by_expected_text(
            ctx,
            "Edit",
            "Edit link",
            x=100,
            y=100,
        )
        assert result is not None
        assert "nth=1" in result["locator"]
        assert "row_anchor_ambiguous" not in result


class TestDescriptionRowAnchor:
    """Description-derived selectors get the same rescue. The path
    returns a dict so the row-anchor provenance (and the base selector
    for honest stability classification) survives to the payload."""

    async def test_link_selector_upgrades(self):
        sel = 'role=link[name="Edit"]'
        composite = f'tr:has-text("{ANCHOR}") >> {sel}'
        ctx = LenientFakeContext(
            {
                sel: FakeLocator(12),
                composite: FakeLocator(1),
            }
        )
        result = await _find_element_by_description(
            ctx,
            "Edit link",
            row_anchor_text=ANCHOR,
        )
        assert result["locator"] == composite
        assert result["row_anchored"] is True
        assert result["row_anchor_base"] == sel

    async def test_nested_link_collapse_carries_base(self):
        """Collapse on the description path: the emitted dict must name
        the base selector so the payload site can keep its stability."""
        sel = 'role=link[name="Edit"]'
        composite = f'tr:has-text("{ANCHOR}") >> {sel}'
        collapsed = f"{composite} >> visible=true >> nth=0"
        ctx = LenientFakeContext(
            {
                sel: FakeLocator(12),
                composite: FakeLocator(2),
                f"{composite} >> visible=true": FakeLocator(2, nested_chain=True),
            }
        )
        result = await _find_element_by_description(
            ctx,
            "Edit link",
            row_anchor_text=ANCHOR,
        )
        assert result == {
            "locator": collapsed,
            "row_anchored": True,
            "row_anchor_base": sel,
        }


class TestStrategyValidatorRowAnchor:
    """STEP-3: a repeated title strategy validates as unique via the
    row composite instead of being recorded not-unique."""

    @staticmethod
    def make_strategies() -> list:
        return [
            {"type": "title", "locator": EDIT, "priority": 5, "strategy": "Title attribute"},
        ]

    async def test_repeated_title_upgrades(self):
        ctx = FakeSearchContext(
            {
                EDIT: FakeLocator(12),
                TR_COMPOSITE: FakeLocator(1),
            }
        )
        result = await _validate_strategy_candidates(
            ctx,
            self.make_strategies(),
            row_anchor_text=ANCHOR,
        )
        entry = result[0]
        assert entry["locator"] == TR_COMPOSITE
        assert entry["unique"] and entry["valid"]
        assert entry["row_anchored"] is True

    async def test_repeated_title_collapses_when_nested(self):
        """STEP-3 strategies get the collapse too — the nested-title
        pattern must validate unique instead of recording not-unique."""
        ctx = FakeSearchContext(
            {
                EDIT: FakeLocator(12),
                TR_COMPOSITE: FakeLocator(2),
                TR_VISIBLE: FakeLocator(2, nested_chain=True),
            }
        )
        result = await _validate_strategy_candidates(
            ctx,
            self.make_strategies(),
            row_anchor_text=ANCHOR,
        )
        entry = result[0]
        assert entry["locator"] == TR_COLLAPSED
        assert entry["unique"] and entry["valid"]
        assert entry["row_anchored"] is True

    async def test_ambiguous_anchor_entry_flagged_not_unique(self):
        ctx = FakeSearchContext(
            {
                EDIT: FakeLocator(12),
                TR_COMPOSITE: FakeLocator(2),
                f"{TR_COMPOSITE} >> visible=true": FakeLocator(2),
                f"{EDIT} >> visible=true": FakeLocator(12),
            }
        )
        result = await _validate_strategy_candidates(
            ctx,
            self.make_strategies(),
            row_anchor_text=ANCHOR,
        )
        entry = result[0]
        assert entry["locator"] == EDIT
        assert not entry["unique"]
        assert entry["row_anchor_ambiguous"] is True


class TestPlumbing:
    """row_anchor_text must exist end-to-end or the LLM can never pass
    it: custom-action param model -> actions wrapper -> locator engine
    -> prompt guidance."""

    def test_engine_signatures_accept_row_anchor(self):
        from browser_service.agent.actions import find_unique_locator_action
        from browser_service.locators.smart_locator import (
            find_unique_locator_at_coordinates,
        )

        for fn in (find_unique_locator_at_coordinates, find_unique_locator_action):
            assert "row_anchor_text" in inspect.signature(fn).parameters, (
                f"{fn.__name__} missing row_anchor_text"
            )

    def test_registration_declares_param(self):
        src = (REPO_ROOT / "browser_service" / "agent" / "registration.py").read_text(
            encoding="utf-8"
        )
        assert "row_anchor_text" in src

    def test_prompt_guidance_exists(self):
        src = (REPO_ROOT / "browser_service" / "prompts" / "templates.py").read_text(
            encoding="utf-8"
        )
        assert "row_anchor_text" in src

    def test_is_collection_guidance_excludes_per_row_actions(self):
        # F3 (2026-07-16): the is_collection bullet's own examples
        # ("table rows") misled the vision agent into flagging per-row
        # clicks as collections (bench q04 r3). The bullet must carry an
        # explicit exclusion pointing at row_anchor_text.
        src = (REPO_ROOT / "browser_service" / "prompts" / "templates.py").read_text(
            encoding="utf-8"
        )
        assert "NOT a collection" in src


class TestDeriveRowAnchorFromDescription:
    """When the caller (vision agent) leaves row_anchor_text unset, a
    per-row action description already names the anchor as a quoted
    phrase — recover it deterministically (bench q04 root cause: 'the
    edit link in the row containing Smith in the first data table')."""

    def test_row_containing_quoted_phrase(self):
        desc = "the 'edit' link in the row containing 'Smith' in the first data table"
        assert _derive_row_anchor_text_from_description(desc) == "Smith"

    def test_double_quotes(self):
        desc = 'the edit link in the row containing "Smith" in the first table'
        assert _derive_row_anchor_text_from_description(desc) == "Smith"

    def test_item_with_phrasing(self):
        desc = "the delete icon for the item with '64625'"
        assert _derive_row_anchor_text_from_description(desc) == "64625"

    def test_relative_clause_that_contains(self):
        # 2026-07-16 bench q04 r1: "row that contains 'Smith'" missed the
        # old pattern and decayed to positional nth=0 — passed by luck.
        desc = "the edit link in the row that contains 'Smith' in the first table"
        assert _derive_row_anchor_text_from_description(desc) == "Smith"

    def test_relative_clause_which_has(self):
        desc = "the checkbox in the row which has 'Pending'"
        assert _derive_row_anchor_text_from_description(desc) == "Pending"

    def test_intervening_noun_before_quoted_anchor(self):
        desc = "the Edit icon in the row for customer '64625'"
        assert _derive_row_anchor_text_from_description(desc) == "64625"

    def test_unquoted_numeric_anchor(self):
        # Grid ids (account numbers, invoice ids) are the canonical row
        # datum and users don't quote them in natural phrasing.
        desc = "the Edit icon in the row for customer 4727985745"
        assert _derive_row_anchor_text_from_description(desc) == "4727985745"

    def test_unquoted_numeric_anchor_without_noun(self):
        desc = "the download link in the row for 64625"
        assert _derive_row_anchor_text_from_description(desc) == "64625"

    def test_plural_head_noun_is_not_an_anchor(self):
        # "all rows with X" is a filtered COLLECTION, not one row's action.
        desc = "all rows with 'pending' status in the data table"
        assert _derive_row_anchor_text_from_description(desc) is None

    def test_every_determiner_is_not_an_anchor(self):
        # "every row containing 'X'" iterates a filtered COLLECTION despite
        # the singular head noun — the plural rejection can't see it.
        desc = "every row containing 'Chennai' in the results table"
        assert _derive_row_anchor_text_from_description(desc) is None

    def test_each_determiner_is_not_an_anchor(self):
        desc = "the checkbox in each row containing 'pending'"
        assert _derive_row_anchor_text_from_description(desc) is None

    def test_single_digit_count_is_not_an_anchor(self):
        # "the row with 3 columns" names a shape, not a row datum.
        desc = "the row with 3 columns"
        assert _derive_row_anchor_text_from_description(desc) is None

    def test_unquoted_word_is_not_an_anchor(self):
        # Only digit-leading tokens qualify unquoted — bare words are
        # too loose ("the row for editing").
        desc = "the row for editing"
        assert _derive_row_anchor_text_from_description(desc) is None

    def test_no_anchor_phrase_returns_none(self):
        assert _derive_row_anchor_text_from_description("the Submit button") is None

    def test_none_description_returns_none(self):
        assert _derive_row_anchor_text_from_description(None) is None

    def test_empty_description_returns_none(self):
        assert _derive_row_anchor_text_from_description("") is None


class TestShouldTreatAsCollection:
    """STEP 0.5 gate trust order: a named row anchor (explicit param or
    description-derived) wins over is_collection=True. REVERSES the
    original explicit-flag-wins contract on 2026-07-16 evidence (owner
    approved): bench q04 r3's vision call set is_collection=true on a
    per-row click and omitted row_anchor_text; the collection path then
    returned a bare 'tbody > tr' with found:true and the assembler
    improvised an unusable locator. An anchor names ONE row — it is
    strictly more specific than the hallucination-prone collection flag
    (8/12 real improvisations in the 507-run scan are this chain).
    Genuine collections ('get all book titles', 'all rows with X') never
    derive a singular row anchor, so they keep the collection path."""

    def test_row_anchor_overrides_explicit_collection(self):
        assert _should_treat_as_collection(True, "irrelevant", "Smith") is False

    def test_explicit_collection_without_anchor_still_wins(self):
        assert _should_treat_as_collection(True, "all book titles on the page", None) is True

    def test_explicit_false_is_not_true_but_keyword_may_still_apply(self):
        # is_collection=False is not the same as "unset" — but the keyword
        # fallback still applies when nothing overrides it.
        desc = "all rows in the data table"
        assert _should_treat_as_collection(False, desc, None) is True

    def test_row_anchor_suppresses_keyword_fallback(self):
        desc = "the 'edit' link in the row containing 'Smith' in the first data table"
        assert _should_treat_as_collection(None, desc, "Smith") is False

    def test_keyword_fallback_fires_without_row_anchor(self):
        desc = "the 'edit' link in the row containing 'Smith' in the first data table"
        assert _should_treat_as_collection(None, desc, None) is True

    def test_no_description_no_row_anchor_not_collection(self):
        assert _should_treat_as_collection(None, None, None) is False

    def test_unrelated_description_not_collection(self):
        assert _should_treat_as_collection(None, "the Submit button", None) is False


class TestCorrectExpectedTextForRowAnchor:
    """ASTPP gate q02 (2026-07-16, 5 of the 6 recorded elem_4 runs): the
    vision agent set expected_text to the ROW ANCHOR DATUM ('4727985745')
    while the indexed element's own DOM text was 'Edit'. TEXT-FIRST then
    built text="4727985745" — UNIQUE on the page because it is the account
    CELL — so count==1, the row-anchor rescue never fired, and the cell
    shipped as the Edit-icon locator in every gate-r2 test.robot.

    The correction: when expected_text IS the anchor datum and the DOM
    evidence disagrees, trust the DOM — search with the element's own
    text. That text repeats across rows (>1 matches), which is exactly
    the condition the existing row-anchor rescue needs to emit the
    tr:has-text composite."""

    Q02_DATA = {"tagName": "a", "textContent": "Edit"}

    def test_q02_replaces_anchor_datum_with_element_text(self):
        text, corrected = correct_expected_text_for_row_anchor(
            "4727985745", "4727985745", self.Q02_DATA
        )
        assert (text, corrected) == ("Edit", True)

    def test_without_anchor_unchanged(self):
        text, corrected = correct_expected_text_for_row_anchor("4727985745", None, self.Q02_DATA)
        assert (text, corrected) == ("4727985745", False)

    def test_expected_differs_from_anchor_unchanged(self):
        # Healthy agent: expected_text already names the control.
        text, corrected = correct_expected_text_for_row_anchor("Edit", "4727985745", self.Q02_DATA)
        assert (text, corrected) == ("Edit", False)

    def test_element_text_equals_anchor_unchanged(self):
        # Legit "click the 4727985745 link" — the element IS the datum.
        data = {"tagName": "a", "textContent": "4727985745"}
        text, corrected = correct_expected_text_for_row_anchor("4727985745", "4727985745", data)
        assert (text, corrected) == ("4727985745", False)

    def test_no_element_data_unchanged(self):
        # No DOM evidence of a mismatch — never correct on speculation.
        text, corrected = correct_expected_text_for_row_anchor("4727985745", "4727985745", None)
        assert (text, corrected) == ("4727985745", False)

    def test_empty_element_text_unchanged(self):
        # Icon-only control: absence of text is not positive evidence.
        data = {"tagName": "a", "textContent": ""}
        text, corrected = correct_expected_text_for_row_anchor("4727985745", "4727985745", data)
        assert (text, corrected) == ("4727985745", False)

    def test_no_expected_text_unchanged(self):
        text, corrected = correct_expected_text_for_row_anchor(None, "4727985745", self.Q02_DATA)
        assert (text, corrected) == (None, False)

    def test_whitespace_collapsed_on_all_sides(self):
        data = {"tagName": "a", "textContent": "  Edit\n "}
        text, corrected = correct_expected_text_for_row_anchor(" 4727985745 ", "4727985745", data)
        assert (text, corrected) == ("Edit", True)

    def test_text_key_fallback(self):
        # element_data producers vary: some emit 'text', not 'textContent'.
        data = {"tagName": "a", "text": "Edit"}
        text, corrected = correct_expected_text_for_row_anchor("4727985745", "4727985745", data)
        assert (text, corrected) == ("Edit", True)

    def test_idempotent_after_correction(self):
        # actions.py corrects, then the engine calls again — no-op.
        text, corrected = correct_expected_text_for_row_anchor("Edit", "4727985745", self.Q02_DATA)
        assert (text, corrected) == ("Edit", False)


class TestCandidateTargetsRowAnchor:
    """The candidate-accept guard's predicate: does the agent's candidate
    locator target the anchor datum itself (the cell), rather than the
    row's control? Anchor text in a row-SCOPING segment before '>>' is
    legitimate and must not fire."""

    A = "4727985745"

    def test_bare_text_engine_locator(self):
        assert candidate_targets_row_anchor(f"text={self.A}", self.A) is True

    def test_quoted_text_engine_locator(self):
        assert candidate_targets_row_anchor(f'text="{self.A}"', self.A) is True

    def test_attribute_selector(self):
        assert candidate_targets_row_anchor(f"[title='{self.A}']", self.A) is True

    def test_has_text_terminal(self):
        assert candidate_targets_row_anchor(f'td:has-text("{self.A}")', self.A) is True

    def test_row_scoped_composite_not_flagged(self):
        assert (
            candidate_targets_row_anchor(f'tr:has-text("{self.A}") >> a[title="Edit"]', self.A)
            is False
        )

    def test_control_candidate_not_flagged(self):
        assert candidate_targets_row_anchor('a[title="Edit"]', self.A) is False

    def test_id_candidate_not_flagged(self):
        assert candidate_targets_row_anchor("id=username", self.A) is False

    def test_no_anchor_never_fires(self):
        assert candidate_targets_row_anchor(f"text={self.A}", None) is False

    def test_no_candidate_never_fires(self):
        assert candidate_targets_row_anchor(None, self.A) is False


class TestAnchorCorrectionEngineFlow:
    """The correction wired into find_unique_locator_at_coordinates must
    also cover the DERIVED-anchor path (agent omitted row_anchor_text but
    the description names the row) — actions.py cannot correct there
    because it never sees the derived anchor."""

    async def test_derived_anchor_corrects_and_row_scopes(self):
        from unittest.mock import MagicMock

        from browser_service.locators.smart_locator import (
            find_unique_locator_at_coordinates,
        )

        anchor = "4727985745"
        text_sel = 'text="Edit"'
        composite = f'tr:has-text("{anchor}") >> {text_sel}'
        box = {"x": 480.0, "y": 670.0, "width": 30.0, "height": 20.0}
        ctx = LenientFakeContext(
            {
                text_sel: FakeLocator(9),
                composite: FakeLocator(1, box=box),
            }
        )
        result = await find_unique_locator_at_coordinates(
            page=MagicMock(),
            x=494,
            y=682,
            element_id="elem_4",
            # Verbatim from the first-gate r1 trace — a phrasing the F2
            # derivation regex recovers. (Two other real q02 phrasings,
            # 'row containing customer ID 4727985745' and 'row
            # corresponding to customer 4727985745', do NOT derive —
            # known F2 gap, reported separately; the agent passed
            # row_anchor_text explicitly in all six observed calls.)
            element_description=(
                "Edit action icon in the row containing 4727985745 in the customer data table"
            ),
            expected_text="4727985745",
            element_data={
                "tagName": "a",
                "id": "",
                "className": "",
                "textContent": "Edit",
                "xpath": "html/body/main/div[3]/div/div/div[2]/table/tbody/tr[9]/td[2]/div/a[1]",
            },
            search_context=ctx,
        )
        assert result["found"] is True
        assert result["best_locator"] == composite
        assert result["best_locator"] != 'text="4727985745"'  # the q02 bug
        assert result.get("row_anchored") is True


class TestCollapsedPayloadStability:
    """The collapse suffix `>> visible=true >> nth=0` is structural
    (parent-first document order within one row's chain), NOT
    DOM-order-dependent — the payload must keep the BASE selector's
    stability so nlrf's positional WARNING comment doesn't cry wolf on
    every ASTPP grid action. Genuinely positional results (ambiguous
    anchor -> nth fallback) keep their honest positional label — that
    path never sets row_anchored."""

    ELEMENT_DATA = {
        "tagName": "a",
        "id": "",
        "className": "",
        "textContent": "Edit",
        "xpath": "html/body/main/div[3]/div/div/div[2]/table/tbody/tr[9]/td[2]/div/a[1]",
    }

    async def test_text_first_collapsed_payload_keeps_base_stability(self):
        from unittest.mock import MagicMock

        from browser_service.locators.smart_locator import (
            find_unique_locator_at_coordinates,
        )

        anchor = "55516"
        title_sel = '[title="Edit"]'
        composite = f'tr:has-text("{anchor}") >> {title_sel}'
        collapsed = f"{composite} >> visible=true >> nth=0"
        box = {"x": 480.0, "y": 670.0, "width": 30.0, "height": 20.0}
        ctx = LenientFakeContext(
            {
                title_sel: FakeLocator(12),
                composite: FakeLocator(2),
                f"{composite} >> visible=true": FakeLocator(2, nested_chain=True),
                collapsed: FakeLocator(1, box=box),
            }
        )
        result = await find_unique_locator_at_coordinates(
            page=MagicMock(),
            x=494,
            y=682,
            element_id="elem_4",
            element_description=(
                "Edit action icon in the row containing 55516 in the customer data table"
            ),
            expected_text="Edit",
            element_data=self.ELEMENT_DATA,
            search_context=ctx,
            row_anchor_text=anchor,
        )
        assert result["found"] is True
        assert result["best_locator"] == collapsed
        assert result.get("row_anchored") is True
        assert result["stability"] == "stable"
        assert result["all_locators"][0]["stability"] == "stable"

    async def test_semantic_collapsed_payload_keeps_base_stability(self):
        from unittest.mock import MagicMock

        from browser_service.locators.smart_locator import (
            find_unique_locator_at_coordinates,
        )

        sel = 'role=link[name="Edit"]'
        composite = f'tr:has-text("{ANCHOR}") >> {sel}'
        collapsed = f"{composite} >> visible=true >> nth=0"
        ctx = LenientFakeContext(
            {
                sel: FakeLocator(12),
                composite: FakeLocator(2),
                f"{composite} >> visible=true": FakeLocator(2, nested_chain=True),
            }
        )
        result = await find_unique_locator_at_coordinates(
            page=MagicMock(),
            x=100,
            y=100,
            element_id="elem_2",
            element_description="Edit link",
            search_context=ctx,
            row_anchor_text=ANCHOR,
        )
        assert result["found"] is True
        assert result["best_locator"] == collapsed
        assert result.get("row_anchored") is True
        assert result["stability"] == "stable"

    async def test_iframe_stable_hop_keeps_base_stability(self):
        """B2 refinement: the iframe override must not relabel a
        row-anchored collapse positional when the HOP itself is stable
        — only an ordinal hop encodes DOM order for these results."""
        from unittest.mock import MagicMock

        from browser_service.locators.smart_locator import (
            find_unique_locator_at_coordinates,
        )

        aria = '[aria-label="Edit"]'
        composite = f'tr:has-text("{ANCHOR}") >> {aria}'
        collapsed = f"{composite} >> visible=true >> nth=0"
        box = {"x": 90.0, "y": 90.0, "width": 40.0, "height": 40.0}
        ctx = LenientFakeContext(
            {
                aria: FakeLocator(12),
                composite: FakeLocator(2),
                f"{composite} >> visible=true": FakeLocator(2, nested_chain=True),
                collapsed: FakeLocator(1, box=box),
            }
        )
        element_data = TestElementDataRowAnchor.make_element_data()
        result = await find_unique_locator_at_coordinates(
            page=MagicMock(),
            x=100,
            y=100,
            element_id="elem_1",
            element_description="Edit link for customer 64625",
            element_data=element_data,
            search_context=ctx,
            iframe_context='iframe[id="app"]',
            row_anchor_text=ANCHOR,
        )
        assert result["found"] is True
        assert result["best_locator"] == f'iframe[id="app"] >>> {collapsed}'
        assert result["stability"] == "stable"

    async def test_text_first_iframe_ordinal_hop_is_positional(self):
        """Same B2 rule on the TEXT-FIRST payload site: the base-selector
        exemption covers the collapse suffix only, never an ordinal
        iframe hop — the whole composite re-resolves the Nth iframe."""
        from unittest.mock import MagicMock

        from browser_service.locators.smart_locator import (
            find_unique_locator_at_coordinates,
        )

        anchor = "55516"
        title_sel = '[title="Edit"]'
        composite = f'tr:has-text("{anchor}") >> {title_sel}'
        collapsed = f"{composite} >> visible=true >> nth=0"
        box = {"x": 480.0, "y": 670.0, "width": 30.0, "height": 20.0}
        ctx = LenientFakeContext(
            {
                title_sel: FakeLocator(12),
                composite: FakeLocator(2),
                f"{composite} >> visible=true": FakeLocator(2, nested_chain=True),
                collapsed: FakeLocator(1, box=box),
            }
        )
        result = await find_unique_locator_at_coordinates(
            page=MagicMock(),
            x=494,
            y=682,
            element_id="elem_5",
            element_description=(
                "Edit action icon in the row containing 55516 in the customer data table"
            ),
            expected_text="Edit",
            element_data=self.ELEMENT_DATA,
            search_context=ctx,
            iframe_context="iframe >> nth=1",
            row_anchor_text=anchor,
        )
        assert result["found"] is True
        assert result["best_locator"] == f"iframe >> nth=1 >>> {collapsed}"
        assert result["stability"] == "positional"

    async def test_semantic_iframe_ordinal_hop_is_positional(self):
        """Same B2 rule on the SEMANTIC (description) payload site."""
        from unittest.mock import MagicMock

        from browser_service.locators.smart_locator import (
            find_unique_locator_at_coordinates,
        )

        sel = 'role=link[name="Edit"]'
        composite = f'tr:has-text("{ANCHOR}") >> {sel}'
        collapsed = f"{composite} >> visible=true >> nth=0"
        ctx = LenientFakeContext(
            {
                sel: FakeLocator(12),
                composite: FakeLocator(2),
                f"{composite} >> visible=true": FakeLocator(2, nested_chain=True),
            }
        )
        result = await find_unique_locator_at_coordinates(
            page=MagicMock(),
            x=100,
            y=100,
            element_id="elem_6",
            element_description="Edit link",
            search_context=ctx,
            iframe_context="iframe >> nth=1",
            row_anchor_text=ANCHOR,
        )
        assert result["found"] is True
        assert result["best_locator"] == f"iframe >> nth=1 >>> {collapsed}"
        assert result["stability"] == "positional"

    async def test_iframe_ordinal_hop_still_positional(self):
        """An ordinal iframe hop DOES encode DOM order — the B2
        override must keep firing for row-anchored results too."""
        from unittest.mock import MagicMock

        from browser_service.locators.smart_locator import (
            find_unique_locator_at_coordinates,
        )

        aria = '[aria-label="Edit"]'
        composite = f'tr:has-text("{ANCHOR}") >> {aria}'
        collapsed = f"{composite} >> visible=true >> nth=0"
        box = {"x": 90.0, "y": 90.0, "width": 40.0, "height": 40.0}
        ctx = LenientFakeContext(
            {
                aria: FakeLocator(12),
                composite: FakeLocator(2),
                f"{composite} >> visible=true": FakeLocator(2, nested_chain=True),
                collapsed: FakeLocator(1, box=box),
            }
        )
        element_data = TestElementDataRowAnchor.make_element_data()
        result = await find_unique_locator_at_coordinates(
            page=MagicMock(),
            x=100,
            y=100,
            element_id="elem_1",
            element_description="Edit link for customer 64625",
            element_data=element_data,
            search_context=ctx,
            iframe_context="iframe >> nth=1",
            row_anchor_text=ANCHOR,
        )
        assert result["found"] is True
        assert result["stability"] == "positional"
