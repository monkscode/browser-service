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
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeLocator:
    """Playwright Locator stand-in: count(), bounding_box() for the
    coordinate identity check, evaluate() for semantic validation, and
    nth() for the disambiguator fallback."""

    def __init__(self, count: int, text: str = '', box: dict = None,
                 nth_boxes: list = None):
        self._count = count
        self._text = text
        self._box = box
        self._nth_boxes = nth_boxes or []

    async def count(self) -> int:
        return self._count

    async def bounding_box(self):
        return self._box

    def nth(self, i: int) -> 'FakeLocator':
        box = self._nth_boxes[i] if i < len(self._nth_boxes) else None
        return FakeLocator(1, box=box)

    async def evaluate(self, js: str) -> dict:
        return {
            'textContent': self._text,
            'textContentLength': len(self._text),
            'innerText': self._text,
            'placeholder': '',
            'ariaLabel': '',
            'value': '',
            'labelText': '',
            'labelledbyText': '',
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


ANCHOR = '64625'
EDIT = '[title="Edit"]'
TR_COMPOSITE = f'tr:has-text("{ANCHOR}") >> {EDIT}'
LI_COMPOSITE = f'li:has-text("{ANCHOR}") >> {EDIT}'


class TestUpgradeToRowAnchor:
    """The shared helper: emit the row-scoped composite only when it
    resolves to exactly one element (that vision saw, if coords given)."""

    async def test_tr_anchor_unique_returns_composite(self):
        ctx = FakeSearchContext({TR_COMPOSITE: FakeLocator(1)})
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR)
        assert result == {'locator': TR_COMPOSITE}

    async def test_ambiguous_two_rows_flagged(self):
        """Anchor matches two rows -> ambiguous, not a silent guess
        (Option 1: caller falls through to today's behavior, flagged)."""
        ctx = FakeSearchContext({
            TR_COMPOSITE: FakeLocator(2),
            f'{TR_COMPOSITE} >> visible=true': FakeLocator(2),
        })
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR)
        assert result == {'ambiguous': True}

    async def test_hidden_duplicate_row_rescued_by_visible(self):
        """A hidden template row (modal grid copy) also matching the
        anchor must not kill the upgrade — visible filter rescues."""
        ctx = FakeSearchContext({
            TR_COMPOSITE: FakeLocator(2),
            f'{TR_COMPOSITE} >> visible=true': FakeLocator(1),
        })
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR)
        assert result == {'locator': f'{TR_COMPOSITE} >> visible=true'}

    async def test_li_fallback_when_no_tr(self):
        """Repeated card/list layouts get the same fix via li."""
        ctx = FakeSearchContext({
            TR_COMPOSITE: FakeLocator(0),
            LI_COMPOSITE: FakeLocator(1),
        })
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR)
        assert result == {'locator': LI_COMPOSITE}

    async def test_xpath_candidate_skipped(self):
        """Chained-xpath scoping (relative vs document root) is
        engine-version-dependent, and the generated test runs under RF
        Browser's own Playwright — the upgrade must not even probe it."""
        ctx = FakeSearchContext({})  # any lookup would KeyError
        for xp in ('xpath=//a[@title="Edit"]', '//a[@title="Edit"]',
                   '(//a[@title="Edit"])'):
            assert await _upgrade_to_row_anchor(ctx, xp, ANCHOR) is None
        assert ctx.probed == []

    async def test_empty_anchor_returns_none(self):
        ctx = FakeSearchContext({})
        assert await _upgrade_to_row_anchor(ctx, EDIT, '') is None
        assert await _upgrade_to_row_anchor(ctx, EDIT, '   ') is None
        assert await _upgrade_to_row_anchor(ctx, EDIT, None) is None
        assert ctx.probed == []

    async def test_quote_in_anchor_escaped(self):
        """Anchor data can carry double quotes (John "JJ" Smith)."""
        escaped = 'tr:has-text("John \\"JJ\\" Smith") >> [title="Edit"]'
        ctx = FakeSearchContext({escaped: FakeLocator(1)})
        result = await _upgrade_to_row_anchor(ctx, EDIT, 'John "JJ" Smith')
        assert result == {'locator': escaped}

    async def test_identity_check_rejects_wrong_row(self):
        """Unique match that is NOT the element vision clicked (anchor
        text present in a different row) must not be emitted."""
        far_box = {'x': 900.0, 'y': 900.0, 'width': 20.0, 'height': 20.0}
        ctx = FakeSearchContext({TR_COMPOSITE: FakeLocator(1, box=far_box)})
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR, x=100, y=100)
        assert result is None

    async def test_identity_check_accepts_right_row(self):
        box = {'x': 90.0, 'y': 90.0, 'width': 40.0, 'height': 40.0}
        ctx = FakeSearchContext({TR_COMPOSITE: FakeLocator(1, box=box)})
        result = await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR, x=100, y=100)
        assert result == {'locator': TR_COMPOSITE}

    async def test_probe_error_returns_none(self):
        ctx = FakeSearchContext({})  # KeyError inside -> swallowed
        assert await _upgrade_to_row_anchor(ctx, EDIT, ANCHOR) is None


class TestElementDataRowAnchor:
    """STEP-0: a row-repeated candidate (aria-label on every row's Edit
    button) must come back row-anchored with the flag in the payload."""

    @staticmethod
    def make_element_data(**overrides) -> dict:
        data = {
            'tagName': 'a',
            'id': '',
            'name': '',
            'className': '',
            'textContent': '',
            'ariaLabel': 'Edit',
            'placeholder': '',
            'title': '',
            'role': '',
            'dataTestId': '',
            'type': '',
            'xpath': '',
            'coordinates': {'x': 100, 'y': 100},
            'parentId': '',
            'parentClass': '',
        }
        data.update(overrides)
        return data

    async def test_repeated_candidate_returns_row_composite(self):
        aria = '[aria-label="Edit"]'
        composite = f'tr:has-text("{ANCHOR}") >> {aria}'
        ctx = LenientFakeContext({
            aria: FakeLocator(12),
            composite: FakeLocator(1),
        })
        result = await _generate_locators_from_element_data(
            ctx, self.make_element_data(), 'elem_1',
            'Edit link for customer 64625',
            row_anchor_text=ANCHOR,
        )
        assert result is not None and result['found']
        assert result['best_locator'] == composite
        assert result['row_anchored'] is True
        assert result['stability'] == 'stable'  # QA-specified data
        assert result['all_locators'][0]['row_anchored'] is True

    async def test_no_anchor_param_keeps_old_behavior(self):
        aria = '[aria-label="Edit"]'
        ctx = LenientFakeContext({
            aria: FakeLocator(12),
            f'{aria} >> visible=true': FakeLocator(12),
        })
        result = await _generate_locators_from_element_data(
            ctx, self.make_element_data(), 'elem_1',
            'Edit link for customer 64625',
        )
        assert result is None  # unchanged pre-fix behavior

    async def test_ambiguous_anchor_flag_rides_on_fallback_result(self):
        """Anchor matches two rows; the anchored candidate falls through
        but a later unique candidate carries row_anchor_ambiguous so
        nlrf can warn (Option 1)."""
        aria = '[aria-label="Edit"]'
        composite = f'tr:has-text("{ANCHOR}") >> {aria}'
        parent_css = '#actions-col a'
        ctx = LenientFakeContext({
            aria: FakeLocator(12),
            composite: FakeLocator(2),
            f'{composite} >> visible=true': FakeLocator(2),
            f'{aria} >> visible=true': FakeLocator(12),
            parent_css: FakeLocator(1),
        })
        result = await _generate_locators_from_element_data(
            ctx, self.make_element_data(parentId='actions-col'), 'elem_1',
            'Edit link for customer 64625',
            row_anchor_text=ANCHOR,
        )
        assert result is not None and result['found']
        assert result['best_locator'] == parent_css
        assert result.get('row_anchored') is None
        assert result['row_anchor_ambiguous'] is True


class TestTextFirstRowAnchor:
    """Text-first: row anchor is tried BEFORE the nth= coordinate
    disambiguation — the anchored composite must win when unique."""

    async def test_row_anchor_beats_nth_fallback(self):
        text_sel = 'text="Edit"'
        composite = f'tr:has-text("{ANCHOR}") >> {text_sel}'
        box = {'x': 90.0, 'y': 90.0, 'width': 40.0, 'height': 40.0}
        ctx = LenientFakeContext({
            text_sel: FakeLocator(12),
            composite: FakeLocator(1, box=box),
        })
        result = await _find_element_by_expected_text(
            ctx, 'Edit', 'Edit link', x=100, y=100,
            row_anchor_text=ANCHOR,
        )
        assert result is not None
        assert result['locator'] == composite
        assert result['row_anchored'] is True

    async def test_ambiguous_anchor_falls_to_nth_with_flag(self):
        """Two rows match the anchor -> today's nth fallback, but the
        result is flagged so nlrf can warn (Option 1)."""
        text_sel = 'text="Edit"'
        composite = f'tr:has-text("{ANCHOR}") >> {text_sel}'
        boxes = [
            {'x': 0.0, 'y': 0.0, 'width': 10.0, 'height': 10.0},
            {'x': 95.0, 'y': 95.0, 'width': 10.0, 'height': 10.0},
        ]
        ctx = LenientFakeContext({
            text_sel: FakeLocator(2, nth_boxes=boxes),
            composite: FakeLocator(2),
            f'{composite} >> visible=true': FakeLocator(2),
            f'{text_sel} >> visible=true': FakeLocator(2, nth_boxes=boxes),
        })
        result = await _find_element_by_expected_text(
            ctx, 'Edit', 'Edit link', x=100, y=100,
            row_anchor_text=ANCHOR,
        )
        assert result is not None
        assert 'nth=1' in result['locator']
        assert result['row_anchor_ambiguous'] is True

    async def test_without_anchor_behavior_unchanged(self):
        text_sel = 'text="Edit"'
        boxes = [
            {'x': 0.0, 'y': 0.0, 'width': 10.0, 'height': 10.0},
            {'x': 95.0, 'y': 95.0, 'width': 10.0, 'height': 10.0},
        ]
        ctx = LenientFakeContext({
            text_sel: FakeLocator(2, nth_boxes=boxes),
            f'{text_sel} >> visible=true': FakeLocator(2, nth_boxes=boxes),
        })
        result = await _find_element_by_expected_text(
            ctx, 'Edit', 'Edit link', x=100, y=100,
        )
        assert result is not None
        assert 'nth=1' in result['locator']
        assert 'row_anchor_ambiguous' not in result


class TestDescriptionRowAnchor:
    """Description-derived selectors get the same rescue."""

    async def test_link_selector_upgrades(self):
        sel = 'role=link[name="Edit"]'
        composite = f'tr:has-text("{ANCHOR}") >> {sel}'
        ctx = LenientFakeContext({
            sel: FakeLocator(12),
            composite: FakeLocator(1),
        })
        locator = await _find_element_by_description(
            ctx, 'Edit link', row_anchor_text=ANCHOR,
        )
        assert locator == composite


class TestStrategyValidatorRowAnchor:
    """STEP-3: a repeated title strategy validates as unique via the
    row composite instead of being recorded not-unique."""

    @staticmethod
    def make_strategies() -> list:
        return [
            {'type': 'title', 'locator': EDIT, 'priority': 5,
             'strategy': 'Title attribute'},
        ]

    async def test_repeated_title_upgrades(self):
        ctx = FakeSearchContext({
            EDIT: FakeLocator(12),
            TR_COMPOSITE: FakeLocator(1),
        })
        result = await _validate_strategy_candidates(
            ctx, self.make_strategies(), row_anchor_text=ANCHOR,
        )
        entry = result[0]
        assert entry['locator'] == TR_COMPOSITE
        assert entry['unique'] and entry['valid']
        assert entry['row_anchored'] is True

    async def test_ambiguous_anchor_entry_flagged_not_unique(self):
        ctx = FakeSearchContext({
            EDIT: FakeLocator(12),
            TR_COMPOSITE: FakeLocator(2),
            f'{TR_COMPOSITE} >> visible=true': FakeLocator(2),
            f'{EDIT} >> visible=true': FakeLocator(12),
        })
        result = await _validate_strategy_candidates(
            ctx, self.make_strategies(), row_anchor_text=ANCHOR,
        )
        entry = result[0]
        assert entry['locator'] == EDIT
        assert not entry['unique']
        assert entry['row_anchor_ambiguous'] is True


class TestPlumbing:
    """row_anchor_text must exist end-to-end or the LLM can never pass
    it: custom-action param model -> actions wrapper -> locator engine
    -> prompt guidance."""

    def test_engine_signatures_accept_row_anchor(self):
        from browser_service.locators.smart_locator import (
            find_unique_locator_at_coordinates,
        )
        from browser_service.agent.actions import find_unique_locator_action
        for fn in (find_unique_locator_at_coordinates,
                   find_unique_locator_action):
            assert 'row_anchor_text' in inspect.signature(fn).parameters, \
                f'{fn.__name__} missing row_anchor_text'

    def test_registration_declares_param(self):
        src = (REPO_ROOT / 'browser_service' / 'agent'
               / 'registration.py').read_text(encoding='utf-8')
        assert 'row_anchor_text' in src

    def test_prompt_guidance_exists(self):
        src = (REPO_ROOT / 'browser_service' / 'prompts'
               / 'templates.py').read_text(encoding='utf-8')
        assert 'row_anchor_text' in src

    def test_is_collection_guidance_excludes_per_row_actions(self):
        # F3 (2026-07-16): the is_collection bullet's own examples
        # ("table rows") misled the vision agent into flagging per-row
        # clicks as collections (bench q04 r3). The bullet must carry an
        # explicit exclusion pointing at row_anchor_text.
        src = (REPO_ROOT / 'browser_service' / 'prompts'
               / 'templates.py').read_text(encoding='utf-8')
        assert 'NOT a collection' in src


class TestDeriveRowAnchorFromDescription:
    """When the caller (vision agent) leaves row_anchor_text unset, a
    per-row action description already names the anchor as a quoted
    phrase — recover it deterministically (bench q04 root cause: 'the
    edit link in the row containing Smith in the first data table')."""

    def test_row_containing_quoted_phrase(self):
        desc = "the 'edit' link in the row containing 'Smith' in the first data table"
        assert _derive_row_anchor_text_from_description(desc) == 'Smith'

    def test_double_quotes(self):
        desc = 'the edit link in the row containing "Smith" in the first table'
        assert _derive_row_anchor_text_from_description(desc) == 'Smith'

    def test_item_with_phrasing(self):
        desc = "the delete icon for the item with '64625'"
        assert _derive_row_anchor_text_from_description(desc) == '64625'

    def test_relative_clause_that_contains(self):
        # 2026-07-16 bench q04 r1: "row that contains 'Smith'" missed the
        # old pattern and decayed to positional nth=0 — passed by luck.
        desc = "the edit link in the row that contains 'Smith' in the first table"
        assert _derive_row_anchor_text_from_description(desc) == 'Smith'

    def test_relative_clause_which_has(self):
        desc = "the checkbox in the row which has 'Pending'"
        assert _derive_row_anchor_text_from_description(desc) == 'Pending'

    def test_intervening_noun_before_quoted_anchor(self):
        desc = "the Edit icon in the row for customer '64625'"
        assert _derive_row_anchor_text_from_description(desc) == '64625'

    def test_unquoted_numeric_anchor(self):
        # Grid ids (account numbers, invoice ids) are the canonical row
        # datum and users don't quote them in natural phrasing.
        desc = "the Edit icon in the row for customer 4727985745"
        assert _derive_row_anchor_text_from_description(desc) == '4727985745'

    def test_unquoted_numeric_anchor_without_noun(self):
        desc = "the download link in the row for 64625"
        assert _derive_row_anchor_text_from_description(desc) == '64625'

    def test_plural_head_noun_is_not_an_anchor(self):
        # "all rows with X" is a filtered COLLECTION, not one row's action.
        desc = "all rows with 'pending' status in the data table"
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
        assert _derive_row_anchor_text_from_description('') is None


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
        assert _should_treat_as_collection(True, 'irrelevant', 'Smith') is False

    def test_explicit_collection_without_anchor_still_wins(self):
        assert _should_treat_as_collection(True, 'all book titles on the page', None) is True

    def test_explicit_false_is_not_true_but_keyword_may_still_apply(self):
        # is_collection=False is not the same as "unset" — but the keyword
        # fallback still applies when nothing overrides it.
        desc = "all rows in the data table"
        assert _should_treat_as_collection(False, desc, None) is True

    def test_row_anchor_suppresses_keyword_fallback(self):
        desc = "the 'edit' link in the row containing 'Smith' in the first data table"
        assert _should_treat_as_collection(None, desc, 'Smith') is False

    def test_keyword_fallback_fires_without_row_anchor(self):
        desc = "the 'edit' link in the row containing 'Smith' in the first data table"
        assert _should_treat_as_collection(None, desc, None) is True

    def test_no_description_no_row_anchor_not_collection(self):
        assert _should_treat_as_collection(None, None, None) is False

    def test_unrelated_description_not_collection(self):
        assert _should_treat_as_collection(None, 'the Submit button', None) is False
