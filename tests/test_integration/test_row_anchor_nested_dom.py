"""
Real-DOM stress tests for the row-anchor containment collapse (q02).

Replicates the live ASTPP customer grid shape (verified 2026-07-17):
every row's Edit link carries title="Edit" on the <a> AND on the icon
<span> inside it, so the row-scoped chain double-counts one control and
the rescue used to declare it ambiguous — shipping the fragile
whole-page positional fallback ([title="Edit"] >> nth=18, observed
drifting from nth=16 overnight as rows were added).

These tests run the actual collapse against real Chromium DOM — the
containment JS, parent-first document order, the visible filter, and
the emitted composite's runtime semantics (uniqueness, actionability,
and following the row's datum through reorder/growth). No network:
pages are built with page.set_content().

Requires: playwright install chromium
Run: pytest tests/test_integration/test_row_anchor_nested_dom.py -m integration -v
"""

import contextlib

import pytest
from playwright.async_api import async_playwright

from browser_service.locators.smart_locator import _upgrade_to_row_anchor

pytestmark = pytest.mark.integration

EDIT = '[title="Edit"]'


def grid_html(customer_ids, extra_rows=''):
    """ASTPP customer-grid replica: id cell + Edit link with the
    nested-title pattern (title on the <a> AND the <span> inside it)."""
    trs = '\n'.join(
        f'<tr><td><span class="cid">{cid}</span></td>'
        f'<td><a href="/accounts/customer_edit/{cid}" title="Edit">'
        f'<span title="Edit" class="fa fa-pencil">E</span></a></td></tr>'
        for cid in customer_ids
    )
    return (
        '<table><tbody>'
        f'{trs}{extra_rows}'
        '</tbody></table>'
    )


@contextlib.asynccontextmanager
async def _page(html):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(html)
        try:
            yield page
        finally:
            await browser.close()


class TestNestedTitleCollapseRealDom:

    async def test_nested_pair_collapses_to_the_link(self):
        """The q02 case verbatim: 2 matches in the row (a + span) are one
        control — the collapse must emit the row-scoped composite and it
        must resolve to exactly the customer's <a>, actionable."""
        async with _page(grid_html(['55514', '55515', '55516'])) as page:
            result = await _upgrade_to_row_anchor(page, EDIT, '55516')
            expected = (
                f'tr:has-text("55516") >> {EDIT} >> visible=true >> nth=0'
            )
            assert result == {'locator': expected}

            target = page.locator(result['locator'])
            assert await target.count() == 1
            assert await target.evaluate('el => el.tagName') == 'A'
            href = await target.get_attribute('href')
            assert href.endswith('/customer_edit/55516')
            # Actionability: the outer link takes the click (the icon
            # span inside it is an accepted hit-target descendant).
            await target.click(trial=True)

    async def test_locator_follows_the_row_through_overnight_growth(self):
        """The failure that motivated the fix: [title="Edit"] >> nth=16
        became nth=18 when rows were added. The collapsed composite must
        keep resolving to customer 55516's Edit link after rows are
        prepended and the table reordered."""
        async with _page(grid_html(['55514', '55515', '55516'])) as page:
            result = await _upgrade_to_row_anchor(page, EDIT, '55516')
            locator = result['locator']

            grown = grid_html(
                ['61001', '61002', '61003', '55515', '61004', '55516',
                 '55514', '61005']
            )
            await page.set_content(grown)

            target = page.locator(locator)
            assert await target.count() == 1
            row_datum = await target.evaluate(
                "el => el.closest('tr').querySelector('.cid').textContent"
            )
            assert row_datum == '55516'

    async def test_two_matching_rows_stay_ambiguous(self):
        """An anchor that substring-matches two rows is genuine
        ambiguity — cross-row matches never nest."""
        html = grid_html(['55516', '155516', '61001'])
        async with _page(html) as page:
            result = await _upgrade_to_row_anchor(page, EDIT, '55516')
            assert result == {'ambiguous': True}

    async def test_second_separate_control_in_row_stays_ambiguous(self):
        """A second non-nested Edit control in the anchor row breaks the
        chain — must stay flagged, not silently collapsed."""
        extra = (
            '<tr><td><span class="cid">55516</span></td>'
            '<td><a href="/e/55516" title="Edit">'
            '<span title="Edit">E</span></a></td>'
            '<td><a href="/quick/55516" title="Edit">quick</a></td></tr>'
        )
        html = grid_html(['55514', '55515'], extra_rows=extra)
        async with _page(html) as page:
            result = await _upgrade_to_row_anchor(page, EDIT, '55516')
            assert result == {'ambiguous': True}

    async def test_hidden_template_row_still_collapses(self):
        """Both ASTPP patterns at once: a display:none duplicate of the
        anchor row (modal grid copy) plus the nested pair. The visible
        filter drops the hidden pair; the visible chain collapses to the
        visible link."""
        hidden = (
            '<tr style="display:none">'
            '<td><span class="cid">55516</span></td>'
            '<td><a href="/hidden/55516" title="Edit">'
            '<span title="Edit">E</span></a></td></tr>'
        )
        html = grid_html(['55514', '55516'], extra_rows=hidden)
        async with _page(html) as page:
            result = await _upgrade_to_row_anchor(page, EDIT, '55516')
            assert result and 'locator' in result
            target = page.locator(result['locator'])
            assert await target.count() == 1
            href = await target.get_attribute('href')
            assert href.endswith('/customer_edit/55516')  # not /hidden/

    async def test_triple_nested_titles_collapse_to_outermost(self):
        """Three name tags on one ancestor line (td > a > span) are
        still one control; nth=0 is the outermost by document order."""
        extra = (
            '<tr><td><span class="cid">55516</span></td>'
            '<td title="Edit"><a href="/e/55516" title="Edit">'
            '<span title="Edit">E</span></a></td></tr>'
        )
        html = grid_html(['55514'], extra_rows=extra)
        async with _page(html) as page:
            result = await _upgrade_to_row_anchor(page, EDIT, '55516')
            assert result and 'locator' in result
            target = page.locator(result['locator'])
            assert await target.count() == 1
            assert await target.evaluate('el => el.tagName') == 'TD'

    async def test_full_engine_emits_collapsed_composite_stable(self):
        """End-to-end through find_unique_locator_at_coordinates with
        the element_data shape from the live q02 traces: the payload
        must carry the row-scoped collapsed composite labeled STABLE
        (no cry-wolf positional warning), and the locator must resolve
        to the customer's Edit link."""
        from browser_service.locators.smart_locator import (
            find_unique_locator_at_coordinates,
        )
        async with _page(grid_html(['55514', '55515', '55516'])) as page:
            box = await page.locator(
                'a[href$="/customer_edit/55516"]'
            ).bounding_box()
            result = await find_unique_locator_at_coordinates(
                page=page,
                x=box['x'] + box['width'] / 2,
                y=box['y'] + box['height'] / 2,
                element_id='elem_4',
                element_description=(
                    'Edit action icon in the row containing 55516 '
                    'in the customer data table'
                ),
                expected_text='Edit',
                element_data={
                    'tagName': 'a', 'id': '', 'name': '', 'className': '',
                    'textContent': 'E', 'ariaLabel': '', 'placeholder': '',
                    'title': 'Edit', 'role': '', 'dataTestId': '',
                    'type': '', 'xpath': '', 'parentId': '',
                    'parentClass': '',
                },
                search_context=page,
                row_anchor_text='55516',
            )
            assert result['found'] is True
            assert 'tr:has-text("55516")' in result['best_locator']
            assert result['best_locator'].endswith('>> nth=0')
            assert result.get('row_anchored') is True
            assert result['stability'] == 'stable'

            target = page.locator(result['best_locator'])
            assert await target.count() == 1
            href = await target.get_attribute('href')
            assert href.endswith('/customer_edit/55516')

    async def test_identity_check_on_real_geometry(self):
        """A5 on real boxes: coordinates on the link accept the collapse;
        coordinates on a different row's link reject it."""
        async with _page(grid_html(['55514', '55515', '55516'])) as page:
            right_box = await page.locator(
                'a[href$="/customer_edit/55516"]'
            ).bounding_box()
            wrong_box = await page.locator(
                'a[href$="/customer_edit/55514"]'
            ).bounding_box()

            cx = right_box['x'] + right_box['width'] / 2
            cy = right_box['y'] + right_box['height'] / 2
            accepted = await _upgrade_to_row_anchor(
                page, EDIT, '55516', x=cx, y=cy
            )
            assert accepted and 'locator' in accepted

            # A far-away point (different row, offset beyond the
            # coordinate threshold) must not be claimed.
            rejected = await _upgrade_to_row_anchor(
                page, EDIT, '55516',
                x=wrong_box['x'] + 600, y=wrong_box['y'] + 600,
            )
            assert rejected is None
