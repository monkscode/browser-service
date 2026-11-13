"""bs #61: the row-datum helpers — which positional candidates skip the row the step names."""

import pytest

from browser_service.locators.smart_locator import carries_row_anchor, unanchored_row_datum

Q04 = "the 'edit' link in the row containing 'Smith' in the first table (Example 1)"


class TestUnanchoredRowDatum:
    def test_q04_positional_candidate_without_the_row_names_smith(self):
        cand = "#table1 tbody tr:nth-child(1) a[href='#edit']"
        assert unanchored_row_datum(cand, Q04) == "Smith"

    def test_q05_names_no_row_datum(self):
        cand = "#table1 tbody tr:nth-child(1) td:nth-child(2)"
        assert (
            unanchored_row_datum(cand, "the cell in the first row and second column of Table 1")
            is None
        )

    def test_candidate_already_carrying_the_row_is_left_alone(self):
        cand = "tr:has-text('Smith') >> a >> nth=0"
        assert unanchored_row_datum(cand, Q04) is None

    def test_non_positional_candidate_is_left_alone(self):
        assert unanchored_row_datum("#table1 tr:has-text('Smith') a", Q04) is None

    def test_datum_under_three_characters_is_ignored(self):
        cand = "#t tr:nth-child(2) a"
        assert unanchored_row_datum(cand, 'the edit link in the row containing "O\'Brien"') is None

    def test_every_row_is_a_collection_not_a_row(self):
        cand = "#t tr:nth-child(2) a"
        assert unanchored_row_datum(cand, "the edit link in every row containing 'Smith'") is None

    @pytest.mark.parametrize("cand", [None, ""])
    def test_no_candidate(self, cand):
        assert unanchored_row_datum(cand, Q04) is None


class TestCarriesRowAnchor:
    @pytest.mark.parametrize(
        "locator, anchor, want",
        [
            ('tr:has-text("Smith") >> text="edit" >> nth=0', "Smith", True),
            ('li:has-text( "  smith ")', "Smith", True),
            ('text="edit" >> nth=0', "Credit card", False),
            ('td:has-text("Smith")', "Smith", False),
            ('iframe[id="t"] >>> tr:has-text("Smith") >> a', "Smith", True),
            ('tr:has-text("Smithson") >> a', "Smith", False),
            ('tr:has-text("Smith") >> a', "", False),
            (None, "Smith", False),
        ],
    )
    def test_cases(self, locator, anchor, want):
        assert carries_row_anchor(locator, anchor) is want
