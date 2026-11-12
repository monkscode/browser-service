"""Tests for locators/css_tokens.py (bs #43): a page's class / id token written without a backslash."""

import pytest

from browser_service.locators.css_tokens import can_quote, css_attr, css_class, css_id


class TestCssClass:
    @pytest.mark.parametrize(
        "token",
        ["btn-primary", "nav_item", "card", "café", "--x", "Header2", "-mt-4", "éclair", "_x"],
    )
    def test_plain_identifier_is_unchanged(self, token):
        assert css_class(token) == f".{token}"

    @pytest.mark.parametrize(
        "token",
        [
            "w-1/2",
            "md:text-lg",
            "form:main",
            "user.name",
            "1st",
            "-4",
            "-",
            "[&>svg]:w-4",
            "bg-[#fff]",
            "w-[50%]",
            "2xl:p-4",
            "a$b",
            "a{b}",
            "x=y",
        ],
    )
    def test_other_token_becomes_a_quoted_attribute(self, token):
        assert css_class(token) == f'[class~="{token}"]'

    @pytest.mark.parametrize(
        "token",
        [
            "",
            None,
            'a"b',
            "a\\b",
            "a\\nb",
            "a b",
            "a\tb",
            "a b",
            "a${b}",
            "a@{b}",
            "a&{b}",
            "a%{b}",
        ],
    )
    def test_unsafe_token_is_skipped(self, token):
        assert css_class(token) == ""


class TestCssId:
    @pytest.mark.parametrize(
        "token", ["formMain", "btn-primary", "nav_item", "café", "Header2", "_x"]
    )
    def test_plain_identifier_is_unchanged(self, token):
        assert css_id(token) == f"#{token}"

    @pytest.mark.parametrize(
        "token",
        ["form:main", "user.name", "1st", "-side", "--x", "-mt-4", "éclair", "100%", "2xl:p-4"],
    )
    def test_id_must_start_with_an_ascii_letter_or_underscore(self, token):
        # NLRF adds css= to a bare "#..." cell only when a letter or "_" follows the "#";
        # any other bare "#..." cell is a Robot comment, so these are quoted.
        assert css_id(token) == f'[id="{token}"]'

    @pytest.mark.parametrize("token", ["", None, 'a"b', "a\\b", "a b", "a${b}"])
    def test_unsafe_token_is_skipped(self, token):
        assert css_id(token) == ""


class TestNoBackslashEver:
    @pytest.mark.parametrize("token", ["w-1/2", "a\\b", 'a"b', "x:y", "é", "--", "-"])
    def test_output_never_holds_a_backslash(self, token):
        for out in (css_class(token), css_id(token)):
            assert "\\" not in out


def test_css_attr_is_the_one_quoting_place():
    assert css_attr("title", "a b") == ""  # whitespace: skipped (bs #60 may extend)
    assert css_attr("data-x", "1/2") == '[data-x="1/2"]'
    assert css_attr("class", "w-1/2", "~=") == '[class~="w-1/2"]'
    assert can_quote("a:b")
    assert not can_quote("a${b}")
