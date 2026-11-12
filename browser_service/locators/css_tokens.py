"""
Robot-safe CSS for a page's class / id token (browser-service #43).

A class or id taken from the page ("w-1/2", "md:flex", "form:main", "1st") used
to be written into a selector with a CSS backslash ("button.w-1\\/2"). Playwright
reads the backslash; the generated test does not: Robot Framework drops a
backslash from a cell, so the keyword receives "button.w-1/2". This module
writes the token without any backslash:

- a token that is a plain CSS identifier stays ".tok" / "#tok" (byte-identical
  to what was shipped before);
- any other token becomes a quoted attribute, '[class~="tok"]' / '[id="tok"]',
  which Robot leaves alone and Playwright reads the same way;
- a token that cannot be quoted safely returns "" and the caller skips that
  candidate (the old helper's contract for a token with a space).

bs #60 adds the rule for a page value inside "...": quote_value picks the quote kind
the value needs, or returns "" for a value Robot would change (backslash, 2+ spaces,
TAB, line break, "${"); robot_safe is the rule a whole locator is checked by. css_attr
writes every quoted attribute value through quote_value.

Referenced by: locators/smart_locator.py
Depends on: nothing
"""

import re

# CSS identifier (non-ASCII is allowed anywhere after the start). A class may start
# with "-" or "--" (Tailwind "-mt-4"); an id may not start with "-" or a non-ASCII
# letter: NLRF puts "css=" in front of a bare cell only when "#" / "." is followed
# by an ASCII letter or "_", and a bare "#..." cell is a Robot comment.
_TAIL = "A-Za-z0-9_\u0080-\U0010ffff-"
_CLASS_IDENT = re.compile(rf"(?:--[{_TAIL}]|-?[A-Za-z_\u0080-\U0010ffff])[{_TAIL}]*\Z")
_ID_IDENT = re.compile(rf"[A-Za-z_][{_TAIL}]*\Z")
# Robot reads ${x} @{x} &{x} %{x} as a variable even inside a quoted value.
_ROBOT_VARIABLE = re.compile(r"[$@&%]\{")
# Robot 7.4.2 splits a cell on 2+ whitespace or one TAB, and a line on str.splitlines.
_ROBOT_SPLIT = re.compile(r"\s{2,}|\t|[\n\r\v\f\x1c-\x1e\x85  ]")


def quote_value(value: str, collapse: bool = False) -> str:
    """'"v"' or "'v'" (the quote kind ``value`` needs), or "" when Robot would change it (bs #60).

    ``collapse`` first folds every whitespace run to one space: for role names, text= and
    :has-text(), whose Playwright engines normalise whitespace themselves.
    """
    value = " ".join(str(value or "").split()) if collapse else (value or "")
    if (
        not value
        or not value.isprintable()  # TAB, line breaks, NBSP and every other non-ASCII space
        or "  " in value
        or "\\" in value
        or _ROBOT_VARIABLE.search(value)
    ):
        return ""
    if '"' not in value:
        return f'"{value}"'
    return f"'{value}'" if "'" not in value else ""


def robot_safe(locator: str) -> bool:
    """True when a Robot Framework cell hands ``locator`` to the keyword unchanged (bs #60).

    The rule, from Robot 7.4.2's tokenizer and NLRF's css= prefix: no backslash, no split
    (2+ whitespace, a TAB, a line break), no variable start, no edge whitespace, and a
    leading '#' only where NLRF adds css= (else the cell is a comment).
    """
    return bool(
        locator
        and "\\" not in locator
        and not _ROBOT_SPLIT.search(locator)
        and not _ROBOT_VARIABLE.search(locator)
        and locator == locator.strip()
        and (not locator.startswith("#") or re.match(r"#[A-Za-z_]", locator))
    )


def css_attr(name: str, value: str, op: str = "=") -> str:
    """'[name="value"]' / "[name='value']" (``op`` "=", "~=", "*=" ...), or "" when the value cannot be quoted."""
    q = quote_value(value)
    return f"[{name}{op}{q}]" if q else ""


def css_class(token: str) -> str:
    """'.tok' for a plain identifier, '[class~="tok"]' otherwise, '' to skip."""
    token = token or ""
    if token.isprintable() and _CLASS_IDENT.match(token):
        return f".{token}"
    return "" if " " in token else css_attr("class", token, "~=")


def css_id(token: str) -> str:
    """'#tok' for a plain identifier, '[id="tok"]' otherwise, '' to skip."""
    token = token or ""
    if token.isprintable() and _ID_IDENT.match(token):
        return f"#{token}"
    return "" if " " in token else css_attr("id", token)
