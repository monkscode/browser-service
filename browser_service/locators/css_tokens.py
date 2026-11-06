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

css_attr is the ONE place that writes a quoted attribute value, so bs #60
(names, text, titles inside "..." with a quote or a backslash) extends it
here — quote-kind choice, the skip rules — without touching a #43 caller.

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


def can_quote(value: str) -> bool:
    """True when ``value`` can sit in a "..." attribute value and survive Robot."""
    return bool(
        value
        and value.isprintable()
        and not any(c.isspace() for c in value)
        and '"' not in value
        and "\\" not in value
        and not _ROBOT_VARIABLE.search(value)
    )


def css_attr(name: str, value: str, op: str = "=") -> str:
    """'[name="value"]' (``op`` "=" or "~="), or "" when the value cannot be quoted."""
    return f'[{name}{op}"{value}"]' if can_quote(value) else ""


def css_class(token: str) -> str:
    """'.tok' for a plain identifier, '[class~="tok"]' otherwise, '' to skip."""
    token = token or ""
    if token.isprintable() and _CLASS_IDENT.match(token):
        return f".{token}"
    return css_attr("class", token, "~=")


def css_id(token: str) -> str:
    """'#tok' for a plain identifier, '[id="tok"]' otherwise, '' to skip."""
    token = token or ""
    if token.isprintable() and _ID_IDENT.match(token):
        return f"#{token}"
    return css_attr("id", token)
