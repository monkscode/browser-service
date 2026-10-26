"""
Elements whose value is never read back (NLRF #131 Part 2, owner decision D8).

An element hides its value when it is ``<input type=password>`` (browser-use's own
serializer never shows a password value) or carries bs's mark attribute, which
``_do_interaction`` sets on a step's field right before typing an org secret into it.
Every bs reader of an element's text then skips the value, so a re-located field reports
its aria-label / title / placeholder / alt / child text -- never what was typed. React
controlled inputs copy typed text into the ``value`` attribute, which browser-use's
``get_meaningful_text_for_llm`` reads first.

Referenced by: agent/registration.py, locators/read_target.py, locators/smart_locator.py
Depends on: nothing
"""

# No ``data-`` prefix on purpose: read_target.py turns any data-* attribute NAME into a token.
SECRET_MARK = "nlrf-secret"


# The JS twin of hides_value: a function expression of one element, called as (HIDES_VALUE_JS)(x).
HIDES_VALUE_JS = (
    "(el => String(el.type || '').toLowerCase() === 'password' || el.hasAttribute('"
    + SECRET_MARK
    + "'))"
)


def hides_value(attrs: dict | None) -> bool:
    """True for a password input or a field carrying the secret mark."""
    if not attrs:
        return False
    return str(attrs.get("type", "")).lower() == "password" or SECRET_MARK in attrs


def node_text(node) -> str:
    """browser-use's ``get_meaningful_text_for_llm`` order WITHOUT ``value``; always a str."""
    attrs = getattr(node, "attributes", None) or {}
    for name in ("aria-label", "title", "placeholder", "alt"):
        if attrs.get(name):
            return str(attrs[name]).strip()
    return (node.get_all_children_text() or "").strip()
