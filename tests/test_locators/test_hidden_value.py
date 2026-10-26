"""
An element that hides its value is never read back (NLRF #131 Part 2, owner decision D8).

A password input, or a field bs marked with ``nlrf-secret`` right before typing an org
secret into it, reports its aria-label / title / placeholder / alt / child text -- never
the value. React controlled inputs copy what was typed into the ``value`` attribute, so
browser-use's own ``get_meaningful_text_for_llm`` (value first) would print it.

Made-up value only. Every leak test asserts the value is in no log line and no returned text.
"""

import logging

from browser_use.dom.views import EnhancedDOMTreeNode

from browser_service.agent.registration import _dom_node_text, _extract_dom_node_attributes
from browser_service.locators.hidden_value import SECRET_MARK, hides_value, node_text
from browser_service.locators.smart_locator import validate_semantic_match

V = "Zq131-made-up-7"


class Node:
    """Duck-typed snapshot node carrying the real browser-use text method."""

    get_meaningful_text_for_llm = EnhancedDOMTreeNode.get_meaningful_text_for_llm

    def __init__(self, attributes, children_text=""):
        self.attributes = attributes
        self.ax_node = None
        self.node_value = ""
        self.tag_name = "input"
        self.node_name = "INPUT"
        self.children_nodes = []
        self._children_text = children_text

    def get_all_children_text(self):
        return self._children_text


SYNCED = {"type": "password", "placeholder": "Password", "id": "password", "value": V}
EMPTY = {"type": "password", "placeholder": "Password", "id": "password", "value": ""}
MARKED = {"type": "text", "placeholder": "API key", "value": V, SECRET_MARK: ""}
PLAIN = {"type": "text", "placeholder": "Name", "value": V}


def test_hides_value_rule():
    assert hides_value({"type": "password"})
    assert hides_value({"type": "PassWord"})
    assert hides_value(MARKED)
    assert hides_value({SECRET_MARK: ""})
    assert not hides_value(PLAIN)
    assert not hides_value(None)
    assert not hides_value({})


def test_mark_name_is_not_a_data_attribute():
    # read_target.py turns any data-* attribute NAME into a token / container base.
    assert not SECRET_MARK.startswith("data-")


def test_node_text_order_never_value():
    assert node_text(Node({"aria-label": "A", "placeholder": "P", "value": V})) == "A"
    assert node_text(Node({"title": "T", "placeholder": "P", "value": V})) == "T"
    assert node_text(Node({"placeholder": "P", "value": V})) == "P"
    assert node_text(Node({"alt": "Al", "value": V})) == "Al"
    assert node_text(Node({"value": V}, children_text=" kid ")) == ""  # never child text (D12)
    assert node_text(Node({"value": V})) == ""


def test_dom_node_text_hides_a_password_value():
    assert _dom_node_text(Node(SYNCED)) == "Password"
    assert _dom_node_text(Node(EMPTY)) == "Password"  # unchanged


def test_dom_node_text_hides_a_marked_field_value():
    assert _dom_node_text(Node(MARKED)) == "API key"


def test_dom_node_text_unmarked_text_field_unchanged():
    assert _dom_node_text(Node(PLAIN)) == V  # accepted residual D4


async def test_semantic_mismatch_never_prints_the_value(caplog):
    caplog.set_level(logging.DEBUG)
    ok, observed = await validate_semantic_match(Node(SYNCED), "Sign in button")
    assert ok is False
    assert V not in observed
    assert V not in caplog.text


async def test_semantic_match_by_placeholder_still_works(caplog):
    caplog.set_level(logging.DEBUG)
    ok, observed = await validate_semantic_match(Node(SYNCED), "Password")
    assert (ok, observed) == (True, "Password")
    assert V not in caplog.text


async def test_semantic_marked_field_never_prints_the_value(caplog):
    caplog.set_level(logging.DEBUG)
    ok, observed = await validate_semantic_match(Node(MARKED), "Sign in button")
    assert ok is False
    assert V not in observed
    assert V not in caplog.text


async def test_semantic_value_of_a_hidden_field_does_not_match(caplog):
    # The value must not even be matched against: expected text equal to it finds nothing.
    ok, _ = await validate_semantic_match(Node(SYNCED), V)
    assert ok is False


def test_extract_attributes_blanks_the_value_when_hidden():
    assert _extract_dom_node_attributes(Node(SYNCED))["value"] == ""
    assert _extract_dom_node_attributes(Node(MARKED))["value"] == ""


def test_extract_attributes_keeps_other_values():
    assert _extract_dom_node_attributes(Node({"type": "radio", "value": "yes"}))["value"] == "yes"
    assert _extract_dom_node_attributes(Node(PLAIN))["value"] == V


# D12: React keeps a controlled textarea's text as a text CHILD, and a contenteditable's text is
# its child text -- the DOM node of a marked one must report none of it.
EDITABLE = {"contenteditable": "true", SECRET_MARK: ""}
TEXTAREA = {"placeholder": "", SECRET_MARK: ""}


def test_node_text_of_a_hidden_node_with_only_child_text_is_empty():
    assert node_text(Node(EDITABLE, children_text=V)) == ""
    assert node_text(Node(TEXTAREA, children_text=V)) == ""


def test_dom_node_text_hides_a_marked_editable_and_textarea_child_text():
    assert V not in _dom_node_text(Node(EDITABLE, children_text=V))
    assert V not in _dom_node_text(Node(TEXTAREA, children_text=V))


def test_dom_node_text_unmarked_editable_unchanged():
    assert _dom_node_text(Node({"contenteditable": "true"}, children_text=V)) == V


async def test_semantic_marked_editable_never_prints_the_value(caplog):
    caplog.set_level(logging.DEBUG)
    # (an empty surface on an interactive tag is accepted by the existing probe-18 carve-out)
    _, observed = await validate_semantic_match(Node(EDITABLE, children_text=V), "Sign in button")
    assert V not in observed
    assert V not in caplog.text
