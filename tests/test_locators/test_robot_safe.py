"""bs #60: a locator the engine returns must reach the keyword unchanged through a Robot Framework cell.

robot_safe is the rule (Robot 7.4.2's tokenizer + NLRF's css= prefix), written without robotframework.
The builder test feeds every value shape Robot changes into every synchronous builder and asserts each
output is robot-safe or absent. The async cascade steps (text-first, the description loop, the
accessibility search, the agent candidate) are covered by the robot_safe filters where they collect
candidates AND by the exit backstop in agent/actions.py.
"""

import types

import pytest

from browser_service.agent.registration import _detect_iframe_context
from browser_service.locators.css_tokens import css_attr, css_class, css_id, quote_value, robot_safe
from browser_service.locators.smart_locator import (
    _build_coordinate_strategies,
    _build_element_data_candidates,
)
from browser_service.locators.validation import convert_to_playwright_locator

B = chr(92)
# Every shape Robot 7.4.2 changes inside a cell (Task 1 runner rows), plus the quote / id shapes.
UNSAFE_VALUES = [
    "Pay  now",
    "Pay\tnow",
    "Pay\nnow",
    "Pay\rnow",
    "Pay  now",
    "Pay now",
    "Pay now",
    "Pay\x85now",
    "Pay ${amount}",
    "Pay @{amount}",
    "Pay &{amount}",
    "Pay %{amount}",
    "Pay ${amount",
    "C:" + B + "run",
    "a" + B + '"b',
]
OTHER_VALUES = [
    'Say "hi"',
    "It's",
    'He said "it\'s"',
    "-side",
    "éclair",
    " lead",
    "trail ",
    "Pay now",
    "x",
]


@pytest.mark.parametrize(
    "locator",
    [
        '[aria-label="Pay now"]',
        "role=button[name='Say \"hi\"']",
        'text="Proceed to checkout "',
        "#plainId",
        "id=x",
        "xpath=//a[@title='Item [3]']",
        'iframe[name="payframe"] >>> text="Inside"',
        "button >> nth=0",
    ],
)
def test_robot_safe_accepts(locator):
    assert robot_safe(locator)


@pytest.mark.parametrize("value", UNSAFE_VALUES)
def test_robot_safe_rejects_every_value_robot_changes(value):
    assert not robot_safe(f'[aria-label="{value}"]')


@pytest.mark.parametrize("locator", ["#-side", "#éclair", "#1st", " text=x", "text=x ", ""])
def test_robot_safe_rejects_comments_and_edge_whitespace(locator):
    assert not robot_safe(locator)


@pytest.mark.parametrize("value", UNSAFE_VALUES + OTHER_VALUES)
def test_quote_value_output_is_always_robot_safe(value):
    for collapse in (False, True):
        q = quote_value(value, collapse=collapse)
        assert q == "" or robot_safe(f"[x={q}]"), (value, collapse, q)


def test_quote_value_with_both_quote_kinds_is_skipped():
    both = 'He said "it\'s"'
    assert quote_value(both) == ""
    assert quote_value(both + " ${x}") == ""
    assert quote_value('say "hi"') == "'say \"hi\"'"
    assert quote_value("it's") == '"it\'s"'


def test_quote_value_collapse_folds_whitespace():
    assert quote_value("  ", collapse=True) == ""
    assert quote_value("\n", collapse=True) == ""
    assert quote_value(" x ", collapse=True) == '"x"'
    assert quote_value("a \t\n b", collapse=True) == '"a b"'
    assert css_attr("name", "  ") == ""


# Printable values with no quote, backslash, 2-space run or variable start: shipped byte for byte.
PLAIN = [
    "Pay now",
    "Sign in",
    "search",
    "user_name",
    "btn-primary",
    "Cart (3 items)",
    "50% off",
    "a:b",
    "x/y",
    "w-1/2",
    "md:flex",
    "form:main",
    "1st",
    "123",
    "Ünïcödé",
    "日本語",
    "👍 Like",
    "a=b",
    "[x]",
    "x.y",
    "Total: $5",
    "{x}",
    "@home",
    "&amp",
    "a#b",
    "Proceed to checkout (3 items)",
    "Item [3]",
    "it>>nth=0",
    "nav_item",
    "Header2",
    "_x",
    "café",
    "Q&A",
    "100%",
    "a,b",
    "x+y",
]


@pytest.mark.parametrize("value", PLAIN)
def test_quote_value_plain_value_is_byte_identical(value):
    assert quote_value(value) == f'"{value}"'


def _iframe_node(**attrs):
    pos = types.SimpleNamespace(x=0, y=0, width=500, height=500)
    return types.SimpleNamespace(node_name="IFRAME", attributes=attrs, absolute_position=pos)


@pytest.mark.parametrize("value", UNSAFE_VALUES + OTHER_VALUES)
def test_every_builder_output_is_robot_safe_or_absent(value):
    out = [css_attr("title", value), css_class(value), css_id(value)]
    out.append(convert_to_playwright_locator(f"name={value}")[0])
    for key in (
        "id",
        "dataTestId",
        "name",
        "ariaLabel",
        "placeholder",
        "parentId",
        "parentClassName",
    ):
        data = {"tagName": "input", "type": "text", key: value}
        out += [c["locator"] for c in _build_element_data_candidates(data)]
    keys = ("id", "name", "dataTestId", "dataTest", "dataQa", "ariaLabel", "placeholder", "title")
    for key in keys + ("innerText", "className", "parentId", "parentClass", "href"):
        data = dict.fromkeys(
            keys + ("innerText", "className", "parentId", "parentClass", "href"), ""
        )
        data.update(tagName="a", role="link", textContent=value, value="", alt="", src="", type="")
        data.update(attributes={}, coordinates={"x": 1, "y": 1}, ariaDescribedby="", siblingIndex=2)
        data[key] = value
        out += [s["locator"] for s in _build_coordinate_strategies(data)]
    for attr in ("id", "name", "title"):
        prefix, _ = _detect_iframe_context({0: _iframe_node(**{attr: value})}, (10, 10))
        out.append(prefix)
    # convert_to_playwright_locator leaves a value it cannot quote as given: the candidate check refuses it
    unsafe = [o for o in out if o and not robot_safe(o) and not o.startswith("name=")]
    assert unsafe == [], (value, unsafe)


@pytest.mark.parametrize(
    ("own_id", "expected"),
    [
        ("-side", '[id="-side"]'),
        ("éclair", '[id="éclair"]'),
        ("Ünïcödé", '[id="Ünïcödé"]'),
        ("plainId", "#plainId"),
        ("123", '[id="123"]'),
    ],
)
def test_own_id_ships_as_id_selector_nlrf_prefixes(own_id, expected):
    cands = _build_element_data_candidates({"tagName": "input", "type": "text", "id": own_id})
    id_cands = [c for c in cands if c["type"] in ("id", "id-attr")]
    assert [c["locator"] for c in id_cands] == [expected]
