"""
The legacy (page) path of validate_semantic_match never reads back a hidden value
(NLRF #131 Part 2, owner decision D8). Real Chromium; made-up value only.
"""

import json
import logging

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import _read_resolved_element
from browser_service.agent.registration import _mark_secret_field
from browser_service.locators.hidden_value import HIDES_VALUE_JS, SECRET_MARK
from browser_service.locators.read_target import resolve_read_address
from browser_service.locators.smart_locator import (
    _get_element_accessibility_info,
    find_unique_locator_at_coordinates,
    validate_semantic_match,
)

pytestmark = pytest.mark.integration

V = "Zq131-made-up-7"


async def _observe(html, caplog):
    caplog.set_level(logging.DEBUG)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.set_content(html)
            await page.locator("#f").fill(V)
            return await validate_semantic_match(None, "Sign in button", page=page, locator="#f")
        finally:
            await browser.close()


async def test_password_input_value_not_read(caplog):
    ok, observed = await _observe('<input id="f" type="password">', caplog)
    assert ok is False
    assert V not in observed
    assert V not in caplog.text


async def test_marked_text_input_value_not_read(caplog):
    ok, observed = await _observe('<input id="f" type="text" nlrf-secret="">', caplog)
    assert ok is False
    assert V not in observed
    assert V not in caplog.text


async def test_unmarked_text_input_value_still_read(caplog):
    """The known positive: the probe can see a value at all (accepted residual D4)."""
    _, observed = await _observe('<input id="f" type="text">', caplog)
    assert observed == V


# ---- D12 / D13: a marked textarea or contenteditable keeps the value as CHILD text (what React
# leaves in the DOM), a label's mark goes to the field it controls. Set directly in the HTML.

TEXTAREA_HTML = '<textarea id="f" role="none" nlrf-secret="">' + V + "</textarea>"
EDITABLE_HTML = '<div id="f" contenteditable nlrf-secret="">' + V + "</div>"
UNMARKED_EDITABLE_HTML = '<div id="f" contenteditable>' + V + "</div>"


async def _on_page(html, fn):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.set_content(html)
            return await fn(page)
        finally:
            await browser.close()


@pytest.mark.parametrize("html", [TEXTAREA_HTML, EDITABLE_HTML], ids=["textarea", "editable"])
async def test_legacy_path_never_reads_marked_child_text(html, caplog):
    caplog.set_level(logging.DEBUG)

    async def run(page):
        return await validate_semantic_match(None, "Sign in button", page=page, locator="#f")

    ok, observed = await _on_page(html, run)
    assert ok is False
    assert V not in observed
    assert V not in caplog.text


async def test_legacy_path_unmarked_child_text_still_read(caplog):
    """The known positive: the probe can see child text at all."""
    caplog.set_level(logging.DEBUG)

    async def run(page):
        return await validate_semantic_match(None, "Sign in button", page=page, locator="#f")

    _, observed = await _on_page(UNMARKED_EDITABLE_HTML, run)
    assert observed == V


async def _coordinate_probe(html, caplog):
    caplog.set_level(logging.DEBUG)

    async def run(page):
        box = await page.locator("#f").bounding_box()
        return await find_unique_locator_at_coordinates(
            page,
            box["x"] + box["width"] / 2,
            box["y"] + box["height"] / 2,
            "elem_1",
            "Sign in button",
        )

    result = await _on_page(html, run)
    return json.dumps(result, default=str), caplog.text


@pytest.mark.parametrize("html", [TEXTAREA_HTML, EDITABLE_HTML], ids=["textarea", "editable"])
async def test_coordinate_path_never_reads_marked_child_text(html, caplog):
    dumped, log = await _coordinate_probe(html, caplog)
    assert V not in dumped
    assert V not in log


async def test_coordinate_path_unmarked_child_text_still_read(caplog):
    dumped, log = await _coordinate_probe(UNMARKED_EDITABLE_HTML, caplog)
    assert V in dumped or V in log


async def test_resolved_element_read_blanks_a_marked_textarea():
    async def run(page):
        return await _read_resolved_element(page, "#f")

    assert (await _on_page(TEXTAREA_HTML, run))["textContent"] == ""
    assert (await _on_page('<textarea id="f">' + V + "</textarea>", run))["textContent"] == V


async def test_mark_goes_to_the_field_a_label_points_to():
    html = '<label for="f" id="l">Token</label><input id="f">'

    async def run(page):
        await _mark_secret_field(page, "#l", None)
        return [
            await page.locator("#l").get_attribute(SECRET_MARK),
            await page.locator("#f").get_attribute(SECRET_MARK),
        ]

    assert await _on_page(html, run) == [None, ""]


async def test_mark_on_a_plain_field_is_unchanged():
    async def run(page):
        await _mark_secret_field(page, "#f", None)
        return await page.locator("#f").get_attribute(SECRET_MARK)

    assert await _on_page('<input id="f">', run) == ""


# ---- D14 / D15: a rich-text editor keeps typed text in CHILD elements (Quill / ProseMirror <p>,
# CodeMirror .cm-line); an element INSIDE a marked element hides its value too. The mark follows
# .control only for a <label>.
_BOX = 'style="width:320px;height:60px;border:1px solid #888"'
_P = 'style="height:40px;margin:0"'
MARKED_EDITOR_HTML = f'<div id="ed" contenteditable nlrf-secret="" {_BOX}><p {_P}>{V}</p></div>'
UNMARKED_EDITOR_HTML = f'<div id="ed" contenteditable {_BOX}><p {_P}>{V}</p></div>'
EMPTY_EDITOR_HTML = f'<div id="ed" contenteditable {_BOX}><p {_P}><br></p></div>'

P_XPATH_JS = """el => { const p = []; for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
  let i = 1; for (let s = n.previousElementSibling; s; s = s.previousElementSibling) if (s.tagName === n.tagName) i++;
  p.unshift(n.tagName.toLowerCase() + '[' + i + ']'); } return p.join('/'); }"""


async def _centre(page, selector):
    box = await page.locator(selector).bounding_box()
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


async def test_predicate_hides_inside_a_marked_element_only():
    html = (
        MARKED_EDITOR_HTML
        + '<input id="pw" type="password"><label id="l" for="t">T</label><input id="t">'
        + '<form id="fm"><input id="m" nlrf-secret=""></form><button id="b">Go</button>'
    )

    async def run(page):
        return {
            sel: await page.locator(sel).evaluate(HIDES_VALUE_JS)
            for sel in ("#ed p", "#ed", "#pw", "#m", "#l", "#fm", "#b")
        }

    assert await _on_page(html, run) == {
        "#ed p": True,
        "#ed": True,
        "#pw": True,
        "#m": True,
        "#l": False,
        "#fm": False,  # an ANCESTOR of a marked element is not hidden (D12)
        "#b": False,
    }


async def _child_probes(html, caplog, prepare=None):
    """Every reader that can land on the editor's <p>; returns (dumped results, log text)."""
    caplog.set_level(logging.DEBUG)

    async def run(page):
        if prepare:
            await prepare(page)
        x, y = await _centre(page, "#ed p")
        coord = await find_unique_locator_at_coordinates(page, x, y, "elem_1", "Editor text")
        address = await resolve_read_address(
            page, "Editor text", {"xpath": await page.locator("#ed p").evaluate(P_XPATH_JS)}, (x, y)
        )
        _, observed = await validate_semantic_match(
            None, "Sign in button", page=page, locator="#ed p"
        )
        resolved = await _read_resolved_element(page, "#ed p")
        return {"coord": coord, "address": address, "observed": observed, "resolved": resolved}

    out = await _on_page(html, run)
    return out, json.dumps(out, default=str), caplog.text


def _assert_no_value(dumped, log):
    for text in (dumped, log):
        assert V not in text
        assert V.casefold() not in text.casefold()


async def test_child_of_a_marked_editor_is_never_read(caplog):
    out, dumped, log = await _child_probes(MARKED_EDITOR_HTML, caplog)
    _assert_no_value(dumped, log)
    assert out["resolved"]["textContent"] == ""


async def test_real_sequence_mark_then_fill_then_read_the_child(caplog):
    async def prepare(page):
        await _mark_secret_field(page, "#ed", None)
        await page.locator("#ed").fill(V)
        assert V in await page.locator("#ed p").text_content()  # the value is really in the child

    _, dumped, log = await _child_probes(EMPTY_EDITOR_HTML, caplog, prepare)
    _assert_no_value(dumped, log)


async def test_child_of_an_unmarked_editor_is_still_read(caplog):
    """The known positive: the probes can see a child's text at all."""
    _, dumped, log = await _child_probes(UNMARKED_EDITOR_HTML, caplog)
    assert V in dumped or V in log


async def test_label_mark_goes_to_its_control_not_the_label():
    html = '<label for="f" id="l">Token</label><input id="f">'

    async def run(page):
        await _mark_secret_field(page, "label", None)
        return [
            await page.locator("#l").get_attribute(SECRET_MARK),
            await page.locator("#f").get_attribute(SECRET_MARK),
        ]

    assert await _on_page(html, run) == [None, ""]


async def test_mark_stays_on_a_custom_element_host_that_exposes_a_control():
    html = """<x-field id="host"></x-field><script>
      customElements.define('x-field', class extends HTMLElement {
        constructor() { super(); this.inner = document.createElement('input'); this.inner.id = 'inner';
          this.appendChild(this.inner); }
        get control() { return this.inner; }
      });</script>"""

    async def run(page):
        await _mark_secret_field(page, "#host", None)
        return [
            await page.locator("#host").get_attribute(SECRET_MARK),
            await page.locator("#inner").get_attribute(SECRET_MARK),
        ]

    assert await _on_page(html, run) == ["", None]


# ---- D16: an editor that auto-links turns a URL-shaped secret into <a href="U">U</a> INSIDE the
# marked editor; such a link gives away neither its text (accessible name) nor its href.
U = "https://hooks.example.invalid/zq131-made-up-7"
U_MARKER = "zq131-made-up-7"
LINK_ED = (  # wide box: a wrapped link has an empty bounding-box centre
    f'<div id="ed" contenteditable nlrf-secret="" style="width:700px;border:1px solid #888">'
    f'<p {_P}>see <a id="lk" href="{U}">{U}</a></p></div>'
)
LINK_ED_UNMARKED = LINK_ED.replace(' nlrf-secret=""', "")
PLAIN_LINK = '<a id="lk" href="/docs/guide">Read the guide</a>'


def _assert_no_link_value(text):
    assert U not in text
    assert U_MARKER not in text.casefold()


async def _link_info(html):
    async def run(page):
        x, y = await _centre(page, "#lk")
        return await _get_element_accessibility_info(page, x, y)

    return await _on_page(html, run)


async def test_link_in_a_marked_editor_has_no_accessible_name():
    info = await _link_info(LINK_ED)
    assert info["role"] == "link"
    assert not info["accessibleName"]
    _assert_no_link_value(json.dumps(info, default=str))


async def test_link_in_an_unmarked_editor_is_still_named_by_its_text():
    """The known positive: step 5 can read a link's text at all."""
    assert (await _link_info(LINK_ED_UNMARKED))["accessibleName"] == U


async def test_a_normal_link_still_gets_its_text_as_name():
    assert (await _link_info(PLAIN_LINK))["accessibleName"] == "Read the guide"


@pytest.mark.parametrize("expected", ["", "Editor link"], ids=["no-text", "with-text"])
async def test_coordinate_path_never_gives_away_a_link_in_a_marked_editor(expected, caplog):
    caplog.set_level(logging.DEBUG)

    async def run(page):
        x, y = await _centre(page, "#lk")
        return await find_unique_locator_at_coordinates(page, x, y, "elem_1", expected)

    out = json.dumps(await _on_page(LINK_ED, run), default=str)
    _assert_no_link_value(out)
    _assert_no_link_value(caplog.text)


async def test_coordinate_path_unmarked_link_href_still_read(caplog):
    caplog.set_level(logging.DEBUG)

    async def run(page):
        x, y = await _centre(page, "#lk")
        return await find_unique_locator_at_coordinates(page, x, y, "elem_1", "")

    out = json.dumps(await _on_page(LINK_ED_UNMARKED, run), default=str)
    assert U_MARKER in out or U_MARKER in caplog.text


def _without_id(html):
    return html.replace(' id="lk"', "")


async def test_coordinate_strategies_never_carry_a_marked_links_href(caplog):
    """A second link makes role=link ambiguous and no id ends the cascade early, so xpath-href runs."""
    caplog.set_level(logging.DEBUG)

    async def run(page):
        x, y = await _centre(page, "#ed a")
        return await find_unique_locator_at_coordinates(page, x, y, "elem_1", "")

    out = await _on_page(_without_id(LINK_ED) + '<a href="/other">Other</a>', run)
    dumped = json.dumps(out, default=str)
    _assert_no_link_value(dumped)
    _assert_no_link_value(caplog.text)


async def test_coordinate_strategies_unmarked_link_href_still_used(caplog):
    caplog.set_level(logging.DEBUG)

    async def run(page):
        x, y = await _centre(page, "#ed a")
        return await find_unique_locator_at_coordinates(page, x, y, "elem_1", "")

    out = await _on_page(_without_id(LINK_ED_UNMARKED) + '<a href="/other">Other</a>', run)
    assert U_MARKER in json.dumps(out, default=str)
