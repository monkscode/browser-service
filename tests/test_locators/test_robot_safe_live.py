"""bs #60 on real Chromium: a shipped locator is one Robot Framework hands over unchanged.

bs validates a locator in Playwright; NLRF then writes it into a Robot test cell, and
Robot 7.4.2 changes some strings on the way: it drops a backslash, splits a cell on 2+
whitespace / a TAB / a line break, and reads "${" as a variable. NLRF adds "css=" only
to "#" or "." followed by an ASCII letter or "_", so a bare "#-side" cell is a comment.
Each row runs the real find_unique_locator_action on an inline page and pins the exact
string shipped, its Playwright count (1) and the element it resolves to (the pointed one).

Pages and call inputs are the bs #60 evaluation's probe pages; the expected strings are
its prototype's recorded output (the #60 repro row is new and was measured here). On
0637613 the item rows fail, each test's docstring names what 0637613 shipped; the span
row and the two controls pass on both.

    pytest tests/test_locators/test_robot_safe_live.py -m fixture
"""

import html
import types

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import find_unique_locator_action
from browser_service.agent.registration import _detect_iframe_context

pytestmark = [pytest.mark.integration, pytest.mark.fixture]

B = chr(92)  # a real backslash
SZ = 'style="display:block;width:420px;min-height:22px;margin:4px"'


def _page(body):
    return f'<!doctype html><html><head><meta charset="utf-8"></head><body>{body}</body></html>'


def _aria_button(label, text="x"):
    return f'<button aria-label="{html.escape(label, quote=True)}" {SZ}>{text}</button>'


def _aria_data(label, text="x"):
    return {"tagName": "button", "ariaLabel": label, "textContent": text, "id": "", "className": ""}


@pytest.fixture
async def page():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True)
        try:
            yield await b.new_page(viewport={"width": 1280, "height": 900})
        finally:
            await b.close()


async def _act(page, body, pick, frame=None, **kw):
    """One call as the evaluation's probe made it: the point is 5 px into the pointed element."""
    await page.set_content(_page(body), wait_until="load")
    box = await (page.frame_locator(frame) if frame else page).locator(pick).first.bounding_box()
    kw.setdefault("element_description", "the pay control")
    kw.setdefault("action", "click")
    return await find_unique_locator_action(
        x=box["x"] + 5, y=box["y"] + 5, element_id="elem_1", page=page, **kw
    )


async def _assert_ships(page, out, expected, pick, frame=None):
    assert out.get("found") is True, out
    assert out["best_locator"] == expected
    if " >>> " in expected:
        left, right = expected.split(" >>> ", 1)
        shipped = page.frame_locator(left).locator(right)
    else:
        shipped = page.locator(expected)
    assert await shipped.count() == 1
    pointed = (
        await (page.frame_locator(frame) if frame else page).locator(pick).first.element_handle()
    )
    assert await (await shipped.first.element_handle()).evaluate("(a, b) => a === b", pointed)


def _assert_withheld(out):
    assert out.get("found") is False, out
    assert not out.get("best_locator")
    assert not out.get("error_type"), out  # the action's catch-all sets it on a crash


class TestItemRows:
    async def test_issue_60_repro_label_with_double_quotes(self, page):
        """Measured here (new page). STEP 0's [aria-label="…"] is not valid CSS for this value on
        either copy, so the role route ships. 0637613 shipped
        role=button[name="Remove \\"Blue shirt\\" from cart"] (Robot drops the backslashes); the
        branch picks the quote kind the value needs."""
        body = _aria_button('Remove "Blue shirt" from cart', "Remove") + _aria_button(
            'Remove "Red shirt" from cart', "Remove"
        )
        out = await _act(
            page,
            body,
            "button >> nth=0",
            element_description="the remove button for the Blue shirt",
            element_data=_aria_data('Remove "Blue shirt" from cart', "Remove"),
        )
        await _assert_ships(
            page, out, "role=button[name='Remove \"Blue shirt\" from cart']", "button >> nth=0"
        )

    async def test_two_spaces_in_a_label_fall_to_the_role_name(self, page):
        """0637613 shipped [aria-label="Pay  now"] (Robot splits the cell at the two spaces)."""
        out = await _act(
            page, _aria_button("Pay  now"), "button", element_data=_aria_data("Pay  now")
        )
        await _assert_ships(page, out, 'role=button[name="Pay now"]', "button")

    async def test_a_variable_start_in_a_label_is_withheld(self, page):
        """0637613 shipped [aria-label="Pay ${amount}"] (Robot reads ${amount} as a variable)."""
        out = await _act(
            page, _aria_button("Pay ${amount}"), "button", element_data=_aria_data("Pay ${amount}")
        )
        _assert_withheld(out)

    async def test_a_label_with_both_quote_kinds_is_withheld(self, page):
        """0637613 shipped role=button[name="He said \\"it's\\" ok"] (Robot drops the backslashes)."""
        both = 'He said "it\'s" ok'
        body = _aria_button(both) + f"<button {SZ}>y</button>"
        out = await _act(page, body, "button >> nth=0", element_data=_aria_data(both))
        _assert_withheld(out)

    async def test_an_own_id_nlrf_does_not_prefix_ships_as_an_id_attribute(self, page):
        """0637613 shipped #-side (a bare "#-side" cell is a Robot comment)."""
        out = await _act(
            page,
            f'<input id="-side" {SZ}><input {SZ}>',
            "input >> nth=0",
            action="fill",
            element_description="the field",
            element_data={"tagName": "input", "id": "-side", "className": "", "type": "text"},
        )
        await _assert_ships(page, out, '[id="-side"]', "input >> nth=0")

    async def test_a_frame_id_with_both_quote_kinds_falls_to_the_frame_name(self, page):
        """0637613 shipped the prefix iframe[id="a\\"b'c"] and
        iframe[id="a\\"b'c"] >>> text="Inside" (Robot drops the backslash)."""
        fid = "a\"b'c"
        inner = f'<meta charset="utf-8"><button {SZ}>Inside</button>'
        body = (
            f"<button {SZ}>Outside</button>"
            f'<iframe id="{html.escape(fid, quote=True)}" name="payframe" title="Payment" '
            f'srcdoc="{html.escape(inner, quote=True)}" style="width:460px;height:120px"></iframe>'
        )
        await page.set_content(_page(body), wait_until="load")
        fbox = await page.locator("iframe").bounding_box()
        node = types.SimpleNamespace(
            node_name="IFRAME",
            attributes={"id": fid, "name": "payframe", "title": "Payment"},
            absolute_position=types.SimpleNamespace(
                x=fbox["x"], y=fbox["y"], width=fbox["width"], height=fbox["height"]
            ),
        )
        ibox = await page.frame_locator("iframe").locator("button").bounding_box()
        prefix, _ = _detect_iframe_context({7: node}, (ibox["x"] + 5, ibox["y"] + 5))
        assert prefix == 'iframe[name="payframe"]'
        out = await _act(
            page,
            body,
            "button",
            frame="iframe",
            expected_text="Inside",
            element_description="the inside control",
            iframe_context=prefix,
        )
        await _assert_ships(
            page, out, 'iframe[name="payframe"] >>> text="Inside"', "button", frame="iframe"
        )

    async def test_text_with_a_line_break_ships_folded(self, page):
        """The July shape: a line break + 20 spaces. 0637613 shipped text="…Here\\n<20 spaces>Find…"
        (Robot ends the line at the break)."""
        text = (
            "Smarter Support Starts Here\n"
            + " " * 20
            + "Find answers faster with intelligent search capabilities."
        )
        body = f"<button {SZ}>{text}</button><button {SZ}>Close</button>"
        out = await _act(
            page,
            body,
            "button >> nth=0",
            expected_text=text,
            element_description="the support panel",
        )
        await _assert_ships(
            page,
            out,
            'text="Smarter Support Starts Here Find answers faster with intelligent search capabilities."',
            "button >> nth=0",
        )

    async def test_an_agent_candidate_with_backslashes_is_skipped(self, page):
        """The candidate text="Say \\"hi\\" now" counts 1 in Playwright; 0637613 shipped it verbatim.
        The branch skips it where candidates are collected and the next step ships the single-quoted
        text= form (only this row pins that half of the rule)."""
        cand = 'text="Say ' + B + '"hi' + B + '" now"'
        body = f"<button {SZ}>Say &quot;hi&quot; now</button><button {SZ}>other</button>"
        out = await _act(
            page, body, "button >> nth=0", candidate_locator=cand, expected_text='Say "hi" now'
        )
        await _assert_ships(page, out, "text='Say \"hi\" now'", "button >> nth=0")


class TestUnchanged:
    """Same string on 0637613 and on the branch."""

    async def test_item_5_span_still_ships_the_immediate_text(self, page):
        body = (
            f"<span {SZ}>Proceed to checkout <b>(3 items)</b></span>"
            f"<span {SZ}>Continue shopping</span>"
        )
        out = await _act(
            page,
            body,
            "span >> nth=0",
            expected_text="Proceed to checkout (3 items)",
            element_description="the checkout note",
        )
        await _assert_ships(page, out, 'text="Proceed to checkout "', "span >> nth=0")

    async def test_control_plain_own_id(self, page):
        out = await _act(
            page,
            f'<input id="plainId" {SZ}><input {SZ}>',
            "input >> nth=0",
            action="fill",
            element_description="the field",
            element_data={"tagName": "input", "id": "plainId", "className": "", "type": "text"},
        )
        await _assert_ships(page, out, "#plainId", "input >> nth=0")

    async def test_control_plain_label(self, page):
        out = await _act(
            page, _aria_button("Pay now"), "button", element_data=_aria_data("Pay now")
        )
        await _assert_ships(page, out, '[aria-label="Pay now"]', "button")
