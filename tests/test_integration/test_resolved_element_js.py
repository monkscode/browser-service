"""
Real-DOM tests for the candidate identity guard's resolved-element read.

The mocked suite (tests/test_agent/test_candidate_identity_guard.py) never
executes ``_RESOLVED_ELEMENT_JS``: its FakeLocator pattern-matches the JS
string and returns a canned dict. So the payload's SHAPE is asserted there,
but its CONTENT — what the JS actually reads off a live element — was
unverified. These tests close that gap by running the real JS in chromium.

Two defects this lane exists to catch, both found by live probe 2026-07-30:

1. Property shadowing. ``HTMLFormElement`` has ``[OverrideBuiltins]``, so a
   named control shadows same-named properties: ``<form><input name="id">``
   makes ``el.id`` return the INPUT NODE, which Playwright serialises as the
   string ``'ref: <Node>'``. Compared as a plain tag/id by
   ``_identity_mismatch`` that is a SPURIOUS REJECT. ``<input name="id">`` is
   ordinary in CRUD forms.

2. tagName casing. ``el.tagName`` is UPPERCASE for HTML elements, but every
   other element_info producer emits lowercase — the full path lowercases
   explicitly (smart_locator.py ``el.tagName.toLowerCase()``), and browser-use's
   element_data is lowercase. nlrf falls back to element_info['tagName'] for
   ``element_type`` (element_identification.py), so a divergence here changes
   assembler prompt text.

No network: pages are built with page.set_content().

Requires: playwright install chromium
Run: pytest tests/test_integration/test_resolved_element_js.py -m integration -v
"""

import contextlib
import time

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import _identity_mismatch, _read_resolved_element

pytestmark = pytest.mark.integration

# Keys classify_element_type() actually reads (classifier.py). If the JS stops
# supplying one of these, the Tier-0 stamp silently degrades.
CLASSIFIER_KEYS = ("tagName", "className", "role", "type")


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


class TestPayloadContent:
    """What the JS reads off real elements."""

    @pytest.mark.asyncio
    async def test_tagname_is_lowercase_matching_every_other_producer(self):
        """smart_locator's extraction JS lowercases; browser-use's element_data
        is lowercase. The resolved read must not be the odd one out."""
        html = """<button id="go" class="btn">Go</button>
            <input id="email" type="text" class="form-control">
            <select id="d"><option value="1">One</option></select>
            <a id="lnk" href="#">L</a>
            <table><tr id="r"><td id="c">x</td></tr></table>"""
        async with _page(html) as page:
            for selector, expected in (
                ("#go", "button"),
                ("#email", "input"),
                ("#d", "select"),
                ("#d option", "option"),
                ("#lnk", "a"),
                ("#r", "tr"),
                ("#c", "td"),
            ):
                resolved = await _read_resolved_element(page, selector)
                assert resolved is not None, selector
                assert resolved["tagName"] == expected, selector

    @pytest.mark.asyncio
    async def test_supplies_every_key_the_classifier_reads(self):
        html = '<input id="dt" class="flatpickr-input" type="text" role="textbox">'
        async with _page(html) as page:
            resolved = await _read_resolved_element(page, "#dt")
        assert resolved is not None
        for key in CLASSIFIER_KEYS:
            assert key in resolved, key
        assert resolved["type"] == "text"
        assert resolved["role"] == "textbox"
        assert "flatpickr-input" in resolved["className"]

    @pytest.mark.asyncio
    async def test_svg_className_is_not_an_object(self):
        """SVG className is an SVGAnimatedString, not a string."""
        async with _page('<svg id="s" class="icon"><path id="p"/></svg>') as page:
            svg = await _read_resolved_element(page, "#s")
            path = await _read_resolved_element(page, "#p")
        assert isinstance(svg["className"], str)
        assert isinstance(path["className"], str)
        # SVG tagName is already lowercase in the DOM; stays lowercase.
        assert svg["tagName"] == "svg"
        assert path["tagName"] == "path"

    @pytest.mark.asyncio
    async def test_textcontent_is_truncated(self):
        async with _page(f'<div id="big">{"A" * 3000}</div>') as page:
            resolved = await _read_resolved_element(page, "#big")
        assert len(resolved["textContent"]) == 500


class TestFormPropertyShadowing:
    """<form> named controls shadow same-named element properties."""

    FORM = """
        <form id="record_form" class="edit">
          <input name="id" value="42">
          <input name="tagName" value="x">
          <input name="className" value="y">
        </form>"""

    @pytest.mark.asyncio
    async def test_shadowed_fields_never_leak_a_node_reference(self):
        async with _page(self.FORM) as page:
            resolved = await _read_resolved_element(page, "#record_form")
        assert resolved is not None
        for key in ("id", "tagName", "className"):
            assert "Node" not in str(resolved[key]), (key, resolved[key])
            assert isinstance(resolved[key], str)

    @pytest.mark.asyncio
    async def test_shadowed_form_does_not_produce_a_spurious_reject(self):
        """The whole point: an unreadable field is UNKNOWN, never a mismatch."""
        async with _page(self.FORM) as page:
            resolved = await _read_resolved_element(page, "#record_form")
        element_data = {"tagName": "form", "id": "record_form"}
        assert _identity_mismatch(element_data, resolved) == ""

    @pytest.mark.asyncio
    async def test_unshadowed_form_still_reads_normally(self):
        async with _page('<form id="plain" class="c"><input name="q"></form>') as page:
            resolved = await _read_resolved_element(page, "#plain")
        assert resolved["tagName"] == "form"
        assert resolved["id"] == "plain"
        assert _identity_mismatch({"tagName": "form", "id": "plain"}, resolved) == ""

    @pytest.mark.asyncio
    async def test_getattribute_shadowing_degrades_instead_of_raising(self):
        """A control named `getAttribute` shadows the METHOD. The read must
        still return a usable dict rather than throwing inside the accept."""
        html = '<form id="f"><input name="getAttribute" value="z"></form>'
        async with _page(html) as page:
            resolved = await _read_resolved_element(page, "#f")
        assert resolved is None or isinstance(resolved, dict)
        if resolved is not None:
            assert _identity_mismatch({"tagName": "form", "id": "f"}, resolved) == ""


class TestQ08OnRealDom:
    """The defect that motivated the guard, against a real <select>."""

    # The live page's own inline onchange handler moves the `selected`
    # content attribute onto the chosen option — that is PAGE SCRIPT, not
    # Chromium behaviour (verified 2026-07-30). Replicated here so the
    # candidate resolves to the <option> exactly as it did on the bench.
    HTML = """
        <select id="dropdown">
          <option value="" disabled="disabled" selected="selected">Please select an option</option>
          <option value="1">Option 1</option>
          <option value="2">Option 2</option>
        </select>
        <script>
          var d = document.getElementById('dropdown');
          d.onchange = function (e) {
            var os = d.getElementsByTagName('option');
            for (var i = 0; i < os.length; i++) os[i].removeAttribute('selected');
            document.querySelector("#dropdown option[value='" + e.target.value + "']")
                    .setAttribute('selected', 'selected');
          };
        </script>"""

    CANDIDATE = "select#dropdown option[selected='selected']"

    @pytest.mark.asyncio
    async def test_option_candidate_is_rejected_against_a_select_element_data(self):
        async with _page(self.HTML) as page:
            await page.select_option("#dropdown", "2")
            # Precondition: the candidate really does resolve, and to Option 2.
            assert await page.locator(self.CANDIDATE).count() == 1
            resolved = await _read_resolved_element(page, self.CANDIDATE)

        assert resolved["tagName"] == "option"
        assert resolved["textContent"] == "Option 2"
        reason = _identity_mismatch({"tagName": "select", "id": "dropdown"}, resolved)
        assert "select" in reason and "option" in reason

    @pytest.mark.asyncio
    async def test_the_select_itself_is_accepted(self):
        async with _page(self.HTML) as page:
            await page.select_option("#dropdown", "2")
            resolved = await _read_resolved_element(page, "#dropdown")
        assert _identity_mismatch({"tagName": "select", "id": "dropdown"}, resolved) == ""


class TestUnreadableDegradesToNone:
    @pytest.mark.asyncio
    async def test_strict_mode_violation_returns_none_fast(self):
        html = "<ul><li>a</li><li>b</li><li>c</li></ul>"
        async with _page(html) as page:
            assert await page.locator("li").count() == 3
            resolved = await _read_resolved_element(page, "li")
        assert resolved is None
        # None is UNKNOWN, never a mismatch.
        assert _identity_mismatch({"tagName": "li", "id": ""}, None) == ""

    @pytest.mark.asyncio
    async def test_detached_between_count_and_read_is_bounded(self):
        """The accept path calls count() then this read. If the node goes away
        in between, Locator.evaluate would otherwise WAIT — Playwright's
        default is 30s, six times the whole cascade budget
        (custom_action_timeout, 5s), and STEP 1 sits outside the
        asyncio.wait_for that bounds STEP 2. Verified live: unbounded, this
        stalls the full 30s.
        """
        async with _page('<button id="go">Go</button>') as page:
            assert await page.locator("#go").count() == 1
            await page.evaluate("document.getElementById('go').remove()")

            started = time.monotonic()
            resolved = await _read_resolved_element(page, "#go", timeout_ms=400)
            elapsed = time.monotonic() - started

        assert resolved is None
        assert elapsed < 5.0, f"read was not bounded: {elapsed:.1f}s"

    @pytest.mark.asyncio
    async def test_bound_does_not_affect_a_present_element(self):
        async with _page('<button id="go" class="b">Go</button>') as page:
            resolved = await _read_resolved_element(page, "#go", timeout_ms=400)
        assert resolved is not None
        assert resolved["tagName"] == "button"
