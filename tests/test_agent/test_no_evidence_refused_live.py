"""bs #72 on real Chromium: a find_unique_locator call with no element evidence is refused.

"No evidence" = no element data (the index missed and so did the point lookup)
AND no expected text. Today bs ships whatever such a call hands it:

* door 1 — the candidate: a selector that counts 1 is accepted on its count
  alone (S1, S6: the Next.js route announcer, an empty ``role=alert`` div in a
  shadow root, is "the toast");
* door 2 — the cascade's Step 3: whatever lies under the agent's point (K3,
  K3b: the page's content container is "the error message").

After the change both doors are closed by one rule at the top of the action:
``found is False``, ``error_type == "NoElementEvidence"``, ``evidence == "none"``.
The pages are inlined from the #133 spike's repro pages (the announcer markup is
modelled from memory of Next.js, not copied from the app); the inputs (points,
viewport, candidates, descriptions, ``action="get_text"``) are the repro's.

Two of the refused cases are the COST of the rule, pinned on purpose: K4 and S4
refuse a call whose element is the right one. They are listed here so nobody
reads the refusal as free. S2 and S3 are not refused by this rule — they carry
evidence and are rejected by the semantic and identity checks, as today.

    pytest tests/test_agent/test_no_evidence_refused_live.py -m fixture
"""

import pytest
from playwright.async_api import async_playwright

from browser_service.agent.actions import find_unique_locator_action

pytestmark = [pytest.mark.integration, pytest.mark.fixture]

UNION = ".toast, [role='status'], [role='alert'], #toast, div[class*='toast']"
POINT = (1843, 162)  # the scaled point of the real call; empty space on every page below
DESC_ANNOUNCER = "message element displaying the verification code"
DESC_LOGIN = "error message on the login page"
LOGIN_POINT = (960, 108)  # falls on the content block of the container page

_ANNOUNCER = """
  <next-route-announcer></next-route-announcer>
  <script>
    const root = document.querySelector('next-route-announcer').attachShadow({mode: 'open'});
    const d = document.createElement('div');
    d.id = '__next-route-announcer__';
    d.setAttribute('aria-live', 'assertive');
    d.setAttribute('role', 'alert');
    d.style.cssText = 'position:absolute;border:0;height:1px;margin:-1px;padding:0;width:1px;clip:rect(0 0 0 0);overflow:hidden;white-space:nowrap;word-wrap:normal';
    root.appendChild(d);
  </script>
"""

# An announcer and no toast: the toast is gone, the announcer is all the union selector matches.
PAGE_ANNOUNCER = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>toast already gone</title></head>
<body>
  <main>
    <button data-testid="send-code-button">Send code</button>
    <input placeholder="Verification code">
    <button data-testid="verify-button">Verify</button>
  </main>
  {_ANNOUNCER}
</body></html>"""

# A login page with an empty flash area and NO error message; the point lands on the content block.
PAGE_CONTAINER = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>a container under the point</title>
<style>
  body{margin:0;font-family:sans-serif}
  #content{width:960px;margin:0 auto}
  .example{padding:60px 20px 20px 20px;background:#eee}
</style></head>
<body>
  <div id="flash-messages" class="large-6 large-centered columns"></div>
  <div id="content" class="large-12 columns">
    <div class="example">
      <h2>Login Page</h2>
      <h4 class="subheader">This is where you can log into the secure area.</h4>
      <form id="login" action="#"><input name="username" id="username"><input name="password" id="password" type="password"><button class="radius" type="submit">Login</button></form>
    </div>
  </div>
</body></html>"""

# A page where the union selector is RIGHT: one visible div.toast, bottom centre.
PAGE_TOAST = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>toast with class toast</title></head>
<body>
  <main><button data-testid="send-code-button">Send code</button></main>
  <div class="toast" style="position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#111;color:#fff;border-radius:8px;padding:12px 20px;font-size:15px;font-weight:600">Your code is 4407</div>
</body></html>"""


@pytest.fixture
async def browser():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True)
        try:
            yield b
        finally:
            await b.close()


async def _open(browser, html):
    ctx = await browser.new_context(viewport={"width": 1920, "height": 1080})
    page = await ctx.new_page()
    await page.set_content(html, wait_until="domcontentloaded")
    return ctx, page


async def _call(page, **kw):
    """One call as registration.py makes it for a get_text step, minus the browser-use session."""
    args = {
        "x": POINT[0],
        "y": POINT[1],
        "element_id": "elem_2",
        "element_description": DESC_ANNOUNCER,
        "expected_text": None,
        "candidate_locator": UNION,
        "element_data": None,
        "page": page,
        "iframe_context": None,
        "is_collection": None,
        "browser_session": None,
        "action": "get_text",
        "vision_point": POINT,
    }
    args.update(kw)
    return await find_unique_locator_action(**args)


def _assert_refused(out):
    assert out["found"] is False
    assert out["error_type"] == "NoElementEvidence"
    assert out["evidence"] == "none"
    assert not out.get("best_locator")


class TestRefused:
    async def test_s1_union_candidate_on_the_announcer_page(self, browser):
        """Door 1: the union counts 1 (the empty announcer) and used to ship as 'the toast'."""
        ctx, page = await _open(browser, PAGE_ANNOUNCER)
        try:
            _assert_refused(await _call(page))
        finally:
            await ctx.close()

    async def test_s6_one_guessed_selector_alone(self, browser):
        """Door 1 again: a single guessed selector, ``[role='alert']``, no evidence."""
        ctx, page = await _open(browser, PAGE_ANNOUNCER)
        try:
            _assert_refused(await _call(page, candidate_locator="[role='alert']"))
        finally:
            await ctx.close()

    async def test_k3_candidate_that_counts_zero_and_a_container_under_the_point(self, browser):
        """Door 2: '#flash' counts 0, the cascade then ships the container under the point."""
        ctx, page = await _open(browser, PAGE_CONTAINER)
        try:
            out = await _call(
                page,
                candidate_locator="#flash",
                element_description=DESC_LOGIN,
                x=LOGIN_POINT[0],
                y=LOGIN_POINT[1],
                vision_point=LOGIN_POINT,
            )
            _assert_refused(out)
        finally:
            await ctx.close()

    async def test_k3b_no_candidate_at_all(self, browser):
        """Door 2 with nothing sent but the point: the same container used to ship."""
        ctx, page = await _open(browser, PAGE_CONTAINER)
        try:
            out = await _call(
                page,
                candidate_locator=None,
                element_description=DESC_LOGIN,
                x=LOGIN_POINT[0],
                y=LOGIN_POINT[1],
                vision_point=LOGIN_POINT,
            )
            _assert_refused(out)
        finally:
            await ctx.close()

    async def test_k4_the_right_element_from_the_point_alone_is_refused_too(self, browser):
        """THE COST, pinned: the point is the visible toast's centre and the cascade used to
        ship it, correctly. With no index and no text bs cannot tell that from a guess, so it
        refuses — the agent must send the index or the toast's text."""
        ctx, page = await _open(browser, PAGE_TOAST)
        try:
            box = await page.locator(".toast").bounding_box()
            pt = (round(box["x"] + box["width"] / 2), round(box["y"] + box["height"] / 2))
            out = await _call(page, candidate_locator=None, x=pt[0], y=pt[1], vision_point=pt)
            _assert_refused(out)
        finally:
            await ctx.close()

    async def test_s4_a_right_candidate_with_a_wrong_point_is_refused_too(self, browser):
        """THE DOOR-1 COST, pinned: ``div.toast`` is the right element and counts 1, so it used
        to be accepted on its count (the point is wrong, on empty space). No evidence, so it
        is refused like any other candidate."""
        ctx, page = await _open(browser, PAGE_TOAST)
        try:
            _assert_refused(await _call(page, candidate_locator="div.toast"))
        finally:
            await ctx.close()


class TestNotRefusedByThisRule:
    """Calls that carry evidence are rejected by their own checks, exactly as before."""

    async def test_s2_expected_text_the_semantic_check_still_rejects(self, browser):
        ctx, page = await _open(browser, PAGE_ANNOUNCER)
        try:
            out = await _call(page, expected_text="Your code is 4407")
            assert out["found"] is False
            assert "evidence" not in out
        finally:
            await ctx.close()

    async def test_s3_element_data_the_identity_check_still_rejects(self, browser):
        ctx, page = await _open(browser, PAGE_ANNOUNCER)
        try:
            out = await _call(
                page,
                element_data={
                    "tagName": "div",
                    "id": "toast-msg",
                    "textContent": "Your code is 4407",
                    "className": "",
                },
            )
            assert out["found"] is False
            assert "evidence" not in out
        finally:
            await ctx.close()


@pytest.mark.parametrize("candidate", [None, "span.sync-chip"])
async def test_the_frame_container_as_element_data_is_refused_in_a_real_frame(browser, candidate):
    """Review I1: the point lookup hands over the <iframe> node when the element under the
    point inside the frame is not indexed. The candidate block may use it (a candidate rejected
    by the identity check, or none, falls on), the cascade drops it, so the call names nothing
    and is refused instead of shipping whatever lies under the point."""
    from tests.test_agent.test_frame_prefix_exits import _in_frame

    ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
    try:
        kwargs = {"candidate_locator": candidate} if candidate else {}
        out = await _in_frame(
            await ctx.new_page(),
            "span.sync-chip",
            element_description="status chip",
            element_data={"tagName": "iframe", "id": "editor-frame", "textContent": ""},
            **kwargs,
        )
        assert out["found"] is False
        assert out["error_type"] == "NoElementEvidence"
        assert out["evidence"] == "none"
        assert "best_locator" not in out
    finally:
        await ctx.close()
