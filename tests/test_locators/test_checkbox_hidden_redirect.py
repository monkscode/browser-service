"""
Unit tests for hidden-input redirection (G3 / Task C).

Styled switches hide the real <input> (display:none) and draw a
slider/label instead (ASTPP: `input#switch23.onoffswitch-checkbox` is
display:none inside `label.switch`). The engine resolved the input —
semantically correct — but RF Browser Click/Check Checkbox needs a
bounding box, so the generated test times out at runtime while
discovery-time JS interaction succeeds: green generation, red test.

Covered seams:
  - resolve_hidden_input_proxy(): probe + proxy selection order
    (label[for] → wrapping label → adjacent sibling)
  - find_locator(): handler results redirect and carry element_info
    {hidden_input, input_locator, proxy_kind}
  - _find_element_by_expected_text(): the text-first checkbox detour
    applies the same redirect
"""

from browser_service.locators.classifier import ElementTypeInfo
from browser_service.locators.handlers.checkbox import (
    find_locator,
    resolve_hidden_input_proxy,
)
from browser_service.locators.smart_locator import (
    _find_element_by_expected_text,
)


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class FakeFirst:
    """Stands in for Locator.first: visibility, probe evaluate, attrs."""

    def __init__(self, visible=True, probe=None, attrs=None):
        self._visible = visible
        self._probe = probe
        self._attrs = attrs or {}

    async def is_visible(self) -> bool:
        return self._visible

    async def evaluate(self, js: str):
        return self._probe

    async def get_attribute(self, name: str):
        return self._attrs.get(name)


class FakeLocator:
    def __init__(self, count: int, first: FakeFirst = None):
        self._count = count
        self.first = first or FakeFirst()

    async def count(self) -> int:
        return self._count


class FakeContext:
    """Maps selector -> FakeLocator; unknown selectors count 0."""

    def __init__(self, locators: dict):
        self._locators = locators

    def locator(self, selector: str) -> FakeLocator:
        return self._locators.get(selector, FakeLocator(0))


def make_probe(**overrides) -> dict:
    probe = {
        'tag': 'input',
        'id': 'switch23',
        'hasLabelFor': False,
        'hasWrappingLabel': False,
        'siblingTag': '',
    }
    probe.update(overrides)
    return probe


def _info(framework: str = "native", primary_type: str = "checkbox"):
    return ElementTypeInfo(
        primary_type=primary_type,
        framework=framework,
        confidence="high",
        signals=["tier:0"],
    )


# ======================================================================
# resolve_hidden_input_proxy — unit
# ======================================================================


class TestResolveHiddenInputProxy:

    async def test_visible_input_no_redirect(self):
        ctx = FakeContext({'id=newsletter': FakeLocator(1, FakeFirst(visible=True))})
        assert await resolve_hidden_input_proxy(ctx, 'id=newsletter') is None

    async def test_label_for_preferred(self):
        ctx = FakeContext({
            'id=agree': FakeLocator(1, FakeFirst(
                visible=False,
                probe=make_probe(id='agree', hasLabelFor=True,
                                 hasWrappingLabel=True, siblingTag='span'),
            )),
            'label[for="agree"]': FakeLocator(1, FakeFirst(visible=True)),
        })
        result = await resolve_hidden_input_proxy(ctx, 'id=agree')
        assert result == {
            'hidden_input': True,
            'locator': 'label[for="agree"]',
            'proxy_kind': 'label-for',
        }

    async def test_wrapping_label_fallback(self):
        """The ASTPP switch shape: no for-attr, input wrapped by
        label.switch with a span.slider sibling."""
        ctx = FakeContext({
            'id=switch23': FakeLocator(1, FakeFirst(
                visible=False,
                probe=make_probe(hasWrappingLabel=True, siblingTag='span'),
            )),
            'label:has(input[id="switch23"])': FakeLocator(1, FakeFirst(visible=True)),
        })
        result = await resolve_hidden_input_proxy(ctx, 'id=switch23')
        assert result['locator'] == 'label:has(input[id="switch23"])'
        assert result['proxy_kind'] == 'wrapping-label'

    async def test_sibling_last_resort(self):
        ctx = FakeContext({
            'id=switch23': FakeLocator(1, FakeFirst(
                visible=False, probe=make_probe(siblingTag='span'),
            )),
            'input[id="switch23"] + span': FakeLocator(1, FakeFirst(visible=True)),
        })
        result = await resolve_hidden_input_proxy(ctx, 'id=switch23')
        assert result['locator'] == 'input[id="switch23"] + span'
        assert result['proxy_kind'] == 'adjacent-sibling'

    async def test_non_unique_proxy_skipped(self):
        """A proxy that matches several elements is not safe — fall to
        the next candidate."""
        ctx = FakeContext({
            'id=switch23': FakeLocator(1, FakeFirst(
                visible=False,
                probe=make_probe(hasWrappingLabel=True, siblingTag='span'),
            )),
            'label:has(input[id="switch23"])': FakeLocator(2, FakeFirst(visible=True)),
            'input[id="switch23"] + span': FakeLocator(1, FakeFirst(visible=True)),
        })
        result = await resolve_hidden_input_proxy(ctx, 'id=switch23')
        assert result['proxy_kind'] == 'adjacent-sibling'

    async def test_hidden_no_proxy_flags_only(self):
        ctx = FakeContext({
            'id=orphan': FakeLocator(1, FakeFirst(
                visible=False, probe=make_probe(id='orphan'),
            )),
        })
        result = await resolve_hidden_input_proxy(ctx, 'id=orphan')
        assert result == {'hidden_input': True, 'locator': None, 'proxy_kind': ''}

    async def test_non_input_tag_no_redirect(self):
        """role=switch custom widgets ARE the visible control."""
        ctx = FakeContext({
            '[role="switch"][aria-label="Dark mode"]': FakeLocator(1, FakeFirst(
                visible=False, probe=make_probe(tag='div', id=''),
            )),
        })
        result = await resolve_hidden_input_proxy(
            ctx, '[role="switch"][aria-label="Dark mode"]'
        )
        assert result is None

    async def test_probe_error_no_redirect(self):
        """Broken probe (e.g. mocked context without is_visible) must
        never break the handler — always-fallback contract."""
        class BrokenContext:
            def locator(self, sel):
                raise RuntimeError("boom")

        assert await resolve_hidden_input_proxy(BrokenContext(), 'id=x') is None


# ======================================================================
# find_locator — handler integration
# ======================================================================


class TestHandlerRedirect:

    async def test_hidden_id_anchored_switch_redirects(self):
        ctx = FakeContext({
            'id=switch23': FakeLocator(1, FakeFirst(
                visible=False,
                probe=make_probe(hasWrappingLabel=True, siblingTag='span'),
            )),
            'label:has(input[id="switch23"])': FakeLocator(1, FakeFirst(visible=True)),
        })
        result = await find_locator(
            page=ctx,
            element_data={"tagName": "input", "type": "checkbox", "id": "switch23"},
            type_info=_info(),
            element_id="elem_sw",
            element_description="Active toggle for customer 64625",
            expected_text=None,
            search_context=ctx,
            iframe_context=None,
            confirmed_coords=None,
        )
        assert result["best_locator"] == 'label:has(input[id="switch23"])'
        assert result["element_type"] == "checkbox"
        assert result["element_info"] == {
            "hidden_input": True,
            "input_locator": "id=switch23",
            "proxy_kind": "wrapping-label",
        }
        assert result["stability"] == "stable"

    async def test_visible_checkbox_payload_unchanged(self):
        ctx = FakeContext({
            'id=newsletter': FakeLocator(1, FakeFirst(visible=True)),
        })
        result = await find_locator(
            page=ctx,
            element_data={"tagName": "input", "type": "checkbox", "id": "newsletter"},
            type_info=_info(),
            element_id="elem_chk",
            element_description="Subscribe checkbox",
            expected_text="Subscribe",
            search_context=ctx,
            iframe_context=None,
            confirmed_coords=None,
        )
        assert result["best_locator"] == "id=newsletter"
        assert "element_info" not in result

    async def test_hidden_no_proxy_keeps_input_with_flag(self):
        ctx = FakeContext({
            'id=orphan': FakeLocator(1, FakeFirst(
                visible=False, probe=make_probe(id='orphan'),
            )),
        })
        result = await find_locator(
            page=ctx,
            element_data={"tagName": "input", "type": "checkbox", "id": "orphan"},
            type_info=_info(),
            element_id="elem",
            element_description="Orphan toggle",
            expected_text=None,
            search_context=ctx,
            iframe_context=None,
            confirmed_coords=None,
        )
        assert result["best_locator"] == "id=orphan"
        assert result["element_info"] == {
            "hidden_input": True,
            "input_locator": "id=orphan",
        }


# ======================================================================
# Text-first checkbox detour — smart_locator integration
# ======================================================================


class TestTextFirstDetourRedirect:

    async def test_detour_redirects_hidden_input(self):
        """iframe_context set → detour evidence gate passes without the
        DOM probe; label-walk finds the input via label[for]; redirect
        swaps in the visible label and keeps the input reference."""
        ctx = FakeContext({
            'label:text-is("agree")': FakeLocator(
                1, FakeFirst(attrs={'for': 'agree_box'})
            ),
            'input[id="agree_box"]': FakeLocator(
                1, FakeFirst(attrs={'type': 'checkbox'})
            ),
            'id=agree_box': FakeLocator(1, FakeFirst(
                visible=False,
                probe=make_probe(id='agree_box', hasLabelFor=True),
            )),
            'label[for="agree_box"]': FakeLocator(1, FakeFirst(visible=True)),
        })
        result = await _find_element_by_expected_text(
            ctx, 'agree', 'the checkbox INPUT element for agree',
            x=None, y=None, iframe_context='iframe[id="main"]',
        )
        assert result is not None
        assert result['locator'] == 'label[for="agree_box"]'
        assert result['element_type'] == 'checkbox'
        assert result['hidden_input'] is True
        assert result['input_locator'] == 'id=agree_box'
        assert result['proxy_kind'] == 'label-for'
