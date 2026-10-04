"""#31 read-address rule, no browser: which read locators carry the observed text, how an element's own texts are matched, and what the gate does with the resolver's answer."""

import ast
import asyncio
import logging
import re
import time
from unittest.mock import AsyncMock, patch

import pytest

from browser_service.config import config
from browser_service.locators.read_target import (
    STRUCTURAL_CANDIDATES_JS,
    TOKENS_JS,
    apply_read_target_policy,
    carries_value,
    displayed_literals,
    own_text_equals,
    own_texts_match,
)

AMAZON = "Portronics Toad 23 Wireless Optical Mouse with 2.4GHz USB Nano Dongle"


@pytest.mark.parametrize(
    "locator, observed, want",
    [
        ("text=Products", "Products", True),
        ('text="John" >> nth=0', "John", True),
        ('text="Cierra"', "Cierra", True),
        ("text=Sony vaio i5", "Sony vaio i5", True),
        ("a:has-text('Sony vaio i5')", "Sony vaio i5", True),
        ("header >> text=Dashboard", "Dashboard", True),
        ('role=link[name="A Light in the Attic"]', "A Light in the Attic", True),
        ('[title="A Light in the Attic"]', "A Light in the ...", True),
        (f'[aria-label="{AMAZON}"]', AMAZON, True),
        ('[aria-label="Men\'s Running Shoes"]', "Men's Running Shoes", True),
        (
            "xpath=//h2[contains(text(),'Logitech B170 Wireless')]",
            "Logitech B170 Wireless Mouse",
            True,
        ),
        ('[placeholder="Email"]', "Email", True),
        ("text=1", "1", True),
        ("li:nth-child(1) > span", "1", False),
        ("input[name='email']", "Email", False),
        ('[data-test="title"]', "Title", False),
        ("#title", "Title", False),
        (".title", "Products", False),
        ('[aria-label="Search for Products, Brands and More"]', AMAZON, False),
        ("text=Products", None, False),
        ("text=Products", "", False),
    ],
)
def test_carries_value(locator, observed, want):
    assert carries_value(locator, observed) is want


def test_displayed_literals_reads_every_quote_style():
    assert displayed_literals('text="a" >> [title=\'b\'] >> role=button[name="c"]') == [
        "a",
        "b",
        "c",
    ]


@pytest.mark.parametrize(
    "own, observed, want",
    [
        ("Products", "Products", True),
        ("  PRODUCTS\n", "products", True),
        ("Products\u00a0", "Products", True),
        ("A Light in the Attic", "A Light in the ...", True),
        ("A Light in the Attic", "A Light in the …", True),
        ("Name (A to Z)\nName (Z to A)", "Products", False),
        ("Products (6)", "Products", False),
        ("", "Products", False),
        ("Products", None, False),
        ("A Light in the ...", "A Light in the Attic", True),  # the PAGE truncated (books)
    ],
)
def test_own_text_equals(own, observed, want):
    assert own_text_equals(own, observed) is want


def test_own_texts_match_any_own_text():
    """flipkart: the agent reported the title attribute; the visible text is shorter."""
    texts = [
        "Trendy Sports Running Shoes",
        "",
        "",
        "",
        "Trendy Sports Running Shoes For Men (Black, 8)",
        "",
    ]
    assert own_texts_match(texts, "Trendy Sports Running Shoes For Men (Black, 8)") is True
    assert own_texts_match(texts, "Premium White Sneakers") is False
    assert own_texts_match([], "Products") is False


BASE = {
    "found": True,
    "best_locator": "text=Products",
    "stability": "volatile",
    "all_locators": [{"locator": "text=Products"}],
    "element_id": "elem_4",
    "description": "heading",
}


async def _gate(
    result=None, action="get_text", expected="Products", iframe=None, verdict=None, in_shadow=False
):
    resolver = AsyncMock(
        return_value=verdict or {"locator": ".title", "kind": "class", "source": "pointed element"}
    )
    with (
        patch("browser_service.locators.read_target.resolve_read_address", new=resolver),
        patch(
            "browser_service.locators.read_target._in_shadow_root",
            new=AsyncMock(return_value=in_shadow),
        ),
    ):
        out = await apply_read_target_policy(
            object(),
            dict(result or BASE),
            action,
            expected,
            iframe,
            element_data={"xpath": "html/body/span"},
            point=(10, 20),
        )
    return out, resolver


class TestGate:
    async def test_rewrites_a_value_bearing_read(self):
        out, resolver = await _gate()
        assert out["best_locator"] == ".title"
        assert [e["locator"] for e in out["all_locators"]] == [".title"]
        assert out["read_target_rewritten_from"] == "text=Products"
        resolver.assert_awaited_once()

    async def test_unconfirmed_is_not_found_with_the_reason(self):
        out, _ = await _gate(verdict={"unconfirmed": "the screen point shows 'x', not 'products'"})
        assert out["found"] is False
        assert out["read_address_unconfirmed"] == "the screen point shows 'x', not 'products'"
        assert out["read_address_rejected"] == "text=Products"
        assert "best_locator" not in out

    async def test_value_free_read_is_untouched(self):
        out, resolver = await _gate(result={**BASE, "best_locator": ".title"})
        assert out["best_locator"] == ".title"
        resolver.assert_not_awaited()

    @pytest.mark.parametrize("action", ["click", "input", "select", None])
    async def test_only_read_actions(self, action):
        out, resolver = await _gate(action=action)
        assert out["best_locator"] == "text=Products"
        resolver.assert_not_awaited()

    async def test_iframe_is_untouched(self):
        out, resolver = await _gate(iframe='iframe[id="main"]')
        assert out["best_locator"] == "text=Products"
        resolver.assert_not_awaited()

    @pytest.mark.parametrize(
        "extra", [{"row_anchored": True}, {"element_type": "collection"}, {"found": False}]
    )
    async def test_anchored_collections_and_misses_are_untouched(self, extra):
        out, resolver = await _gate(result={**BASE, **extra})
        resolver.assert_not_awaited()

    async def test_shadow_root_keeps_todays_locator(self):
        out, resolver = await _gate(in_shadow=True)
        assert out["best_locator"] == "text=Products"
        resolver.assert_not_awaited()


REWRITE = {"locator": ".title", "kind": "class", "source": "pointed element"}


def _answers_late(value):
    """A page call that would answer after a second — far past the budget."""

    async def answer(*_args, **_kwargs):
        await asyncio.sleep(1)
        return value

    return answer


class TestGateBudget:
    """The address checks are bounded by the custom-action budget; a page that does not answer in time is an unconfirmed verdict — the value-carrying locator is never handed over."""

    BUDGET = 0.05
    REASON = f"the page did not answer the address checks within {BUDGET} s"

    async def _run(self, shadow, resolver):
        with (
            patch.object(config.locator, "custom_action_timeout", self.BUDGET),
            patch("browser_service.locators.read_target.resolve_read_address", new=resolver),
            patch("browser_service.locators.read_target._in_shadow_root", new=shadow),
        ):
            started = time.monotonic()
            out = await apply_read_target_policy(
                object(),
                dict(BASE),
                "get_text",
                "Products",
                element_data={"xpath": "html/body/span"},
                point=(10, 20),
            )
            return out, time.monotonic() - started

    def _assert_unconfirmed(self, out, elapsed):
        assert out["found"] is False
        assert out["read_address_unconfirmed"] == self.REASON
        assert out["read_address_rejected"] == "text=Products"
        assert out["error"] == f"the read locator would carry the value being read; {self.REASON}"
        assert "best_locator" not in out
        assert elapsed < 0.5

    async def test_a_resolver_slower_than_the_budget_is_unconfirmed(self, caplog):
        caplog.set_level(logging.INFO, logger="browser_service.locators.read_target")
        out, elapsed = await self._run(
            AsyncMock(return_value=False), AsyncMock(side_effect=_answers_late(REWRITE))
        )
        self._assert_unconfirmed(out, elapsed)
        assert "signal: read-address-unconfirmed" in caplog.text

    async def test_a_shadow_guard_slower_than_the_budget_is_unconfirmed(self):
        resolver = AsyncMock(return_value=REWRITE)
        out, elapsed = await self._run(AsyncMock(side_effect=_answers_late(False)), resolver)
        self._assert_unconfirmed(out, elapsed)
        resolver.assert_not_awaited()

    async def test_answers_inside_the_budget_are_unchanged(self):
        resolver = AsyncMock(return_value=REWRITE)
        out, _ = await self._run(AsyncMock(return_value=False), resolver)
        assert out["best_locator"] == ".title"
        resolver.assert_awaited_once()


def test_tokens_script_reads_every_value_the_candidate_script_builds_on():
    """TOKENS_JS reads the VALUES of exactly the attributes STRUCTURAL_CANDIDATES_JS builds on (its SHARED_ATTRS, the test ids among them): one missing would let a volatile value through."""
    values = re.search(r"const VALUE_ATTRS = (\[[^\]]*\]);", TOKENS_JS)
    assert values, TOKENS_JS
    read = ast.literal_eval(values.group(1))
    shared = re.search(r"const SHARED_ATTRS = (\[[^\]]*\]);", STRUCTURAL_CANDIDATES_JS)
    assert read == ast.literal_eval(shared.group(1))
    for listed in re.findall(r"for \(const \w+ of (\[[^\]]*\])\)", STRUCTURAL_CANDIDATES_JS):
        assert set(ast.literal_eval(listed)) <= set(read), listed
