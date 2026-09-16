"""
E2b read-target policy, no browser: which locators count as embedding the
value being read, and what the policy does with the builder's answer.
"""

from unittest.mock import AsyncMock, patch

import pytest

from browser_service.locators import read_target
from browser_service.locators.read_target import apply_read_target_policy, data_bound_literal

AMAZON = "Portronics Toad 23 Wireless Optical Mouse with 2.4GHz USB Nano Dongle"
FLIPKART = "Sports Sneaker Running And Outdoor Walking Shoes For Men"
LOGI = "Logitech B170 Wireless Mouse, 2.4 GHz with USB Nano Receiver"


@pytest.mark.parametrize(
    "locator, expected, want",
    [
        (f'[aria-label="{AMAZON}"]', AMAZON, True),  # u01
        ('text="Sports Sneaker Running And Outdoor Walking Shoes For Me..."', FLIPKART, True),
        ("a[title*='Sports Sneaker Running And Outdoor Walking Shoes']", FLIPKART, True),  # u02 r2
        ('role=link[name="Logitech B170 Wireless Mouse, 2.4 GHz with USB"]', LOGI, True),
        ("xpath=//h2[contains(text(),'Logitech B170 Wireless Mouse, 2.4 GHz')]", LOGI, True),
        (
            '[aria-label="Men\'s Running Shoes With Extra Cushioning"]',
            "Men's Running Shoes With Extra Cushioning",
            True,
        ),
        ('text="John" >> nth=0', "John", False),  # q05: short literal, 40/42 passed
        ('text="buy milk"', "buy milk", False),  # u10: short
        ('[aria-label="Search for Products, Brands and More"]', AMAZON, False),  # unrelated
        ("id=twotabsearchtextbox", AMAZON, False),
        (f'[aria-label="{AMAZON}"]', None, False),
    ],
)
def test_data_bound_literal(locator, expected, want):
    assert bool(data_bound_literal(locator, expected)) is want


REWRITTEN = '[data-component-type="s-search-result"] >> nth=0 >> h2'
BASE = {
    "found": True,
    "best_locator": f'[aria-label="{AMAZON}"]',
    "stability": "volatile",
    "all_locators": [{"locator": f'[aria-label="{AMAZON}"]'}, {"locator": "xpath=//h2[1]"}],
    "element_info": {"id": ""},
}


async def _apply(result=None, action="get_text", expected=AMAZON, iframe=None, built=REWRITTEN):
    builder = AsyncMock(return_value=built)
    with patch("browser_service.locators.read_target.build_container_ordinal", new=builder):
        out = await apply_read_target_policy(
            object(), dict(result or BASE), action, expected, iframe
        )
    return out, builder


async def _apply_id(id_validates, result, expected=AMAZON, built=REWRITTEN):
    builder = AsyncMock(return_value=built)
    id_validator = AsyncMock(return_value=id_validates)
    with (
        patch("browser_service.locators.read_target.build_container_ordinal", new=builder),
        patch("browser_service.locators.read_target._resolves_to_same_element", new=id_validator),
    ):
        out = await apply_read_target_policy(object(), dict(result), "get_text", expected, None)
    return out, builder, id_validator


class TestPolicy:
    async def test_rewrites_a_data_bound_read(self):
        out, _ = await _apply()
        assert out["best_locator"] == REWRITTEN
        assert out["stability"] == "positional"
        assert [e["locator"] for e in out["all_locators"]] == [REWRITTEN]
        assert out["read_target_rewritten_from"] == BASE["best_locator"]

    async def test_keeps_it_when_no_container_validates(self):
        out, _ = await _apply(built=None)
        assert out["best_locator"] == BASE["best_locator"]
        assert "read_target_rewritten_from" not in out

    @pytest.mark.parametrize("action", ["click", "input", None])
    async def test_only_read_actions(self, action):
        out, builder = await _apply(action=action)
        assert out["best_locator"] == BASE["best_locator"]
        builder.assert_not_awaited()

    async def test_skips_iframes(self):
        out, builder = await _apply(iframe='iframe[id="main"]')
        builder.assert_not_awaited()

    @pytest.mark.parametrize(
        "extra", [{"row_anchored": True}, {"element_type": "collection"}, {"found": False}]
    )
    async def test_skips_anchored_collections_and_misses(self, extra):
        out, builder = await _apply(result={**BASE, **extra})
        builder.assert_not_awaited()

    async def test_uses_the_stable_id_when_it_validates(self):
        result = {**BASE, "element_info": {"id": "result-title"}}
        out, builder, id_validator = await _apply_id(True, result)
        assert out["best_locator"] == "id=result-title"
        assert [e["locator"] for e in out["all_locators"]] == ["id=result-title"]
        assert out["read_target_rewritten_from"] == BASE["best_locator"]
        id_validator.assert_awaited_once()
        builder.assert_not_awaited()

    async def test_falls_through_to_container_when_the_id_does_not_validate(self):
        result = {**BASE, "element_info": {"id": "result-title"}}
        out, builder, id_validator = await _apply_id(False, result)
        assert out["best_locator"] == REWRITTEN
        id_validator.assert_awaited_once()
        builder.assert_awaited_once()

    async def test_volatile_id_skips_id_validation(self):
        result = {**BASE, "element_info": {"id": "ember472"}}
        out, builder, id_validator = await _apply_id(True, result)
        assert out["best_locator"] == REWRITTEN
        id_validator.assert_not_awaited()
        builder.assert_awaited_once()

    async def test_short_literal_is_left_alone(self):
        result = {**BASE, "best_locator": 'text="John" >> nth=0'}
        out, builder = await _apply(result=result, expected="John")
        assert out["best_locator"] == 'text="John" >> nth=0'
        builder.assert_not_awaited()


def test_container_walk_max_ancestors_is_16():
    """E2b depth fix (2026-09-16): live amazon.in nests the product title
    14 ancestors below the repeated card; the walk must reach past that."""
    assert read_target.CONTAINER_WALK_MAX_ANCESTORS == 16
    assert "depth < 16" in read_target.CONTAINER_ORDINAL_JS
    assert "depth < 8" not in read_target.CONTAINER_ORDINAL_JS
