"""
Live integration tests for find_unique_locator_at_coordinates (Tier 2).

Requires:
  - Playwright with Chromium: playwright install chromium
  - Internet access (tests hit example.com)

Run with:
  pytest tests/test_integration/test_live_locator.py -m integration -v
"""

import pytest
from playwright.async_api import async_playwright

pytestmark = pytest.mark.integration

LIVE_URL = "https://example.com"


class TestFindUniqueLocatorLive:
    """Integration tests for find_unique_locator_at_coordinates against a real page."""

    @pytest.mark.asyncio
    async def test_returns_dict_with_expected_keys(self):
        """Result dict contains all required top-level keys."""
        from browser_service.locators import find_unique_locator_at_coordinates

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(LIVE_URL, wait_until="domcontentloaded")

            h1 = page.locator("h1").first
            bbox = await h1.bounding_box()
            x = bbox["x"] + bbox["width"] / 2 if bbox else 300.0
            y = bbox["y"] + bbox["height"] / 2 if bbox else 150.0

            result = await find_unique_locator_at_coordinates(
                page=page,
                x=x,
                y=y,
                element_id="elem_1",
                element_description="main heading",
            )
            await browser.close()

        assert isinstance(result, dict)
        for key in ("found", "best_locator", "all_locators", "validation_summary", "element_id"):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_found_true_for_real_element(self):
        """Known-good element coordinates return found=True with a non-empty best_locator."""
        from browser_service.locators import find_unique_locator_at_coordinates

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(LIVE_URL, wait_until="domcontentloaded")

            h1 = page.locator("h1").first
            bbox = await h1.bounding_box()
            if not bbox:
                await browser.close()
                pytest.skip("Could not get bounding box for h1 on example.com")

            result = await find_unique_locator_at_coordinates(
                page=page,
                x=bbox["x"] + bbox["width"] / 2,
                y=bbox["y"] + bbox["height"] / 2,
                element_id="elem_1",
                element_description="main heading",
                expected_text="Example Domain",
            )
            await browser.close()

        assert result["found"] is True
        assert isinstance(result["best_locator"], str)
        assert len(result["best_locator"]) > 0

    @pytest.mark.asyncio
    async def test_validation_summary_has_expected_keys(self):
        """validation_summary always contains total_generated and validation_method."""
        from browser_service.locators import find_unique_locator_at_coordinates

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(LIVE_URL, wait_until="domcontentloaded")

            h1 = page.locator("h1").first
            bbox = await h1.bounding_box()
            x = bbox["x"] + 20 if bbox else 300.0
            y = bbox["y"] + 10 if bbox else 150.0

            result = await find_unique_locator_at_coordinates(
                page=page,
                x=x,
                y=y,
                element_id="elem_1",
                element_description="heading",
            )
            await browser.close()

        assert "validation_summary" in result
        summary = result["validation_summary"]
        assert "total_generated" in summary
        assert "validation_method" in summary

    @pytest.mark.asyncio
    async def test_out_of_bounds_coords_returns_found_false(self):
        """Coordinates far outside the viewport produce found=False without raising."""
        from browser_service.locators import find_unique_locator_at_coordinates

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(LIVE_URL, wait_until="domcontentloaded")

            result = await find_unique_locator_at_coordinates(
                page=page,
                x=99999.0,
                y=99999.0,
                element_id="elem_1",
                element_description="nothing",
            )
            await browser.close()

        assert result["found"] is False
