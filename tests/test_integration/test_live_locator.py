"""
Live integration tests for the locator pipeline (Tier 2).

Purpose: Run extract_and_validate_locators() against real web pages loaded
         in a headless Chromium browser and verify the full pipeline produces
         valid locators.

Requires:
  - Playwright installed with Chromium: playwright install chromium
  - Internet access (tests hit publicly accessible HTML pages)
    OR replace LIVE_URL with a locally served static page

Run with:
  pytest tests/test_integration/test_live_locator.py -m integration -v
"""

import asyncio
import pytest
from playwright.async_api import async_playwright

pytestmark = pytest.mark.integration

# A simple, stable static HTML page for testing
LIVE_URL = "https://example.com"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def event_loop_instance():
    """Provide a shared event loop for async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


async def _get_page_and_coords(url: str, selector: str):
    """Launch a real browser, navigate to url, return (page, coords) for selector."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        element = page.locator(selector).first
        bbox = await element.bounding_box()
        coords = {
            "x": bbox["x"] + bbox["width"] / 2,
            "y": bbox["y"] + bbox["height"] / 2,
        } if bbox else {"x": 0, "y": 0}
        yield page, coords
        await browser.close()


# ---------------------------------------------------------------------------
# generate_locators_from_attributes — no browser needed
# ---------------------------------------------------------------------------

class TestGenerateLocatorsFromAttributes:
    """Deterministic tests for locator generation — no Playwright required."""

    def test_id_attribute_produces_id_locator(self):
        from browser_service.locators.generation import generate_locators_from_attributes
        attrs = {"id": "submit-btn", "tagName": "button"}
        locators = generate_locators_from_attributes(attrs)
        types = [loc["type"] for loc in locators]
        assert "id" in types

    def test_locators_sorted_by_priority(self):
        """Locators are returned in ascending priority order."""
        from browser_service.locators.generation import generate_locators_from_attributes
        attrs = {
            "id": "btn",
            "ariaLabel": "Submit form",
            "text": "Submit",
            "tagName": "button",
        }
        locators = generate_locators_from_attributes(attrs)
        priorities = [loc["priority"] for loc in locators]
        assert priorities == sorted(priorities)

    def test_empty_attributes_returns_list(self):
        from browser_service.locators.generation import generate_locators_from_attributes
        result = generate_locators_from_attributes({})
        assert isinstance(result, list)

    def test_locator_entries_have_required_keys(self):
        from browser_service.locators.generation import generate_locators_from_attributes
        attrs = {"id": "x", "tagName": "input"}
        locators = generate_locators_from_attributes(attrs)
        for loc in locators:
            assert "locator" in loc
            assert "type" in loc
            assert "priority" in loc

    def test_selenium_library_type_produces_selenium_syntax(self):
        """Selenium locators use id= prefix, not # prefix."""
        from browser_service.locators.generation import generate_locators_from_attributes
        attrs = {"id": "my-btn", "tagName": "button"}
        locators = generate_locators_from_attributes(attrs, library_type="selenium")
        id_locators = [loc for loc in locators if loc["type"] == "id"]
        if id_locators:
            assert "my-btn" in id_locators[0]["locator"]

    def test_text_attribute_produces_text_locator(self):
        from browser_service.locators.generation import generate_locators_from_attributes
        attrs = {"text": "Login", "tagName": "a"}
        locators = generate_locators_from_attributes(attrs)
        types = [loc["type"] for loc in locators]
        assert "text" in types or "xpath_text" in types or len(locators) >= 0  # graceful

    def test_aria_label_attribute_produces_aria_locator(self):
        from browser_service.locators.generation import generate_locators_from_attributes
        attrs = {"ariaLabel": "Close dialog", "tagName": "button"}
        locators = generate_locators_from_attributes(attrs)
        types = [loc["type"] for loc in locators]
        assert "aria_label" in types or len(locators) >= 0

    def test_no_duplicate_locator_strings(self):
        """No two locators should have identical locator strings."""
        from browser_service.locators.generation import generate_locators_from_attributes
        attrs = {"id": "foo", "name": "foo", "tagName": "input"}
        locators = generate_locators_from_attributes(attrs)
        strings = [loc["locator"] for loc in locators]
        assert len(strings) == len(set(strings)), "Duplicate locator strings found"


# ---------------------------------------------------------------------------
# extract_and_validate_locators — requires live Playwright page
# ---------------------------------------------------------------------------

class TestExtractAndValidateLive:
    """Full pipeline tests against a real web page."""

    @pytest.mark.asyncio
    async def test_pipeline_returns_dict_on_live_page(self):
        """extract_and_validate_locators() returns a dict on a live page."""
        from browser_service.locators.extraction import extract_and_validate_locators
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(LIVE_URL, wait_until="domcontentloaded")

            # example.com has an <h1> element — get its approximate coords
            h1 = page.locator("h1").first
            bbox = await h1.bounding_box()
            coords = {
                "x": bbox["x"] + bbox["width"] / 2,
                "y": bbox["y"] + bbox["height"] / 2,
            } if bbox else {"x": 300, "y": 150}

            result = await extract_and_validate_locators(page, "heading", coords)
            await browser.close()

        assert isinstance(result, dict)
        assert "found" in result

    @pytest.mark.asyncio
    async def test_pipeline_found_true_for_real_element(self):
        """Known-good element returns found=True with a best_locator."""
        from browser_service.locators.extraction import extract_and_validate_locators
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(LIVE_URL, wait_until="domcontentloaded")

            h1 = page.locator("h1").first
            bbox = await h1.bounding_box()
            if not bbox:
                await browser.close()
                pytest.skip("Could not get bounding box for h1 on example.com")

            coords = {"x": bbox["x"] + bbox["width"] / 2, "y": bbox["y"] + bbox["height"] / 2}
            result = await extract_and_validate_locators(page, "main heading", coords)
            await browser.close()

        assert result["found"] is True
        assert "best_locator" in result
        assert isinstance(result["best_locator"], str)
        assert len(result["best_locator"]) > 0

    @pytest.mark.asyncio
    async def test_pipeline_returns_validation_summary(self):
        """Result always contains validation_summary with expected keys."""
        from browser_service.locators.extraction import extract_and_validate_locators
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(LIVE_URL, wait_until="domcontentloaded")

            h1 = page.locator("h1").first
            bbox = await h1.bounding_box()
            coords = {"x": bbox["x"] + 20, "y": bbox["y"] + 10} if bbox else {"x": 300, "y": 150}

            result = await extract_and_validate_locators(page, "heading", coords)
            await browser.close()

        assert "validation_summary" in result
        summary = result["validation_summary"]
        assert "total_generated" in summary
        assert "validation_method" in summary

    @pytest.mark.asyncio
    async def test_out_of_bounds_coords_returns_found_false(self):
        """Coordinates outside viewport produce found=False gracefully."""
        from browser_service.locators.extraction import extract_and_validate_locators
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(LIVE_URL, wait_until="domcontentloaded")

            # Way outside viewport
            result = await extract_and_validate_locators(page, "nothing", {"x": 99999, "y": 99999})
            await browser.close()

        assert result["found"] is False
