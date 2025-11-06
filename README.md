# Browser Service 🤖

**AI-powered browser automation made simple.** Automatically identify web elements and generate reliable locators for your automation scripts.

[![PyPI version](https://badge.fury.io/py/browser-service.svg)](https://pypi.org/project/browser-service/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## ✨ What is Browser Service?

Browser Service is an intelligent browser automation library that uses AI to automatically identify and generate reliable selectors for web elements. No more fragile XPath or CSS selectors that break when websites change!

**Perfect for:**
- 🤖 Automated testing
- 🔄 Web scraping
- 📊 Data extraction
- 🎯 UI automation
- 🚀 Robotic Process Automation (RPA)

## 🚀 Quick Start

### 1. Install

```bash
pip install browser-service
```

### 2. Basic Usage

```python
import asyncio
from browser_service.config import config
from browser_service.tasks import process_workflow_task

async def automate_website():
    # Configure your API key (get from Google AI Studio)
    config.GEMINI_API_KEY = "your-gemini-api-key-here"

    # Define what you want to automate
    workflow = {
        "url": "https://example.com",
        "steps": [
            {"action": "click", "element": "Login button"},
            {"action": "type", "element": "Email field", "text": "user@example.com"},
            {"action": "click", "element": "Submit button"}
        ]
    }

    # Run the automation
    result = await process_workflow_task(workflow)
    print(f"✅ Automation completed: {result}")

# Run it
asyncio.run(automate_website())
```

That's it! The AI automatically finds the right elements on the page.

## 🎯 Key Features

### 🧠 AI-Powered Element Detection
- **Natural Language**: Describe elements in plain English ("Login button", "Search box")
- **Smart Locators**: Automatically generates reliable XPath, CSS, and Playwright selectors
- **Self-Healing**: Adapts when websites change their structure

### 🔧 Easy Configuration
```python
from browser_service.config import config

# Configure once, use everywhere
config.GEMINI_API_KEY = "your-api-key"
config.HEADLESS = True  # Run without browser window
config.TIMEOUT = 30     # Seconds to wait for elements
```

### 🌐 Multiple Browser Support
- Chrome/Chromium
- Firefox
- Safari (macOS)
- Edge

### 📊 Built-in Monitoring
```python
from browser_service.utils import record_workflow_metrics

# Track your automation performance
metrics = record_workflow_metrics(workflow_result)
print(f"⏱️  Execution time: {metrics['duration']}s")
print(f"🎯 Success rate: {metrics['success_rate']}%")
```

## 📚 Usage Examples

### Form Filling Automation

```python
workflow = {
    "url": "https://myapp.com/contact",
    "steps": [
        {"action": "type", "element": "Name field", "text": "John Doe"},
        {"action": "type", "element": "Email field", "text": "john@example.com"},
        {"action": "select", "element": "Country dropdown", "value": "United States"},
        {"action": "click", "element": "Submit button"}
    ]
}
```

### Data Extraction

```python
workflow = {
    "url": "https://news-site.com",
    "steps": [
        {"action": "extract", "element": "Article titles", "multiple": True},
        {"action": "extract", "element": "Publication dates", "multiple": True}
    ]
}
```

### E-commerce Automation

```python
workflow = {
    "url": "https://store.com",
    "steps": [
        {"action": "type", "element": "Search box", "text": "wireless headphones"},
        {"action": "click", "element": "Search button"},
        {"action": "click", "element": "First product"},
        {"action": "click", "element": "Add to cart button"}
    ]
}
```

## 🔧 Configuration

### Required Setup

```python
from browser_service.config import config

# Required: Your Google Gemini API key
config.GEMINI_API_KEY = "your-gemini-api-key-here"

# Optional: Customize behavior
config.HEADLESS = True          # Run without browser UI
config.TIMEOUT = 30             # Element wait timeout (seconds)
config.BROWSER_TYPE = "chrome"  # chrome, firefox, safari, edge
```

### Advanced Configuration

```python
from browser_service.config import BrowserServiceConfig

# Create custom config
custom_config = BrowserServiceConfig(
    gemini_api_key="your-key",
    headless=False,
    timeout=60,
    browser_type="firefox"
)
```

## 🎮 Action Types

| Action | Description | Example |
|--------|-------------|---------|
| `click` | Click an element | `{"action": "click", "element": "Submit button"}` |
| `type` | Type text into field | `{"action": "type", "element": "Email field", "text": "user@email.com"}` |
| `select` | Select dropdown option | `{"action": "select", "element": "Country", "value": "USA"}` |
| `extract` | Get element text | `{"action": "extract", "element": "Price"}` |
| `wait` | Wait for element | `{"action": "wait", "element": "Loading spinner"}` |
| `scroll` | Scroll to element | `{"action": "scroll", "element": "Bottom of page"}` |

## 🏗️ API Reference

### Core Functions

```python
from browser_service.tasks import process_workflow_task
from browser_service.locators import generate_locators_from_attributes
from browser_service.browser import BrowserSessionManager

# Process a complete workflow
result = await process_workflow_task(workflow)

# Generate locators for an element
locators = generate_locators_from_attributes(element_attributes)

# Manage browser sessions
browser = BrowserSessionManager()
await browser.start_session()
# ... use browser ...
await browser.cleanup()
```

### Configuration Classes

```python
from browser_service.config import (
    config,                    # Global config instance
    BrowserServiceConfig,      # Main config class
    BatchConfig,              # Batch processing config
    LocatorConfig,            # Locator generation config
    LLMConfig                 # AI model config
)
```

## 🐛 Troubleshooting

### Common Issues

**❌ "Element not found" errors**
```python
# Solution: Make element descriptions more specific
{"action": "click", "element": "Blue Login button on the right"}  # Better
{"action": "click", "element": "Login button"}                   # Worse
```

**❌ "API key invalid" error**
```python
# Make sure you set your Gemini API key
config.GEMINI_API_KEY = "your-actual-api-key-from-google-ai-studio"
```

**❌ "Timeout" errors**
```python
# Increase timeout for slow pages
config.TIMEOUT = 60  # 60 seconds instead of default 30
```

### Debug Mode

```python
import logging
from browser_service.utils import setup_logging

# Enable detailed logging
setup_logging(level=logging.DEBUG)

# Now you'll see detailed logs about element detection
```

## 📋 Requirements

- **Python**: 3.8 or higher
- **API Key**: Google Gemini API key (free tier available)
- **Browser**: Chrome/Chromium recommended (others supported)

## 🤝 Contributing

Found a bug or want to improve Browser Service?

1. **🐛 Report Issues**: [GitHub Issues](https://github.com/monkscode/browser-service/issues)
2. **💡 Feature Requests**: Open an issue with "Feature Request" label
3. **🔧 Code Contributions**: Fork → Branch → PR

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- **📖 Documentation**: More examples in `/examples` folder
- **💬 Questions**: [GitHub Discussions](https://github.com/monkscode/browser-service/discussions)
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/monkscode/browser-service/issues)

---

**Ready to automate?** 🚀 Install now and start building intelligent browser automation!

```bash
pip install browser-service
```
