# Browser Service

AI-assisted element locator extraction for web pages. Give it a page URL and a
plain-English description of an element; it drives a real browser, finds the
element, and returns validated locators.

[![PyPI version](https://badge.fury.io/py/browser-service.svg)](https://pypi.org/project/browser-service/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Scope — read this first

This package was extracted from
[Natural-Language-to-Robot-Framework](https://github.com/monkscode/Natural-Language-to-Robot-Framework)
(NLRF) and exists to serve it. It is published to PyPI so that project can pin a
version, not because it is a general-purpose automation library.

Concretely:

- It ships **no runnable entrypoint** — no `__main__`, no console script. You
  mount it into a Flask app yourself.
- Its public API is deliberately small: a config object and four config classes.
  Everything else is internal and changes without notice.
- Configuration falls back to NLRF's settings when that package happens to be
  importable (`src.backend.core.config`). Every such import is wrapped in
  `try/except ImportError`, so the service runs standalone — it just defaults to
  `MODEL_PROVIDER=vertex` instead of reading NLRF's choice.

If you want a browser-automation library, use
[browser-use](https://github.com/browser-use/browser-use) directly — this is a
thin, opinionated service around it.

## Install

```bash
pip install browser-service
```

Requires **Python 3.11 or newer** (`requires-python = ">=3.11,<4.0"`).

## Running it

There is no built-in server. Register the routes onto a Flask app you own and
supply a task processor:

```python
from flask import Flask
from browser_service.api import register_routes

app = Flask(__name__)
register_routes(app, task_processor)   # task_processor: your queue/executor
app.run(host="0.0.0.0", port=4999)
```

NLRF does exactly this in `tools/browser_use_service.py` and talks to the result
over HTTP. It never imports this package's internals.

## HTTP endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Service banner |
| `GET` | `/health` | Health check |
| `GET` | `/probe` | Legacy health check |
| `POST` | `/workflow` | Submit a workflow task — **the primary endpoint** |
| `POST` | `/batch` | Deprecated alias for `/workflow` |
| `GET` | `/query/<task_id>` | Task status by id |
| `GET` | `/tasks` | List tasks with summaries |

## Configuration

Config is nested, not flat. Import the shared instance:

```python
from browser_service.config import config

config.llm.model_provider      # "vertex" (default) | "gemini" | "local"
config.headless                # bool
config.batch                   # BatchConfig
config.locator                 # LocatorConfig
config.agent_vision_mode
config.enable_custom_actions
config.max_concurrent_tasks
config.robot_library
```

Settings resolve in this order: **environment variable → NLRF settings (if
importable) → built-in default.** The environment variables read are:

| Variable | Notes |
|---|---|
| `MODEL_PROVIDER` | `vertex` (default), `gemini`, or `local` |
| `GEMINI_API_KEY` | only used when `MODEL_PROVIDER=gemini` |
| `VERTEXAI_CREDENTIALS` | service-account JSON path, for `vertex` |
| `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION` | Vertex targeting |
| `GOOGLE_MODEL` | model name |
| `BROWSER_HEADLESS` | run without a visible browser |
| `AGENT_VISION_MODE` | vision behaviour for element finding |
| `ENABLE_CUSTOM_ACTIONS` | enable the custom action set |
| `ROBOT_LIBRARY` | locator syntax target |

## Public API

```python
from browser_service import (
    __version__,
    config,                 # shared BrowserServiceConfig instance
    BrowserServiceConfig,
    BatchConfig,
    LocatorConfig,
    LLMConfig,
)

from browser_service.api import (
    register_routes,
    validate_workflow_request,
    format_task_response,
    format_error_response,
)
```

That is the whole supported surface. Anything else you can import is internal.

## Element actions

The action dispatcher in `browser_service/agent/registration.py` handles:

| Action | Aliases | Effect |
|---|---|---|
| `input` | `type` | Type text into a field |
| `click` | `submit` | Click an element |
| `select` | — | Choose a dropdown option |
| `check` | `uncheck` | Toggle a checkbox |

## Example

See [`examples/basic_usage.py`](examples/basic_usage.py).

## Contributing

Issues and pull requests: [GitHub](https://github.com/monkscode/browser-service/issues).

Releases are automated — merging to `main` triggers `.github/workflows/publish.yml`,
which derives the next version from PyPI and publishes. Do not hand-edit the
version in `pyproject.toml`; CI overwrites it at build time.

## License

MIT — see [LICENSE](LICENSE).
