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
- Its public API is deliberately small: a version string, a config object, four
  config classes, and four Flask helpers. Everything else is internal and
  changes without notice.
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

config.llm.model_provider      # "vertex" (default) | "gemini"  ("local" is rejected)
config.headless                # bool
config.batch                   # BatchConfig
config.locator                 # LocatorConfig
config.agent_vision_mode
config.enable_custom_actions
config.max_concurrent_tasks
config.robot_library
```

There is no single resolution rule — it differs per setting. Four read the
environment first and fall back to NLRF's settings when `src.backend.core.config`
is importable:

| Variable | Resolution | Notes |
|---|---|---|
| `MODEL_PROVIDER` | env → NLRF → `vertex` | `gemini` or `vertex`. **`local` is rejected** — this service requires a Google vision model, and `validate()` reports an error for it |
| `GEMINI_API_KEY` | env → NLRF → unset | required when `MODEL_PROVIDER=gemini` |
| `VERTEXAI_PROJECT` | env → NLRF → unset | required when `MODEL_PROVIDER=vertex` |
| `VERTEXAI_LOCATION` | env → NLRF → unset | required when `MODEL_PROVIDER=vertex` |

One resolves the *other* way round — NLRF wins over the environment:

| Variable | Resolution | Notes |
|---|---|---|
| `GOOGLE_MODEL` | **NLRF → env** → `gemini-2.5-flash` | when NLRF is importable and sets `ONLINE_MODEL`, `GOOGLE_MODEL` is ignored. Any provider prefix is stripped |

The rest read the environment and fall straight through to a built-in default.
They never consult NLRF:

| Variable | Default | Notes |
|---|---|---|
| `VERTEXAI_CREDENTIALS` | unset | service-account JSON path; not an NLRF settings field |
| `ROBOT_LIBRARY` | `browser` | any other value raises at construction |
| `BROWSER_HEADLESS` | `true` | run without a visible browser |
| `AGENT_VISION_MODE` | `auto` | `auto` or `on`; any other value raises at construction |
| `ENABLE_CUSTOM_ACTIONS` | `true` | enable the custom action set |
| `MAX_CONCURRENT_TASKS` | `10` | each task spawns a headless Chrome (~250MB) |
| `MAX_AGENT_STEPS` | `15` | agent step ceiling per workflow |
| `MAX_RETRIES_PER_ELEMENT` | `2` | |
| `ELEMENT_TIMEOUT` | `120` | seconds |
| `CONTENT_BASED_RETRIES` | `7` | |
| `COORDINATE_BASED_RETRIES` | `7` | |
| `ELEMENT_TYPE_RETRIES` | `5` | |
| `COORDINATE_OFFSET_ATTEMPTS` | `7` | |
| `CUSTOM_ACTION_TIMEOUT` | `5` | seconds |

These tables are asserted against `config.py` by
`tests/test_docs_match_code.py` — adding a variable without documenting it
fails CI.

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

That is the whole supported surface — it is exactly the two `__all__` lists,
asserted by `tests/test_docs_match_code.py`. Anything else you can import is
internal.

## Element actions

The action dispatcher in `browser_service/agent/registration.py` handles:

| Action | Aliases | Effect |
|---|---|---|
| `input` | `type` | Type text into a field |
| `click` | `submit` | Click an element |
| `select` | — | Choose a dropdown option |
| `check` | `uncheck` | Toggle a checkbox |

## Example

Submit a workflow, then poll for the result. This is the only supported way to
drive the service — `process_workflow_task` and the rest of `browser_service.tasks`
are internal, and NLRF itself never imports them.

```bash
curl -X POST http://localhost:4999/workflow   -H 'Content-Type: application/json'   -d '{
    "url": "https://example.com/login",
    "user_query": "Log in with a username",
    "elements": [
      {"id": "elem_1", "description": "Username field", "action": "input"},
      {"id": "elem_2", "description": "Login button",   "action": "click"}
    ]
  }'
# → 202 {"task_id": "...", "status": "processing", "message": "..."}

curl http://localhost:4999/query/<task_id>
```

`elements` is required and must be a non-empty list; `url`, `user_query`,
`session_config` and `enable_custom_actions` are optional. Submission returns
`429` when `MAX_CONCURRENT_TASKS` tasks are already active.

## Contributing

Issues and pull requests: [GitHub](https://github.com/monkscode/browser-service/issues).

Releases are automated — merging to `main` triggers `.github/workflows/publish.yml`,
which derives the next version from PyPI and publishes. Do not hand-edit the
version in `pyproject.toml`; CI overwrites it at build time.

## License

MIT — see [LICENSE](LICENSE).
