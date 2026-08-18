"""Assert every table in README.md still matches the code it describes.

Why this file exists
--------------------
The README drifted far enough that every code example in it failed. The
replacement was verified by hand, and the hand-verification was itself wrong
twice: an ``os.getenv`` grep missed the nine variables read through the
``_int_env`` wrapper, and an ``inspect.signature().bind()`` check passed on a
call that raises ``ValueError`` at runtime. Both checks confirmed shape and
never behaviour.

So the README's four tables are asserted here instead of by hand. Each test
reads the table out of README.md and the corresponding fact out of the source,
and fails on any difference in either direction — a documented variable that
no longer exists, and an added variable nobody documented.

These are deliberately content assertions, not formatting ones: they compare
sets of names, so the tables can be reflowed freely. A failure means the README
and the code disagree, not that markdown moved.

Referenced by: CI fast lane (unmarked, no browser, no network).
Depends on: README.md, browser_service/config.py,
            browser_service/agent/registration.py, browser_service/api/routes.py
"""

import ast
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"


# ---------------------------------------------------------------------------
# README parsing helpers
# ---------------------------------------------------------------------------


def _section(title: str) -> str:
    """Return the body of a top-level ``## <title>`` section."""
    text = README.read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(title)}$(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, f"README section '## {title}' not found"
    return match.group(1)


def _table_cells(section: str, column: int) -> list[str]:
    """Return the raw text of one column across every table row in a section.

    Header rows (``| Variable |``) and separators (``|---|---|``) carry no
    backticks, so callers filter on the backtick pattern rather than on
    position — that keeps the parse working when a table gains a column.
    """
    cells = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        fields = [f.strip() for f in line.strip("|").split("|")]
        if column < len(fields):
            cells.append(fields[column])
    return cells


def _backticked_names(cells: list[str], pattern: str) -> set[str]:
    """Extract every ``name`` in backticks from the given cells."""
    found: set[str] = set()
    for cell in cells:
        found.update(re.findall(rf"`({pattern})`", cell))
    return found


# ---------------------------------------------------------------------------
# 1. Environment variables
# ---------------------------------------------------------------------------


def _env_vars_read_by_config() -> set[str]:
    """Every environment variable name ``config.py`` actually reads.

    Covers both access paths: direct ``os.getenv("NAME", ...)`` and the
    ``self._int_env("NAME", default)`` wrapper. Missing the wrapper is exactly
    how the previous hand-check reported "10 for 10, neither direction
    missing" while the file read nineteen.
    """
    source = (REPO_ROOT / "browser_service" / "config.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        is_getenv = isinstance(func, ast.Attribute) and func.attr == "getenv"
        is_int_env = isinstance(func, ast.Attribute) and func.attr == "_int_env"
        if not (is_getenv or is_int_env):
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            names.add(first.value)
    return names


def test_readme_documents_every_environment_variable():
    documented = _backticked_names(_table_cells(_section("Configuration"), 0), r"[A-Z][A-Z0-9_]+")
    actual = _env_vars_read_by_config()

    assert documented == actual, (
        f"README env table and config.py disagree.\n"
        f"  documented but not read: {sorted(documented - actual)}\n"
        f"  read but not documented: {sorted(actual - documented)}"
    )


def _env_defaults_read_by_config() -> dict[str, str]:
    """The literal fallback each variable is read with.

    Only meaningful for the variables that fall straight through to a built-in
    default — the NLRF-aware ones are read with an empty sentinel and get their
    real default further down the chain, so they are excluded by the caller.
    """
    source = (REPO_ROOT / "browser_service" / "config.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    defaults: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) != 2:
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in ("getenv", "_int_env")):
            continue
        name, fallback = node.args
        if isinstance(name, ast.Constant) and isinstance(fallback, ast.Constant):
            defaults[name.value] = str(fallback.value)
    return defaults


def test_readme_documents_the_real_default_for_each_plain_variable():
    """The third Configuration table publishes 14 concrete values — gate them.

    A retry count tuned in ``config.py`` must not leave the README quoting the
    old number. Only the third table is checked: the first two describe
    fallback chains rather than literal defaults.
    """
    section = _section("Configuration")
    marker = "The rest read the environment and fall straight through"
    assert marker in section, "third Configuration table's lead-in text moved"
    plain_table = section.split(marker, 1)[1]

    actual = _env_defaults_read_by_config()
    mismatches = []
    for line in plain_table.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        fields = [f.strip() for f in line.strip("|").split("|")]
        if len(fields) < 2:
            continue
        name = re.fullmatch(r"`([A-Z][A-Z0-9_]+)`", fields[0])
        if not name:
            continue
        # "unset" is how the table renders an empty-string fallback.
        documented = "" if fields[1] == "unset" else fields[1].strip("`")
        real = actual.get(name.group(1))
        if documented != real:
            mismatches.append(f"{name.group(1)}: README says {documented!r}, code uses {real!r}")

    assert not mismatches, "README defaults disagree with config.py:\n  " + "\n  ".join(mismatches)


def test_google_model_resolves_nlrf_before_environment():
    """Pin the one inverted resolution order the README calls out.

    ``_get_google_model`` consults NLRF's settings *first* and only then the
    environment, unlike every other NLRF-aware setting. The README documents
    that inversion; this asserts the code still has it, so the note cannot
    silently become false.
    """
    source = (REPO_ROOT / "browser_service" / "config.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    func = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_get_google_model"
    )

    settings_line = next(
        n.lineno
        for n in ast.walk(func)
        if isinstance(n, ast.ImportFrom) and n.module == "src.backend.core.config"
    )
    getenv_line = next(
        n.lineno
        for n in ast.walk(func)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "getenv"
    )

    assert settings_line < getenv_line, (
        "_get_google_model no longer reads NLRF settings before GOOGLE_MODEL; "
        "the README's inverted-resolution note is now wrong"
    )


# ---------------------------------------------------------------------------
# 2. Public API
# ---------------------------------------------------------------------------


def test_readme_public_api_block_matches_dunder_all():
    import browser_service
    import browser_service.api

    block = _section("Public API")
    imports = dict(re.findall(r"from ([\w.]+) import \(\s*(.*?)\)", block, re.S))
    assert set(imports) == {"browser_service", "browser_service.api"}, (
        f"README Public API block imports from unexpected modules: {sorted(imports)}"
    )

    for module, body in imports.items():
        documented = set(re.findall(r"^\s*(\w+),", body, re.M))
        actual = set(vars(__import__(module, fromlist=["__all__"]))["__all__"])
        assert documented == actual, (
            f"README Public API block and {module}.__all__ disagree.\n"
            f"  documented but not exported: {sorted(documented - actual)}\n"
            f"  exported but not documented: {sorted(actual - documented)}"
        )


# ---------------------------------------------------------------------------
# 3. Element actions
# ---------------------------------------------------------------------------


def _actions_accepted_by_dispatcher() -> set[str]:
    """The dispatcher's canonical accept-list.

    ``registration.py`` guards with ``if action not in (...)`` and rejects
    anything else, so that single tuple is the whole supported action set —
    every ``elif`` further down is a subset of it.
    """
    source = (REPO_ROOT / "browser_service" / "agent" / "registration.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], ast.NotIn):
            continue
        if not (isinstance(node.left, ast.Name) and node.left.id == "action"):
            continue
        right = node.comparators[0]
        if isinstance(right, (ast.Tuple, ast.List)) and all(
            isinstance(e, ast.Constant) for e in right.elts
        ):
            return {e.value for e in right.elts}

    pytest.fail("no `action not in (...)` accept-list found in registration.py")


def test_readme_action_table_matches_dispatcher():
    section = _section("Element actions")
    # Column 0 is the primary action, column 1 its aliases ("—" when none).
    documented = _backticked_names(_table_cells(section, 0), r"[a-z_]+")
    documented |= _backticked_names(_table_cells(section, 1), r"[a-z_]+")
    actual = _actions_accepted_by_dispatcher()

    assert documented == actual, (
        f"README action table and the dispatcher disagree.\n"
        f"  documented but rejected: {sorted(documented - actual)}\n"
        f"  accepted but undocumented: {sorted(actual - documented)}"
    )


# ---------------------------------------------------------------------------
# 4. HTTP endpoints
# ---------------------------------------------------------------------------


def test_readme_endpoint_table_matches_registered_routes():
    from flask import Flask

    from browser_service.api import register_routes

    app = Flask(__name__)
    register_routes(app, MagicMock())

    actual = {
        (method, str(rule))
        for rule in app.url_map.iter_rules()
        if rule.endpoint != "static"
        for method in rule.methods
        if method in ("GET", "POST", "PUT", "PATCH", "DELETE")
    }

    documented = set()
    section = _section("HTTP endpoints")
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        fields = [f.strip() for f in line.strip("|").split("|")]
        if len(fields) < 2:
            continue
        method = re.fullmatch(r"`([A-Z]+)`", fields[0])
        path = re.fullmatch(r"`(/[^`]*)`", fields[1])
        if method and path:
            documented.add((method.group(1), path.group(1)))

    assert documented == actual, (
        f"README endpoint table and the registered routes disagree.\n"
        f"  documented but not registered: {sorted(documented - actual)}\n"
        f"  registered but not documented: {sorted(actual - documented)}"
    )
