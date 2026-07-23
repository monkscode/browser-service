#!/usr/bin/env python3
"""Resolve the next release version using the latest PyPI publication.

Fetching the most recent version directly from PyPI removes the dependency on
Git tags or GitHub releases, which keeps automated publishing consistent even if
no manual tagging occurs.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Optional, Tuple

SEMVER_PATTERN = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def _parse_version(value: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not value:
        return None
    value = value.strip()
    if value.startswith(("v", "V")):
        value = value[1:]
    match = SEMVER_PATTERN.fullmatch(value)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def _format_version(version: Optional[Tuple[int, int, int]]) -> str:
    if version is None:
        return "none"
    return ".".join(str(part) for part in version)


def _latest_pypi_version(package: str) -> Optional[Tuple[int, int, int]]:
    url = f"https://pypi.org/pypi/{package}/json"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise
    except urllib.error.URLError:
        return None

    info = payload.get("info", {})
    return _parse_version(info.get("version"))


def _bump(version: Tuple[int, int, int], level: str) -> Tuple[int, int, int]:
    major, minor, patch = version
    if level == "major":
        return major + 1, 0, 0
    if level == "minor":
        return major, minor + 1, 0
    return major, minor, patch + 1


def main() -> int:
    package_name = os.environ.get("PYPI_PACKAGE_NAME", "browser-service").strip()
    if not package_name:
        print("PYPI_PACKAGE_NAME cannot be empty", file=sys.stderr)
        return 1

    bump_level = os.environ.get("AUTO_BUMP_LEVEL", "minor").strip().lower()
    if bump_level not in {"major", "minor", "patch"}:
        print(
            f"Unsupported AUTO_BUMP_LEVEL '{bump_level}', defaulting to patch.",
            file=sys.stderr,
        )
        bump_level = "patch"

    base_version = _latest_pypi_version(package_name) or (0, 0, 0)
    next_version = _bump(base_version, bump_level)

    print(
        "Resolved versions -> PyPI={} | next={} ({})".format(
            _format_version(base_version),
            _format_version(next_version),
            bump_level,
        ),
        file=sys.stderr,
    )

    print(_format_version(next_version))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
