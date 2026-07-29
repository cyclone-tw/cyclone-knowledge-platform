"""Shared pytest fixtures and repo-walking helpers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that never hold tracked source.
_SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    "htmlcov",
}


def iter_repo_files(
    suffixes: frozenset[str] | None = None,
    names: frozenset[str] | None = None,
) -> Iterator[Path]:
    """Yield tracked-ish repo files matching ``suffixes`` or exact ``names``.

    Walks the working tree rather than shelling out to ``git ls-files`` so the
    helper works in a plain source export as well as a checkout.
    """
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        if (names and path.name in names) or (suffixes and path.suffix in suffixes):
            yield path


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
