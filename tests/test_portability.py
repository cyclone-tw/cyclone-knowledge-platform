"""Portability guard: no absolute host paths in executable/shipped source.

Contract Phase 8 requires this package to cold-rebuild on a Mac mini and then
on a Mac Studio with no source edits, so host-specific locations must arrive
through config or environment only (AGENTS.md §7).

Documentation is deliberately out of scope -- prose legitimately names host
paths when describing conventions. This guard covers what actually runs or
ships.
"""

from __future__ import annotations

from pathlib import Path

from conftest import REPO_ROOT, iter_repo_files

SCANNED_SUFFIXES = frozenset({".py", ".toml", ".yml", ".yaml", ".sh", ".cfg", ".json"})
SCANNED_NAMES = frozenset({"Dockerfile", "docker-compose.yml", ".dockerignore"})

# Absolute user-home roots that pin a file to one machine.
FORBIDDEN_PREFIXES = ("/Users/", "/home/")

# This guard is the only file allowed to skip itself: it has to contain the
# very literals it forbids. Keep this list at exactly one entry -- the
# assertion below is what stops it from quietly growing into an escape hatch.
SELF_EXEMPT = (Path(__file__).resolve(),)


def test_self_exemption_stays_a_single_file() -> None:
    """The exemption list must not grow. One file, and it must be this one."""
    assert len(SELF_EXEMPT) == 1
    assert SELF_EXEMPT[0] == Path(__file__).resolve()


def test_no_absolute_host_paths_in_source() -> None:
    offenders: list[str] = []
    scanned = 0

    for path in iter_repo_files(suffixes=SCANNED_SUFFIXES, names=SCANNED_NAMES):
        if path.resolve() in SELF_EXEMPT:
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for prefix in FORBIDDEN_PREFIXES:
                if prefix in line:
                    rel = path.relative_to(REPO_ROOT)
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")

    # A guard that silently scans zero files reports success while checking
    # nothing. Pin a floor so an over-eager skip rule shows up as a failure.
    assert scanned >= 3, f"portability guard only scanned {scanned} files"
    assert not offenders, "absolute host paths in tracked source:\n" + "\n".join(
        offenders
    )
