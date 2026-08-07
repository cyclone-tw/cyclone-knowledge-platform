"""This repo must never accumulate Cyclone-Wiki note content.

AGENTS.md §1: the platform owns the API schema, not the notes. Fixtures are
synthetic or public. The guard exists before the first fixture lands on
purpose -- a privacy rule that arrives after the data does is not a rule.

It matches on Cyclone Profile frontmatter markers rather than on wording, so a
copied note trips it regardless of what the note says.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from conftest import REPO_ROOT, iter_repo_files

# A synthetic fixture may legitimately carry Cyclone Profile frontmatter, so
# the guard is about the *privacy class*, not about frontmatter existing.
# Privacy classes that may never appear in this repo in any form:
#
# `internal` joined this list for issue #23 (D5/RP1 hardening): a survey of
# the whole Cyclone-Wiki `Core/` tree found zero notes declaring `public` --
# 145 `internal`, 3 `sensitive`, 2 `student-private`. Leaving `internal` off
# this list would in practice permit any real note into this repo, which
# defeats the guard's own stated purpose ("Fixtures are synthetic or
# public"). `public` remains the only privacy class this repo may carry.
FORBIDDEN_PRIVACY_VALUES = (
    "student-private",
    "restricted",
    "sensitive",
    "internal",
)

_PRIVACY_LINE = re.compile(
    r"^privacy:\s*['\"]?(?P<value>[a-z-]+)['\"]?\s*$",
    re.MULTILINE,
)


def _frontmatter(text: str) -> str | None:
    """Return the YAML frontmatter block, or None when the file has none."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    return text[3:end]


def _display_path(path: Path) -> str:
    """Render ``path`` relative to ``REPO_ROOT`` when it lives under it.

    Paths outside ``REPO_ROOT`` (the planted-note regression tests below use
    ``tmp_path`` fixtures) fall back to the path as given -- there is no
    relative form to show, and forcing one would raise.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _scan_offenders(paths: Iterable[Path]) -> list[str]:
    """The actual scanner. Both the repo-wide test and the planted-note
    regression tests below call this same function -- a planted-note test
    that duplicates the matching logic instead of calling this can go green
    while the real scanner is broken (Codex review finding on this file:
    the previous version of the `internal` regression test only asserted
    tuple membership, never exercised this function at all).

    Offender strings show paths relative to ``REPO_ROOT`` (issue #36 --
    the message previously showed absolute paths, which disagreed with the
    calling test's own "paths shown relative to ..." wording).
    """
    offenders: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        block = _frontmatter(text)
        if block is None:
            continue
        for match in _PRIVACY_LINE.finditer(block):
            value = match.group("value")
            if value in FORBIDDEN_PRIVACY_VALUES:
                offenders.append(f"{_display_path(path)}: privacy: {value}")
    return offenders


def test_no_forbidden_privacy_class_in_any_markdown() -> None:
    paths = list(iter_repo_files(suffixes=frozenset({".md", ".mdf", ".markdown"})))
    offenders = _scan_offenders(paths)

    assert not offenders, (
        "Wiki content with a non-public privacy class is present in this repo "
        f"(paths shown relative to {REPO_ROOT}):\n" + "\n".join(offenders)
    )


def test_guard_detects_a_planted_student_private_note(tmp_path) -> None:
    """The guard is only worth having if it actually fires.

    Plants real files on disk and calls the real `_scan_offenders`, the same
    function `test_no_forbidden_privacy_class_in_any_markdown` uses -- not a
    hand-rolled copy of its matching logic. A mutation that breaks the
    scanner's comparison (not just the forbidden-values tuple) must turn
    this red too.
    """
    planted = tmp_path / "planted.md"
    planted.write_text(
        "---\ntype: Source\nprivacy: student-private\n---\n\n# planted\n",
        encoding="utf-8",
    )
    benign = tmp_path / "benign.md"
    benign.write_text(
        "---\ntype: Source\nprivacy: public\n---\n\n# benign\n",
        encoding="utf-8",
    )

    offenders = _scan_offenders([planted, benign])

    assert len(offenders) == 1
    assert str(planted) in offenders[0]
    assert "student-private" in offenders[0]
    assert str(benign) not in "".join(offenders)


def test_guard_detects_a_planted_internal_note(tmp_path) -> None:
    """Issue #23 hardening: `internal` must trip the *real scanner*, not
    just be present in the `FORBIDDEN_PRIVACY_VALUES` tuple.

    A real Cyclone-Wiki note vendored into this repo would almost always
    declare `internal`, not `student-private` -- the survey behind adding
    `internal` to `FORBIDDEN_PRIVACY_VALUES` found zero `public` notes in
    Core. This calls `_scan_offenders` against a planted file on disk, so it
    catches both classes of regression: `internal` dropped from the tuple,
    and the scanner's comparison logic itself broken while the tuple stays
    intact (Codex review finding: the previous version only asserted
    `"internal" in FORBIDDEN_PRIVACY_VALUES`, which the latter mutation
    would not have caught).
    """
    planted = tmp_path / "planted-internal.md"
    planted.write_text(
        "---\ntype: Procedure\nprivacy: internal\n---\n\n# planted\n",
        encoding="utf-8",
    )
    benign = tmp_path / "benign.md"
    benign.write_text(
        "---\ntype: Procedure\nprivacy: public\n---\n\n# benign\n",
        encoding="utf-8",
    )

    offenders = _scan_offenders([planted, benign])

    assert len(offenders) == 1
    assert str(planted) in offenders[0]
    assert "internal" in offenders[0]
    assert str(benign) not in "".join(offenders)
