"""This repo must never accumulate Cyclone-Wiki note content.

AGENTS.md §1: the platform owns the API schema, not the notes. Fixtures are
synthetic or public. The guard exists before the first fixture lands on
purpose -- a privacy rule that arrives after the data does is not a rule.

It matches on Cyclone Profile frontmatter markers rather than on wording, so a
copied note trips it regardless of what the note says.
"""

from __future__ import annotations

import re

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


def test_no_forbidden_privacy_class_in_any_markdown() -> None:
    offenders: list[str] = []
    for path in iter_repo_files(suffixes=frozenset({".md", ".mdf", ".markdown"})):
        text = path.read_text(encoding="utf-8", errors="replace")
        block = _frontmatter(text)
        if block is None:
            continue
        for match in _PRIVACY_LINE.finditer(block):
            value = match.group("value")
            if value in FORBIDDEN_PRIVACY_VALUES:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: privacy: {value}")

    assert not offenders, (
        "Wiki content with a non-public privacy class is present in this repo:\n"
        + "\n".join(offenders)
    )


def test_guard_detects_a_planted_note(tmp_path) -> None:
    """The guard is only worth having if it actually fires.

    Mutation check in test form: a synthetic student-private note must be
    detected by the same matcher the real scan uses.
    """
    planted = tmp_path / "planted.md"
    planted.write_text(
        "---\ntype: Source\nprivacy: student-private\n---\n\n# planted\n",
        encoding="utf-8",
    )
    block = _frontmatter(planted.read_text(encoding="utf-8"))
    assert block is not None
    found = [m.group("value") for m in _PRIVACY_LINE.finditer(block)]
    assert "student-private" in found

    benign = tmp_path / "benign.md"
    benign.write_text(
        "---\ntype: Source\nprivacy: public\n---\n\n# benign\n",
        encoding="utf-8",
    )
    benign_block = _frontmatter(benign.read_text(encoding="utf-8"))
    assert benign_block is not None
    benign_found = [m.group("value") for m in _PRIVACY_LINE.finditer(benign_block)]
    assert benign_found == ["public"]
    assert not set(benign_found) & set(FORBIDDEN_PRIVACY_VALUES)


def test_guard_detects_a_planted_internal_note(tmp_path) -> None:
    """Issue #23 hardening: `internal` must trip the same matcher.

    A real Cyclone-Wiki note vendored into this repo would almost always
    declare `internal`, not `student-private` -- the survey behind adding
    `internal` to ``FORBIDDEN_PRIVACY_VALUES`` found zero `public` notes in
    Core. This is the mutation check for that specific addition: removing
    `internal` from the tuple must turn this red.
    """
    planted = tmp_path / "planted-internal.md"
    planted.write_text(
        "---\ntype: Procedure\nprivacy: internal\n---\n\n# planted\n",
        encoding="utf-8",
    )
    block = _frontmatter(planted.read_text(encoding="utf-8"))
    assert block is not None
    found = [m.group("value") for m in _PRIVACY_LINE.finditer(block)]
    assert "internal" in found
    assert "internal" in FORBIDDEN_PRIVACY_VALUES
