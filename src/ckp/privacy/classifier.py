"""Resolve a note's privacy class from its frontmatter, or refuse to.

The outcome is one of two types, and the split is the point:

* :class:`Classified` -- the note declares exactly one recognisable privacy
  class. Carries the class and where the evidence came from (AGENTS.md §8).
* :class:`Unclassified` -- anything else. It has **no** ``privacy`` attribute
  at all, so code cannot read a class off an undetermined item even by
  accident; red line R1 ("no item proceeds with its privacy class still
  undetermined") starts at the type layer, not at a runtime check.

Parsing is deliberately strict and every deviation lands on the reject side.
The classifier recognises one shape only: a frontmatter block delimited by
``---`` lines, containing exactly one column-zero ``privacy:`` line whose
value is one of the four literal tokens of Decision §1. In particular:

* **Duplicate declarations are refused, not resolved.** A YAML parser would
  quietly let the last key win, which turns ``privacy: student-private``
  followed by ``privacy: public`` into a downgrade smuggled past the gate
  (Decision §3 forbids downgrades). Two declarations mean the note's class is
  ambiguous, and ambiguous is undetermined.
* **Only column-zero keys count.** An indented ``privacy:`` line belongs to
  some nested structure and says nothing about the note itself.
* **Case and spelling are exact.** ``Public`` is not a class. Guessing what a
  near-miss meant is how a misspelling becomes an admission.

Reason-code names here are provisional (Decision §8): the *semantics* --
refuse, stable machine-readable code, no payload echo -- are frozen; the
literal names get aligned with the Writer's error namespace when Phase 3
freezes it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, final

from ckp.privacy.classes import PrivacyClass
from ckp.revision import bundle_member_key

#: Provisional reason codes for undetermined outcomes. Stable within this
#: repo; renamed only via the Writer namespace mapping table, never ad hoc.
REASON_NOT_A_BUNDLE_MEMBER = "not-a-bundle-member"
REASON_UNREADABLE = "note-unreadable"
REASON_FRONTMATTER_MISSING = "frontmatter-missing"
REASON_FRONTMATTER_UNTERMINATED = "frontmatter-unterminated"
REASON_PRIVACY_MISSING = "privacy-missing"
REASON_PRIVACY_DUPLICATE = "privacy-duplicate"
REASON_PRIVACY_INVALID = "privacy-invalid"

#: A column-zero privacy declaration. The value group is matched against the
#: class tokens separately so that *any* other line shape -- trailing comment,
#: block-scalar marker, list item -- fails as a whole line, not as a value.
#: The separator after the colon is ``+``, not ``*``: YAML block mappings
#: require whitespace there, so ``privacy:public`` is not a mapping and a
#: parser that accepted it would classify a note the rule source would not.
_PRIVACY_KEY = re.compile(r"^privacy:")
_PRIVACY_DECLARATION = re.compile(
    r"^privacy:[ \t]+(?P<quote>['\"]?)(?P<value>[a-z-]+)(?P=quote)[ \t]*$"
)


@dataclass(frozen=True)
class Classified:
    """A determined privacy class plus the evidence trail it came from."""

    privacy: PrivacyClass
    #: Where the class was read from. Currently always ``"frontmatter"``;
    #: recorded so a future second evidence source stays distinguishable
    #: (AGENTS.md §8 -- self-declared and derived evidence must not blur).
    source: str


@dataclass(frozen=True)
class Unclassified:
    """No determined class. Deliberately carries no ``privacy`` attribute.

    ``reason`` is a stable code, never the offending text: a malformed
    privacy value is payload, and payload is exactly what a refusal receipt
    must not echo (Decision §7) -- the misplaced text could itself be the
    sensitive content.
    """

    reason: str


Classification = Classified | Unclassified


class Classifier(Protocol):
    """What the gate requires. Anything classifying notes must look like this."""

    def classify(self, path: Path) -> Classification: ...


def _split_lines(text: str) -> list[str]:
    """Split on ``\\n`` only, tolerating CRLF.

    ``splitlines`` also breaks on U+2028 and friends, so a frontmatter line
    containing one would be silently split into two and could change what
    looks like a declaration. Same root cause as the commit-stamp parser in
    ``ckp.revision`` -- kept consistent deliberately.
    """
    return [line.rstrip("\r") for line in text.split("\n")]


def _frontmatter_lines(text: str) -> tuple[list[str] | None, str | None]:
    """The frontmatter body lines, or ``(None, reason)`` when there are none.

    Strict shape: the first line is exactly ``---`` and a later line is
    exactly ``---``. No BOM allowance, no ``----`` prefix-matching -- a note
    whose frontmatter this cannot see is a note whose privacy is undetermined,
    which fails closed downstream.
    """
    lines = _split_lines(text)
    if not lines or lines[0] != "---":
        return None, REASON_FRONTMATTER_MISSING
    for index in range(1, len(lines)):
        if lines[index] == "---":
            return lines[1:index], None
    return None, REASON_FRONTMATTER_UNTERMINATED


@final
class FrontmatterClassifier:
    """Classify a note by the single privacy declaration in its frontmatter.

    Constructed against a bundle root, which is **required**: this is a
    separate read point from the C1 bundle walk and inherits none of its
    containment, so it applies the same membership rule itself via
    ``bundle_member_key`` -- one implementation, reused, so the two read
    points cannot drift. That refuses symlinks at *every* component below the
    root (a ``bundle/inbox -> /outside`` link must not pull external content
    into classification), refuses paths that resolve outside the root, and
    refuses non-regular files, while keeping C1's allowance for the root
    itself being a symlink (mounting a checkout at a stable path is normal).

    The remaining check-to-read race is the same known residue as C1's,
    owned by the openat-anchored walk in C3 (Epic #1).
    """

    def __init__(self, bundle_root: Path) -> None:
        self._bundle_root = bundle_root

    def classify(self, path: Path) -> Classification:
        if bundle_member_key(path, self._bundle_root) is None:
            return Unclassified(reason=REASON_NOT_A_BUNDLE_MEMBER)
        try:
            raw = path.read_bytes()
        except (OSError, ValueError):
            return Unclassified(reason=REASON_UNREADABLE)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return Unclassified(reason=REASON_UNREADABLE)

        body, failure = _frontmatter_lines(text)
        if body is None:
            assert failure is not None
            return Unclassified(reason=failure)

        declarations = [line for line in body if _PRIVACY_KEY.match(line)]
        if not declarations:
            return Unclassified(reason=REASON_PRIVACY_MISSING)
        if len(declarations) > 1:
            return Unclassified(reason=REASON_PRIVACY_DUPLICATE)

        match = _PRIVACY_DECLARATION.match(declarations[0])
        if match is None:
            return Unclassified(reason=REASON_PRIVACY_INVALID)
        try:
            privacy = PrivacyClass(match.group("value"))
        except ValueError:
            return Unclassified(reason=REASON_PRIVACY_INVALID)
        return Classified(privacy=privacy, source="frontmatter")


__all__ = [
    "REASON_FRONTMATTER_MISSING",
    "REASON_FRONTMATTER_UNTERMINATED",
    "REASON_NOT_A_BUNDLE_MEMBER",
    "REASON_PRIVACY_DUPLICATE",
    "REASON_PRIVACY_INVALID",
    "REASON_PRIVACY_MISSING",
    "REASON_UNREADABLE",
    "Classification",
    "Classified",
    "Classifier",
    "FrontmatterClassifier",
    "Unclassified",
]
