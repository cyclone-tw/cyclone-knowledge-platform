"""Synthetic note factories for the privacy tests.

Every note here is fabricated at test time into a temporary directory and
never lands as tracked Markdown. That is deliberate: ``test_no_wiki_content``
forbids any tracked Markdown carrying a non-public privacy class, and carving
a fixtures allowlist into that guard would weaken exactly the check these
tests exist to strengthen. The guard stays untouched at full strength; the
fixtures cover every class anyway.

The ``student-private`` sample is unmistakably fake -- an invented student at
an invented school, labelled as such in its own body -- because a synthetic
fixture that *looks* real is indistinguishable from a leak in a diff.
"""

from __future__ import annotations

from pathlib import Path

#: Appears in refusal-echo tests: if any verdict field ever contains this,
#: the gate leaked payload into a receipt.
PAYLOAD_SENTINEL = "PAYLOAD-SENTINEL-9f3a"

_BODIES = {
    "public": (
        "---\n"
        "uuid: 0192f000-0000-7000-8000-00000000c201\n"
        "title: Synthetic public note\n"
        "type: Source\n"
        "privacy: public\n"
        "---\n\n"
        "# Synthetic public note\n\n"
        "SYNTHETIC FIXTURE. Placeholder prose about kettles.\n"
    ),
    "internal": (
        "---\n"
        "uuid: 0192f000-0000-7000-8000-00000000c202\n"
        "title: Synthetic internal note\n"
        "type: Procedure\n"
        "privacy: internal\n"
        "---\n\n"
        "# Synthetic internal note\n\n"
        "SYNTHETIC FIXTURE. Pretend operational runbook, no real content.\n"
    ),
    "sensitive": (
        "---\n"
        "uuid: 0192f000-0000-7000-8000-00000000c203\n"
        "title: Synthetic sensitive note\n"
        "type: Source\n"
        "privacy: sensitive\n"
        "---\n\n"
        "# Synthetic sensitive note\n\n"
        "SYNTHETIC FIXTURE. A pretend private meeting between two invented\n"
        "adults, Alex Example and Sam Placeholder, about nothing real.\n"
    ),
    "student-private": (
        "---\n"
        "uuid: 0192f000-0000-7000-8000-00000000c204\n"
        "title: Synthetic student-private note\n"
        "type: Source\n"
        "privacy: student-private\n"
        "---\n\n"
        "# Synthetic student-private note\n\n"
        "SYNTHETIC FIXTURE -- NOT A REAL PERSON. Invented student\n"
        '"Zaphod Example-Student" at the invented "Synthetic Academy for\n'
        'Fabricated Data", invented guardian "Trillian Example-Guardian".\n'
        "Every detail on this page is fabricated for gate testing.\n"
    ),
}


def write_note(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_bytes(text.encode("utf-8"))
    return path


def classed_note(directory: Path, privacy: str) -> Path:
    """A well-formed synthetic note declaring the given privacy class."""
    return write_note(directory, f"note-{privacy}.md", _BODIES[privacy])


def note_without_frontmatter(directory: Path) -> Path:
    return write_note(
        directory,
        "no-frontmatter.md",
        "# Synthetic note\n\nNo frontmatter at all.\nprivacy: public\n",
    )


def note_with_unterminated_frontmatter(directory: Path) -> Path:
    return write_note(
        directory,
        "unterminated.md",
        "---\ntitle: Synthetic unterminated\nprivacy: public\n\n# body\n",
    )


def note_without_privacy(directory: Path) -> Path:
    return write_note(
        directory,
        "no-privacy.md",
        "---\nuuid: 0192f000-0000-7000-8000-00000000c210\n"
        "title: Synthetic note with no privacy declaration\ntype: Source\n"
        "---\n\n# Synthetic note\n",
    )


def note_with_duplicate_privacy(directory: Path) -> Path:
    """The downgrade smuggle: a strict class followed by a lax one.

    A last-wins parser reads this as ``public`` and admits it. The classifier
    must instead refuse the ambiguity outright.
    """
    return write_note(
        directory,
        "duplicate-privacy.md",
        "---\ntitle: Synthetic duplicate declaration\n"
        "privacy: student-private\nprivacy: public\n---\n\n# body\n",
    )


def note_with_invalid_privacy(directory: Path) -> Path:
    """An unknown token -- and the token doubles as the payload sentinel."""
    return write_note(
        directory,
        "invalid-privacy.md",
        f"---\ntitle: Synthetic invalid value\nprivacy: {PAYLOAD_SENTINEL}\n"
        "---\n\n# body\n",
    )


def note_with_no_space_after_colon(directory: Path) -> Path:
    """Not a YAML mapping: block mappings require whitespace after the colon.

    A parser that accepts this classifies a note the rule source would not.
    """
    return write_note(
        directory,
        "no-space-privacy.md",
        "---\ntitle: Synthetic missing separator\nprivacy:public\n---\n\n# body\n",
    )


def note_with_no_space_quoted(directory: Path) -> Path:
    return write_note(
        directory,
        "no-space-quoted-privacy.md",
        '---\ntitle: Synthetic missing separator\nprivacy:"public"\n---\n\n# body\n',
    )


def note_with_miscased_privacy(directory: Path) -> Path:
    return write_note(
        directory,
        "miscased-privacy.md",
        "---\ntitle: Synthetic miscased value\nprivacy: Public\n---\n\n# body\n",
    )


def note_with_list_privacy(directory: Path) -> Path:
    return write_note(
        directory,
        "list-privacy.md",
        "---\ntitle: Synthetic non-scalar value\nprivacy: [public]\n---\n\n# body\n",
    )


def note_with_nested_privacy_only(directory: Path) -> Path:
    """A privacy key that belongs to a nested mapping, not to the note."""
    return write_note(
        directory,
        "nested-privacy.md",
        "---\ntitle: Synthetic nested key\nexport:\n  privacy: public\n---\n\n# body\n",
    )


def note_with_privacy_in_body_only(directory: Path) -> Path:
    """A declaration-shaped line after the frontmatter closed."""
    return write_note(
        directory,
        "body-privacy.md",
        "---\ntitle: Synthetic body-only declaration\ntype: Source\n---\n\n"
        "privacy: public\n\n# body\n",
    )


def note_with_commented_privacy(directory: Path) -> Path:
    return write_note(
        directory,
        "commented-privacy.md",
        "---\ntitle: Synthetic trailing comment\nprivacy: public # trusted\n"
        "---\n\n# body\n",
    )


def note_with_quoted_privacy(directory: Path) -> Path:
    """Quoted declarations are still declarations."""
    return write_note(
        directory,
        "quoted-privacy.md",
        '---\ntitle: Synthetic quoted value\nprivacy: "internal"\n---\n\n# body\n',
    )


def note_with_crlf_line_endings(directory: Path) -> Path:
    return write_note(
        directory,
        "crlf.md",
        "---\r\ntitle: Synthetic CRLF note\r\nprivacy: internal\r\n---\r\n"
        "\r\n# body\r\n",
    )


def undecodable_note(directory: Path) -> Path:
    path = directory / "undecodable.md"
    path.write_bytes(b"---\nprivacy: public\n---\n\xff\xfe broken \x80\n")
    return path
