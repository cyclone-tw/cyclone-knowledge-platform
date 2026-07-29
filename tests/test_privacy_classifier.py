"""FrontmatterClassifier: one recognisable shape in, everything else refused."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from ckp.privacy import Classified, FrontmatterClassifier, PrivacyClass, Unclassified
from ckp.privacy.classifier import (
    REASON_FRONTMATTER_MISSING,
    REASON_FRONTMATTER_UNTERMINATED,
    REASON_PRIVACY_DUPLICATE,
    REASON_PRIVACY_INVALID,
    REASON_PRIVACY_MISSING,
    REASON_UNREADABLE,
)
from privacy_fixtures import (
    PAYLOAD_SENTINEL,
    classed_note,
    note_with_commented_privacy,
    note_with_crlf_line_endings,
    note_with_duplicate_privacy,
    note_with_invalid_privacy,
    note_with_list_privacy,
    note_with_miscased_privacy,
    note_with_nested_privacy_only,
    note_with_privacy_in_body_only,
    note_with_quoted_privacy,
    note_with_unterminated_frontmatter,
    note_without_frontmatter,
    note_without_privacy,
    undecodable_note,
)

CLASSIFIER = FrontmatterClassifier()


@pytest.mark.parametrize(
    "token,expected",
    [
        ("public", PrivacyClass.PUBLIC),
        ("internal", PrivacyClass.INTERNAL),
        ("sensitive", PrivacyClass.SENSITIVE),
        ("student-private", PrivacyClass.STUDENT_PRIVATE),
    ],
)
def test_each_class_token_classifies_to_its_class(
    tmp_path: Path, token: str, expected: PrivacyClass
) -> None:
    outcome = CLASSIFIER.classify(classed_note(tmp_path, token))
    assert isinstance(outcome, Classified)
    assert outcome.privacy is expected
    assert outcome.source == "frontmatter"


def test_quoted_declaration_is_a_declaration(tmp_path: Path) -> None:
    outcome = CLASSIFIER.classify(note_with_quoted_privacy(tmp_path))
    assert isinstance(outcome, Classified)
    assert outcome.privacy is PrivacyClass.INTERNAL


def test_crlf_note_classifies(tmp_path: Path) -> None:
    outcome = CLASSIFIER.classify(note_with_crlf_line_endings(tmp_path))
    assert isinstance(outcome, Classified)
    assert outcome.privacy is PrivacyClass.INTERNAL


@pytest.mark.parametrize(
    "factory,reason",
    [
        (note_without_frontmatter, REASON_FRONTMATTER_MISSING),
        (note_with_unterminated_frontmatter, REASON_FRONTMATTER_UNTERMINATED),
        (note_without_privacy, REASON_PRIVACY_MISSING),
        (note_with_duplicate_privacy, REASON_PRIVACY_DUPLICATE),
        (note_with_invalid_privacy, REASON_PRIVACY_INVALID),
        (note_with_miscased_privacy, REASON_PRIVACY_INVALID),
        (note_with_list_privacy, REASON_PRIVACY_INVALID),
        (note_with_commented_privacy, REASON_PRIVACY_INVALID),
        (note_with_nested_privacy_only, REASON_PRIVACY_MISSING),
        (note_with_privacy_in_body_only, REASON_PRIVACY_MISSING),
        (undecodable_note, REASON_UNREADABLE),
    ],
)
def test_every_deviation_is_undetermined(tmp_path: Path, factory, reason) -> None:
    outcome = CLASSIFIER.classify(factory(tmp_path))
    assert isinstance(outcome, Unclassified)
    assert outcome.reason == reason


def test_duplicate_declaration_never_resolves_to_the_laxer_class(
    tmp_path: Path,
) -> None:
    """The downgrade smuggle: student-private followed by public.

    A last-wins parser admits this note as public. Whatever else changes in
    the classifier, this must never classify.
    """
    outcome = CLASSIFIER.classify(note_with_duplicate_privacy(tmp_path))
    assert isinstance(outcome, Unclassified)


def test_missing_file_is_unreadable(tmp_path: Path) -> None:
    outcome = CLASSIFIER.classify(tmp_path / "does-not-exist.md")
    assert isinstance(outcome, Unclassified)
    assert outcome.reason == REASON_UNREADABLE


def test_directory_is_unreadable(tmp_path: Path) -> None:
    (tmp_path / "a-directory.md").mkdir()
    outcome = CLASSIFIER.classify(tmp_path / "a-directory.md")
    assert isinstance(outcome, Unclassified)
    assert outcome.reason == REASON_UNREADABLE


def test_symlink_is_refused_even_when_target_is_classifiable(
    tmp_path: Path,
) -> None:
    """This is a separate read point; C1's bundle-walk symlink refusal does
    not protect it, so it refuses links itself."""
    target = classed_note(tmp_path, "public")
    link = tmp_path / "link.md"
    link.symlink_to(target)
    outcome = CLASSIFIER.classify(link)
    assert isinstance(outcome, Unclassified)
    assert outcome.reason == REASON_UNREADABLE


def test_permission_denied_is_unreadable(tmp_path: Path) -> None:
    note = classed_note(tmp_path, "public")
    note.chmod(0)
    try:
        # Premise from an independent source, not from the code under test:
        # under root (some containers) chmod 0 still reads fine and the
        # scenario cannot be built. os.access answers without ckp's help.
        if os.access(note, os.R_OK):
            pytest.skip("running with privileges that ignore file modes")
        outcome = CLASSIFIER.classify(note)
        assert isinstance(outcome, Unclassified)
        assert outcome.reason == REASON_UNREADABLE
    finally:
        note.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_unclassified_never_echoes_the_offending_value(tmp_path: Path) -> None:
    """A refusal receipt carries a code, never payload (Decision §7)."""
    outcome = CLASSIFIER.classify(note_with_invalid_privacy(tmp_path))
    assert isinstance(outcome, Unclassified)
    assert PAYLOAD_SENTINEL not in outcome.reason
    assert PAYLOAD_SENTINEL not in repr(outcome)
