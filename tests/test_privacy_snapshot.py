"""Privacy admission is bound to the exact immutable snapshot member."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ckp.bundle import AnchoredBundleReader, BundleMember
from ckp.privacy import (
    Admitted,
    Classified,
    FrontmatterClassifier,
    PrivacyClass,
    PrivacyGate,
)


def _member(content: bytes) -> BundleMember:
    return BundleMember(
        relative_path="note.md",
        digest_key="note.md",
        content=content,
        content_sha256=hashlib.sha256(content).hexdigest(),
    )


def test_classifier_can_classify_the_bytes_already_in_a_snapshot(
    tmp_path: Path,
) -> None:
    member = _member(b"---\nprivacy: public\n---\n\n# synthetic\n")

    result = FrontmatterClassifier(AnchoredBundleReader(tmp_path)).classify_member(
        member
    )

    assert isinstance(result, Classified)
    assert result.privacy is PrivacyClass.PUBLIC
    assert result.member_key == member.relative_path
    assert result.content_sha256 == member.content_sha256


def test_gate_admission_is_bound_to_member_path_and_bytes(tmp_path: Path) -> None:
    member = _member(b"---\nprivacy: public\n---\n\n# synthetic\n")
    gate = PrivacyGate(
        FrontmatterClassifier(AnchoredBundleReader(tmp_path)),
        frozenset({PrivacyClass.PUBLIC}),
    )

    admitted = gate.admit_member(member)

    assert isinstance(admitted, Admitted)
    assert admitted.member_key == member.relative_path
    assert admitted.content_sha256 == member.content_sha256


def test_path_classification_uses_the_anchored_reader(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_bytes(b"---\nprivacy: public\n---\n\n# synthetic\n")
    reader = AnchoredBundleReader(tmp_path)

    result = FrontmatterClassifier(reader).classify(note)

    assert isinstance(result, Classified)
    assert result.member_key == "note.md"
