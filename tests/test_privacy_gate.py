"""PrivacyGate: admission, refusal receipts, and the student-private ceiling."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from ckp.privacy import (
    Admitted,
    FrontmatterClassifier,
    PrivacyClass,
    PrivacyGate,
    PrivacyPolicyError,
    Refused,
)
from ckp.privacy.classifier import Classified
from ckp.privacy.gate import REASON_NOT_ADMISSIBLE, REASON_STUDENT_PRIVATE_SINK
from privacy_fixtures import (
    PAYLOAD_SENTINEL,
    classed_note,
    note_with_duplicate_privacy,
    note_with_invalid_privacy,
    note_without_privacy,
)

CORE_ROUTE = frozenset({PrivacyClass.PUBLIC, PrivacyClass.INTERNAL})


def core_gate(bundle_root: Path) -> PrivacyGate:
    """The Decision §6 Core-route acceptance set: {public, internal}."""
    return PrivacyGate(FrontmatterClassifier(bundle_root), CORE_ROUTE)


def test_admits_a_class_inside_the_admissible_set(tmp_path: Path) -> None:
    note = classed_note(tmp_path, "internal")
    verdict = core_gate(tmp_path).admit(note)
    assert isinstance(verdict, Admitted)
    assert verdict.privacy is PrivacyClass.INTERNAL
    assert verdict.path == note


def test_refuses_a_determined_class_outside_the_admissible_set(
    tmp_path: Path,
) -> None:
    note = classed_note(tmp_path, "sensitive")
    verdict = core_gate(tmp_path).admit(note)
    assert isinstance(verdict, Refused)
    assert verdict.reason == REASON_NOT_ADMISSIBLE


@pytest.mark.parametrize(
    "factory",
    [note_without_privacy, note_with_duplicate_privacy, note_with_invalid_privacy],
)
def test_an_undetermined_item_cannot_pass(tmp_path: Path, factory) -> None:
    """Red line R1, end to end.

    Also the named mutation target: mutate the classifier to default to
    ``public`` when no declaration is found, and every one of these goes
    green-to-admitted -- so this test goes red. Having *a* value is not the
    same as having *determined* one.
    """
    verdict = core_gate(tmp_path).admit(factory(tmp_path))
    assert isinstance(verdict, Refused)


def test_student_private_is_refused_with_its_dedicated_code(
    tmp_path: Path,
) -> None:
    """Verdict-layer half of the double enforcement.

    Pinning the *specific* code matters: were the verdict-layer branch
    removed, a student-private note would still be refused as merely
    "not admissible", and this assertion is what turns that mutation red.
    """
    note = classed_note(tmp_path, "student-private")
    verdict = core_gate(tmp_path).admit(note)
    assert isinstance(verdict, Refused)
    assert verdict.reason == REASON_STUDENT_PRIVATE_SINK


def test_a_gate_admitting_student_private_cannot_be_built(tmp_path: Path) -> None:
    """Construction-layer half: widening a refused combination fails closed."""
    with pytest.raises(PrivacyPolicyError):
        PrivacyGate(
            FrontmatterClassifier(tmp_path),
            frozenset({PrivacyClass.PUBLIC, PrivacyClass.STUDENT_PRIVATE}),
        )


def test_admitted_cannot_be_minted_outside_the_gate(tmp_path: Path) -> None:
    """The seal: the only way to hold an Admitted is to have run the gate."""
    with pytest.raises(TypeError):
        Admitted(path=tmp_path / "x.md", privacy=PrivacyClass.PUBLIC)  # type: ignore[call-arg]
    with pytest.raises(PrivacyPolicyError):
        Admitted(
            path=tmp_path / "x.md",
            privacy=PrivacyClass.PUBLIC,
            _seal=object(),  # type: ignore[arg-type]
        )


def test_admission_cannot_be_rewritten_into_a_different_one(
    tmp_path: Path,
) -> None:
    """A real seal must not survive a field rewrite.

    ``dataclasses.replace`` re-runs construction with the *original* seal, so
    an unbound seal would let an admitted public note be rewritten into an
    "admitted" student-private one. The seal binds path and privacy; both
    rewrites must refuse.
    """
    note = classed_note(tmp_path, "public")
    admitted = core_gate(tmp_path).admit(note)
    assert isinstance(admitted, Admitted)
    with pytest.raises(PrivacyPolicyError):
        dataclasses.replace(admitted, privacy=PrivacyClass.STUDENT_PRIVATE)
    with pytest.raises(PrivacyPolicyError):
        dataclasses.replace(admitted, path=tmp_path / "other.md")


def test_refusal_receipts_never_echo_payload(tmp_path: Path) -> None:
    note = note_with_invalid_privacy(tmp_path)
    verdict = core_gate(tmp_path).admit(note)
    assert isinstance(verdict, Refused)
    assert PAYLOAD_SENTINEL not in verdict.reason
    assert PAYLOAD_SENTINEL not in repr(verdict)


def test_partition_accounts_for_every_path(tmp_path: Path) -> None:
    """Privacy before aggregation (contract §2.4): nothing silently dropped."""
    paths = [
        classed_note(tmp_path, "public"),
        classed_note(tmp_path, "internal"),
        classed_note(tmp_path, "sensitive"),
        classed_note(tmp_path, "student-private"),
        note_without_privacy(tmp_path),
    ]
    admitted, refused = core_gate(tmp_path).partition(paths)
    assert {a.path for a in admitted} == set(paths[:2])
    assert {r.path for r in refused} == set(paths[2:])
    assert len(admitted) + len(refused) == len(paths)


def test_gate_consults_the_classifier_it_was_built_with(tmp_path: Path) -> None:
    """The classifier parameter is load-bearing, not decorative."""

    class Stub:
        def __init__(self) -> None:
            self.calls: list[Path] = []

        def classify(self, path: Path) -> Classified:
            self.calls.append(path)
            return Classified(privacy=PrivacyClass.PUBLIC, source="frontmatter")

    stub = Stub()
    note = tmp_path / "anything.md"
    verdict = PrivacyGate(stub, CORE_ROUTE).admit(note)
    assert stub.calls == [note]
    assert isinstance(verdict, Admitted)
