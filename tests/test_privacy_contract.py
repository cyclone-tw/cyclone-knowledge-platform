"""The R1 contract, pinned structurally.

These tests are about the *shape* of the privacy package: what must be
required, what must not exist, and what later children are forced to wire.
They are the tests the Epic's named mutations are aimed at.
"""

from __future__ import annotations

import inspect
import types

import ckp.privacy as privacy_pkg
from ckp.privacy import PrivacyClass, PrivacyGate, Unclassified
from conftest import REPO_ROOT

#: The four literal tokens of decision-cyclone-privacy-sink-v1 §1, written
#: out independently here so the enum and this test are two separate
#: transcriptions of the Decision -- they can only agree by both being right.
DECISION_TOKENS = {"public", "internal", "sensitive", "student-private"}

#: Packages that will host read or enqueue paths (Epic #1 file ownership).
#: The moment one appears without referencing the privacy package, this test
#: turns red. Vacuously green today; armed the day C3 or C8 starts.
FUTURE_GATED_PACKAGES = ("catalog", "gateway", "outbox", "writer", "packer")


def test_the_enum_is_a_faithful_transcription_of_the_decision() -> None:
    assert {c.value for c in PrivacyClass} == DECISION_TOKENS


def test_strictness_ordering_matches_decision_s3() -> None:
    """Privacy may be maintained or raised, never lowered."""
    ordered = sorted(PrivacyClass, key=lambda c: c.strictness)
    assert ordered == [
        PrivacyClass.PUBLIC,
        PrivacyClass.INTERNAL,
        PrivacyClass.SENSITIVE,
        PrivacyClass.STUDENT_PRIVATE,
    ]
    assert PrivacyClass.SENSITIVE.is_at_least(PrivacyClass.INTERNAL)
    assert not PrivacyClass.INTERNAL.is_at_least(PrivacyClass.SENSITIVE)


def test_gate_parameters_are_required_not_optional() -> None:
    """R1's wiring rule: a required constructor parameter, never a flag.

    Mutation target: give either parameter a default and this is red. A
    default classifier would make "nobody chose one" indistinguishable from
    "the right one was chosen".
    """
    parameters = inspect.signature(PrivacyGate.__init__).parameters
    assert parameters["classifier"].default is inspect.Parameter.empty
    assert parameters["admissible"].default is inspect.Parameter.empty


def test_undetermined_carries_no_privacy_class_at_all() -> None:
    """The type-layer half of R1.

    ``Unclassified`` must never grow a ``privacy`` attribute -- not even one
    holding ``None`` -- because the moment it has one, code downstream can
    read a class off an undetermined item and the type distinction stops
    meaning anything.
    """
    outcome = Unclassified(reason="privacy-missing")
    assert not hasattr(outcome, "privacy")
    assert "privacy" not in {
        f for f in getattr(Unclassified, "__dataclass_fields__", {})
    }


def test_no_prebuilt_gate_or_default_classifier_is_exported() -> None:
    """The package must not ship a ready-made admission decision.

    A module-level gate or classifier singleton is a default in disguise:
    callers would import the decision instead of constructing and owning it.
    Classes are fine; instances are not.
    """
    for name in dir(privacy_pkg):
        if name.startswith("_"):
            continue
        member = getattr(privacy_pkg, name)
        # Union aliases (Classification, Verdict) are types, not instances.
        if isinstance(member, types.UnionType):
            continue
        assert inspect.isclass(member) or inspect.ismodule(member), (
            f"ckp.privacy exports non-class {name!r}; "
            "admission decisions belong to call sites, not to the package"
        )


def test_future_read_and_enqueue_paths_reference_the_privacy_package() -> None:
    """Tripwire for C3/C8: a read or enqueue package that never mentions the
    privacy package cannot be taking it as a required constructor parameter.

    A grep is deliberately crude -- it cannot prove correct wiring, only make
    silently-absent wiring impossible. Review still owns the rest.
    """
    for package in FUTURE_GATED_PACKAGES:
        root = REPO_ROOT / "src" / "ckp" / package
        if not root.exists():
            continue
        sources = list(root.rglob("*.py"))
        assert sources, f"src/ckp/{package} exists but holds no Python source"
        combined = "\n".join(p.read_text(encoding="utf-8") for p in sources)
        assert "ckp.privacy" in combined or "PrivacyGate" in combined, (
            f"src/ckp/{package} exists but never references ckp.privacy; "
            "red line R1 requires the classifier as a required constructor "
            "parameter on every read and enqueue path (Epic #1)"
        )
