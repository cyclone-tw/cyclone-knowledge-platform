"""The R1 contract, pinned structurally.

These tests are about the *shape* of the privacy package: what must be
required, what must not exist, and what later children are forced to wire.
They are the tests the Epic's named mutations are aimed at.
"""

from __future__ import annotations

import ast
import inspect
import types
from pathlib import Path

import ckp.privacy as privacy_pkg
from ckp.privacy import FrontmatterClassifier, PrivacyClass, PrivacyGate, Unclassified
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
    # Same rule for the classifier itself: the bundle root it contains reads
    # to is a decision, and decisions do not get defaults.
    classifier_params = inspect.signature(FrontmatterClassifier.__init__).parameters
    assert classifier_params["bundle_root"].default is inspect.Parameter.empty


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


def _imports_privacy_package(package_root: Path) -> bool:
    """Does any module under ``package_root`` actually import ``ckp.privacy``?

    Checked on the AST, not by string search: a comment or docstring that
    merely *mentions* the package must not satisfy the tripwire. An import
    statement is the weakest thing that cannot be faked in prose -- and an
    import that is never used fails CI separately (ruff F401), so
    imported-and-unused cannot slip through either.
    """
    for source in package_root.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(
                    alias.name == "ckp.privacy" or alias.name.startswith("ckp.privacy.")
                    for alias in node.names
                ):
                    return True
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "ckp.privacy" or module.startswith("ckp.privacy."):
                    return True
                if module == "ckp" and any(
                    alias.name == "privacy" for alias in node.names
                ):
                    return True
    return False


def test_future_read_and_enqueue_paths_import_the_privacy_package() -> None:
    """Tripwire for C3/C8: a read or enqueue package that never imports the
    privacy package cannot be taking it as a required constructor parameter.

    An import cannot prove *correct* wiring -- review still owns that -- but
    its absence proves absent wiring, and that is what must be impossible to
    land silently.
    """
    for package in FUTURE_GATED_PACKAGES:
        root = REPO_ROOT / "src" / "ckp" / package
        if not root.exists():
            continue
        assert list(root.rglob("*.py")), (
            f"src/ckp/{package} exists but holds no Python source"
        )
        assert _imports_privacy_package(root), (
            f"src/ckp/{package} exists but never imports ckp.privacy; "
            "red line R1 requires the classifier as a required constructor "
            "parameter on every read and enqueue path (Epic #1)"
        )


def test_the_tripwire_cannot_be_satisfied_by_a_comment(tmp_path: Path) -> None:
    """The tripwire is a guard, so it gets the planted-package treatment.

    A package whose only reference to ckp.privacy lives in a comment and a
    docstring must not pass; a real import must.
    """
    faked = tmp_path / "faked"
    faked.mkdir()
    (faked / "__init__.py").write_text(
        '"""Mentions ckp.privacy and PrivacyGate in prose only."""\n'
        "# TODO: wire up ckp.privacy some day\n",
        encoding="utf-8",
    )
    assert not _imports_privacy_package(faked)

    wired = tmp_path / "wired"
    wired.mkdir()
    (wired / "__init__.py").write_text(
        'from ckp.privacy import PrivacyGate\n\n__all__ = ["PrivacyGate"]\n',
        encoding="utf-8",
    )
    assert _imports_privacy_package(wired)
