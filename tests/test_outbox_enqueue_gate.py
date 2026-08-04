"""Red line R1: privacy routing completes before enqueue, structurally.

The Epic's named mutation for this file: defaulting the classifier to
``public`` must keep these tests red -- "has a value" is not "was
determined". Every rejection here also asserts zero durable residue, which
is what makes "admission before persistence" a tested property instead of
prose.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from ckp.outbox.errors import OutboxErrorCode
from ckp.outbox.models import OutboxRejectReceipt, OutboxState
from ckp.outbox.service import OutboxService
from ckp.writer.errors import WriterErrorCode
from outbox_fixtures import (
    assert_no_plaintext_at_rest,
    build_outbox_harness,
    enqueue,
    operation_commits,
    outbox_request,
)
from writer_fixtures import RejectingValidator


def _residue_free(harness) -> None:
    assert not list((harness.store_root / "records").iterdir())
    assert not list((harness.store_root / "idempotency").iterdir())
    assert not list((harness.store_root / "tmp").iterdir())
    assert_no_plaintext_at_rest(harness.store_root)


def test_an_undetermined_privacy_class_cannot_be_enqueued(
    tmp_path: Path,
) -> None:
    """No privacy declaration means refusal -- not a default, not later."""
    harness = build_outbox_harness(tmp_path)
    request = outbox_request()
    frontmatter = dict(request["frontmatter"])
    del frontmatter["privacy"]
    request["frontmatter"] = frontmatter
    receipt = enqueue(harness.service, request)
    assert isinstance(receipt, OutboxRejectReceipt)
    assert receipt.code == OutboxErrorCode.PRIVACY_UNCLASSIFIED.value
    _residue_free(harness)
    assert operation_commits(harness.repository) == ()


def test_admissible_but_undeclared_variants_all_refuse(tmp_path: Path) -> None:
    harness = build_outbox_harness(tmp_path)
    for privacy_value in ("", "Public", "unknown", "public\npublic"):
        request = outbox_request()
        frontmatter = dict(request["frontmatter"])
        frontmatter["privacy"] = privacy_value
        request["frontmatter"] = frontmatter
        receipt = enqueue(harness.service, request)
        assert isinstance(receipt, OutboxRejectReceipt), privacy_value
        assert receipt.code == OutboxErrorCode.PRIVACY_UNCLASSIFIED.value, privacy_value
    _residue_free(harness)


def test_every_admission_gate_rejects_before_any_persistence(
    tmp_path: Path,
) -> None:
    harness = build_outbox_harness(tmp_path)
    cases: list[tuple[dict[str, object], dict[str, object], str]] = [
        (
            {},
            {"actor_credential": "not-a-registered-credential"},
            OutboxErrorCode.ACTOR_DENIED.value,
        ),
        (
            {},
            {"model_id": None},
            OutboxErrorCode.MODEL_IDENTITY_MISSING.value,
        ),
        (
            outbox_request(
                frontmatter={
                    "type": "Source",
                    "title": "Spoof",
                    "privacy": "internal",
                    "generated": {"by": "spoof", "at": "2026-08-04T00:00:00Z"},
                }
            ),
            {},
            OutboxErrorCode.PROVENANCE_FORBIDDEN.value,
        ),
        (
            outbox_request(target_path="Core/formal-root-note.md"),
            {},
            OutboxErrorCode.TARGET_DENIED.value,
        ),
        (
            outbox_request(target_path="Core/_inbox/c7-synthetic/../escape.md"),
            {},
            OutboxErrorCode.REQUEST_INVALID.value,
        ),
        (
            outbox_request(privacy="sensitive"),
            {},
            OutboxErrorCode.CROSS_ZONE_DENIED.value,
        ),
        (
            outbox_request(body="# Synthetic\n\nSYNTHETIC_SECRET_FIXTURE_DO_NOT_USE\n"),
            {},
            OutboxErrorCode.SECRET_DETECTED.value,
        ),
    ]
    for request_overrides, invoke_overrides, expected in cases:
        request = request_overrides or outbox_request()
        receipt = enqueue(harness.service, request, **invoke_overrides)
        assert isinstance(receipt, OutboxRejectReceipt), expected
        assert receipt.code == expected
    _residue_free(harness)
    assert operation_commits(harness.repository) == ()


def test_worktree_validators_run_in_a_sandbox_before_persistence(
    tmp_path: Path,
) -> None:
    for stage, code in (
        ("profile", OutboxErrorCode.PROFILE_VALIDATION_FAILED),
        ("lint", OutboxErrorCode.LINT_FAILED),
        ("privacy", OutboxErrorCode.PRIVACY_SCAN_FAILED),
    ):
        writer_code = WriterErrorCode(code.value.replace("outbox/", "writer/"))
        stage_path = tmp_path / stage
        stage_path.mkdir()
        harness = build_outbox_harness(
            stage_path,
            **{f"{stage}": RejectingValidator(writer_code)},
        )
        receipt = enqueue(harness.service)
        assert isinstance(receipt, OutboxRejectReceipt)
        assert receipt.code == code.value
        _residue_free(harness)


def test_successful_enqueue_ran_every_validator_and_persisted_once(
    tmp_path: Path,
) -> None:
    harness = build_outbox_harness(tmp_path)
    receipt = enqueue(harness.service)
    assert not isinstance(receipt, OutboxRejectReceipt)
    assert receipt.state is OutboxState.QUEUED
    assert harness.profile.calls and harness.lint.calls and harness.privacy.calls
    # The sandbox is gone; the only durable artifacts are the record and its
    # derived index, and neither holds plaintext.
    assert len(list((harness.store_root / "records").iterdir())) == 1
    assert len(list((harness.store_root / "idempotency").iterdir())) == 1
    assert not list((harness.store_root / "tmp").iterdir())
    assert not list((harness.store_root / "quarantine").iterdir())
    assert_no_plaintext_at_rest(harness.store_root)
    # Enqueue is not a commit: the synthetic repository is untouched.
    assert operation_commits(harness.repository) == ()


def test_gate_and_classifier_are_required_constructor_parameters() -> None:
    """R1's wiring rule, pinned on the service and composition signatures."""
    from ckp.outbox.composition import build_c8_synthetic_outbox

    for callable_ in (OutboxService.__init__, build_c8_synthetic_outbox):
        parameters = inspect.signature(callable_).parameters
        assert "privacy_gate" in parameters
        assert all(
            parameter.default is inspect.Parameter.empty
            for name, parameter in parameters.items()
            if name != "self"
        )
