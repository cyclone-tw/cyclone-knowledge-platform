"""Red line R2: student-private never enters the generic outbox.

Encryption does not make it acceptable -- the Epic's named mutation is
enqueueing the same student-private payload again in encrypted form and the
tests must stay red. The only approved destination is a local-only
restricted store that this codebase deliberately does not implement.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from ckp.outbox.errors import OutboxErrorCode
from ckp.outbox.models import OutboxRejectReceipt
from ckp.privacy import FrontmatterClassifier, PrivacyClass, PrivacyGate
from ckp.privacy.gate import PrivacyPolicyError
from outbox_fixtures import (
    assert_no_plaintext_at_rest,
    build_outbox_harness,
    enqueue,
    operation_commits,
    outbox_request,
)

_STUDENT_BODY = (
    "# SYNTHETIC FIXTURE\n\n"
    "Invented student Zaphod Example-Student, entirely synthetic.\n"
)


def _assert_refused_without_residue(harness, receipt, code: str) -> None:
    assert isinstance(receipt, OutboxRejectReceipt)
    assert receipt.code == code
    assert not list((harness.store_root / "records").iterdir())
    assert not list((harness.store_root / "idempotency").iterdir())
    assert_no_plaintext_at_rest(harness.store_root)
    assert operation_commits(harness.repository) == ()


def test_raw_student_private_is_refused_with_its_dedicated_code(
    tmp_path: Path,
) -> None:
    harness = build_outbox_harness(tmp_path)
    receipt = enqueue(
        harness.service,
        outbox_request(privacy="student-private", body=_STUDENT_BODY),
    )
    _assert_refused_without_residue(
        harness, receipt, OutboxErrorCode.STUDENT_PRIVATE_DENIED.value
    )


def test_encrypting_the_payload_first_is_not_an_exemption(
    tmp_path: Path,
) -> None:
    """The most likely compliant-looking bypass, pinned shut.

    A caller that pre-encrypts the body but still declares the true class is
    refused on the class. A caller that hides the class along with the body
    is refused as undetermined. Neither route reaches the store.
    """
    harness = build_outbox_harness(tmp_path)
    disguised = base64.b64encode(_STUDENT_BODY.encode("utf-8")).decode("ascii")

    declared = enqueue(
        harness.service,
        outbox_request(
            privacy="student-private",
            body=f"encrypted-payload: {disguised}\n",
        ),
    )
    _assert_refused_without_residue(
        harness, declared, OutboxErrorCode.STUDENT_PRIVATE_DENIED.value
    )

    request = outbox_request(body=f"encrypted-payload: {disguised}\n")
    frontmatter = dict(request["frontmatter"])
    del frontmatter["privacy"]
    request["frontmatter"] = frontmatter
    hidden = enqueue(harness.service, request)
    _assert_refused_without_residue(
        harness, hidden, OutboxErrorCode.PRIVACY_UNCLASSIFIED.value
    )


def test_synthetic_student_identifier_in_payload_is_refused(
    tmp_path: Path,
) -> None:
    """The deterministic payload scanner is a second, independent net."""
    harness = build_outbox_harness(tmp_path)
    receipt = enqueue(
        harness.service,
        outbox_request(body="# Synthetic\n\nSYNTHETIC_STUDENT_IDENTIFIER_DO_NOT_USE\n"),
    )
    _assert_refused_without_residue(
        harness, receipt, OutboxErrorCode.STUDENT_PRIVATE_DENIED.value
    )


def test_the_outbox_gate_cannot_be_built_to_admit_student_private(
    tmp_path: Path,
) -> None:
    """Widening the admissible set is refused at construction (Decision §3)."""
    with pytest.raises(PrivacyPolicyError):
        PrivacyGate(
            FrontmatterClassifier(tmp_path),
            frozenset(
                {
                    PrivacyClass.PUBLIC,
                    PrivacyClass.INTERNAL,
                    PrivacyClass.SENSITIVE,
                    PrivacyClass.STUDENT_PRIVATE,
                }
            ),
        )


def test_cross_zone_and_secret_payloads_are_refused(tmp_path: Path) -> None:
    harness = build_outbox_harness(tmp_path)
    sensitive_to_core = enqueue(harness.service, outbox_request(privacy="sensitive"))
    assert isinstance(sensitive_to_core, OutboxRejectReceipt)
    assert sensitive_to_core.code == OutboxErrorCode.CROSS_ZONE_DENIED.value

    secret = enqueue(
        harness.service,
        outbox_request(body="# Synthetic\n\nSYNTHETIC_SECRET_FIXTURE_DO_NOT_USE\n"),
    )
    assert isinstance(secret, OutboxRejectReceipt)
    assert secret.code == OutboxErrorCode.SECRET_DETECTED.value
    assert not list((harness.store_root / "records").iterdir())
    assert operation_commits(harness.repository) == ()
