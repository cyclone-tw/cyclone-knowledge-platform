"""The outbox/v1 namespace and the C7 admission mapping are frozen."""

from __future__ import annotations

from ckp.outbox.errors import (
    ERROR_PREFIX,
    WRITER_ADMISSION_CODES,
    OutboxErrorCode,
    writer_code_to_outbox_code,
)
from ckp.writer.errors import WriterErrorCode

FROZEN_CODES = {
    "outbox/v1/request-invalid",
    "outbox/v1/identity-missing",
    "outbox/v1/actor-denied",
    "outbox/v1/model-identity-missing",
    "outbox/v1/provenance-forbidden",
    "outbox/v1/privacy-unclassified",
    "outbox/v1/privacy-denied",
    "outbox/v1/student-private-denied",
    "outbox/v1/secret-detected",
    "outbox/v1/target-denied",
    "outbox/v1/cross-zone-denied",
    "outbox/v1/symlink-denied",
    "outbox/v1/profile-validation-failed",
    "outbox/v1/lint-failed",
    "outbox/v1/privacy-scan-failed",
    "outbox/v1/binding-conflict",
    "outbox/v1/storage-failed",
    "outbox/v1/lease-unavailable",
    "outbox/v1/record-expired",
    "outbox/v1/record-corrupt",
    "outbox/v1/key-denied",
    "outbox/v1/version-unknown",
    "outbox/v1/replay-conflict",
    "outbox/v1/replay-quarantined",
    "outbox/v1/internal-error",
}


def test_outbox_v1_namespace_and_codes_are_frozen() -> None:
    assert ERROR_PREFIX == "outbox/v1"
    assert {code.value for code in OutboxErrorCode} == FROZEN_CODES
    assert all(code.value.startswith("outbox/v1/") for code in OutboxErrorCode)


def test_admission_mapping_is_exact_and_never_widens() -> None:
    """Only enqueue-reachable writer refusals map; the suffixes must agree."""
    expected_admission = {
        WriterErrorCode.REQUEST_INVALID,
        WriterErrorCode.IDENTITY_MISSING,
        WriterErrorCode.ACTOR_DENIED,
        WriterErrorCode.MODEL_IDENTITY_MISSING,
        WriterErrorCode.PROVENANCE_FORBIDDEN,
        WriterErrorCode.PRIVACY_UNCLASSIFIED,
        WriterErrorCode.PRIVACY_DENIED,
        WriterErrorCode.STUDENT_PRIVATE_DENIED,
        WriterErrorCode.SECRET_DETECTED,
        WriterErrorCode.TARGET_DENIED,
        WriterErrorCode.CROSS_ZONE_DENIED,
        WriterErrorCode.SYMLINK_DENIED,
        WriterErrorCode.PROFILE_VALIDATION_FAILED,
        WriterErrorCode.LINT_FAILED,
        WriterErrorCode.PRIVACY_SCAN_FAILED,
    }
    assert set(WRITER_ADMISSION_CODES) == expected_admission
    for writer_code, outbox_code in WRITER_ADMISSION_CODES.items():
        assert writer_code.value.removeprefix(
            "writer/v1/"
        ) == outbox_code.value.removeprefix("outbox/v1/")


def test_transaction_phase_and_unknown_codes_fail_closed() -> None:
    """Base movement belongs to the replay policy, never to admission."""
    for transaction_code in (
        WriterErrorCode.BASE_MOVED,
        WriterErrorCode.PATH_CONFLICT,
        WriterErrorCode.IDEMPOTENCY_CONFLICT,
        WriterErrorCode.LEADER_UNAVAILABLE,
        WriterErrorCode.REPOSITORY_DENIED,
        WriterErrorCode.DIFF_DENIED,
        WriterErrorCode.COMMIT_FAILED,
        WriterErrorCode.INTERNAL_ERROR,
    ):
        assert (
            writer_code_to_outbox_code(transaction_code)
            is OutboxErrorCode.INTERNAL_ERROR
        )
