"""Stable, payload-free outbox v1 refusal codes and the C7 admission mapping."""

from __future__ import annotations

from enum import StrEnum

from ckp.writer.errors import WriterErrorCode

ERROR_PREFIX = "outbox/v1"


class OutboxErrorCode(StrEnum):
    # Admission mirrors: the C7 gates run before enqueue persistence, and
    # their refusals surface under this namespace without renaming writer/v1.
    REQUEST_INVALID = "outbox/v1/request-invalid"
    IDENTITY_MISSING = "outbox/v1/identity-missing"
    ACTOR_DENIED = "outbox/v1/actor-denied"
    MODEL_IDENTITY_MISSING = "outbox/v1/model-identity-missing"
    PROVENANCE_FORBIDDEN = "outbox/v1/provenance-forbidden"
    PRIVACY_UNCLASSIFIED = "outbox/v1/privacy-unclassified"
    PRIVACY_DENIED = "outbox/v1/privacy-denied"
    STUDENT_PRIVATE_DENIED = "outbox/v1/student-private-denied"
    SECRET_DETECTED = "outbox/v1/secret-detected"
    TARGET_DENIED = "outbox/v1/target-denied"
    CROSS_ZONE_DENIED = "outbox/v1/cross-zone-denied"
    SYMLINK_DENIED = "outbox/v1/symlink-denied"
    PROFILE_VALIDATION_FAILED = "outbox/v1/profile-validation-failed"
    LINT_FAILED = "outbox/v1/lint-failed"
    PRIVACY_SCAN_FAILED = "outbox/v1/privacy-scan-failed"
    # Outbox-owned codes.
    BINDING_CONFLICT = "outbox/v1/binding-conflict"
    STORAGE_FAILED = "outbox/v1/storage-failed"
    LEASE_UNAVAILABLE = "outbox/v1/lease-unavailable"
    RECORD_EXPIRED = "outbox/v1/record-expired"
    RECORD_CORRUPT = "outbox/v1/record-corrupt"
    KEY_DENIED = "outbox/v1/key-denied"
    VERSION_UNKNOWN = "outbox/v1/version-unknown"
    REPLAY_CONFLICT = "outbox/v1/replay-conflict"
    REPLAY_QUARANTINED = "outbox/v1/replay-quarantined"
    INTERNAL_ERROR = "outbox/v1/internal-error"


#: Writer refusals that the enqueue admission path can legitimately produce.
#: Transaction-phase writer codes (base-moved, path-conflict, ...) belong to
#: replay dispositions, not to this table; anything unlisted fails closed.
WRITER_ADMISSION_CODES: dict[WriterErrorCode, OutboxErrorCode] = {
    WriterErrorCode.REQUEST_INVALID: OutboxErrorCode.REQUEST_INVALID,
    WriterErrorCode.IDENTITY_MISSING: OutboxErrorCode.IDENTITY_MISSING,
    WriterErrorCode.ACTOR_DENIED: OutboxErrorCode.ACTOR_DENIED,
    WriterErrorCode.MODEL_IDENTITY_MISSING: OutboxErrorCode.MODEL_IDENTITY_MISSING,
    WriterErrorCode.PROVENANCE_FORBIDDEN: OutboxErrorCode.PROVENANCE_FORBIDDEN,
    WriterErrorCode.PRIVACY_UNCLASSIFIED: OutboxErrorCode.PRIVACY_UNCLASSIFIED,
    WriterErrorCode.PRIVACY_DENIED: OutboxErrorCode.PRIVACY_DENIED,
    WriterErrorCode.STUDENT_PRIVATE_DENIED: OutboxErrorCode.STUDENT_PRIVATE_DENIED,
    WriterErrorCode.SECRET_DETECTED: OutboxErrorCode.SECRET_DETECTED,
    WriterErrorCode.TARGET_DENIED: OutboxErrorCode.TARGET_DENIED,
    WriterErrorCode.CROSS_ZONE_DENIED: OutboxErrorCode.CROSS_ZONE_DENIED,
    WriterErrorCode.SYMLINK_DENIED: OutboxErrorCode.SYMLINK_DENIED,
    WriterErrorCode.PROFILE_VALIDATION_FAILED: (
        OutboxErrorCode.PROFILE_VALIDATION_FAILED
    ),
    WriterErrorCode.LINT_FAILED: OutboxErrorCode.LINT_FAILED,
    WriterErrorCode.PRIVACY_SCAN_FAILED: OutboxErrorCode.PRIVACY_SCAN_FAILED,
}


def writer_code_to_outbox_code(code: WriterErrorCode) -> OutboxErrorCode:
    """Map known admission refusals; unknown or transaction codes fail closed."""
    return WRITER_ADMISSION_CODES.get(code, OutboxErrorCode.INTERNAL_ERROR)


class OutboxRefusal(ValueError):
    """A transport-safe outbox refusal carrying a code, never payload details."""

    def __init__(self, code: OutboxErrorCode) -> None:
        self.code = code
        super().__init__(code)


__all__ = [
    "ERROR_PREFIX",
    "WRITER_ADMISSION_CODES",
    "OutboxErrorCode",
    "OutboxRefusal",
    "writer_code_to_outbox_code",
]
