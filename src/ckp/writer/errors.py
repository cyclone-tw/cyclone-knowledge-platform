"""Stable, payload-free Writer v1 refusal codes and C2 mapping."""

from __future__ import annotations

from enum import StrEnum

ERROR_PREFIX = "writer/v1"


class WriterErrorCode(StrEnum):
    REQUEST_INVALID = "writer/v1/request-invalid"
    IDENTITY_MISSING = "writer/v1/identity-missing"
    ACTOR_DENIED = "writer/v1/actor-denied"
    MODEL_IDENTITY_MISSING = "writer/v1/model-identity-missing"
    PROVENANCE_FORBIDDEN = "writer/v1/provenance-forbidden"
    REPOSITORY_DENIED = "writer/v1/repository-denied"
    PRIVACY_UNCLASSIFIED = "writer/v1/privacy-unclassified"
    PRIVACY_DENIED = "writer/v1/privacy-denied"
    STUDENT_PRIVATE_DENIED = "writer/v1/student-private-denied"
    SECRET_DETECTED = "writer/v1/secret-detected"
    TARGET_DENIED = "writer/v1/target-denied"
    CROSS_ZONE_DENIED = "writer/v1/cross-zone-denied"
    SYMLINK_DENIED = "writer/v1/symlink-denied"
    PATH_CONFLICT = "writer/v1/path-conflict"
    IDEMPOTENCY_CONFLICT = "writer/v1/idempotency-conflict"
    LEADER_UNAVAILABLE = "writer/v1/leader-unavailable"
    BASE_MOVED = "writer/v1/base-moved"
    PROFILE_VALIDATION_FAILED = "writer/v1/profile-validation-failed"
    LINT_FAILED = "writer/v1/lint-failed"
    PRIVACY_SCAN_FAILED = "writer/v1/privacy-scan-failed"
    DIFF_DENIED = "writer/v1/diff-denied"
    COMMIT_FAILED = "writer/v1/commit-failed"
    INTERNAL_ERROR = "writer/v1/internal-error"

    @property
    def receipt_code(self) -> str:
        """Compatibility spelling for code-to-receipt call sites."""
        return self.value


C2_REASON_CODES: dict[str, WriterErrorCode] = {
    "not-a-bundle-member": WriterErrorCode.REQUEST_INVALID,
    "note-unreadable": WriterErrorCode.REQUEST_INVALID,
    "frontmatter-missing": WriterErrorCode.PRIVACY_UNCLASSIFIED,
    "frontmatter-unterminated": WriterErrorCode.PRIVACY_UNCLASSIFIED,
    "privacy-missing": WriterErrorCode.PRIVACY_UNCLASSIFIED,
    "privacy-duplicate": WriterErrorCode.PRIVACY_UNCLASSIFIED,
    "privacy-invalid": WriterErrorCode.PRIVACY_UNCLASSIFIED,
    "student-private-sink": WriterErrorCode.STUDENT_PRIVATE_DENIED,
    "privacy-not-admissible": WriterErrorCode.PRIVACY_DENIED,
}


def c2_reason_to_writer_code(reason: str) -> WriterErrorCode:
    """Map known C2 refusal semantics; unknown input fails closed."""
    return C2_REASON_CODES.get(reason, WriterErrorCode.INTERNAL_ERROR)


def map_c2_reason(reason: str) -> WriterErrorCode:
    """Compatibility alias; prefer :func:`c2_reason_to_writer_code`."""
    return c2_reason_to_writer_code(reason)


class WriterRefusal(ValueError):
    """A transport-safe Writer refusal carrying a code, never payload details."""

    def __init__(self, code: WriterErrorCode) -> None:
        self.code = code
        super().__init__(code)


__all__ = [
    "C2_REASON_CODES",
    "ERROR_PREFIX",
    "WriterErrorCode",
    "WriterRefusal",
    "c2_reason_to_writer_code",
    "map_c2_reason",
]
