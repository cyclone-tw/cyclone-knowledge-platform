from __future__ import annotations

from ckp.writer.errors import (
    C2_REASON_CODES,
    ERROR_PREFIX,
    WriterErrorCode,
    c2_reason_to_writer_code,
)


def test_writer_v1_namespace_and_codes_are_frozen() -> None:
    assert ERROR_PREFIX == "writer/v1"
    codes = {code.value.removeprefix(f"{ERROR_PREFIX}/") for code in WriterErrorCode}
    assert codes == {
        "request-invalid",
        "identity-missing",
        "actor-denied",
        "model-identity-missing",
        "provenance-forbidden",
        "repository-denied",
        "privacy-unclassified",
        "privacy-denied",
        "student-private-denied",
        "secret-detected",
        "target-denied",
        "cross-zone-denied",
        "symlink-denied",
        "path-conflict",
        "idempotency-conflict",
        "leader-unavailable",
        "base-moved",
        "profile-validation-failed",
        "lint-failed",
        "privacy-scan-failed",
        "diff-denied",
        "commit-failed",
        "internal-error",
    }
    assert all(code.value.startswith(f"{ERROR_PREFIX}/") for code in WriterErrorCode)


def test_c2_reason_mapping_is_exact_and_unknown_reasons_fail_closed() -> None:
    assert C2_REASON_CODES == {
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
    assert (
        c2_reason_to_writer_code("not-a-bundle-member")
        is WriterErrorCode.REQUEST_INVALID
    )
    assert (
        c2_reason_to_writer_code("untrusted-detail") is WriterErrorCode.INTERNAL_ERROR
    )
