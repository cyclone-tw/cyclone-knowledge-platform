"""Stable, payload-free ``index/v1`` refusal codes.

Index inputs are note-derived vectors and note identifiers. As everywhere
else in this repo, a refusal carries a code and nothing else -- echoing a
query vector, a note body, or a host string back into an exception message
would leak payload into logs.
"""

from __future__ import annotations

from enum import StrEnum

INDEX_ERROR_PREFIX = "index/v1"


class IndexErrorCode(StrEnum):
    # Input shape.
    QUERY_INVALID = "index/v1/query-invalid"
    TOP_K_INVALID = "index/v1/top-k-invalid"
    DIMENSION_MISMATCH = "index/v1/dimension-mismatch"
    # Provider admission and composition.
    PROVIDER_INVALID = "index/v1/provider-invalid"
    CONTRACT_VERSION_UNKNOWN = "index/v1/contract-version-unknown"
    # Phase 3 offline boundary.
    REMOTE_DENIED = "index/v1/remote-denied"
    # Lifecycle.
    NOT_BUILT = "index/v1/not-built"
    PLAN_INVALID = "index/v1/plan-invalid"
    SNAPSHOT_INVALID = "index/v1/snapshot-invalid"
    UNAVAILABLE = "index/v1/unavailable"


class IndexRefusal(ValueError):
    """A transport-safe refusal carrying a code, never the offending value."""

    def __init__(self, code: IndexErrorCode) -> None:
        self.code = code
        super().__init__(code)


__all__ = [
    "INDEX_ERROR_PREFIX",
    "IndexErrorCode",
    "IndexRefusal",
]
