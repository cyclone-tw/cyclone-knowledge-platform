"""Stable, payload-free ``embedding/v1`` refusal codes.

A refusal carries a code and nothing else. Embedding inputs are note text,
so echoing the offending value back into an error message would turn every
validation failure into a content leak.
"""

from __future__ import annotations

from enum import StrEnum

ERROR_PREFIX = "embedding/v1"


class EmbeddingErrorCode(StrEnum):
    # Input shape.
    TEXT_INVALID = "embedding/v1/text-invalid"
    TEXT_EMPTY = "embedding/v1/text-empty"
    BATCH_INVALID = "embedding/v1/batch-invalid"
    TOP_K_INVALID = "embedding/v1/top-k-invalid"
    CANDIDATE_DUPLICATE = "embedding/v1/candidate-duplicate"
    # Descriptor and provider registration.
    DESCRIPTOR_INVALID = "embedding/v1/descriptor-invalid"
    CONTRACT_VERSION_UNKNOWN = "embedding/v1/contract-version-unknown"
    PROVIDER_UNKNOWN = "embedding/v1/provider-unknown"
    PROVIDER_DUPLICATE = "embedding/v1/provider-duplicate"
    PROVIDER_KIND_MISMATCH = "embedding/v1/provider-kind-mismatch"
    # Phase 3 offline boundary.
    NETWORK_PROVIDER_DENIED = "embedding/v1/network-provider-denied"
    NONDETERMINISTIC_PROVIDER_DENIED = "embedding/v1/nondeterministic-provider-denied"


class EmbeddingRefusal(ValueError):
    """A transport-safe refusal carrying a code, never the offending text."""

    def __init__(self, code: EmbeddingErrorCode) -> None:
        self.code = code
        super().__init__(code)


__all__ = [
    "ERROR_PREFIX",
    "EmbeddingErrorCode",
    "EmbeddingRefusal",
]
