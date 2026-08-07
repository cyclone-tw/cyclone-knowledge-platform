"""Stable, payload-free ``embedding-semantic/v1`` refusal codes.

Same shape as ``ckp.embedding.errors`` on purpose: a refusal carries a code
and nothing else. Model asset paths and directory listings never belong in
an exception message either -- on a shared host that is filesystem layout,
not note content, but it is still not this module's business to echo.
"""

from __future__ import annotations

from enum import StrEnum

SEMANTIC_ERROR_PREFIX = "embedding-semantic/v1"


class SemanticErrorCode(StrEnum):
    # Asset resolution (issue #25: pinned local model cache, no runtime
    # network path).
    ASSET_DIR_INVALID = "embedding-semantic/v1/asset-dir-invalid"
    ASSETS_MISSING = "embedding-semantic/v1/assets-missing"
    ASSET_DIGEST_MISMATCH = "embedding-semantic/v1/asset-digest-mismatch"
    # Raised instead of letting an ImportError escape when the ``semantic``
    # extra was never installed.
    INFERENCE_BACKEND_UNAVAILABLE = (
        "embedding-semantic/v1/inference-backend-unavailable"
    )


class SemanticRefusal(ValueError):
    """A transport-safe refusal carrying a code, never a path or a value."""

    def __init__(self, code: SemanticErrorCode) -> None:
        self.code = code
        super().__init__(code)


__all__ = [
    "SEMANTIC_ERROR_PREFIX",
    "SemanticErrorCode",
    "SemanticRefusal",
]
