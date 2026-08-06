"""The composed ``index/v1`` revision.

``compute_composed_index_revision`` folds the three inputs that fully
determine what a rebuilt vector index contains: the bundle content revision
(which notes, which bytes), the C4 embedding provider revision (which vector
for each byte sequence), and the index schema version (how points are laid
out). Leave any one out and two indexes with different contents could claim
the same revision -- exactly what contract §5.5 forbids.
"""

from __future__ import annotations

import hashlib

#: Bumped only by a breaking change to point layout or payload fields.
INDEX_SCHEMA_VERSION = "1"

REVISION_DOMAIN = b"ckp-index-composed-v1"

_SHA256_HEX = 64


def _require_revision_string(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != len("sha256:") + _SHA256_HEX
    ):
        raise ValueError("revision inputs must be sha256:<64 hex> strings")
    return value


def compute_composed_index_revision(
    *,
    bundle_index_revision: str,
    embedding_revision: str,
    index_schema_version: str,
) -> str:
    """Length-framed digest over every input that shapes the index."""
    if not isinstance(index_schema_version, str) or not index_schema_version:
        raise ValueError("index_schema_version must be a non-empty string")
    digest = hashlib.sha256()
    for value in (
        REVISION_DOMAIN,
        _require_revision_string(bundle_index_revision).encode("ascii"),
        _require_revision_string(embedding_revision).encode("ascii"),
        index_schema_version.encode("utf-8"),
    ):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return f"sha256:{digest.hexdigest()}"


__all__ = [
    "INDEX_SCHEMA_VERSION",
    "REVISION_DOMAIN",
    "compute_composed_index_revision",
]
