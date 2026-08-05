"""The provider-neutral ``index/v1`` interface.

A consumer depends on this shape and on nothing else (contract §2: swapping
the index provider must not change the consumer contract). This module must
stay ignorant of every concrete provider; ``tests/test_c5_contract.py`` pins
that, as with the C4 embedding layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ckp.index.errors import IndexErrorCode, IndexRefusal
from ckp.index.models import (
    INDEX_CONTRACT,
    IndexDescriptor,
    RebuildPlan,
    RebuildReport,
    SearchResult,
)
from ckp.privacy import PrivacyClass


@runtime_checkable
class VectorIndexProvider(Protocol):
    """Rebuild from a plan, search, snapshot -- deterministically.

    Frozen invariants every implementation owes its caller:

    1. ``rebuild`` is full, never incremental: wipe first, then store exactly
       the plan's points, then verify what was stored against the plan's
       ``payload_digest``. A partial rebuild that keeps stale points is how
       an index stops being recomputable from a commit.
    2. ``search`` before any rebuild fails closed (``index/v1/not-built``);
       it never invents an empty success.
    3. Result order is score descending, then ``relative_path`` ascending --
       enforced by :class:`SearchResult`, honored by the provider.
    4. ``snapshot``/``restore`` round-trip the exact point set; ``restore``
       re-verifies the payload digest before reporting success.
    5. No remote endpoint, credential, or model download, ever.
    """

    @property
    def descriptor(self) -> IndexDescriptor: ...

    def rebuild(self, plan: RebuildPlan) -> RebuildReport: ...

    def search(
        self,
        query_vector: tuple[float, ...],
        *,
        top_k: int,
        filter_privacy: frozenset[PrivacyClass],
    ) -> SearchResult: ...

    def snapshot(self, target_dir: Path) -> Path: ...

    def restore(self, snapshot_path: Path) -> RebuildReport: ...

    def wipe(self) -> None: ...


def require_index_provider(provider: object) -> IndexDescriptor:
    """Admission for anything claiming to be a vector index provider.

    Checked once at composition time (and re-checked by consumers that hold
    a provider long-term), so a mistyped object fails at the boundary with a
    stable code instead of deep inside a rebuild.
    """
    if not isinstance(provider, VectorIndexProvider):
        raise IndexRefusal(IndexErrorCode.PROVIDER_INVALID)
    descriptor = provider.descriptor
    if not isinstance(descriptor, IndexDescriptor):
        raise IndexRefusal(IndexErrorCode.PROVIDER_INVALID)
    if descriptor.contract_version != INDEX_CONTRACT:
        raise IndexRefusal(IndexErrorCode.CONTRACT_VERSION_UNKNOWN)
    for name in ("rebuild", "search", "snapshot", "restore", "wipe"):
        if not callable(getattr(provider, name, None)):
            raise IndexRefusal(IndexErrorCode.PROVIDER_INVALID)
    return descriptor


__all__ = [
    "VectorIndexProvider",
    "require_index_provider",
]
