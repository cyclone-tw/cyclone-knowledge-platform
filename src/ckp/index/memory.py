"""The in-memory reference index shipped with C5.

Pure Python, no service, no dependency: unit tests and the shadow benchmark
run against this without Docker, and the Qdrant implementation is checked
for behavioral parity against it. Determinism comes from ``math.fsum`` dot
products over already-normalized vectors and the frozen total order.
"""

from __future__ import annotations

import math
from pathlib import Path

from ckp.index.errors import IndexErrorCode, IndexRefusal
from ckp.index.models import (
    INDEX_CONTRACT,
    IndexDescriptor,
    RebuildPlan,
    RebuildReport,
    SearchHit,
    SearchResult,
    compute_payload_digest,
    read_plan_snapshot,
    require_public_filter,
    require_query_vector,
    require_top_k,
    write_plan_snapshot,
)
from ckp.index.revision import INDEX_SCHEMA_VERSION
from ckp.privacy import PrivacyClass

MEMORY_INDEX_ID = "memory-cosine"
MEMORY_INDEX_VERSION = "1"


class InMemoryVectorIndex:
    """Deterministic cosine index over plan points, held in process memory."""

    def __init__(self) -> None:
        self._descriptor = IndexDescriptor(
            contract_version=INDEX_CONTRACT,
            provider_id=MEMORY_INDEX_ID,
            provider_version=MEMORY_INDEX_VERSION,
            schema_version=INDEX_SCHEMA_VERSION,
        )
        self._plan: RebuildPlan | None = None

    @property
    def descriptor(self) -> IndexDescriptor:
        return self._descriptor

    def rebuild(self, plan: RebuildPlan) -> RebuildReport:
        if not isinstance(plan, RebuildPlan):
            raise IndexRefusal(IndexErrorCode.PLAN_INVALID)
        self.wipe()
        stored = plan.points
        # Verify what was stored, not what was promised: recompute the digest
        # from the retained points and refuse a rebuild that does not match.
        if compute_payload_digest(stored) != plan.payload_digest:
            raise IndexRefusal(IndexErrorCode.PLAN_INVALID)
        self._plan = plan
        return self._report(plan)

    def search(
        self,
        query_vector: tuple[float, ...],
        *,
        top_k: int,
        filter_privacy: frozenset[PrivacyClass],
    ) -> SearchResult:
        plan = self._require_built()
        require_public_filter(filter_privacy)
        limit = require_top_k(top_k)
        vector = require_query_vector(query_vector, dimension=plan.dimension)
        scored = [(point, _dot(vector, point.vector)) for point in plan.points]
        scored.sort(key=lambda item: (-item[1], item[0].relative_path))
        return SearchResult(
            composed_revision=plan.composed_revision,
            hits=tuple(
                SearchHit(
                    relative_path=point.relative_path,
                    content_sha256=point.content_sha256,
                    score=score,
                    rank=rank,
                )
                for rank, (point, score) in enumerate(scored[:limit])
            ),
        )

    def snapshot(self, target_dir: Path) -> Path:
        plan = self._require_built()
        return write_plan_snapshot(
            plan, provider_id=self._descriptor.provider_id, target_dir=target_dir
        )

    def restore(self, snapshot_path: Path) -> RebuildReport:
        plan = read_plan_snapshot(snapshot_path)
        self.wipe()
        self._plan = plan
        return self._report(plan)

    def wipe(self) -> None:
        self._plan = None

    def _require_built(self) -> RebuildPlan:
        if self._plan is None:
            raise IndexRefusal(IndexErrorCode.NOT_BUILT)
        return self._plan

    def _report(self, plan: RebuildPlan) -> RebuildReport:
        return RebuildReport(
            provider_id=self._descriptor.provider_id,
            composed_revision=plan.composed_revision,
            point_count=plan.indexed_count,
            indexed_count=plan.indexed_count,
            excluded_count=plan.excluded_count,
            payload_digest=plan.payload_digest,
        )


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


__all__ = [
    "MEMORY_INDEX_ID",
    "MEMORY_INDEX_VERSION",
    "InMemoryVectorIndex",
]
