"""The Qdrant-backed index. Local service only, verified after every write.

Design notes that keep this honest (AGENTS.md §8):

* **The fence is structural.** The constructor accepts a host and refuses
  anything but loopback with a stable code. There is no URL parameter, no
  API-key parameter, and no TLS parameter -- a cloud endpoint is not a
  configuration away, it is a contract change (open an issue first).
* **Qdrant stores float32.** The canonical plan is float64, so post-write
  verification compares what Qdrant returns against the float32 projection
  of the plan -- bit-exact at the precision Qdrant can deliver -- while the
  reported ``payload_digest`` stays the canonical float64 one. Claiming
  float64 fidelity for a float32 store would be a fabricated capability.
* **The provider keeps the canonical plan.** Snapshots are written from the
  float64 plan (provider-neutral format shared with the in-memory index),
  never re-read from the quantized store.
* **Search re-sorts in process.** Qdrant orders by score alone; equal scores
  come back in arbitrary order. The frozen total order (score descending,
  ``relative_path`` ascending) is imposed here by fetching every point and
  sorting, which is exact at Phase 3 synthetic-corpus scale.

``qdrant-client`` is imported lazily so the package -- and the shipped
runtime image, which deliberately excludes the ``index`` extra -- imports
cleanly without it.
"""

from __future__ import annotations

import contextlib
import math
import struct
import uuid
from pathlib import Path

from ckp.index.errors import IndexErrorCode, IndexRefusal
from ckp.index.models import (
    INDEX_CONTRACT,
    IndexDescriptor,
    RebuildPlan,
    RebuildReport,
    SearchHit,
    SearchResult,
    read_plan_snapshot,
    require_public_filter,
    require_query_vector,
    require_top_k,
    write_plan_snapshot,
)
from ckp.index.revision import INDEX_SCHEMA_VERSION
from ckp.privacy import PrivacyClass

QDRANT_INDEX_ID = "qdrant-local"
QDRANT_INDEX_VERSION = "1"

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_COLLECTION_PATTERN_MAX = 64


def _as_float32(values: tuple[float, ...]) -> tuple[float, ...]:
    """Project a float64 vector onto what a float32 store can hold."""
    count = len(values)
    return struct.unpack(f">{count}f", struct.pack(f">{count}f", *values))


def _point_uuid(point_id: str) -> str:
    """Qdrant point ids must be UUIDs or ints; derive one from the sha256 id."""
    return str(uuid.UUID(point_id[:32]))


class QdrantVectorIndex:
    """Full-rebuild Qdrant collection with verified writes and restores."""

    def __init__(self, *, host: str, port: int, collection: str) -> None:
        if not isinstance(host, str) or host not in _LOOPBACK_HOSTS:
            raise IndexRefusal(IndexErrorCode.REMOTE_DENIED)
        port_ok = (
            isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535
        )
        if not port_ok:
            raise IndexRefusal(IndexErrorCode.REMOTE_DENIED)
        if (
            not isinstance(collection, str)
            or not collection
            or len(collection) > _COLLECTION_PATTERN_MAX
            or not collection.replace("-", "").replace("_", "").isalnum()
        ):
            raise IndexRefusal(IndexErrorCode.PROVIDER_INVALID)
        try:
            from qdrant_client import QdrantClient
        except ImportError as error:
            raise IndexRefusal(IndexErrorCode.UNAVAILABLE) from error
        self._descriptor = IndexDescriptor(
            contract_version=INDEX_CONTRACT,
            provider_id=QDRANT_INDEX_ID,
            provider_version=QDRANT_INDEX_VERSION,
            schema_version=INDEX_SCHEMA_VERSION,
        )
        self._collection = collection
        self._client = QdrantClient(host=host, port=port, https=False, timeout=30)
        self._plan: RebuildPlan | None = None

    @property
    def descriptor(self) -> IndexDescriptor:
        return self._descriptor

    def rebuild(self, plan: RebuildPlan) -> RebuildReport:
        if not isinstance(plan, RebuildPlan):
            raise IndexRefusal(IndexErrorCode.PLAN_INVALID)
        from qdrant_client import models as qmodels

        self.wipe()
        try:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(
                    size=plan.dimension,
                    distance=qmodels.Distance.DOT,
                ),
            )
            if plan.points:
                self._client.upsert(
                    collection_name=self._collection,
                    wait=True,
                    points=[
                        qmodels.PointStruct(
                            id=_point_uuid(point.point_id),
                            vector=list(point.vector),
                            payload={
                                "point_id": point.point_id,
                                "digest_key": point.digest_key,
                                "relative_path": point.relative_path,
                                "content_sha256": point.content_sha256,
                            },
                        )
                        for point in plan.points
                    ],
                )
        except IndexRefusal:
            raise
        except Exception as error:  # qdrant/transport errors: payload-free code
            raise IndexRefusal(IndexErrorCode.UNAVAILABLE) from error
        self._verify_stored(plan)
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
        if not plan.points:
            return SearchResult(composed_revision=plan.composed_revision, hits=())
        try:
            response = self._client.query_points(
                collection_name=self._collection,
                query=list(vector),
                limit=plan.indexed_count,
                with_payload=True,
            )
        except Exception as error:
            raise IndexRefusal(IndexErrorCode.UNAVAILABLE) from error
        scored = []
        for hit in response.points:
            payload = hit.payload or {}
            relative_path = payload.get("relative_path")
            content_sha256 = payload.get("content_sha256")
            identity_ok = isinstance(relative_path, str) and isinstance(
                content_sha256, str
            )
            if not identity_ok:
                raise IndexRefusal(IndexErrorCode.UNAVAILABLE)
            score = float(hit.score)
            if not math.isfinite(score):
                raise IndexRefusal(IndexErrorCode.UNAVAILABLE)
            scored.append((relative_path, content_sha256, score))
        scored.sort(key=lambda item: (-item[2], item[0]))
        return SearchResult(
            composed_revision=plan.composed_revision,
            hits=tuple(
                SearchHit(
                    relative_path=relative_path,
                    content_sha256=content_sha256,
                    score=score,
                    rank=rank,
                )
                for rank, (relative_path, content_sha256, score) in enumerate(
                    scored[:limit]
                )
            ),
        )

    def snapshot(self, target_dir: Path) -> Path:
        plan = self._require_built()
        return write_plan_snapshot(
            plan, provider_id=self._descriptor.provider_id, target_dir=target_dir
        )

    def restore(self, snapshot_path: Path) -> RebuildReport:
        plan = read_plan_snapshot(snapshot_path)
        return self.rebuild(plan)

    def wipe(self) -> None:
        # Absent collection is the desired post-state; anything else will
        # resurface on the next operation with a stable code.
        with contextlib.suppress(Exception):
            self._client.delete_collection(collection_name=self._collection)
        self._plan = None

    def _require_built(self) -> RebuildPlan:
        if self._plan is None:
            raise IndexRefusal(IndexErrorCode.NOT_BUILT)
        return self._plan

    def _verify_stored(self, plan: RebuildPlan) -> None:
        """Read every point back and compare against the plan, fail closed.

        Identity fields must match exactly; vectors must match the float32
        projection of the canonical plan bit-for-bit.
        """
        try:
            stored = self._client.retrieve(
                collection_name=self._collection,
                ids=[_point_uuid(point.point_id) for point in plan.points],
                with_payload=True,
                with_vectors=True,
            )
        except Exception as error:
            raise IndexRefusal(IndexErrorCode.UNAVAILABLE) from error
        by_point_id = {}
        for record in stored:
            payload = record.payload or {}
            by_point_id[payload.get("point_id")] = record
        if len(by_point_id) != len(plan.points):
            raise IndexRefusal(IndexErrorCode.PLAN_INVALID)
        for point in plan.points:
            record = by_point_id.get(point.point_id)
            if record is None:
                raise IndexRefusal(IndexErrorCode.PLAN_INVALID)
            payload = record.payload or {}
            if (
                payload.get("digest_key") != point.digest_key
                or payload.get("relative_path") != point.relative_path
                or payload.get("content_sha256") != point.content_sha256
            ):
                raise IndexRefusal(IndexErrorCode.PLAN_INVALID)
            vector = record.vector
            if vector is None:
                raise IndexRefusal(IndexErrorCode.PLAN_INVALID)
            returned = tuple(float(value) for value in vector)
            if _as_float32(returned) != _as_float32(point.vector):
                raise IndexRefusal(IndexErrorCode.PLAN_INVALID)

    def _report(self, plan: RebuildPlan) -> RebuildReport:
        return RebuildReport(
            provider_id=self._descriptor.provider_id,
            composed_revision=plan.composed_revision,
            point_count=plan.indexed_count,
            indexed_count=plan.indexed_count,
            excluded_count=plan.excluded_count,
            payload_digest=plan.payload_digest,
        )


__all__ = [
    "QDRANT_INDEX_ID",
    "QDRANT_INDEX_VERSION",
    "QdrantVectorIndex",
]
