"""C5 provider-neutral vector index: rebuild, snapshot, search.

The package exports interfaces, shapes, the rebuild planner, and the two
local providers. The default application composes none of this --
``tests/test_c5_contract.py`` pins that, along with the loopback-only fence
and the absence of any cloud or credential path.

Phase 6 owns any cutover; here the index exists to be rebuilt, snapshotted,
and shadow-benchmarked against the lexical baseline.
"""

from ckp.index.errors import INDEX_ERROR_PREFIX, IndexErrorCode, IndexRefusal
from ckp.index.memory import (
    MEMORY_INDEX_ID,
    MEMORY_INDEX_VERSION,
    InMemoryVectorIndex,
)
from ckp.index.models import (
    INDEX_CONTRACT,
    SNAPSHOT_SCHEMA,
    IndexDescriptor,
    IndexedPoint,
    RebuildPlan,
    RebuildReport,
    SearchHit,
    SearchResult,
    compute_payload_digest,
    compute_point_id,
    plan_rebuild,
    read_plan_snapshot,
    require_public_filter,
    require_query_vector,
    require_top_k,
    write_plan_snapshot,
)
from ckp.index.provider import VectorIndexProvider, require_index_provider
from ckp.index.qdrant import (
    QDRANT_INDEX_ID,
    QDRANT_INDEX_VERSION,
    QdrantVectorIndex,
)
from ckp.index.revision import (
    INDEX_SCHEMA_VERSION,
    REVISION_DOMAIN,
    compute_composed_index_revision,
)

__all__ = [
    "INDEX_CONTRACT",
    "INDEX_ERROR_PREFIX",
    "INDEX_SCHEMA_VERSION",
    "MEMORY_INDEX_ID",
    "MEMORY_INDEX_VERSION",
    "QDRANT_INDEX_ID",
    "QDRANT_INDEX_VERSION",
    "REVISION_DOMAIN",
    "SNAPSHOT_SCHEMA",
    "InMemoryVectorIndex",
    "IndexDescriptor",
    "IndexErrorCode",
    "IndexRefusal",
    "IndexedPoint",
    "QdrantVectorIndex",
    "RebuildPlan",
    "RebuildReport",
    "SearchHit",
    "SearchResult",
    "VectorIndexProvider",
    "compute_composed_index_revision",
    "compute_payload_digest",
    "compute_point_id",
    "plan_rebuild",
    "read_plan_snapshot",
    "require_index_provider",
    "require_public_filter",
    "require_query_vector",
    "require_top_k",
    "write_plan_snapshot",
]
