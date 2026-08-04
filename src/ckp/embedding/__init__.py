"""C4 provider-neutral embedding and reranker interfaces.

The package exports interfaces, schemas, and builders only -- no registry
instance, no default provider, no dimension chosen for you. The default
application composes nothing from here; ``tests/test_c4_contract.py`` pins
that, along with the absence of any network, credential, or model-download
path in this package.

C5 is what connects this to Qdrant and ``index_revision``.
"""

from ckp.embedding.composition import (
    EmbeddingStack,
    build_c4_deterministic_registry,
    build_c4_deterministic_stack,
)
from ckp.embedding.errors import (
    ERROR_PREFIX,
    EmbeddingErrorCode,
    EmbeddingRefusal,
)
from ckp.embedding.hashing import (
    COSINE_RERANKER_ID,
    HASH_EMBEDDING_ID,
    PROVIDER_VERSION,
    CosineReranker,
    HashEmbeddingProvider,
    tokenize,
)
from ckp.embedding.models import (
    EMBEDDING_CONTRACT,
    NORM_TOLERANCE,
    REVISION_DOMAIN,
    EmbeddingBatch,
    EmbeddingVector,
    ProviderDescriptor,
    ProviderKind,
    RankedCandidate,
    RerankCandidate,
    RerankResult,
    SimilarityMetric,
    compute_provider_revision,
    require_text,
)
from ckp.embedding.provider import EmbeddingProvider, RerankerProvider
from ckp.embedding.registry import ProviderRegistry

__all__ = [
    "COSINE_RERANKER_ID",
    "EMBEDDING_CONTRACT",
    "ERROR_PREFIX",
    "HASH_EMBEDDING_ID",
    "NORM_TOLERANCE",
    "PROVIDER_VERSION",
    "REVISION_DOMAIN",
    "CosineReranker",
    "EmbeddingBatch",
    "EmbeddingErrorCode",
    "EmbeddingProvider",
    "EmbeddingRefusal",
    "EmbeddingStack",
    "EmbeddingVector",
    "HashEmbeddingProvider",
    "ProviderDescriptor",
    "ProviderKind",
    "ProviderRegistry",
    "RankedCandidate",
    "RerankCandidate",
    "RerankResult",
    "RerankerProvider",
    "SimilarityMetric",
    "build_c4_deterministic_registry",
    "build_c4_deterministic_stack",
    "compute_provider_revision",
    "require_text",
    "tokenize",
]
