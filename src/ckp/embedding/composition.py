"""Explicit composition for the C4 offline providers.

Every parameter is required. There is no module-level registry, no default
dimension, and no default provider name: the caller names what it wants and
the registry either has it or refuses. The default application composes none
of this -- C5 owns wiring an index to it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ckp.embedding.hashing import CosineReranker, HashEmbeddingProvider
from ckp.embedding.models import compute_provider_revision
from ckp.embedding.provider import EmbeddingProvider, RerankerProvider
from ckp.embedding.registry import ProviderRegistry


@dataclass(frozen=True)
class EmbeddingStack:
    """A resolved pair plus the revisions C5 folds into ``index_revision``."""

    embedding: EmbeddingProvider
    reranker: RerankerProvider
    embedding_revision: str
    reranker_revision: str


def build_c4_deterministic_registry(
    *,
    dimension: int,
    embedding_name: str,
    reranker_name: str,
) -> ProviderRegistry:
    """Register the two offline providers under caller-chosen names."""
    embedding = HashEmbeddingProvider(dimension=dimension)
    reranker = CosineReranker(embedder=embedding)
    registry = ProviderRegistry()
    registry.register_embedding(embedding_name, embedding)
    registry.register_reranker(reranker_name, reranker)
    return registry


def build_c4_deterministic_stack(
    *,
    dimension: int,
    embedding_name: str,
    reranker_name: str,
) -> EmbeddingStack:
    """Resolve the pair back out of the registry, never straight from the class."""
    registry = build_c4_deterministic_registry(
        dimension=dimension,
        embedding_name=embedding_name,
        reranker_name=reranker_name,
    )
    embedding = registry.resolve_embedding(embedding_name)
    reranker = registry.resolve_reranker(reranker_name)
    return EmbeddingStack(
        embedding=embedding,
        reranker=reranker,
        embedding_revision=compute_provider_revision(embedding.descriptor),
        reranker_revision=compute_provider_revision(reranker.descriptor),
    )


__all__ = [
    "EmbeddingStack",
    "build_c4_deterministic_registry",
    "build_c4_deterministic_stack",
]
