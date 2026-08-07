"""Explicit composition for the semantic provider (issue #25).

Mirrors ``ckp.embedding.composition``: every parameter required, no default
model directory, no module-level registry, no default provider name. The
caller decides where the verified asset cache lives and what names to
register under; nothing here guesses one. The reranker is
``ckp.embedding.hashing.CosineReranker`` reused unmodified -- proof that
swapping the embedding provider really does not change any caller, not even
the one immediately downstream of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ckp.embedding.errors import EmbeddingErrorCode, EmbeddingRefusal
from ckp.embedding.hashing import CosineReranker
from ckp.embedding.models import ProviderKind, compute_provider_revision
from ckp.embedding.provider import (
    EmbeddingProvider,
    RerankerProvider,
    require_provider,
)
from ckp.embedding.registry import ProviderRegistry
from ckp.semantic.provider import SemanticEmbeddingProvider


@dataclass(frozen=True)
class SemanticEmbeddingStack:
    """A resolved pair plus the revisions C5 folds into ``index_revision``.

    Validates itself rather than trusting the builder below, exactly like
    ``ckp.embedding.composition.EmbeddingStack`` -- a stack assembled by
    hand around an unadmitted provider, or carrying a revision string that
    does not match its own descriptor, would hand ``index_revision`` a
    value nothing can recompute.
    """

    embedding: EmbeddingProvider
    reranker: RerankerProvider
    embedding_revision: str
    reranker_revision: str

    def __post_init__(self) -> None:
        embedding_descriptor = require_provider(self.embedding, ProviderKind.EMBEDDING)
        reranker_descriptor = require_provider(self.reranker, ProviderKind.RERANKER)
        for revision, descriptor in (
            (self.embedding_revision, embedding_descriptor),
            (self.reranker_revision, reranker_descriptor),
        ):
            if revision != compute_provider_revision(descriptor):
                raise EmbeddingRefusal(EmbeddingErrorCode.DESCRIPTOR_INVALID)


def build_semantic_registry(
    *,
    model_dir: Path,
    embedding_name: str,
    reranker_name: str,
) -> ProviderRegistry:
    """Register the semantic embedder and its cosine reranker under
    caller-chosen names. Construction alone verifies the pinned assets
    (``ckp.semantic.assets.resolve_semantic_assets``); a bad cache refuses
    here, before anything is registered."""
    embedding = SemanticEmbeddingProvider(model_dir=model_dir)
    reranker = CosineReranker(embedder=embedding)
    registry = ProviderRegistry()
    registry.register_embedding(embedding_name, embedding)
    registry.register_reranker(reranker_name, reranker)
    return registry


def build_semantic_stack(
    *,
    model_dir: Path,
    embedding_name: str,
    reranker_name: str,
) -> SemanticEmbeddingStack:
    """Resolve the pair back out of the registry, never straight from the
    class -- the same discipline ``build_c4_deterministic_stack`` applies,
    so nothing can end up composed without going through
    ``ProviderRegistry``."""
    registry = build_semantic_registry(
        model_dir=model_dir,
        embedding_name=embedding_name,
        reranker_name=reranker_name,
    )
    embedding = registry.resolve_embedding(embedding_name)
    reranker = registry.resolve_reranker(reranker_name)
    return SemanticEmbeddingStack(
        embedding=embedding,
        reranker=reranker,
        embedding_revision=compute_provider_revision(embedding.descriptor),
        reranker_revision=compute_provider_revision(reranker.descriptor),
    )


__all__ = [
    "SemanticEmbeddingStack",
    "build_semantic_registry",
    "build_semantic_stack",
]
