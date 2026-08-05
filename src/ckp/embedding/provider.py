"""The provider-neutral ``embedding/v1`` interfaces.

These two protocols are the whole point of C4: a caller depends on the shape
below and on nothing else, so swapping the provider behind it is a
composition change rather than a code change. Contract §2: a consumer does
not need to know the index provider's details.

This module must stay ignorant of every concrete provider -- no provider id,
no import of an implementation. ``tests/test_c4_contract.py`` pins that.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ckp.embedding.errors import EmbeddingErrorCode, EmbeddingRefusal
from ckp.embedding.models import (
    EMBEDDING_CONTRACT,
    EmbeddingBatch,
    EmbeddingVector,
    ProviderDescriptor,
    RerankCandidate,
    RerankResult,
)

#: The methods each kind of provider owes, checked for callability rather
#: than mere presence: ``runtime_checkable`` only asks ``hasattr``, so
#: ``embed_query = None`` would otherwise pass as an embedding provider and
#: fail at the first query instead of at registration.
EMBEDDING_METHODS = ("embed_documents", "embed_query")
RERANKER_METHODS = ("rerank",)


def require_offline_provider(descriptor: ProviderDescriptor) -> None:
    """The Phase 3 offline boundary, in one place.

    It is enforced at registration *and* wherever one provider is composed
    into another. Checking only the outer descriptor would let a reranker
    wrap a network-calling embedder and still register as offline -- the
    reranker's own flags would be a lie it never had to back.
    """
    if descriptor.contract_version != EMBEDDING_CONTRACT:
        raise EmbeddingRefusal(EmbeddingErrorCode.CONTRACT_VERSION_UNKNOWN)
    if descriptor.requires_network:
        # Phase 3 non-goal: no cloud provider, no model download.
        raise EmbeddingRefusal(EmbeddingErrorCode.NETWORK_PROVIDER_DENIED)
    if not descriptor.deterministic:
        # A non-recomputable vector cannot back a recomputable
        # ``index_revision`` (contract §5.5).
        raise EmbeddingRefusal(EmbeddingErrorCode.NONDETERMINISTIC_PROVIDER_DENIED)


def require_callable_interface(provider: object, methods: tuple[str, ...]) -> None:
    """Every declared method must actually be callable on this object."""
    for method in methods:
        if not callable(getattr(provider, method, None)):
            raise EmbeddingRefusal(EmbeddingErrorCode.PROVIDER_KIND_MISMATCH)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Text in, vectors out, deterministically and with no I/O.

    Frozen invariants every implementation owes its caller:

    1. Pure in ``(text, descriptor)``: no clock, no RNG, no environment, no
       filesystem, no network.
    2. ``embed_query(t)`` is bit-identical to ``embed_documents([t])`` row 0.
       Two code paths for the same question is how a query silently stops
       matching its own documents.
    3. Output length equals input length, in input order.
    4. Empty or whitespace-only text is refused, never mapped to a zero
       vector -- a zero vector is a fabricated value (AGENTS.md §8).
    """

    @property
    def descriptor(self) -> ProviderDescriptor: ...

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch: ...

    def embed_query(self, text: str) -> EmbeddingVector: ...


@runtime_checkable
class RerankerProvider(Protocol):
    """Query plus candidates in, a total order out.

    Frozen invariants:

    1. ``top_k`` is a required keyword, at least 1; the result holds
       ``min(top_k, len(candidates))`` items.
    2. The order is score descending, then ``candidate_id`` ascending. Never
       input order.
    3. Duplicate candidate ids are refused.
    4. The result carries ids, scores, and ranks only -- never candidate text.
    """

    @property
    def descriptor(self) -> ProviderDescriptor: ...

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int,
    ) -> RerankResult: ...


__all__ = [
    "EMBEDDING_METHODS",
    "RERANKER_METHODS",
    "EmbeddingProvider",
    "RerankerProvider",
    "require_callable_interface",
    "require_offline_provider",
]
