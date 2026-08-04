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

from ckp.embedding.models import (
    EmbeddingBatch,
    EmbeddingVector,
    ProviderDescriptor,
    RerankCandidate,
    RerankResult,
)


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
    "EmbeddingProvider",
    "RerankerProvider",
]
