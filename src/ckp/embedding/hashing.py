"""The offline deterministic providers shipped with C4.

Neither of these downloads a model, reads a credential, or opens a socket.
They exist so the interfaces in :mod:`ckp.embedding.provider` have a real
implementation to be checked against, and so C5 has a fixed baseline for its
shadow benchmark. Both declare ``semantic=False``: hashed term overlap is not
semantic similarity and must never be reported as if it were.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from ckp.embedding.errors import EmbeddingErrorCode, EmbeddingRefusal
from ckp.embedding.models import (
    EMBEDDING_CONTRACT,
    EmbeddingBatch,
    EmbeddingVector,
    ProviderDescriptor,
    ProviderKind,
    RankedCandidate,
    RerankCandidate,
    RerankResult,
    SimilarityMetric,
    require_text,
)
from ckp.embedding.provider import EmbeddingProvider

HASH_EMBEDDING_ID = "hash-sha256"
COSINE_RERANKER_ID = "cosine-rerank"
PROVIDER_VERSION = "1"

_TOKEN_DOMAIN = b"ckp-embedding-hash-token-v1"

#: Frozen tokenizer: ASCII alphanumeric runs, plus every other single word
#: character on its own. The second branch is what keeps Traditional Chinese
#: from being silently dropped -- a whitespace tokenizer would map a whole
#: Chinese note to one token. Punctuation matches neither branch and is
#: ignored. Changing this changes every vector, so it is versioned by
#: ``PROVIDER_VERSION`` rather than tuned in place.
_TOKEN = re.compile(r"[a-z0-9]+|[^\W_]")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


class HashEmbeddingProvider:
    """Deterministic feature-hashing embeddings over ``dimension`` buckets.

    Term counts are accumulated as integers and normalized exactly once at
    the end. Accumulating in floating point instead would make the result
    depend on token order at the last bit, which is precisely the kind of
    drift that breaks a recomputable ``index_revision`` (AGENTS.md §8).

    Weights are unsigned on purpose. Signed feature hashing lets two tokens
    cancel to the zero vector, which has no direction to normalize -- and the
    only ways out of that are a content-dependent refusal or a fabricated
    fallback vector. Neither is acceptable, so the collision cost is paid
    instead.
    """

    def __init__(self, *, dimension: int) -> None:
        self._descriptor = ProviderDescriptor(
            contract_version=EMBEDDING_CONTRACT,
            provider_id=HASH_EMBEDDING_ID,
            provider_version=PROVIDER_VERSION,
            kind=ProviderKind.EMBEDDING,
            dimension=dimension,
            metric=SimilarityMetric.COSINE,
            normalized=True,
            deterministic=True,
            requires_network=False,
            semantic=False,
        )

    @property
    def descriptor(self) -> ProviderDescriptor:
        return self._descriptor

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        if isinstance(texts, str) or not isinstance(texts, Sequence):
            raise EmbeddingRefusal(EmbeddingErrorCode.BATCH_INVALID)
        return EmbeddingBatch(
            descriptor=self._descriptor,
            vectors=tuple(self._embed_one(text) for text in texts),
        )

    def embed_query(self, text: str) -> EmbeddingVector:
        # Deliberately the same code path as a one-element batch: a separate
        # query path is how a query stops matching its own documents.
        return EmbeddingVector(
            descriptor=self._descriptor,
            values=self._embed_one(text),
        )

    def _embed_one(self, text: str) -> tuple[float, ...]:
        tokens = tokenize(require_text(text))
        if not tokens:
            # Non-empty but with nothing indexable in it (punctuation only).
            # Reporting a zero vector would be inventing a value.
            raise EmbeddingRefusal(EmbeddingErrorCode.TEXT_EMPTY)
        dimension = self._descriptor.dimension
        assert dimension is not None  # guaranteed by the descriptor validator
        counts = [0] * dimension
        for token in tokens:
            digest = hashlib.sha256(_TOKEN_DOMAIN + token.encode("utf-8")).digest()
            counts[int.from_bytes(digest[:8], "big") % dimension] += 1
        norm = math.sqrt(sum(count * count for count in counts))
        return tuple(count / norm for count in counts)


class CosineReranker:
    """Order candidates by cosine similarity against an injected embedder.

    This reuses whatever embedding provider it is given rather than carrying
    a second tokenizer of its own -- a duplicated implementation would drift
    away from the vectors the index was built with.

    It is a deterministic baseline, not a cross-encoder. ``semantic=False``
    says so, and C5's benchmark is expected to beat it.
    """

    def __init__(self, *, embedder: EmbeddingProvider) -> None:
        if not embedder.descriptor.normalized:
            # Unit vectors are what makes the dot product a cosine. Rather
            # than dividing by a norm that might be zero, refuse the
            # composition up front.
            raise EmbeddingRefusal(EmbeddingErrorCode.DESCRIPTOR_INVALID)
        self._embedder = embedder
        self._descriptor = ProviderDescriptor(
            contract_version=EMBEDDING_CONTRACT,
            provider_id=COSINE_RERANKER_ID,
            provider_version=PROVIDER_VERSION,
            kind=ProviderKind.RERANKER,
            dimension=None,
            metric=SimilarityMetric.COSINE,
            # For a reranker this reads as "scores come from unit vectors",
            # so they stay inside [-1, 1].
            normalized=True,
            deterministic=True,
            requires_network=False,
            semantic=False,
        )

    @property
    def descriptor(self) -> ProviderDescriptor:
        return self._descriptor

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int,
    ) -> RerankResult:
        if isinstance(candidates, str) or not isinstance(candidates, Sequence):
            raise EmbeddingRefusal(EmbeddingErrorCode.BATCH_INVALID)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise EmbeddingRefusal(EmbeddingErrorCode.TOP_K_INVALID)
        for candidate in candidates:
            if not isinstance(candidate, RerankCandidate):
                raise EmbeddingRefusal(EmbeddingErrorCode.BATCH_INVALID)
        identifiers = [candidate.candidate_id for candidate in candidates]
        if len(set(identifiers)) != len(identifiers):
            raise EmbeddingRefusal(EmbeddingErrorCode.CANDIDATE_DUPLICATE)

        query_vector = self._embedder.embed_query(query).values
        batch = self._embedder.embed_documents(
            [candidate.text for candidate in candidates]
        )
        scored = [
            (identifier, _dot(query_vector, row))
            for identifier, row in zip(identifiers, batch.vectors, strict=True)
        ]
        # Score descending, then candidate_id ascending. Input order never
        # participates: it is a caller-side accident, not part of the result.
        scored.sort(key=lambda item: (-item[1], item[0]))
        return RerankResult(
            descriptor=self._descriptor,
            ranked=tuple(
                RankedCandidate(candidate_id=identifier, score=score, rank=rank)
                for rank, (identifier, score) in enumerate(scored[:top_k])
            ),
        )


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


__all__ = [
    "COSINE_RERANKER_ID",
    "HASH_EMBEDDING_ID",
    "PROVIDER_VERSION",
    "CosineReranker",
    "HashEmbeddingProvider",
    "tokenize",
]
