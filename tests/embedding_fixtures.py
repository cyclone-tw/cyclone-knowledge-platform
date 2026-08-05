"""Synthetic providers and the one neutral caller the C4 tests share.

Everything here is invented text. No Wiki note, no fixture bundle, and no
network is involved in any C4 test.

The providers below are deliberately defined *in the tests*: they are the
third-party implementations that prove the interface is provider-neutral. If
a caller has to change to accept one of them, the interface has failed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

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
from ckp.embedding.provider import EmbeddingProvider, RerankerProvider

#: Mixed script on purpose: a whitespace tokenizer would collapse the
#: Traditional Chinese half into a single token and the tests would not notice.
SAMPLE_TEXTS = (
    "kettle temperature control",
    "手沖 咖啡 水溫 控制",
    "espresso grinder burr alignment",
    "特教 個別化 教育 計畫",
)


class OrdinalEmbedding:
    """A third-party embedding provider that shares no code with the shipped one."""

    def __init__(self, *, dimension: int) -> None:
        self._descriptor = ProviderDescriptor(
            contract_version=EMBEDDING_CONTRACT,
            provider_id="ordinal-test",
            provider_version="1",
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
        return EmbeddingBatch(
            descriptor=self._descriptor,
            vectors=tuple(self._embed_one(text) for text in texts),
        )

    def embed_query(self, text: str) -> EmbeddingVector:
        return EmbeddingVector(
            descriptor=self._descriptor, values=self._embed_one(text)
        )

    def _embed_one(self, text: str) -> tuple[float, ...]:
        dimension = self._descriptor.dimension
        assert dimension is not None
        counts = [0] * dimension
        for character in require_text(text):
            counts[ord(character) % dimension] += 1
        norm = math.sqrt(sum(count * count for count in counts))
        return tuple(count / norm for count in counts)


class LengthReranker:
    """A third-party reranker with no embedding provider behind it at all."""

    def __init__(self) -> None:
        self._descriptor = ProviderDescriptor(
            contract_version=EMBEDDING_CONTRACT,
            provider_id="length-test",
            provider_version="1",
            kind=ProviderKind.RERANKER,
            dimension=None,
            metric=SimilarityMetric.COSINE,
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
        scored = [
            (
                candidate.candidate_id,
                1.0 / (1.0 + abs(len(candidate.text) - len(query))),
            )
            for candidate in candidates
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return RerankResult(
            descriptor=self._descriptor,
            ranked=tuple(
                RankedCandidate(candidate_id=identifier, score=score, rank=rank)
                for rank, (identifier, score) in enumerate(scored[:top_k])
            ),
        )


def descriptor_with(**overrides: object) -> ProviderDescriptor:
    """A valid embedding descriptor with individual fields overridden."""
    fields: dict[str, object] = {
        "contract_version": EMBEDDING_CONTRACT,
        "provider_id": "probe-test",
        "provider_version": "1",
        "kind": ProviderKind.EMBEDDING,
        "dimension": 8,
        "metric": SimilarityMetric.COSINE,
        "normalized": True,
        "deterministic": True,
        "requires_network": False,
        "semantic": False,
    }
    fields.update(overrides)
    return ProviderDescriptor(**fields)  # type: ignore[arg-type]


class DescriptorOnlyEmbedding:
    """Structurally a provider, but carrying whatever descriptor a test hands it.

    Used to drive the registration boundary (network, non-determinism, wrong
    contract version) without inventing a fake cloud client.
    """

    def __init__(self, descriptor: ProviderDescriptor) -> None:
        self._descriptor = descriptor

    @property
    def descriptor(self) -> ProviderDescriptor:
        return self._descriptor

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        raise AssertionError("registration must fail before any call")

    def embed_query(self, text: str) -> EmbeddingVector:
        raise AssertionError("registration must fail before any call")


class DescriptorOnlyReranker(DescriptorOnlyEmbedding):
    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int,
    ) -> RerankResult:
        raise AssertionError("registration must fail before any call")


class NonCallableEmbedding:
    """Has every attribute the protocol names, none of them callable.

    Values are deliberately not ``None``: a runtime protocol check special-
    cases ``None`` for method members and would reject that on its own. Any
    other non-callable value sails straight through ``isinstance``, so this
    is what an isinstance-only registration would admit -- and then fail on
    at the first query, with the provider already in the index path.
    """

    embed_documents = "not a method"
    embed_query = 42

    def __init__(self, descriptor: ProviderDescriptor) -> None:
        self._descriptor = descriptor

    @property
    def descriptor(self) -> ProviderDescriptor:
        return self._descriptor


class TwoFacedEmbedding(OrdinalEmbedding):
    """Answers admission with a clean descriptor, then changes its story.

    ``descriptor`` is a property call on a foreign object, so every read is a
    fresh answer. An admission point that validates the first read and then
    re-reads the property has checked nothing.
    """

    def __init__(self, *, dimension: int, later: ProviderDescriptor) -> None:
        super().__init__(dimension=dimension)
        self._later = later
        self._reads = 0

    @property
    def descriptor(self) -> ProviderDescriptor:
        self._reads += 1
        if self._reads == 1:
            return self._descriptor
        return self._later


def candidates_from(texts: Sequence[str]) -> tuple[RerankCandidate, ...]:
    return tuple(
        RerankCandidate(candidate_id=f"note-{index:02d}", text=text)
        for index, text in enumerate(texts)
    )


def retrieve_ids(
    embedding: EmbeddingProvider,
    reranker: RerankerProvider,
    *,
    query: str,
    candidates: Sequence[RerankCandidate],
    top_k: int,
) -> tuple[str, ...]:
    """The neutral caller under test.

    It names no provider and no provider id, imports no implementation, and
    branches on nothing except the declared dimension. Swapping either
    provider must not require editing one character of this function --
    ``tests/test_embedding_neutrality.py`` asserts exactly that.
    """
    query_vector = embedding.embed_query(query)
    assert len(query_vector.values) == embedding.descriptor.dimension
    embedding.embed_documents([candidate.text for candidate in candidates])
    result = reranker.rerank(query, candidates, top_k=top_k)
    return tuple(item.candidate_id for item in result.ranked)


__all__ = [
    "SAMPLE_TEXTS",
    "DescriptorOnlyEmbedding",
    "DescriptorOnlyReranker",
    "LengthReranker",
    "NonCallableEmbedding",
    "OrdinalEmbedding",
    "TwoFacedEmbedding",
    "candidates_from",
    "descriptor_with",
    "retrieve_ids",
]
