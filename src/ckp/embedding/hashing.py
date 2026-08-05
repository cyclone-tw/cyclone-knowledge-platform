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
import unicodedata
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
from ckp.embedding.provider import EmbeddingProvider, require_provider

HASH_EMBEDDING_ID = "hash-sha256"
COSINE_RERANKER_ID = "cosine-rerank"
PROVIDER_VERSION = "1"

_TOKEN_DOMAIN = b"ckp-embedding-hash-token-v1"

#: ASCII-only case folding, as an explicit table. ``str.casefold`` and
#: ``str.lower`` read the interpreter's Unicode case tables, which gain
#: entries between Unicode releases -- the same note would then vectorize
#: differently on two Python versions.
_ASCII_FOLD = {codepoint: codepoint + 32 for codepoint in range(ord("A"), ord("Z") + 1)}

#: Non-ASCII characters that carry no retrieval signal: ideographic and
#: fullwidth space, and the punctuation Chinese prose is full of. Frozen as a
#: literal for the same reason as the fold table -- deciding "is this
#: punctuation?" from ``unicodedata`` would make tokenization depend on which
#: Unicode version the interpreter was built against.
_IGNORED_NON_ASCII = (
    # Spaces and zero-width marks, spelled out so the set is auditable.
    "\u00a0\u2002\u2003\u2009\u200b\u200c\u200d\u3000\ufeff"
    # CJK and fullwidth punctuation.
    "\u3001\u3002\uff0c\uff0e\u00b7\uff1b\uff1a\uff1f\uff01"
    "\u2026\u2025\u2014\u2015\uff5e\uff0d"
    "\u300c\u300d\u300e\u300f\uff08\uff09\u3008\u3009"
    "\u300a\u300b\u3010\u3011\u3014\u3015\u3016\u3017"
    "\u201c\u201d\u2018\u2019"
    # Bullets and separators. The set is deliberately finite: deciding
    # "is this punctuation?" from unicodedata would put tokenization back on
    # a moving table. Anything not listed becomes its own token, which costs
    # one noisy bucket and never drops real text.
    "\u2022\u2023\u2027\u2043\u30fb\uff65\u2010\u2011\u2012\u2013"
)

#: Frozen tokenizer, defined purely by codepoint ranges: runs of ASCII
#: alphanumerics, plus every other single non-ASCII character on its own. The
#: second branch is what keeps Traditional Chinese from being dropped -- a
#: whitespace tokenizer would map a whole Chinese note to one token. It
#: deliberately uses no ``\w``/``\s``/``\d`` class, because those are
#: Unicode-database-driven and therefore interpreter-version-dependent.
#: Changing any of this changes every vector, so it is versioned by
#: ``PROVIDER_VERSION`` rather than tuned in place.
_TOKEN = re.compile(r"[a-z0-9]+|[^\x00-\x7f" + re.escape(_IGNORED_NON_ASCII) + "]")


def tokenize(text: str) -> list[str]:
    r"""Normalize to NFC, fold ASCII case, then split on the frozen classes.

    NFC matters because the same Chinese or accented text arrives decomposed
    from macOS and composed from almost everywhere else; without it
    ``"caf\u00e9"`` and ``"cafe\u0301"`` land in different buckets and a note
    stops matching its own query. Normalization is the one Unicode table this
    provider does consult, and it is the safe one: the Unicode Normalization
    Stability Policy guarantees the normalized form of an already-assigned
    string never changes in a later version, which is exactly the guarantee
    ``\w`` and ``casefold`` do not give.
    """
    normalized = unicodedata.normalize("NFC", text)
    return _TOKEN.findall(normalized.translate(_ASCII_FOLD))


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
        # Taking an embedder is an admission point, so it runs the same
        # admission check the registry does: real interface, embedding kind,
        # inside the Phase 3 offline fence. Checking only the flags would
        # still accept an object with no embed_query, or a reranker in an
        # embedder's place (AGENTS.md §9: same root cause, other point on
        # the path).
        descriptor = require_provider(embedder, ProviderKind.EMBEDDING)
        if not descriptor.normalized:
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
            # Inherited, not asserted. The guard above already refuses a
            # network or non-deterministic embedder, so these are False/True
            # today -- but if that guard were ever removed, the descriptor
            # would still report what this reranker can actually back, and
            # the registry would reject it. A hardcoded pair would lie.
            deterministic=embedder.descriptor.deterministic,
            requires_network=embedder.descriptor.requires_network,
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
