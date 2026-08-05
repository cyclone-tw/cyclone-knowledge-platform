"""Frozen ``embedding/v1`` descriptor, vector, and rerank schemas.

Two invariants are enforced by the models themselves rather than left to
each provider:

* a vector never travels without the descriptor that produced it -- a bare
  float list has no dimension, metric, or provider identity, so it cannot be
  compared against anything or folded into a revision;
* a rerank result is a *total* order (score descending, then candidate id
  ascending). Input order is a caller-side accident; tie-breaking on it would
  make two callers with the same candidates disagree.

Nothing here knows the name of any concrete provider.
"""

from __future__ import annotations

import hashlib
import math
import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ckp.embedding.errors import EmbeddingErrorCode, EmbeddingRefusal

#: Bumped only by a breaking change to the shapes in this module.
EMBEDDING_CONTRACT = "embedding/v1"

REVISION_DOMAIN = b"ckp-embedding-provider-v1"

#: The largest L2 deviation a ``normalized`` provider may report. Integer
#: accumulation followed by a single normalization lands far inside this;
#: anything looser would hide a provider that forgot to normalize at all.
NORM_TOLERANCE = 1e-12

_PROVIDER_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_PROVIDER_VERSION = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")


class ProviderKind(StrEnum):
    EMBEDDING = "embedding"
    RERANKER = "reranker"


class SimilarityMetric(StrEnum):
    COSINE = "cosine"


class ProviderDescriptor(BaseModel):
    """Everything a consumer needs to know about a provider, and no more.

    ``semantic`` is an honesty field (AGENTS.md §8): a provider that computes
    a lexical or hashed score must declare ``semantic=False`` so a downstream
    benchmark cannot quietly present it as model quality.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: str = Field(min_length=1)
    provider_id: str = Field(min_length=1, max_length=64)
    provider_version: str = Field(min_length=1, max_length=64)
    kind: ProviderKind
    dimension: int | None = Field(default=None, ge=1, le=65536)
    metric: SimilarityMetric
    normalized: bool
    deterministic: bool
    requires_network: bool
    semantic: bool

    @field_validator("provider_id")
    @classmethod
    def require_provider_id_shape(cls, value: str) -> str:
        if _PROVIDER_ID.fullmatch(value) is None:
            raise ValueError("provider_id must be lowercase [a-z0-9._-]")
        return value

    @field_validator("provider_version")
    @classmethod
    def require_provider_version_shape(cls, value: str) -> str:
        if _PROVIDER_VERSION.fullmatch(value) is None:
            raise ValueError("provider_version must be lowercase [a-z0-9._-]")
        return value

    @model_validator(mode="after")
    def require_dimension_to_match_kind(self) -> ProviderDescriptor:
        if self.kind is ProviderKind.EMBEDDING and self.dimension is None:
            raise ValueError("an embedding provider must declare a dimension")
        if self.kind is ProviderKind.RERANKER and self.dimension is not None:
            raise ValueError("a reranker provider must not declare a dimension")
        return self

    @property
    def revision(self) -> str:
        return compute_provider_revision(self)


def _framed_hash(*values: bytes) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return f"sha256:{digest.hexdigest()}"


def compute_provider_revision(descriptor: ProviderDescriptor) -> str:
    """Length-framed digest over every descriptor field.

    C5 folds this into ``index_revision``; a field left out here would let two
    providers that produce different vectors claim the same index revision.
    """
    return _framed_hash(
        REVISION_DOMAIN,
        descriptor.contract_version.encode("utf-8"),
        descriptor.provider_id.encode("utf-8"),
        descriptor.provider_version.encode("utf-8"),
        descriptor.kind.value.encode("ascii"),
        b"" if descriptor.dimension is None else str(descriptor.dimension).encode(),
        descriptor.metric.value.encode("ascii"),
        _flag(descriptor.normalized),
        _flag(descriptor.deterministic),
        _flag(descriptor.requires_network),
        _flag(descriptor.semantic),
    )


def _flag(value: bool) -> bytes:
    return b"1" if value else b"0"


def _require_result_descriptor(
    descriptor: ProviderDescriptor, kind: ProviderKind
) -> None:
    """These models *are* the v1 shapes, so a result must be a v1 result.

    Without this a reranker could return a result stamped with an embedding
    descriptor, or with ``embedding/v9``, and every consumer downstream would
    read a provider identity that never produced these numbers.
    """
    if descriptor.contract_version != EMBEDDING_CONTRACT:
        raise ValueError("result descriptor must declare this contract version")
    if descriptor.kind is not kind:
        raise ValueError(f"result descriptor must be of kind {kind.value}")


def _validate_row(values: tuple[float, ...], descriptor: ProviderDescriptor) -> None:
    if descriptor.dimension is None or len(values) != descriptor.dimension:
        raise ValueError("vector length must equal the declared dimension")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("vector components must all be finite")
    if descriptor.normalized:
        norm = math.sqrt(math.fsum(value * value for value in values))
        if abs(norm - 1.0) > NORM_TOLERANCE:
            raise ValueError("a normalized provider must return unit vectors")


class EmbeddingVector(BaseModel):
    """One vector, inseparable from the descriptor that produced it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    descriptor: ProviderDescriptor
    values: tuple[float, ...]

    @model_validator(mode="after")
    def require_declared_shape(self) -> EmbeddingVector:
        _require_result_descriptor(self.descriptor, ProviderKind.EMBEDDING)
        _validate_row(self.values, self.descriptor)
        return self


class EmbeddingBatch(BaseModel):
    """Vectors for one ``embed_documents`` call, in the caller's input order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    descriptor: ProviderDescriptor
    vectors: tuple[tuple[float, ...], ...]

    @model_validator(mode="after")
    def require_declared_shape(self) -> EmbeddingBatch:
        _require_result_descriptor(self.descriptor, ProviderKind.EMBEDDING)
        for row in self.vectors:
            _validate_row(row, self.descriptor)
        return self

    def vector_at(self, index: int) -> EmbeddingVector:
        """Detach one row, carrying the descriptor with it."""
        return EmbeddingVector(descriptor=self.descriptor, values=self.vectors[index])


class RerankCandidate(BaseModel):
    """A rerank input. Text enters here and never comes back out."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1, max_length=256)
    text: str

    @field_validator("candidate_id")
    @classmethod
    def require_candidate_id_shape(cls, value: str) -> str:
        if _CANDIDATE_ID.fullmatch(value) is None:
            raise ValueError("candidate_id must match the frozen id pattern")
        return value


class RankedCandidate(BaseModel):
    """One placed candidate: id, score, rank. Deliberately no text field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1, max_length=256)
    score: float
    rank: int = Field(ge=0)

    @field_validator("candidate_id")
    @classmethod
    def require_candidate_id_shape(cls, value: str) -> str:
        if _CANDIDATE_ID.fullmatch(value) is None:
            raise ValueError("candidate_id must match the frozen id pattern")
        return value

    @field_validator("score")
    @classmethod
    def require_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


class RerankResult(BaseModel):
    """A total order over the placed candidates, checked here, not per provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    descriptor: ProviderDescriptor
    ranked: tuple[RankedCandidate, ...]

    @model_validator(mode="after")
    def require_frozen_total_order(self) -> RerankResult:
        _require_result_descriptor(self.descriptor, ProviderKind.RERANKER)
        seen: set[str] = set()
        for position, item in enumerate(self.ranked):
            if item.rank != position:
                raise ValueError("rank must be the dense 0-based position")
            if item.candidate_id in seen:
                raise ValueError("candidate_id must not repeat in a result")
            seen.add(item.candidate_id)
        for previous, item in zip(self.ranked, self.ranked[1:], strict=False):
            key_previous = (-previous.score, previous.candidate_id)
            key_item = (-item.score, item.candidate_id)
            if key_previous >= key_item:
                raise ValueError(
                    "ranked order must be score descending, then candidate_id ascending"
                )
        return self


def require_text(value: object) -> str:
    """Accept one non-empty text, failing closed rather than coercing."""
    if not isinstance(value, str):
        raise EmbeddingRefusal(EmbeddingErrorCode.TEXT_INVALID)
    if not value.strip():
        raise EmbeddingRefusal(EmbeddingErrorCode.TEXT_EMPTY)
    try:
        # A lone surrogate is a legal ``str`` that cannot be encoded. Every
        # provider hashes bytes eventually, so without this the first
        # ``encode`` raises UnicodeEncodeError -- an unstable exception type
        # escaping instead of a coded refusal.
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise EmbeddingRefusal(EmbeddingErrorCode.TEXT_INVALID) from error
    return value


__all__ = [
    "EMBEDDING_CONTRACT",
    "NORM_TOLERANCE",
    "REVISION_DOMAIN",
    "EmbeddingBatch",
    "EmbeddingVector",
    "ProviderDescriptor",
    "ProviderKind",
    "RankedCandidate",
    "RerankCandidate",
    "RerankResult",
    "SimilarityMetric",
    "compute_provider_revision",
    "require_text",
]
