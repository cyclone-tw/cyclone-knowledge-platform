"""The frozen embedding/v1 schemas: descriptor, vectors, and the total order."""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from ckp.embedding.models import (
    EMBEDDING_CONTRACT,
    EmbeddingBatch,
    EmbeddingVector,
    ProviderDescriptor,
    ProviderKind,
    RankedCandidate,
    RerankResult,
    SimilarityMetric,
    compute_provider_revision,
)
from embedding_fixtures import descriptor_with


def test_an_embedding_descriptor_must_declare_a_dimension() -> None:
    with pytest.raises(ValidationError):
        descriptor_with(dimension=None)


def test_a_reranker_descriptor_must_not_declare_a_dimension() -> None:
    with pytest.raises(ValidationError):
        descriptor_with(kind=ProviderKind.RERANKER, dimension=8)
    # ... and is valid without one.
    descriptor = descriptor_with(kind=ProviderKind.RERANKER, dimension=None)
    assert descriptor.dimension is None


def test_descriptor_is_frozen_and_rejects_unknown_fields() -> None:
    descriptor = descriptor_with()
    with pytest.raises(ValidationError):
        descriptor.provider_id = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        descriptor_with(quality="excellent")


def test_descriptor_rejects_a_provider_id_that_is_not_the_frozen_shape() -> None:
    for bad in ("Upper", "has space", "-leading", ""):
        with pytest.raises(ValidationError):
            descriptor_with(provider_id=bad)


def test_provider_revision_changes_when_any_varying_field_changes() -> None:
    """Two providers that produce different vectors must not share a revision."""
    baseline = compute_provider_revision(descriptor_with())
    variants = {
        "contract_version": descriptor_with(contract_version="embedding/v2"),
        "provider_id": descriptor_with(provider_id="other-test"),
        "provider_version": descriptor_with(provider_version="2"),
        "kind": descriptor_with(kind=ProviderKind.RERANKER, dimension=None),
        "dimension": descriptor_with(dimension=16),
        "normalized": descriptor_with(normalized=False),
        "deterministic": descriptor_with(deterministic=False),
        "requires_network": descriptor_with(requires_network=True),
        "semantic": descriptor_with(semantic=True),
    }
    seen = {baseline}
    for field, descriptor in variants.items():
        revision = compute_provider_revision(descriptor)
        assert revision != baseline, f"{field} does not reach the revision"
        seen.add(revision)
    assert len(seen) == len(variants) + 1


def test_provider_revision_reads_every_descriptor_field() -> None:
    """``metric`` has one legal value today, so pin it at the source level."""
    source = inspect.getsource(compute_provider_revision)
    for needle in (
        "REVISION_DOMAIN",
        "descriptor.contract_version",
        "descriptor.provider_id",
        "descriptor.provider_version",
        "descriptor.kind",
        "descriptor.dimension",
        "descriptor.metric",
        "descriptor.normalized",
        "descriptor.deterministic",
        "descriptor.requires_network",
        "descriptor.semantic",
    ):
        assert needle in source


def test_provider_revision_is_a_domain_separated_sha256() -> None:
    revision = compute_provider_revision(descriptor_with())
    assert revision.startswith("sha256:")
    assert len(revision) == len("sha256:") + 64
    assert descriptor_with().revision == revision


def test_a_vector_must_match_the_declared_dimension_and_be_finite() -> None:
    descriptor = descriptor_with(dimension=2, normalized=False)
    EmbeddingVector(descriptor=descriptor, values=(0.5, 0.5))
    with pytest.raises(ValidationError):
        EmbeddingVector(descriptor=descriptor, values=(0.5,))
    with pytest.raises(ValidationError):
        EmbeddingVector(descriptor=descriptor, values=(float("nan"), 0.5))
    with pytest.raises(ValidationError):
        EmbeddingVector(descriptor=descriptor, values=(float("inf"), 0.5))


def test_a_normalized_provider_cannot_return_a_non_unit_vector() -> None:
    descriptor = descriptor_with(dimension=2, normalized=True)
    EmbeddingVector(descriptor=descriptor, values=(1.0, 0.0))
    with pytest.raises(ValidationError):
        EmbeddingVector(descriptor=descriptor, values=(1.0, 1.0))
    with pytest.raises(ValidationError):
        # A zero vector has no direction; it is the shape a dropped
        # normalization or an invented fallback would take.
        EmbeddingVector(descriptor=descriptor, values=(0.0, 0.0))


def test_batch_rows_carry_the_batch_descriptor_when_detached() -> None:
    descriptor = descriptor_with(dimension=2, normalized=True)
    batch = EmbeddingBatch(descriptor=descriptor, vectors=((1.0, 0.0), (0.0, 1.0)))
    assert batch.vector_at(1) == EmbeddingVector(
        descriptor=descriptor, values=(0.0, 1.0)
    )
    with pytest.raises(ValidationError):
        EmbeddingBatch(descriptor=descriptor, vectors=((1.0, 0.0), (0.3, 0.3)))


def _ranked(*pairs: tuple[str, float]) -> tuple[RankedCandidate, ...]:
    return tuple(
        RankedCandidate(candidate_id=identifier, score=score, rank=rank)
        for rank, (identifier, score) in enumerate(pairs)
    )


def _result(*pairs: tuple[str, float]) -> RerankResult:
    return RerankResult(
        descriptor=descriptor_with(kind=ProviderKind.RERANKER, dimension=None),
        ranked=_ranked(*pairs),
    )


def test_rerank_result_accepts_the_frozen_total_order() -> None:
    result = _result(("note-a", 0.9), ("note-b", 0.5), ("note-c", 0.5))
    assert [item.rank for item in result.ranked] == [0, 1, 2]


def test_rerank_result_rejects_ascending_scores() -> None:
    with pytest.raises(ValidationError):
        _result(("note-a", 0.1), ("note-b", 0.9))


def test_rerank_result_rejects_a_tie_broken_against_candidate_id() -> None:
    """Input order is a caller-side accident and must never decide a tie."""
    with pytest.raises(ValidationError):
        _result(("note-c", 0.5), ("note-b", 0.5))


def test_rerank_result_rejects_duplicate_ids_and_sparse_ranks() -> None:
    with pytest.raises(ValidationError):
        _result(("note-a", 0.9), ("note-a", 0.5))
    with pytest.raises(ValidationError):
        RerankResult(
            descriptor=descriptor_with(kind=ProviderKind.RERANKER, dimension=None),
            ranked=(
                RankedCandidate(candidate_id="note-a", score=0.9, rank=0),
                RankedCandidate(candidate_id="note-b", score=0.5, rank=2),
            ),
        )


def test_a_ranked_candidate_has_no_text_field() -> None:
    """Rerank output is ids and scores. Text goes in and does not come back."""
    assert "text" not in RankedCandidate.model_fields
    with pytest.raises(ValidationError):
        RankedCandidate(candidate_id="note-a", score=0.9, rank=0, text="leak")


def test_ranked_candidate_rejects_a_non_finite_score() -> None:
    with pytest.raises(ValidationError):
        RankedCandidate(candidate_id="note-a", score=float("nan"), rank=0)


def test_contract_and_metric_constants_are_the_frozen_values() -> None:
    assert EMBEDDING_CONTRACT == "embedding/v1"
    assert SimilarityMetric.COSINE == "cosine"
    assert set(ProviderKind) == {ProviderKind.EMBEDDING, ProviderKind.RERANKER}
    assert ProviderDescriptor.model_config["frozen"] is True
