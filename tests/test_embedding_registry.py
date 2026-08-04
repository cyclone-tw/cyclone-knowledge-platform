"""Registration is the boundary. Everything it refuses, it refuses closed."""

from __future__ import annotations

import pytest

from ckp.embedding.composition import (
    build_c4_deterministic_registry,
    build_c4_deterministic_stack,
)
from ckp.embedding.errors import EmbeddingErrorCode, EmbeddingRefusal
from ckp.embedding.hashing import CosineReranker, HashEmbeddingProvider
from ckp.embedding.models import ProviderKind, compute_provider_revision
from ckp.embedding.registry import ProviderRegistry
from embedding_fixtures import (
    DescriptorOnlyEmbedding,
    DescriptorOnlyReranker,
    LengthReranker,
    OrdinalEmbedding,
    descriptor_with,
)

DIMENSION = 32


def _registry_with_hash() -> ProviderRegistry:
    registry = ProviderRegistry()
    embedding = HashEmbeddingProvider(dimension=DIMENSION)
    registry.register_embedding("hash", embedding)
    registry.register_reranker("cosine", CosineReranker(embedder=embedding))
    return registry


def test_a_new_registry_is_empty() -> None:
    """No builtins auto-register. Nothing runs that nobody chose."""
    registry = ProviderRegistry()
    assert registry.embedding_names() == ()
    assert registry.reranker_names() == ()


def test_an_unknown_name_refuses_instead_of_falling_back() -> None:
    registry = _registry_with_hash()
    for name in ("nope", "", "HASH", None):
        with pytest.raises(EmbeddingRefusal) as refusal:
            registry.resolve_embedding(name)  # type: ignore[arg-type]
        assert refusal.value.code is EmbeddingErrorCode.PROVIDER_UNKNOWN
    with pytest.raises(EmbeddingRefusal):
        registry.resolve_reranker("hash")


def test_resolve_returns_the_exact_registered_instance() -> None:
    registry = ProviderRegistry()
    embedding = HashEmbeddingProvider(dimension=DIMENSION)
    registry.register_embedding("hash", embedding)
    assert registry.resolve_embedding("hash") is embedding
    assert registry.embedding_names() == ("hash",)


def test_registering_the_same_name_twice_refuses_rather_than_overrides() -> None:
    """A silent override makes the winner depend on import order."""
    registry = _registry_with_hash()
    with pytest.raises(EmbeddingRefusal) as refusal:
        registry.register_embedding("hash", HashEmbeddingProvider(dimension=8))
    assert refusal.value.code is EmbeddingErrorCode.PROVIDER_DUPLICATE
    assert registry.resolve_embedding("hash").descriptor.dimension == DIMENSION


def test_a_network_provider_cannot_be_registered_in_phase_3() -> None:
    registry = ProviderRegistry()
    provider = DescriptorOnlyEmbedding(descriptor_with(requires_network=True))
    with pytest.raises(EmbeddingRefusal) as refusal:
        registry.register_embedding("cloud", provider)
    assert refusal.value.code is EmbeddingErrorCode.NETWORK_PROVIDER_DENIED
    assert registry.embedding_names() == ()


def test_a_nondeterministic_provider_cannot_be_registered() -> None:
    """A non-recomputable vector cannot back a recomputable index_revision."""
    registry = ProviderRegistry()
    provider = DescriptorOnlyEmbedding(descriptor_with(deterministic=False))
    with pytest.raises(EmbeddingRefusal) as refusal:
        registry.register_embedding("drifty", provider)
    assert refusal.value.code is EmbeddingErrorCode.NONDETERMINISTIC_PROVIDER_DENIED


def test_a_foreign_contract_version_cannot_be_registered() -> None:
    registry = ProviderRegistry()
    provider = DescriptorOnlyEmbedding(descriptor_with(contract_version="embedding/v9"))
    with pytest.raises(EmbeddingRefusal) as refusal:
        registry.register_embedding("future", provider)
    assert refusal.value.code is EmbeddingErrorCode.CONTRACT_VERSION_UNKNOWN


def test_a_provider_cannot_be_registered_into_the_wrong_slot() -> None:
    registry = ProviderRegistry()
    reranker_descriptor = descriptor_with(kind=ProviderKind.RERANKER, dimension=None)
    with pytest.raises(EmbeddingRefusal) as refusal:
        registry.register_reranker(
            "mislabelled", DescriptorOnlyReranker(descriptor_with())
        )
    assert refusal.value.code is EmbeddingErrorCode.PROVIDER_KIND_MISMATCH
    with pytest.raises(EmbeddingRefusal) as refusal:
        registry.register_embedding(
            "mislabelled", DescriptorOnlyEmbedding(reranker_descriptor)
        )
    assert refusal.value.code is EmbeddingErrorCode.PROVIDER_KIND_MISMATCH


def test_an_object_missing_the_interface_cannot_be_registered() -> None:
    registry = ProviderRegistry()
    with pytest.raises(EmbeddingRefusal) as refusal:
        registry.register_embedding("bare", object())
    assert refusal.value.code is EmbeddingErrorCode.PROVIDER_KIND_MISMATCH
    with pytest.raises(EmbeddingRefusal):
        # Structurally an embedding provider, so it must not pass as a reranker.
        registry.register_reranker("bare", HashEmbeddingProvider(dimension=DIMENSION))


def test_third_party_providers_register_through_the_same_door() -> None:
    registry = ProviderRegistry()
    registry.register_embedding("ordinal", OrdinalEmbedding(dimension=DIMENSION))
    registry.register_reranker("length", LengthReranker())
    assert registry.embedding_names() == ("ordinal",)
    assert registry.reranker_names() == ("length",)


def test_the_c4_builders_resolve_through_the_registry() -> None:
    stack = build_c4_deterministic_stack(
        dimension=DIMENSION,
        embedding_name="hash",
        reranker_name="cosine",
    )
    assert stack.embedding.descriptor.dimension == DIMENSION
    assert stack.embedding_revision == compute_provider_revision(
        stack.embedding.descriptor
    )
    assert stack.reranker_revision == compute_provider_revision(
        stack.reranker.descriptor
    )
    assert stack.embedding_revision != stack.reranker_revision

    registry = build_c4_deterministic_registry(
        dimension=DIMENSION,
        embedding_name="e",
        reranker_name="r",
    )
    assert registry.embedding_names() == ("e",)
    assert registry.reranker_names() == ("r",)
    with pytest.raises(EmbeddingRefusal):
        registry.resolve_embedding("hash")


def test_the_builders_take_no_positional_arguments() -> None:
    with pytest.raises(TypeError):
        build_c4_deterministic_stack(DIMENSION, "hash", "cosine")  # type: ignore[misc]
