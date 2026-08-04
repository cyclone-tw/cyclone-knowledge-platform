"""The C4 acceptance criterion: swapping the provider does not touch the caller."""

from __future__ import annotations

import inspect

from ckp.embedding import hashing, models, provider, registry
from ckp.embedding.hashing import (
    COSINE_RERANKER_ID,
    HASH_EMBEDDING_ID,
    CosineReranker,
    HashEmbeddingProvider,
)
from ckp.embedding.registry import ProviderRegistry
from embedding_fixtures import (
    SAMPLE_TEXTS,
    LengthReranker,
    OrdinalEmbedding,
    candidates_from,
    retrieve_ids,
)

DIMENSION = 32
QUERY = "kettle temperature"


def _pairs() -> list[tuple[str, object, object]]:
    shipped = HashEmbeddingProvider(dimension=DIMENSION)
    third_party = OrdinalEmbedding(dimension=DIMENSION)
    return [
        ("shipped", shipped, CosineReranker(embedder=shipped)),
        ("third-party", third_party, LengthReranker()),
        # Mixed on purpose: a reranker must not be coupled to one embedder.
        ("mixed", third_party, CosineReranker(embedder=third_party)),
        ("crossed", shipped, LengthReranker()),
    ]


def test_the_same_caller_runs_against_every_provider_pair() -> None:
    candidates = candidates_from(SAMPLE_TEXTS)
    expected_ids = {candidate.candidate_id for candidate in candidates}
    for label, embedding, reranker in _pairs():
        ranked = retrieve_ids(
            embedding,
            reranker,
            query=QUERY,
            candidates=candidates,
            top_k=3,
        )
        assert len(ranked) == 3, label
        assert len(set(ranked)) == 3, label
        assert set(ranked) <= expected_ids, label


def test_different_providers_really_do_produce_different_answers() -> None:
    """Otherwise the test above would pass on a caller that ignores them."""
    candidates = candidates_from(SAMPLE_TEXTS)
    answers = {
        label: retrieve_ids(
            embedding, reranker, query=QUERY, candidates=candidates, top_k=4
        )
        for label, embedding, reranker in _pairs()
    }
    assert answers["shipped"] != answers["third-party"]


def test_the_caller_never_names_a_provider() -> None:
    """Provider neutrality, read straight off the caller's source."""
    source = inspect.getsource(retrieve_ids)
    for forbidden in (
        HASH_EMBEDDING_ID,
        COSINE_RERANKER_ID,
        "ordinal-test",
        "length-test",
        "HashEmbeddingProvider",
        "CosineReranker",
        "hashing",
    ):
        assert forbidden not in source


def test_the_abstraction_layer_does_not_know_any_concrete_provider() -> None:
    """models / provider / registry must be swappable-provider agnostic.

    If a provider id leaks into them, "neutral" has quietly become "neutral
    except for the one we shipped".
    """
    for module in (models, provider, registry):
        source = inspect.getsource(module)
        for forbidden in (
            HASH_EMBEDDING_ID,
            COSINE_RERANKER_ID,
            "HashEmbeddingProvider",
            "CosineReranker",
            "ckp.embedding.hashing",
        ):
            assert forbidden not in source, module.__name__


def test_only_the_implementation_module_defines_the_shipped_ids() -> None:
    assert HASH_EMBEDDING_ID in inspect.getsource(hashing)


def test_a_registry_can_serve_both_worlds_under_caller_chosen_names() -> None:
    store = ProviderRegistry()
    shipped = HashEmbeddingProvider(dimension=DIMENSION)
    store.register_embedding("primary", shipped)
    store.register_embedding("shadow", OrdinalEmbedding(dimension=DIMENSION))
    store.register_reranker("primary", CosineReranker(embedder=shipped))
    store.register_reranker("shadow", LengthReranker())

    candidates = candidates_from(SAMPLE_TEXTS)
    for name in ("primary", "shadow"):
        ranked = retrieve_ids(
            store.resolve_embedding(name),
            store.resolve_reranker(name),
            query=QUERY,
            candidates=candidates,
            top_k=2,
        )
        assert len(ranked) == 2
