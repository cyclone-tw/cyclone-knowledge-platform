"""The shipped offline providers, against the frozen interface invariants."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys

import pytest

from ckp.embedding.errors import EmbeddingErrorCode, EmbeddingRefusal
from ckp.embedding.hashing import (
    COSINE_RERANKER_ID,
    HASH_EMBEDDING_ID,
    CosineReranker,
    HashEmbeddingProvider,
    tokenize,
)
from ckp.embedding.models import ProviderKind, RerankCandidate, require_text
from ckp.embedding.provider import EmbeddingProvider, RerankerProvider
from embedding_fixtures import SAMPLE_TEXTS, candidates_from

DIMENSION = 64

#: Golden digests over the packed float64 vectors. These pin the tokenizer,
#: the bucket function, and the normalization together. A change here is a
#: change to every stored vector, so it needs a ``PROVIDER_VERSION`` bump and
#: a reindex -- not a quiet edit.
GOLDEN = {
    "kettle temperature control": (
        "83597b7973bc92b2b3839db69218ed1d396b0c5f1029fa6f3dc090236097872c"
    ),
    "手沖 咖啡 水溫 控制": (
        "67d72b59a5d87529b7a81e6a8ee7751c5c27f507774a6a59d91f475241408912"
    ),
}

_GOLDEN_SCRIPT = """
import hashlib, json, struct, sys
from ckp.embedding.hashing import HashEmbeddingProvider

provider = HashEmbeddingProvider(dimension={dimension})
out = {{}}
for text in json.loads(sys.stdin.read()):
    values = provider.embed_query(text).values
    packed = struct.pack("<%dd" % len(values), *values)
    out[text] = hashlib.sha256(packed).hexdigest()
sys.stdout.write(json.dumps(out))
"""


def _digest(values: tuple[float, ...]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()


def _digests_in_subprocess(hash_seed: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = hash_seed
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", _GOLDEN_SCRIPT.format(dimension=DIMENSION)],
        input=json.dumps(list(GOLDEN)),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_vectors_are_identical_across_processes_and_hash_seeds() -> None:
    """The oracle lives outside the unit: two interpreters, two hash seeds.

    Nothing inside one process can tell you whether a provider reached for
    the built-in ``hash()``, which is salted per process. Two subprocesses
    can, and so can a digest frozen into this file.
    """
    first = _digests_in_subprocess("12345")
    second = _digests_in_subprocess("98765")
    assert first == second
    assert first == GOLDEN


def test_the_in_process_provider_matches_the_frozen_golden_digests() -> None:
    provider = HashEmbeddingProvider(dimension=DIMENSION)
    for text, expected in GOLDEN.items():
        assert _digest(provider.embed_query(text).values) == expected


def test_query_and_document_paths_are_bit_identical() -> None:
    """One path. Two would let a query stop matching its own documents."""
    provider = HashEmbeddingProvider(dimension=DIMENSION)
    for text in SAMPLE_TEXTS:
        assert (
            provider.embed_query(text).values
            == (provider.embed_documents([text]).vectors[0])
        )


def test_a_batch_preserves_length_and_order() -> None:
    provider = HashEmbeddingProvider(dimension=DIMENSION)
    batch = provider.embed_documents(list(SAMPLE_TEXTS))
    assert len(batch.vectors) == len(SAMPLE_TEXTS)
    for index, text in enumerate(SAMPLE_TEXTS):
        assert batch.vectors[index] == provider.embed_query(text).values
    assert provider.embed_documents([]).vectors == ()


def test_traditional_chinese_is_tokenized_per_character_not_dropped() -> None:
    assert tokenize("手沖 咖啡。") == [
        "手",
        "沖",
        "咖",
        "啡",
    ]
    provider = HashEmbeddingProvider(dimension=DIMENSION)
    coffee = provider.embed_query("手沖 咖啡 水溫")
    teaching = provider.embed_query("特教 個別化 計畫")
    assert coffee.values != teaching.values


def test_empty_untokenizable_and_non_text_inputs_fail_closed() -> None:
    provider = HashEmbeddingProvider(dimension=DIMENSION)
    for empty in ("", "   ", "\n\t"):
        with pytest.raises(EmbeddingRefusal) as refusal:
            provider.embed_query(empty)
        assert refusal.value.code is EmbeddingErrorCode.TEXT_EMPTY
    # Non-empty, but with nothing indexable in it. A zero vector here would
    # be an invented value (AGENTS.md §8).
    with pytest.raises(EmbeddingRefusal) as refusal:
        provider.embed_query("。。。!!!")
    assert refusal.value.code is EmbeddingErrorCode.TEXT_EMPTY
    for bad in (None, 7, b"bytes"):
        with pytest.raises(EmbeddingRefusal) as refusal:
            provider.embed_query(bad)  # type: ignore[arg-type]
        assert refusal.value.code is EmbeddingErrorCode.TEXT_INVALID


def test_a_bare_string_is_not_a_batch_of_documents() -> None:
    """``embed_documents("abc")`` would otherwise embed three characters."""
    provider = HashEmbeddingProvider(dimension=DIMENSION)
    with pytest.raises(EmbeddingRefusal) as refusal:
        provider.embed_documents("kettle")  # type: ignore[arg-type]
    assert refusal.value.code is EmbeddingErrorCode.BATCH_INVALID
    with pytest.raises(EmbeddingRefusal):
        provider.embed_documents(42)  # type: ignore[arg-type]


def test_require_text_is_the_single_input_gate() -> None:
    assert require_text("ok") == "ok"
    with pytest.raises(EmbeddingRefusal):
        require_text(" ")


def test_the_shipped_embedding_descriptor_is_honest() -> None:
    descriptor = HashEmbeddingProvider(dimension=DIMENSION).descriptor
    assert descriptor.provider_id == HASH_EMBEDDING_ID
    assert descriptor.kind is ProviderKind.EMBEDDING
    assert descriptor.dimension == DIMENSION
    assert descriptor.normalized is True
    assert descriptor.deterministic is True
    assert descriptor.requires_network is False
    # Hashed term overlap is not semantic similarity, and C5's benchmark must
    # not be handed a provider that claims otherwise.
    assert descriptor.semantic is False


def test_dimension_is_required_and_reaches_the_revision() -> None:
    with pytest.raises(TypeError):
        HashEmbeddingProvider()  # type: ignore[call-arg]
    small = HashEmbeddingProvider(dimension=16).descriptor
    large = HashEmbeddingProvider(dimension=DIMENSION).descriptor
    assert small.revision != large.revision


def test_the_shipped_providers_satisfy_the_protocols() -> None:
    provider = HashEmbeddingProvider(dimension=DIMENSION)
    assert isinstance(provider, EmbeddingProvider)
    assert isinstance(CosineReranker(embedder=provider), RerankerProvider)
    assert not isinstance(provider, RerankerProvider)


def _reranker() -> CosineReranker:
    return CosineReranker(embedder=HashEmbeddingProvider(dimension=DIMENSION))


def test_rerank_orders_by_score_and_truncates_to_top_k() -> None:
    reranker = _reranker()
    candidates = candidates_from(
        (
            "kettle temperature control for pour over",
            "個別化 教育 計畫",
            "temperature control",
        )
    )
    result = reranker.rerank("kettle temperature", candidates, top_k=2)
    assert [item.candidate_id for item in result.ranked] == ["note-00", "note-02"]
    assert [item.rank for item in result.ranked] == [0, 1]
    assert result.ranked[0].score > result.ranked[1].score
    assert result.descriptor.provider_id == COSINE_RERANKER_ID


def test_rerank_returns_at_most_the_candidates_it_was_given() -> None:
    reranker = _reranker()
    candidates = candidates_from(("kettle", "grinder"))
    assert len(reranker.rerank("kettle", candidates, top_k=99).ranked) == 2


def test_identical_texts_tie_break_on_candidate_id_not_input_order() -> None:
    reranker = _reranker()
    forward = (
        RerankCandidate(candidate_id="note-zz", text="kettle"),
        RerankCandidate(candidate_id="note-aa", text="kettle"),
    )
    reversed_input = tuple(reversed(forward))
    expected = ["note-aa", "note-zz"]
    for candidates in (forward, reversed_input):
        result = reranker.rerank("kettle", candidates, top_k=2)
        assert [item.candidate_id for item in result.ranked] == expected


def test_rerank_never_echoes_candidate_text() -> None:
    reranker = _reranker()
    secret = "kettle brew ratio unique-marker"
    candidates = (RerankCandidate(candidate_id="note-00", text=secret),)
    result = reranker.rerank("kettle", candidates, top_k=1)
    assert "unique-marker" not in result.model_dump_json()


def test_rerank_fails_closed_on_top_k_duplicates_and_bad_candidates() -> None:
    reranker = _reranker()
    candidates = candidates_from(("kettle", "grinder"))
    for bad_top_k in (0, -1, True, "2", None):
        with pytest.raises(EmbeddingRefusal) as refusal:
            reranker.rerank("kettle", candidates, top_k=bad_top_k)  # type: ignore[arg-type]
        assert refusal.value.code is EmbeddingErrorCode.TOP_K_INVALID
    duplicated = (
        RerankCandidate(candidate_id="note-00", text="kettle"),
        RerankCandidate(candidate_id="note-00", text="grinder"),
    )
    with pytest.raises(EmbeddingRefusal) as refusal:
        reranker.rerank("kettle", duplicated, top_k=2)
    assert refusal.value.code is EmbeddingErrorCode.CANDIDATE_DUPLICATE
    with pytest.raises(EmbeddingRefusal) as refusal:
        reranker.rerank("kettle", ["not-a-candidate"], top_k=1)  # type: ignore[list-item]
    assert refusal.value.code is EmbeddingErrorCode.BATCH_INVALID
    with pytest.raises(EmbeddingRefusal):
        reranker.rerank("kettle", "not-a-sequence", top_k=1)  # type: ignore[arg-type]


def test_top_k_is_keyword_only() -> None:
    reranker = _reranker()
    with pytest.raises(TypeError):
        reranker.rerank("kettle", candidates_from(("kettle",)), 1)  # type: ignore[misc]


def test_a_cosine_reranker_refuses_an_embedder_that_is_not_normalized() -> None:
    """Without unit vectors the dot product is not a cosine."""

    class Unnormalized(HashEmbeddingProvider):
        def __init__(self) -> None:
            super().__init__(dimension=DIMENSION)
            self._descriptor = self._descriptor.model_copy(update={"normalized": False})

    with pytest.raises(EmbeddingRefusal) as refusal:
        CosineReranker(embedder=Unnormalized())
    assert refusal.value.code is EmbeddingErrorCode.DESCRIPTOR_INVALID


def test_embedder_is_required_and_keyword_only() -> None:
    with pytest.raises(TypeError):
        CosineReranker()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        CosineReranker(HashEmbeddingProvider(dimension=DIMENSION))  # type: ignore[misc]
