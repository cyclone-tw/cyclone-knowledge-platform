"""The shipped offline providers, against the frozen interface invariants."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import struct
import subprocess
import sys

import pytest

from ckp.embedding.errors import EmbeddingErrorCode, EmbeddingRefusal
from ckp.embedding.hashing import (
    _TOKEN,
    COSINE_RERANKER_ID,
    HASH_EMBEDDING_ID,
    CosineReranker,
    HashEmbeddingProvider,
    tokenize,
)
from ckp.embedding.models import ProviderKind, RerankCandidate, require_text
from ckp.embedding.provider import EmbeddingProvider, RerankerProvider
from embedding_fixtures import (
    SAMPLE_TEXTS,
    DescriptorOnlyEmbedding,
    DescriptorOnlyReranker,
    NonCallableEmbedding,
    TwoFacedEmbedding,
    candidates_from,
    descriptor_with,
)

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


def test_the_tokenizer_reads_no_unicode_database() -> None:
    """``\\w``/``\\s``/``\\d`` and ``casefold`` change between Unicode releases.

    A character assigned in a later Unicode version would then tokenize on one
    interpreter and be ignored on another, so the same note would vectorize
    differently on Python 3.12 and 3.14 -- with nothing in the golden digests
    to notice, because they only cover characters assigned years ago.
    """
    for unicode_class in (r"\w", r"\W", r"\s", r"\S", r"\d", r"\D", r"\b"):
        assert unicode_class not in _TOKEN.pattern
    # Read the compiled names, not the source: the docstring has to be free
    # to say the word "casefold" while explaining why it is not used.
    names = set(tokenize.__code__.co_names)
    assert "casefold" not in names
    assert "lower" not in names
    # NFC is the one table it does consult, and only for normalization.
    assert names & {"unicodedata", "normalize"} == {"unicodedata", "normalize"}


def test_case_folding_is_ascii_only_and_declared_as_such() -> None:
    assert tokenize("KETTLE Brew") == ["kettle", "brew"]
    # Non-ASCII case is deliberately *not* folded: doing so would read the
    # interpreter's Unicode case tables. Both forms are still tokenized.
    assert tokenize("Ölkanne") == ["Ö", "lkanne"]
    assert tokenize("ölkanne") == ["ö", "lkanne"]


def test_cjk_punctuation_is_ignored_while_cjk_text_is_kept() -> None:
    assert tokenize("手沖，咖啡。「水溫」") == ["手", "沖", "咖", "啡", "水", "溫"]
    assert tokenize("　​。、（）") == []


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


def test_text_that_cannot_be_encoded_fails_closed_with_a_code() -> None:
    """A lone surrogate is a legal str that no provider can hash.

    Without the gate the first ``encode`` raises UnicodeEncodeError, so an
    unstable exception type escapes where a coded refusal was promised.
    """
    provider = HashEmbeddingProvider(dimension=DIMENSION)
    for unencodable in ("\ud800", "kettle\udfff", "\udc00\ud800"):
        with pytest.raises(EmbeddingRefusal) as refusal:
            provider.embed_query(unencodable)
        assert refusal.value.code is EmbeddingErrorCode.TEXT_INVALID
        with pytest.raises(EmbeddingRefusal):
            provider.embed_documents([unencodable])


def test_composed_and_decomposed_text_embed_identically() -> None:
    """macOS hands over NFD, almost everything else NFC.

    Without normalization the same note filed from two machines lands in
    different buckets and stops matching its own query.
    """
    provider = HashEmbeddingProvider(dimension=DIMENSION)
    for composed, decomposed in (
        ("caf\u00e9 brew", "cafe\u0301 brew"),
        ("\u9ad8\u9f61", "\u9ad8\u9f61"),
        ("\u30ac\u30c8", "\u30ab\u3099\u30c8"),
    ):
        assert provider.embed_query(composed).values == (
            provider.embed_query(decomposed).values
        )
    assert tokenize("cafe\u0301") == ["caf", "\u00e9"]


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


def test_a_cosine_reranker_refuses_an_offline_violating_embedder() -> None:
    """The fence has to hold where one provider is composed into another.

    Checking it only at registration would let a reranker wrap a cloud client
    and still register as offline: the registry sees the reranker's own
    descriptor, which says requires_network=False.
    """
    cases = {
        "requires_network": (
            {"requires_network": True},
            EmbeddingErrorCode.NETWORK_PROVIDER_DENIED,
        ),
        "deterministic": (
            {"deterministic": False},
            EmbeddingErrorCode.NONDETERMINISTIC_PROVIDER_DENIED,
        ),
        "contract_version": (
            {"contract_version": "embedding/v9"},
            EmbeddingErrorCode.CONTRACT_VERSION_UNKNOWN,
        ),
    }
    for label, (overrides, expected) in cases.items():
        embedder = DescriptorOnlyEmbedding(descriptor_with(**overrides))
        with pytest.raises(EmbeddingRefusal) as refusal:
            CosineReranker(embedder=embedder)
        assert refusal.value.code is expected, label


def test_admission_reads_the_descriptor_once_and_keeps_that_answer() -> None:
    """``descriptor`` is a property on a foreign object: each read is a fresh
    answer, and admission only vouches for the first one. A provider that
    answers admission with a clean descriptor and then reports
    ``requires_network=True`` must not be able to smuggle the second answer
    into the reranker's own descriptor -- that is the value the registry
    trusts."""
    two_faced = TwoFacedEmbedding(
        dimension=DIMENSION,
        later=descriptor_with(requires_network=True, deterministic=False),
    )
    reranker = CosineReranker(embedder=two_faced)
    assert reranker.descriptor.requires_network is False
    assert reranker.descriptor.deterministic is True


def test_a_cosine_reranker_refuses_an_embedder_that_is_not_one() -> None:
    """Taking an embedder is an admission point, not just a flag check."""
    not_an_embedder = DescriptorOnlyReranker(
        descriptor_with(kind=ProviderKind.RERANKER, dimension=None)
    )
    with pytest.raises(EmbeddingRefusal) as refusal:
        CosineReranker(embedder=not_an_embedder)
    assert refusal.value.code is EmbeddingErrorCode.PROVIDER_KIND_MISMATCH

    hollow = NonCallableEmbedding(descriptor_with())
    with pytest.raises(EmbeddingRefusal) as refusal:
        CosineReranker(embedder=hollow)
    assert refusal.value.code is EmbeddingErrorCode.PROVIDER_KIND_MISMATCH


def test_a_reranker_reports_the_flags_of_the_embedder_it_wraps() -> None:
    """Inherited, not asserted: a hardcoded pair would be a claim it cannot
    back -- and inherited from the *admitted* descriptor, never a re-read of
    the property, which is a fresh answer admission never saw."""
    source = inspect.getsource(CosineReranker)
    assert "deterministic=descriptor.deterministic" in source
    assert "requires_network=descriptor.requires_network" in source
    assert "embedder.descriptor.deterministic" not in source
    assert "embedder.descriptor.requires_network" not in source


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
