"""The real semantic provider, against the frozen ``EmbeddingProvider``
invariants -- and against the pinned model weights themselves.

Needs the local semantic model cache (``CKP_SEMANTIC_MODEL_DIR``, populated
by ``scripts/fetch-semantic-model.sh``). Locally the suite skips when the
cache is absent or fails digest verification; in CI
``CKP_REQUIRE_SEMANTIC=1`` turns that skip into a failure so this path can
never silently stop running -- mirroring ``tests/test_index_qdrant.py``'s
``CKP_REQUIRE_QDRANT`` exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from ckp.embedding.errors import EmbeddingErrorCode, EmbeddingRefusal
from ckp.embedding.hashing import CosineReranker
from ckp.embedding.models import ProviderKind
from ckp.embedding.provider import EmbeddingProvider, RerankerProvider
from ckp.semantic.assets import resolve_semantic_assets
from ckp.semantic.errors import SemanticRefusal
from ckp.semantic.manifest import SEMANTIC_MODEL_DIMENSION
from ckp.semantic.provider import SEMANTIC_EMBEDDING_ID, SemanticEmbeddingProvider

_MODEL_DIR_RAW = os.environ.get("CKP_SEMANTIC_MODEL_DIR", "")
_REQUIRED = os.environ.get("CKP_REQUIRE_SEMANTIC") == "1"


def _assets_available() -> Path | None:
    if not _MODEL_DIR_RAW:
        return None
    candidate = Path(_MODEL_DIR_RAW)
    try:
        resolve_semantic_assets(model_dir=candidate)
    except SemanticRefusal:
        return None
    return candidate


_MODEL_DIR = _assets_available()
if _REQUIRED and _MODEL_DIR is None:
    pytest.fail(
        "CKP_REQUIRE_SEMANTIC=1 but no verified semantic model cache is "
        "reachable at CKP_SEMANTIC_MODEL_DIR"
    )

needs_semantic = pytest.mark.skipif(
    _MODEL_DIR is None,
    reason="no local semantic model cache reachable (CKP_SEMANTIC_MODEL_DIR)",
)

#: Golden digests over the packed float64 embed_query vectors, produced
#: against the pinned model in this repo's own evaluation session (two
#: separate interpreter processes, one with OMP_NUM_THREADS=4 forced, both
#: identical). A change here means either the pinned weights moved (a
#: manifest edit) or the pooling/normalization code changed -- never a
#: quiet edit either way.
GOLDEN = {
    "kettle temperature control": (
        "845bc2708a53931b8d644c92ad34b000580b2344bf281ecce1c7f382dab82fcc"
    ),
    "手沖 咖啡 水溫 控制": (
        "3ee3291db29966601fd389e6b368cd46a8250342a52f51761807251d602fabd7"
    ),
    "特教 個別化 教育 計畫": (
        "108033347dfe8623953910b1cf4bcd72143d767837d109fcc4665e3484be83e7"
    ),
    "個別化教育計畫要怎麼撰寫": (
        "0361cc5b435a109042bc41120191e25268e31f22402acaa0608000fc040c813e"
    ),
}

_GOLDEN_SCRIPT = """
import hashlib, json, os, struct, sys
from pathlib import Path
from ckp.semantic.provider import SemanticEmbeddingProvider

model_dir = Path(os.environ["CKP_SEMANTIC_MODEL_DIR"])
provider = SemanticEmbeddingProvider(model_dir=model_dir)
out = {}
for text in json.loads(sys.stdin.read()):
    values = provider.embed_query(text).values
    packed = struct.pack("<%dd" % len(values), *values)
    out[text] = hashlib.sha256(packed).hexdigest()
sys.stdout.write(json.dumps(out))
"""


def _digest(values: tuple[float, ...]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()


def _digests_in_subprocess(extra_env: dict[str, str]) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["CKP_SEMANTIC_MODEL_DIR"] = str(_MODEL_DIR)
    environment.update(extra_env)
    completed = subprocess.run(
        [sys.executable, "-c", _GOLDEN_SCRIPT],
        input=json.dumps(list(GOLDEN)),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


@needs_semantic
def test_vectors_are_identical_across_processes_and_thread_counts() -> None:
    """The oracle lives outside the unit, same reasoning as the hash
    provider's equivalent test: nothing inside one process proves the ONNX
    session is actually pinned to single-threaded execution. Two
    subprocesses, one of them with OMP_NUM_THREADS forced to 4, can."""
    first = _digests_in_subprocess({})
    second = _digests_in_subprocess({"OMP_NUM_THREADS": "4"})
    assert first == second
    assert first == GOLDEN


@needs_semantic
def test_the_in_process_provider_matches_the_frozen_golden_digests() -> None:
    provider = SemanticEmbeddingProvider(model_dir=_MODEL_DIR)
    for text, expected in GOLDEN.items():
        assert _digest(provider.embed_query(text).values) == expected


@needs_semantic
def test_query_and_document_paths_are_bit_identical() -> None:
    """Frozen ``EmbeddingProvider`` invariant #2. Also the reason this
    provider carries no query-side instruction prefix -- see the module
    docstring in ``ckp.semantic.provider``."""
    provider = SemanticEmbeddingProvider(model_dir=_MODEL_DIR)
    for text in GOLDEN:
        assert (
            provider.embed_query(text).values
            == provider.embed_documents([text]).vectors[0]
        )


@needs_semantic
def test_a_batch_preserves_length_and_order() -> None:
    provider = SemanticEmbeddingProvider(model_dir=_MODEL_DIR)
    texts = list(GOLDEN)
    batch = provider.embed_documents(texts)
    assert len(batch.vectors) == len(texts)
    for index, text in enumerate(texts):
        assert batch.vectors[index] == provider.embed_query(text).values
    assert provider.embed_documents([]).vectors == ()


@needs_semantic
def test_the_shipped_descriptor_is_honest() -> None:
    """RP3: ``semantic`` must reflect reality, not a hardcoded claim. This
    is checked against the *actually loaded* provider, backed by the
    pinned weights -- not a fixture that merely asserts the field's type."""
    descriptor = SemanticEmbeddingProvider(model_dir=_MODEL_DIR).descriptor
    assert descriptor.provider_id == SEMANTIC_EMBEDDING_ID
    assert descriptor.kind is ProviderKind.EMBEDDING
    assert descriptor.dimension == SEMANTIC_MODEL_DIMENSION
    assert descriptor.normalized is True
    assert descriptor.deterministic is True
    assert descriptor.requires_network is False
    assert descriptor.semantic is True


@needs_semantic
def test_related_chinese_phrases_are_closer_than_unrelated_ones() -> None:
    """A quality sanity check, not just a shape check: this is the whole
    point of issue #25 -- the hash baseline cannot tell these apart at all
    beyond token overlap, and these two share zero substring overlap."""
    provider = SemanticEmbeddingProvider(model_dir=_MODEL_DIR)
    related_a = provider.embed_query("特教 個別化 教育 計畫").values
    related_b = provider.embed_query("個別化教育計畫要怎麼撰寫").values
    unrelated = provider.embed_query("kettle temperature control").values

    def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=True))

    sim_related = cosine(related_a, related_b)
    sim_unrelated = cosine(related_a, unrelated)
    assert sim_related > sim_unrelated + 0.2, (sim_related, sim_unrelated)


@needs_semantic
def test_empty_and_non_text_inputs_fail_closed() -> None:
    provider = SemanticEmbeddingProvider(model_dir=_MODEL_DIR)
    for empty in ("", "   ", "\n\t"):
        with pytest.raises(EmbeddingRefusal) as refusal:
            provider.embed_query(empty)
        assert refusal.value.code is EmbeddingErrorCode.TEXT_EMPTY
    for bad in (None, 7, b"bytes"):
        with pytest.raises(EmbeddingRefusal) as refusal:
            provider.embed_query(bad)  # type: ignore[arg-type]
        assert refusal.value.code is EmbeddingErrorCode.TEXT_INVALID


@needs_semantic
def test_the_provider_satisfies_the_protocol_and_composes_with_the_reranker() -> None:
    """No new reranker was written for issue #25 -- ``CosineReranker`` from
    C4's ``hashing.py`` is reused unmodified, proof that swapping the
    embedding provider changes no caller, not even the one immediately
    downstream of it."""
    provider = SemanticEmbeddingProvider(model_dir=_MODEL_DIR)
    assert isinstance(provider, EmbeddingProvider)
    reranker = CosineReranker(embedder=provider)
    assert isinstance(reranker, RerankerProvider)
    assert reranker.descriptor.deterministic is True
    assert reranker.descriptor.requires_network is False


@needs_semantic
def test_offline_admission_still_holds_for_this_provider() -> None:
    """Same fence C4 has always enforced -- constructing this provider does
    not create a way around ``require_offline_provider``. Registration is
    exercised end to end via ``build_semantic_stack`` in
    ``tests/test_semantic_contract.py``; this test pins the descriptor
    flags the fence actually reads."""
    from ckp.embedding.provider import require_offline_provider

    provider = SemanticEmbeddingProvider(model_dir=_MODEL_DIR)
    require_offline_provider(provider.descriptor)  # must not raise
